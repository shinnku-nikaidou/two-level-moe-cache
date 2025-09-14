"""
Mixture of Experts (MoE) implementations with on-demand weight loading.

This module provides memory-efficient MoE components that load expert weights
only when needed, significantly reducing GPU memory usage.
"""

import os
import torch
import torch.distributed as dist
from safetensors import safe_open

from ...boilerplate.gpt_oss.model import ModelConfig, swiglu, RMSNorm


class LazyExpertTensor:
    """
    Lazy-loading wrapper for expert weight tensors.
    
    Stores metadata about expert weights and loads only specific experts
    from safetensor files when requested, rather than keeping all experts
    in memory.
    """
    
    def __init__(self, checkpoint_path: str, param_name: str, 
                 expected_shape: tuple, dtype: torch.dtype, device: torch.device):
        """
        Initialize lazy expert tensor.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
            param_name: Parameter name in safetensor files (e.g., "model.layers.0.mlp.experts.gate_up_proj_blocks")
            expected_shape: Expected full tensor shape (num_experts, ...)
            dtype: Target data type for loaded tensors
            device: Target device for loaded tensors
        """
        self.checkpoint_path = checkpoint_path
        self.param_name = param_name
        self.expected_shape = expected_shape
        self.dtype = dtype
        self.device = device
        self.num_experts = expected_shape[0]
        
        # Build mapping from tensor names to safetensor files
        self._build_file_mapping()
    
    def _build_file_mapping(self):
        """Build mapping from parameter names to safetensor file paths."""
        self.tensor_to_file = {}
        
        # Read safetensor index to find which file contains our parameter
        index_path = os.path.join(self.checkpoint_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            import json
            with open(index_path, 'r') as f:
                index = json.load(f)
                weight_map = index.get('weight_map', {})
                if self.param_name in weight_map:
                    filename = weight_map[self.param_name]
                    self.tensor_to_file[self.param_name] = os.path.join(self.checkpoint_path, filename)
        
        # Fallback: search all safetensor files
        if self.param_name not in self.tensor_to_file:
            for fname in os.listdir(self.checkpoint_path):
                if fname.endswith('.safetensors'):
                    filepath = os.path.join(self.checkpoint_path, fname)
                    with safe_open(filepath, framework="pt", device="cpu") as f:
                        if self.param_name in f.keys():
                            self.tensor_to_file[self.param_name] = filepath
                            break
    
    def load_experts(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Load specific experts from safetensor files.
        
        Args:
            expert_indices: Tensor of expert indices to load, shape (batch_size, experts_per_token)
            
        Returns:
            Selected expert weights, shape (batch_size, experts_per_token, ...)
        """
        if self.param_name not in self.tensor_to_file:
            raise FileNotFoundError(f"Parameter {self.param_name} not found in checkpoint files")
        
        file_path = self.tensor_to_file[self.param_name]
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            # Load full tensor to CPU first
            full_tensor = f.get_tensor(self.param_name)
            
            # Slice only the required experts
            # expert_indices shape: (batch_size, experts_per_token)
            # Result shape: (batch_size, experts_per_token, ...)
            selected_tensor = full_tensor[expert_indices, ...]
            
            # Move to target device and convert dtype
            selected_tensor = selected_tensor.to(device=self.device, dtype=self.dtype)
        
        return selected_tensor


class LazyMLPBlock(torch.nn.Module):
    """
    Memory-efficient MLP block with on-demand expert weight loading.
    
    This block loads only the expert weights selected by the router,
    significantly reducing GPU memory usage compared to pre-loading
    all expert weights.
    """
    
    def __init__(self, config: ModelConfig, checkpoint_path: str,
                 layer_idx: int, device: torch.device | None = None):
        """
        Initialize lazy MLP block.
        
        Args:
            config: Model configuration
            checkpoint_path: Path to checkpoint directory
            layer_idx: Layer index for parameter naming
            device: Target device for computations
        """
        super().__init__()
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.layer_idx = layer_idx
        self.device = device or torch.device('cuda')
        
        # MoE configuration
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Non-expert components (loaded normally)
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, 
            device=device, dtype=torch.bfloat16
        )
        
        # Lazy expert tensors (not loaded until needed)
        intermediate_size = config.intermediate_size // self.world_size
        self._init_lazy_expert_tensors(intermediate_size)
    
    def _init_lazy_expert_tensors(self, intermediate_size: int):
        """Initialize lazy loading tensors for expert weights."""
        # MLP1 (gate_up) tensors
        self.mlp1_weight = LazyExpertTensor(
            checkpoint_path=self.checkpoint_path,
            param_name=f"model.layers.{self.layer_idx}.mlp.experts.gate_up_proj_blocks",
            expected_shape=(self.num_experts, intermediate_size * 2, self.config.hidden_size),
            dtype=torch.bfloat16,
            device=self.device
        )
        
        self.mlp1_bias = LazyExpertTensor(
            checkpoint_path=self.checkpoint_path,
            param_name=f"model.layers.{self.layer_idx}.mlp.experts.gate_up_proj_bias", 
            expected_shape=(self.num_experts, intermediate_size * 2),
            dtype=torch.bfloat16,
            device=self.device
        )
        
        # MLP2 (down) tensors  
        self.mlp2_weight = LazyExpertTensor(
            checkpoint_path=self.checkpoint_path,
            param_name=f"model.layers.{self.layer_idx}.mlp.experts.down_proj_blocks",
            expected_shape=(self.num_experts, self.config.hidden_size, intermediate_size),
            dtype=torch.bfloat16,
            device=self.device
        )
        
        self.mlp2_bias = LazyExpertTensor(
            checkpoint_path=self.checkpoint_path,
            param_name=f"model.layers.{self.layer_idx}.mlp.experts.down_proj_bias",
            expected_shape=(self.num_experts, self.config.hidden_size),
            dtype=torch.bfloat16,
            device=self.device
        )
    
    def _load_gate_weights_if_needed(self):
        """Load gate weights if not already loaded (for initialization)."""
        # This would be called during model loading to initialize gate weights
        # For now, assume gate weights are loaded normally during model initialization
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with on-demand expert weight loading.
        
        Args:
            x: Input tensor, shape (sequence_length, hidden_size)
            
        Returns:
            Output tensor after MoE computation
        """
        # Normalize input
        t = self.norm(x)
        
        # Router computation
        g = self.gate(t) 
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices
        
        try:
            # Load only the selected expert weights
            mlp1_weight = self.mlp1_weight.load_experts(expert_indices)
            mlp1_bias = self.mlp1_bias.load_experts(expert_indices)
            
            # MLP1 computation with SwiGLU activation
            t_mlp1 = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
            t_activated = swiglu(t_mlp1, limit=self.swiglu_limit)
            
            # Load MLP2 weights
            mlp2_weight = self.mlp2_weight.load_experts(expert_indices)  
            mlp2_bias = self.mlp2_bias.load_experts(expert_indices)
            
            # MLP2 computation
            t_mlp2 = torch.einsum("beck,bek->bec", mlp2_weight, t_activated)
            
            # All-reduce for distributed training
            if self.world_size > 1:
                dist.all_reduce(t_mlp2, op=dist.ReduceOp.SUM)
            
            t_mlp2 += mlp2_bias
            
            # Weighted sum of expert outputs
            t_output = torch.einsum("bec,be->bc", t_mlp2, expert_weights)
            
        finally:
            # Explicit memory cleanup to free GPU memory
            if 'mlp1_weight' in locals():
                del mlp1_weight # pyright: ignore[reportPossiblyUnboundVariable]
            if 'mlp1_bias' in locals(): 
                del mlp1_bias # pyright: ignore[reportPossiblyUnboundVariable]
            if 'mlp2_weight' in locals():
                del mlp2_weight # pyright: ignore[reportPossiblyUnboundVariable]
            if 'mlp2_bias' in locals():
                del mlp2_bias  # pyright: ignore[reportPossiblyUnboundVariable]
            if 't_mlp1' in locals():
                del t_mlp1 # pyright: ignore[reportPossiblyUnboundVariable]
            if 't_activated' in locals():
                del t_activated # pyright: ignore[reportPossiblyUnboundVariable]
            if 't_mlp2' in locals():
                del t_mlp2 # pyright: ignore[reportPossiblyUnboundVariable]
            
            # Force GPU memory cleanup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Residual connection
        return x + t_output