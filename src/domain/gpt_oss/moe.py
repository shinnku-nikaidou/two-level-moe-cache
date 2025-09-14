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
from ...boilerplate.gpt_oss.weights import Checkpoint


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
            param_name: Parameter name in checkpoint (e.g., "block.0.mlp.mlp1_weight")
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
        
        # Use Checkpoint class for proper MXFP4 handling
        self.checkpoint = Checkpoint(checkpoint_path, torch.device("cpu"))  # Load to CPU first
    
    def load_experts(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Load specific experts from checkpoint using Checkpoint class.
        
        Args:
            expert_indices: Tensor of expert indices to load, shape (batch_size, experts_per_token)
            
        Returns:
            Selected expert weights, shape (batch_size, experts_per_token, ...)
        """
        # Load full tensor from checkpoint (handles MXFP4 automatically)
        full_tensor = self.checkpoint.get(self.param_name)
        
        # Slice only the required experts
        # expert_indices should already be on CPU
        selected_tensor = full_tensor[expert_indices, ...]
        
        # Move to target device and convert dtype if needed
        if selected_tensor.device != self.device or selected_tensor.dtype != self.dtype:
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
            device: Target device for computations (None for auto-detection)
        """
        super().__init__()
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.layer_idx = layer_idx
        self.device = device  # Will be set on first forward pass if None
        
        # MoE configuration
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Non-expert components (device will be set during forward if needed)
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, 
            device=device, dtype=torch.bfloat16, bias=False
        )
        
        # Lazy expert tensors will be initialized on first forward pass
        self._lazy_tensors_initialized = False
    
    def _ensure_device_and_tensors(self, input_device: torch.device):
        """Ensure device is set and lazy tensors are initialized."""
        if self.device is None:
            self.device = input_device
            # Move components to detected device
            self.norm = self.norm.to(input_device)
            self.gate = self.gate.to(input_device)
        
        if not self._lazy_tensors_initialized:
            intermediate_size = self.config.intermediate_size // self.world_size
            self._init_lazy_expert_tensors(intermediate_size)
            self._lazy_tensors_initialized = True
    
    def _init_lazy_expert_tensors(self, intermediate_size: int):
        """Initialize lazy loading tensors for expert weights."""
        # Ensure device is set
        assert self.device is not None, "Device must be set before initializing lazy tensors"
        
        # Use actual parameter names from the original checkpoint
        # MLP1 (gate_up) tensors
        self.mlp1_weight = LazyExpertTensor(
            checkpoint_path=self.checkpoint_path,
            param_name=f"block.{self.layer_idx}.mlp.mlp1_weight.blocks",
            expected_shape=(self.num_experts, intermediate_size * 2, self.config.hidden_size),
            dtype=torch.bfloat16,
            device=self.device
        )
        
        self.mlp1_bias = LazyExpertTensor(
            checkpoint_path=self.checkpoint_path,
            param_name=f"block.{self.layer_idx}.mlp.mlp1_bias", 
            expected_shape=(self.num_experts, intermediate_size * 2),
            dtype=torch.bfloat16,
            device=self.device
        )
        
        # MLP2 (down) tensors  
        self.mlp2_weight = LazyExpertTensor(
            checkpoint_path=self.checkpoint_path,
            param_name=f"block.{self.layer_idx}.mlp.mlp2_weight.blocks",
            expected_shape=(self.num_experts, self.config.hidden_size, intermediate_size),
            dtype=torch.bfloat16,
            device=self.device
        )
        
        self.mlp2_bias = LazyExpertTensor(
            checkpoint_path=self.checkpoint_path,
            param_name=f"block.{self.layer_idx}.mlp.mlp2_bias",
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
        # Auto-detect device from input and initialize if needed
        self._ensure_device_and_tensors(x.device)
        
        # Normalize input
        t = self.norm(x)
        
        # Router computation
        g = self.gate(t) 
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices
        
        try:
            # Move expert_indices to CPU for safetensor indexing
            expert_indices_cpu = expert_indices.cpu()
            
            # Load only the selected expert weights
            mlp1_weight = self.mlp1_weight.load_experts(expert_indices_cpu)
            mlp1_bias = self.mlp1_bias.load_experts(expert_indices_cpu)
            
            # MLP1 computation with SwiGLU activation
            t_mlp1 = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
            t_activated = swiglu(t_mlp1, limit=self.swiglu_limit)
            
            # Load MLP2 weights
            mlp2_weight = self.mlp2_weight.load_experts(expert_indices_cpu)  
            mlp2_bias = self.mlp2_bias.load_experts(expert_indices_cpu)
            
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
            if self.device and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Residual connection
        return x + t_output