"""
Mixture of Experts (MoE) implementations with expert cache integration.

This module provides memory-efficient MoE components that use the expert
caching system for automatic memory management across VRAM, RAM, and DISK.
"""

import torch
import torch.distributed as dist

from ...boilerplate.gpt_oss.model import ModelConfig, swiglu, RMSNorm
from ..cache.interfaces.expert_cache import IExpertCache
from .lazy_tensor import LazyExpertTensor


class LazyMLPBlock(torch.nn.Module):
    """
    Memory-efficient MLP block with expert cache integration.

    This block uses the expert caching system for automatic memory management
    across VRAM, RAM, and DISK tiers, eliminating manual memory cleanup.
    """

    def __init__(
        self,
        config: ModelConfig,
        expert_cache: IExpertCache,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        """
        Initialize lazy MLP block with expert cache integration.

        Args:
            config: Model configuration
            expert_cache: Expert cache instance for loading weights
            layer_idx: Layer index for parameter naming
            device: Target device for computations (None for auto-detection)
        """
        super().__init__()
        self.config = config
        self.expert_cache = expert_cache
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
            config.hidden_size,
            config.num_experts,
            device=device,
            dtype=torch.bfloat16,
            bias=False,
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
        """Initialize lazy loading tensors for expert weights using expert cache."""
        # Ensure device is set
        assert (
            self.device is not None
        ), "Device must be set before initializing lazy tensors"

        # Create LazyExpertTensor instances using expert cache
        self.mlp1_weight = LazyExpertTensor(
            expert_cache=self.expert_cache,
            layer_idx=self.layer_idx,
            param_type="mlp1_weight",
            expected_shape=(
                self.num_experts,
                intermediate_size * 2,
                self.config.hidden_size,
            ),
            dtype=torch.bfloat16,
            device=self.device,
        )

        self.mlp1_bias = LazyExpertTensor(
            expert_cache=self.expert_cache,
            layer_idx=self.layer_idx,
            param_type="mlp1_bias",
            expected_shape=(self.num_experts, intermediate_size * 2),
            dtype=torch.bfloat16,
            device=self.device,
        )

        # MLP2 (down) tensors
        self.mlp2_weight = LazyExpertTensor(
            expert_cache=self.expert_cache,
            layer_idx=self.layer_idx,
            param_type="mlp2_weight",
            expected_shape=(
                self.num_experts,
                self.config.hidden_size,
                intermediate_size,
            ),
            dtype=torch.bfloat16,
            device=self.device,
        )

        self.mlp2_bias = LazyExpertTensor(
            expert_cache=self.expert_cache,
            layer_idx=self.layer_idx,
            param_type="mlp2_bias",
            expected_shape=(self.num_experts, self.config.hidden_size),
            dtype=torch.bfloat16,
            device=self.device,
        )

    def _load_gate_weights_if_needed(self):
        """Load gate weights if not already loaded (for initialization)."""
        # This would be called during model loading to initialize gate weights
        # For now, assume gate weights are loaded normally during model initialization
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic expert memory management.

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

        # Load expert weights using cache system (automatic memory management)
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

        # Residual connection - no manual memory management needed!
        return x + t_output
