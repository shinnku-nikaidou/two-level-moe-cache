"""
Mixture of Experts (MoE) implementations with expert cache integration.

This module provides memory-efficient MoE components that use the expert
caching system for automatic memory management across VRAM, RAM, and DISK.
"""

import torch
import torch.distributed as dist

from ...boilerplate.gpt_oss.model import ModelConfig, swiglu, RMSNorm
from ..cache.interfaces.expert_cache import IExpertCacheManager
from ..cache.entities.types import ExpertKey, ExpertParamType
from src.config import TORCH_VRAM_DEVICE
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
        expert_cache: IExpertCacheManager,
        layer_idx: int,
    ):
        """
        Initialize lazy MLP block with expert cache integration.

        Args:
            config: Model configuration
            expert_cache: Expert cache instance for loading weights
            layer_idx: Layer index for parameter naming
        """
        super().__init__()
        self.config = config
        self.expert_cache = expert_cache
        self.layer_idx = layer_idx

        # MoE configuration
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Non-expert components using global device config
        self.norm = RMSNorm(config.hidden_size, device=TORCH_VRAM_DEVICE)
        self.gate = torch.nn.Linear(
            config.hidden_size,
            config.num_experts,
            device=TORCH_VRAM_DEVICE,
            dtype=torch.bfloat16,
            bias=False,
        )

        # Initialize expert tensors directly
        intermediate_size = self.config.intermediate_size // self.world_size

        self.mlp1_weight = LazyExpertTensor(
            expert_cache=self.expert_cache,
            layer_idx=self.layer_idx,
            param_type=ExpertParamType.MLP1_WEIGHT,
            expected_shape=(
                self.num_experts,
                intermediate_size * 2,
                self.config.hidden_size,
            ),
            dtype=torch.bfloat16,
            device=TORCH_VRAM_DEVICE,
        )

        self.mlp1_bias = LazyExpertTensor(
            expert_cache=self.expert_cache,
            layer_idx=self.layer_idx,
            param_type=ExpertParamType.MLP1_BIAS,
            expected_shape=(self.num_experts, intermediate_size * 2),
            dtype=torch.bfloat16,
            device=TORCH_VRAM_DEVICE,
        )

        # MLP2 (down) tensors
        self.mlp2_weight = LazyExpertTensor(
            expert_cache=self.expert_cache,
            layer_idx=self.layer_idx,
            param_type=ExpertParamType.MLP2_WEIGHT,
            expected_shape=(
                self.num_experts,
                self.config.hidden_size,
                intermediate_size,
            ),
            dtype=torch.bfloat16,
            device=TORCH_VRAM_DEVICE,
        )

        self.mlp2_bias = LazyExpertTensor(
            expert_cache=self.expert_cache,
            layer_idx=self.layer_idx,
            param_type=ExpertParamType.MLP2_BIAS,
            expected_shape=(self.num_experts, self.config.hidden_size),
            dtype=torch.bfloat16,
            device=TORCH_VRAM_DEVICE,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic expert memory management.

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
