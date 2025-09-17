import torch
from typing import List
from ..cache.interfaces.expert_cache import IExpertCacheManager
from src.common.types import ExpertKey, ExpertParamType


class LazyExpertTensor:
    """
    Lazy-loading wrapper for expert weight tensors using expert cache.

    Instead of directly loading from checkpoint files, this class uses
    the expert caching system to efficiently manage expert weights across
    different memory tiers (VRAM, RAM, DISK).
    """

    def __init__(
        self,
        expert_cache_manager: IExpertCacheManager,
        layer_idx: int,
        param_type: ExpertParamType,
        expected_shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Initialize lazy expert tensor with cache-based loading.

        Args:
            expert_cache: Expert cache instance for loading weights
            layer_idx: Layer index for parameter naming
            param_type: Parameter type enum (ExpertParamType.MLP1_WEIGHT, etc.)
            expected_shape: Expected full tensor shape (num_experts, ...)
            dtype: Target data type for loaded tensors
            device: Target device for loaded tensors
        """
        self.expert_cache_manager = expert_cache_manager
        self.layer_idx = layer_idx
        self.param_type = param_type
        self.expected_shape = expected_shape
        self.dtype = dtype
        self.device = device
        self.num_experts = expected_shape[0]

        # Pre-compute expert keys for all experts in this layer/param
        self._expert_keys: List[ExpertKey] = []
        for expert_idx in range(self.num_experts):
            expert_key = ExpertKey(
                layer_idx=layer_idx,
                expert_id=expert_idx,  # Use expert_id instead of expert_idx
                param_type=param_type,  # Use enum directly for ExpertKey
            )
            self._expert_keys.append(expert_key)

    def load_experts(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Load specific experts using the expert cache system.

        Args:
            expert_indices: Tensor of expert indices, shape (batch_size, experts_per_token)

        Returns:
            Selected expert weights, shape (batch_size, experts_per_token, ...)
        """
        # Simple validation - fail fast
        assert (
            expert_indices.dim() == 2
        ), f"Expected 2D tensor, got {expert_indices.dim()}D"
        batch_size, experts_per_token = expert_indices.shape

        # Linear collection of expert keys (no optimization)
        expert_keys: List[ExpertKey] = []
        for b in range(batch_size):
            for e in range(experts_per_token):
                expert_id = int(expert_indices[b, e].item())
                expert_keys.append(self._expert_keys[expert_id])

        # Batch load experts
        experts = self.expert_cache_manager.get_batch(expert_keys)

        # Direct result construction with null check
        expert_tensors = []
        for expert in experts:
            if expert.data_vram is None:
                raise RuntimeError(f"Expert data is None after loading")
            expert_tensors.append(expert.data_vram)

        result = torch.stack(expert_tensors).view(
            batch_size, experts_per_token, *self.expected_shape[1:]
        )

        return result
