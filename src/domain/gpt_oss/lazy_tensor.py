"""
Lazy loading tensor implementation for expert weights.

This module provides L        # Load experts from cache in batch
        expert_dict: Dict[ExpertKey, Expert] = self.expert_cache.get_batch(required_keys)

        # Extract tensors and build index mapping
        expert_tensors = {}
        for key, expert in expert_dict.items():
            if expert.data is None:
                raise RuntimeError(f"Expert {key} data is None after loading")
            expert_tensors[key.expert_id] = expert.data  # Use expert_id instead of expert_idxTensor that uses expert caching system
instead of direct checkpoint access for memory-efficient expert loading.
"""

import torch
from typing import Dict, List, Any
from ..cache.interfaces.expert_cache import IExpertCache
from ..cache.entities.expert import Expert
from ..cache.entities.types import ExpertKey, MemoryTier


class LazyExpertTensor:
    """
    Lazy-loading wrapper for expert weight tensors using expert cache.

    Instead of directly loading from checkpoint files, this class uses
    the expert caching system to efficiently manage expert weights across
    different memory tiers (VRAM, RAM, DISK).
    """

    def __init__(
        self,
        expert_cache: IExpertCache,
        layer_idx: int,
        param_type: str,  # 'mlp1_weight', 'mlp1_bias', 'mlp2_weight', 'mlp2_bias'
        expected_shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Initialize lazy expert tensor with cache-based loading.

        Args:
            expert_cache: Expert cache instance for loading weights
            layer_idx: Layer index for parameter naming
            param_type: Parameter type ('mlp1_weight', 'mlp1_bias', etc.)
            expected_shape: Expected full tensor shape (num_experts, ...)
            dtype: Target data type for loaded tensors
            device: Target device for loaded tensors
        """
        self.expert_cache = expert_cache
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
                param_type=param_type,
            )
            self._expert_keys.append(expert_key)

    def load_experts(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Load specific experts using the expert cache system.

        Args:
            expert_indices: Tensor of expert indices to load, shape (batch_size, experts_per_token)

        Returns:
            Selected expert weights, shape (batch_size, experts_per_token, ...)
        """
        # Convert tensor indices to list for processing
        if expert_indices.dim() == 1:
            # Single batch dimension
            indices_list = expert_indices.cpu().tolist()
        else:
            # Multi-batch dimension: flatten and get unique indices
            indices_list = expert_indices.flatten().unique().cpu().tolist()

        # Get expert keys for the requested indices
        required_keys = [self._expert_keys[idx] for idx in indices_list]

        # Load experts from cache in batch
        expert_dict: Dict[ExpertKey, Expert] = self.expert_cache.get_batch(
            required_keys
        )

        # Extract tensors and build index mapping
        expert_tensors = {}
        for key, expert in expert_dict.items():
            if expert.data is None:
                raise RuntimeError(f"Expert {key} data is None after loading")
            expert_tensors[key.expert_id] = expert.data

        # Build result tensor by indexing
        if expert_indices.dim() == 1:
            # Simple 1D case
            result_tensors = []
            for idx in expert_indices.cpu().tolist():
                tensor = expert_tensors[idx]
                # Ensure correct device and dtype
                tensor = tensor.to(device=self.device, dtype=self.dtype)
                result_tensors.append(tensor)

            result = torch.stack(result_tensors, dim=0)
        else:
            # Multi-dimensional case: preserve original shape
            batch_size, experts_per_token = expert_indices.shape
            result_shape = (batch_size, experts_per_token) + self.expected_shape[1:]
            result = torch.empty(result_shape, device=self.device, dtype=self.dtype)

            for b in range(batch_size):
                for e in range(experts_per_token):
                    expert_idx = expert_indices[b, e].item()
                    tensor = expert_tensors[expert_idx]
                    tensor = tensor.to(device=self.device, dtype=self.dtype)
                    result[b, e] = tensor

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for this tensor.

        Returns:
            Dictionary with cache performance metrics
        """
        return self.expert_cache.get_cache_stats()
