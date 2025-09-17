"""
Two-Tire Watermark expert cache manager.

This module provides a Python wrapper around the Rust-based TwoTireWmExpertCacheManager
for watermark-based expert caching with dual-tier memory management.
"""

from typing import List
from ..cache.interfaces.expert_cache import IExpertCacheManager
from ..cache.entities.expert import Expert
from ..cache.entities.types import ExpertKey
from .. import ModelType
from .utils import rust_model_type
from rust_core import TwoTireWmExpertCacheManager as RustTwoTireWmExpertCacheManager


class TwoTireWmExpertCacheManager(IExpertCacheManager):
    """
    Two-Tier Watermark-based expert cache manager.

    This implementation uses the watermark algorithm for dual-tier caching
    with VRAM and RAM memory tiers, managing experts based on benefit density
    and adaptive watermark thresholds.

    Key Features:
    - Pure watermark algorithm implementation in Rust
    - Receives fused predictions from external policy components
    - Adaptive dual-tier memory management
    - Benefit density-based caching decisions
    """

    def __init__(
        self,
        model_type: ModelType,
        vram_capacity_mb: int = 5120,
        ram_capacity_mb: int = 20480,
    ):
        """
        Initialize two-tier watermark cache manager.

        Args:
            model_type: Type of model for configuration
            vram_capacity_mb: VRAM capacity limit in MB
            ram_capacity_mb: RAM capacity limit in MB
        """
        super().__init__(model_type)

        self.layer_idx_now = 0

        # Initialize Rust implementation (simplified interface)
        # Note: Using positional args to match Rust #[new] signature:
        # (model_type, total_layers, vram_capacity, ram_capacity)
        self._rust_cache = RustTwoTireWmExpertCacheManager(
            rust_model_type(model_type),
            self._config.num_hidden_layers,
            vram_capacity_mb * 1024 * 1024,
            ram_capacity_mb * 1024 * 1024,
        )

    def update_activations(self, activated_experts: List[int]) -> None:
        self._rust_cache.update_activations(activated_experts)

    def get(self, key: ExpertKey) -> Expert:
        expert = self._get_expert(key)
        return expert

    def get_batch(self, keys: List[ExpertKey]) -> List[Expert]:
        if not keys:
            return []

        # Assert all keys are from the same layer
        first_layer = keys[0].layer_idx
        assert all(
            key.layer_idx == first_layer for key in keys
        ), f"All keys must be from the same layer, but got layers: {[key.layer_idx for key in keys]}"

        # Assert all keys have the same ExpertParamType
        first_param_type = keys[0].param_type
        assert all(
            key.param_type == first_param_type for key in keys
        ), f"All keys must have the same ExpertParamType, but got types: {[key.param_type for key in keys]}"

        if first_layer != self.layer_idx_now:
            self.layer_idx_now = first_layer
            self.update_activations([key.expert_id for key in keys])

        experts = []
        for key in keys:
            expert = self.get(key)
            experts.append(expert)
        return experts

    def clear(self) -> None: ...

    def next(self) -> None:
        """Advance to next time step and apply watermark decisions."""
        # Use step_forward which is the actual method in Rust implementation
        self._rust_cache.step_forward()
