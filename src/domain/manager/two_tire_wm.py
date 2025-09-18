"""
Two-Tire Watermark expert cache manager.

This module provides a Python wrapper around the Rust-based TwoTireWmExpertCacheManager
for watermark-based expert caching with dual-tier memory management.
"""

from typing import List
from ..cache.interfaces.expert_cache import IExpertCacheManager
from ..cache.entities.expert import Expert
from src.common.types import ExpertKey, MemoryTier, ExpertParamType
from .. import ModelType
from .utils import rust_model_type
from rust_core import RustTwoTireWmExpertCacheManager


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

        # Initialize to -1 to ensure first layer (layer 0) triggers activation update
        self.layer_idx_now = -1

        # Initialize Rust implementation (simplified interface)
        # Note: Using positional args to match Rust #[new] signature:
        # (model_type, total_layers, vram_capacity, ram_capacity)
        self._rust_cache = RustTwoTireWmExpertCacheManager(
            rust_model_type(model_type), vram_capacity_mb, ram_capacity_mb
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

        # Layer transition detection: Each layer has 4 parameter types (MLP1_WEIGHT, MLP1_BIAS, MLP2_WEIGHT, MLP2_BIAS)
        # so get_batch() is called 4 times per layer. We must only update activations and sync cache state
        # ONCE per layer to avoid:
        # 1. Incorrect EWMA statistics accumulation (4x the actual activation count)
        # 2. Wrong layer-local time advancement (4 steps instead of 1)
        # 3. Erroneous watermark threshold calculations
        # This check ensures watermark algorithm correctness by preventing repeated updates within the same layer.
        if first_layer != self.layer_idx_now:
            self.layer_idx_now = first_layer
            self.update_activations([key.expert_id for key in keys])
            self.sync_back()

        experts = []
        for key in keys:
            expert = self.get(key)
            experts.append(expert)
        return experts

    def clear(self) -> None: ...

    def sync_back(self) -> None:
        """
        Synchronize expert states from Rust backend to Python Expert instances.
        """
        rust_experts_status = self._rust_cache.experts_status()

        # Sync Rust-side states to Expert instances in self._experts
        for rust_expert_status in rust_experts_status:
            rust_expert_key = rust_expert_status.expert_key

            # Convert to Python key
            expert_key = ExpertKey(
                layer_idx=rust_expert_key.layer_idx,
                expert_id=rust_expert_key.expert_id,
                param_type=ExpertParamType(rust_expert_key.param_type.name),
            )

            # Get the corresponding Expert instance and sync
            expert = self._experts[expert_key]
            target_tier = MemoryTier(rust_expert_status.current_tier)

            self._sync_expert_tier(expert, target_tier)

    def _sync_expert_tier(self, expert: Expert, target_tier: MemoryTier) -> None:
        """
        Synchronize a single Expert's memory tier to the target tier.

        Args:
            expert: Expert instance to synchronize
            target_tier: Target tier indicated by Rust backend
        """
        current_tier = expert.current_tier

        if current_tier == target_tier:
            return  # Already at correct tier

        # Adjust Expert state based on target tier
        if target_tier == MemoryTier.VRAM:
            # Need to promote to VRAM
            if current_tier == MemoryTier.DISK:
                expert.nvme_to_vram()  # DISK -> VRAM (also creates RAM copy)
            elif current_tier == MemoryTier.RAM:
                expert.ram_to_vram()  # RAM -> VRAM

        elif target_tier == MemoryTier.RAM:
            # Need to be at RAM tier
            if current_tier == MemoryTier.DISK:
                expert.nvme_to_ram()  # DISK -> RAM
            elif current_tier == MemoryTier.VRAM:
                expert.vram_to_ram()  # VRAM -> RAM

        elif target_tier == MemoryTier.DISK:
            # Need to unload to DISK
            expert.unload()  # Clear all memory copies

    def step_forward(self) -> None:
        """Advance to next time step and apply watermark decisions."""
        # Use step_forward which is the actual method in Rust implementation
        self._rust_cache.step_forward()
