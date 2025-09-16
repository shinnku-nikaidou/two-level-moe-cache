"""
Two-Tire Watermark expert cache manager.

This module provides a Python wrapper around the Rust-based TwoTireWmExpertCacheManager
for watermark-based expert caching with dual-tier memory management.
"""

from typing import List, Dict, Optional, Any
from ..cache.interfaces.expert_cache import IExpertCacheManager
from ..cache.entities.expert import Expert
from ..cache.entities.types import ExpertKey
from .. import ModelType

# Import the Rust implementation
from rust_core import TwoTireWmExpertCacheManager as RustTwoTireWmExpertCacheManager
from rust_core import WatermarkConfig as RustWatermarkConfig
from rust_core import ExpertKey as RustExpertKey
from rust_core import ExpertRef as RustExpertRef
from rust_core import ExpertParamType as RustExpertParamType
from rust_core import MemoryTier as RustMemoryTier


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
        vram_capacity_mb: int = 512,
        ram_capacity_mb: int = 2048,
        **kwargs,
    ):
        """
        Initialize two-tier watermark cache manager.

        Args:
            model_type: Type of model for configuration
            vram_capacity_mb: VRAM capacity limit in MB
            ram_capacity_mb: RAM capacity limit in MB
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_type)

        # Initialize Rust implementation (simplified interface)
        # Note: Using positional args to match Rust #[new] signature:
        # (model_type, total_layers, vram_capacity, ram_capacity)
        self._rust_cache = RustTwoTireWmExpertCacheManager(
            self._rust_model_type(model_type),
            self._config.num_hidden_layers,
            vram_capacity_mb * 1024 * 1024,
            ram_capacity_mb * 1024 * 1024,
        )

        # Cache for tracking accessed experts
        self._accessed_experts = set()

    def update_activations(self, activated_experts: List[int]) -> None:
        """
        Update layer activations - delegates to simplified core interface.

        Args:
            activated_experts: List of expert IDs activated in current layer
        """
        self._rust_cache.update_activations(activated_experts)

    def get(self, key: ExpertKey) -> Expert:
        """
        Get expert by key, loading if necessary.

        Args:
            key: Expert identifier

        Returns:
            Expert instance with loaded weights

        Raises:
            KeyError: If expert cannot be loaded
        """
        # Convert our ExpertKey to Rust format
        rust_key = self._python_to_rust_key(key)
        rust_expert_ref = self._rust_cache.get(rust_key)

        # Get the Python Expert instance and update it
        expert = self._get_expert(key)
        self._sync_expert_from_rust(expert, rust_expert_ref)

        # Track access
        self._accessed_experts.add(key)

        return expert

    def get_batch(self, keys: List[ExpertKey]) -> List[Expert]:
        """
        Get batch of experts (implemented using sequential get calls).

        Args:
            keys: List of expert identifiers

        Returns:
            List of expert instances in the same order as keys

        Raises:
            KeyError: If any expert cannot be loaded
        """
        # Note: Rust implementation doesn't have get_batch, use sequential calls
        experts = []
        for key in keys:
            expert = self.get(key)
            experts.append(expert)
        return experts

    def clear(self) -> None:
        """Clear all cached experts and reset state (simplified implementation)."""
        # Note: Rust implementation doesn't have explicit clear method
        # Reset our tracking
        self._accessed_experts.clear()

    def next(self) -> None:
        """Advance to next time step and apply watermark decisions."""
        # Use step_forward which is the actual method in Rust implementation
        self._rust_cache.step_forward()

    def get_watermark_stats(self) -> Dict[str, Any]:
        """
        Get watermark statistics for monitoring and debugging.

        Returns:
            Dictionary with watermark values, capacity usage, and other stats
        """
        # Get stats from the Rust implementation
        stats = self._rust_cache.get_stats()

        # Add our Python-side tracking info
        stats["accessed_experts_count"] = len(self._accessed_experts)

        return stats

    # Private helper methods
    def _python_to_rust_key(self, key: ExpertKey) -> RustExpertKey:
        """Convert Python ExpertKey to Rust ExpertKey."""
        return RustExpertKey(
            layer_idx=key.layer_idx,
            expert_id=key.expert_id,
            param_type=self._python_to_rust_param_type(key.param_type),
        )

    def _python_to_rust_param_type(self, param_type) -> RustExpertParamType:
        """Convert Python ExpertParamType to Rust ExpertParamType."""
        # Import the Python ExpertParamType enum
        from ..cache.entities.types import ExpertParamType

        mapping = {
            ExpertParamType.MLP1_WEIGHT: RustExpertParamType.MLP1_WEIGHT,
            ExpertParamType.MLP1_BIAS: RustExpertParamType.MLP1_BIAS,
            ExpertParamType.MLP2_WEIGHT: RustExpertParamType.MLP2_WEIGHT,
            ExpertParamType.MLP2_BIAS: RustExpertParamType.MLP2_BIAS,
        }
        return mapping[param_type]

    def _rust_model_type(self, model_type: ModelType) -> str:
        """Convert Python ModelType to Rust-compatible string."""
        mapping = {
            ModelType.GPT_OSS_20B: "gpt-oss-20b",
            ModelType.GPT_OSS_120B: "gpt-oss-120b",
            ModelType.PHI_TINY_MOE: "phi-tiny-moe",
        }
        return mapping[model_type]

    def _expert_key_to_string(self, key: ExpertKey) -> str:
        """Convert ExpertKey to string format expected by Rust."""
        return f"L{key.layer_idx}_E{key.expert_id}_{key.param_type}"

    def _sync_expert_from_rust(self, expert: Expert, rust_ref: RustExpertRef) -> None:
        """Synchronize Python Expert state from Rust reference."""
        # For the simplified implementation, we don't actually sync the expert state
        # since that would require actual tensor loading/moving operations.
        # The Rust reference contains tier information that's used for caching decisions,
        # but the Python Expert maintains its own tier based on actual tensor data presence.

        # In a full implementation, we would:
        # - Check rust_ref.tier to determine target location
        # - Load/move tensor data accordingly using expert.load_to_ram(), expert.load_to_vram(), etc.
        # - Update expert's memory usage tracking

        # For now, this is a no-op since we're demonstrating the architecture pattern
        pass
