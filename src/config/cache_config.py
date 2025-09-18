"""
Cache configuration management for expert caching system.

This module provides configuration classes for managing cache parameters,
capacity limits, and performance settings across different implementations.
"""

from dataclasses import dataclass
from typing import Optional

from ..domain import ModelType


@dataclass
class CacheConfig:
    """
    Configuration for expert cache system.

    This class encapsulates all configuration parameters needed to create
    and configure expert cache instances with appropriate capacity limits
    and performance settings.
    """

    # Model configuration
    model_type: ModelType

    # Capacity limits
    max_vram_experts: int = 8
    max_ram_experts: int = 32

    # Performance tuning
    prefetch_enabled: bool = True
    batch_loading_enabled: bool = True

    # Memory management
    aggressive_eviction: bool = False
    memory_pressure_threshold: float = 0.8  # Trigger eviction when memory usage > 80%

    # Statistics and monitoring
    enable_stats: bool = True
    stats_update_interval: int = 100  # Update stats every N operations

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_vram_experts <= 0:
            raise ValueError("max_vram_experts must be positive")
        if self.max_ram_experts <= 0:
            raise ValueError("max_ram_experts must be positive")
        if not 0.0 < self.memory_pressure_threshold <= 1.0:
            raise ValueError("memory_pressure_threshold must be between 0 and 1")
        if self.stats_update_interval <= 0:
            raise ValueError("stats_update_interval must be positive")

    @classmethod
    def for_model(
        cls,
        model_type: ModelType,
        vram_limit_mb: Optional[int] = None,
        ram_limit_mb: Optional[int] = None,
        **kwargs,
    ) -> "CacheConfig":
        """
        Create cache configuration optimized for specific model.

        Args:
            model_type: Target model type
            vram_limit_mb: VRAM limit in MB (auto-detect if None)
            ram_limit_mb: RAM limit in MB (auto-detect if None)
            **kwargs: Additional configuration parameters

        Returns:
            Optimized cache configuration
        """
        # Estimate expert sizes based on model type
        expert_size_mb = cls._estimate_expert_size(model_type)

        # Calculate capacity limits based on memory constraints
        if vram_limit_mb is not None:
            max_vram_experts = max(
                1, int(vram_limit_mb * 0.6 // expert_size_mb)
            )  # Use 60% of VRAM
        else:
            # Default conservative limits - increased for better performance
            if model_type == ModelType.GPT_OSS_120B:
                max_vram_experts = 8  # Larger model, fewer experts in VRAM
            elif model_type == ModelType.GPT_OSS_20B:
                max_vram_experts = 16  # Reasonable VRAM capacity
            else:
                max_vram_experts = 32  # Default for unknown models

        if ram_limit_mb is not None:
            max_ram_experts = max(
                1, int(ram_limit_mb * 0.3 // expert_size_mb)
            )  # Use 30% of RAM
        else:
            # Default limits based on model size - increased for better performance
            # GPT-OSS requires more experts for full forward pass
            if model_type == ModelType.GPT_OSS_120B:
                max_ram_experts = 64  # Larger model, fewer experts in RAM
            elif model_type == ModelType.GPT_OSS_20B:
                max_ram_experts = 64  # Reasonable capacity, should evict when full
            else:
                max_ram_experts = 128  # Default for unknown models

        return cls(
            model_type=model_type,
            max_vram_experts=max_vram_experts,
            max_ram_experts=max_ram_experts,
            **kwargs,
        )

    @staticmethod
    def _estimate_expert_size(model_type: ModelType) -> float:
        """
        Estimate expert weight size in MB for capacity planning.

        Args:
            model_type: Model type to estimate for

        Returns:
            Estimated expert size in MB
        """
        # Rough estimates based on typical model architectures
        # These are approximate values for planning purposes
        size_estimates = {
            ModelType.GPT_OSS_20B: 50.0,  # ~50MB per expert
            ModelType.GPT_OSS_120B: 250.0,  # ~250MB per expert
            ModelType.PHI_TINY_MOE: 10.0,  # ~10MB per expert
        }

        return size_estimates.get(model_type, 100.0)  # Default 100MB

    def get_memory_summary(self) -> dict:
        """
        Get estimated memory usage summary.

        Returns:
            Dictionary with memory usage estimates
        """
        expert_size_mb = self._estimate_expert_size(self.model_type)

        return {
            "expert_size_mb": expert_size_mb,
            "max_vram_usage_mb": self.max_vram_experts * expert_size_mb,
            "max_ram_usage_mb": self.max_ram_experts * expert_size_mb,
            "total_cache_capacity": self.max_vram_experts + self.max_ram_experts,
        }

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        return {
            "model_type": self.model_type.value,
            "max_vram_experts": self.max_vram_experts,
            "max_ram_experts": self.max_ram_experts,
            "prefetch_enabled": self.prefetch_enabled,
            "batch_loading_enabled": self.batch_loading_enabled,
            "aggressive_eviction": self.aggressive_eviction,
            "memory_pressure_threshold": self.memory_pressure_threshold,
            "enable_stats": self.enable_stats,
            "stats_update_interval": self.stats_update_interval,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheConfig":
        """Create configuration from dictionary."""
        # Convert string model type back to enum
        if isinstance(data.get("model_type"), str):
            data["model_type"] = ModelType(data["model_type"])

        return cls(**data)
