"""
Factory for creating expert cache instances.

This module provides factory methods for creating different types of expert
cache implementations with appropriate configuration and dependencies.
"""

from typing import Dict, Optional

from src.config.cache_config import CacheConfig
from src.domain import ModelType
from src.domain.cache.interfaces.expert_cache import IExpertCacheManager
from src.domain.manager import (DirectNVMEExpertCacheManager,
                                DirectRAMExpertCacheManager,
                                LRUExpertCacheManager,
                                TwoTierWmExpertCacheManager)


class ExpertCacheFactory:
    """
    Factory for creating expert cache instances with different strategies.

    This factory encapsulates the creation logic for expert cache systems,
    allowing easy switching between different implementations and configurations.
    """

    # Registry of available cache implementations
    _cache_implementations: Dict[str, type] = {
        "lru": LRUExpertCacheManager,
        "direct_vram": DirectNVMEExpertCacheManager,
        "direct_ram": DirectRAMExpertCacheManager,
        "two_tier_wm": TwoTierWmExpertCacheManager,
    }

    @classmethod
    def create_lru_cache_manager(
        cls,
        model_type: ModelType,
        config: Optional[CacheConfig] = None,
    ) -> IExpertCacheManager:
        """
        Create an LRU-based expert cache.

        Args:
            model_type: Type of model for configuration
            config: Cache configuration (optional)
        Returns:
            LRU expert cache manager
        """
        # Use provided config or create default
        if config is None:
            config = CacheConfig.for_model(model_type)

        # Create LRU cache with configuration
        return LRUExpertCacheManager(
            model_type=model_type,
            max_vram_experts=config.max_vram_experts,
            max_ram_experts=config.max_ram_experts,
        )

    @classmethod
    def create_direct_nvme_cache_manager(
        cls,
        model_type: ModelType,
    ) -> IExpertCacheManager:
        """
        Create a direct VRAM cache for use-and-delete strategy.

        This creates the simplest possible cache that loads experts directly
        to VRAM and unloads them immediately after use. No capacity management,
        no LRU tracking - maximum simplicity for maximum performance.

        Args:
            model_type: Type of model for expert loading

        Returns:
            Direct VRAM expert cache
        """
        return DirectNVMEExpertCacheManager(model_type=model_type)

    @classmethod
    def create_direct_ram_cache_manager(
        cls,
        model_type: ModelType,
    ) -> IExpertCacheManager:
        """
        Create a Direct RAM expert cache.

        Args:
            model_type: Type of model

        Returns:
            Direct RAM cache manager
        """
        return DirectRAMExpertCacheManager(model_type=model_type)

    @classmethod
    def create_two_tier_wm_cache_manager(
        cls,
        model_type: ModelType,
        vram_capacity_mb: int = 5120,
        ram_capacity_mb: int = 20480,
        **kwargs,
    ) -> IExpertCacheManager:
        """
        Create a Two-Tier Watermark expert cache.

        Args:
            model_type: Type of model
            vram_capacity_mb: VRAM capacity limit in MB
            ram_capacity_mb: RAM capacity limit in MB
            vram_learning_rate: Learning rate for VRAM watermark updates
            ram_learning_rate: Learning rate for RAM watermark updates
            **kwargs: Additional configuration parameters

        Returns:
            Two-tier watermark cache manager

        Raises:
            RuntimeError: If Rust core library is not available
        """
        return TwoTierWmExpertCacheManager(
            model_type=model_type,
            vram_capacity_mb=vram_capacity_mb,
            ram_capacity_mb=ram_capacity_mb,
            **kwargs,
        )
