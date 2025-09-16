"""
Factory for creating expert cache instances.

This module provides factory methods for creating different types of expert
cache implementations with appropriate configuration and dependencies.
"""

from typing import Optional, Dict
from src.config.cache_config import CacheConfig
from src.domain.cache.interfaces.expert_cache import IExpertCacheManager
from src.domain.manager import (
    LRUExpertCacheManager,
    DirectNVMEExpertCacheManager, 
    DirectRAMExpertCacheManager
)
from src.domain import ModelType


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
        Create a pre-warmed RAM cache for maximum inference performance.

        This creates a cache that loads ALL experts to RAM during initialization,
        then provides ultra-fast RAM-to-VRAM copying during inference.
        No disk I/O during get() operations - maximum speed at the cost of
        high RAM usage and longer initialization time.

        Args:
            model_type: Type of model for expert loading

        Returns:
            Pre-warmed DirectRAM expert cache

        Note:
            This will consume significant RAM and take time to initialize,
            but provides the fastest possible inference performance.
        """
        return DirectRAMExpertCacheManager(model_type=model_type)
