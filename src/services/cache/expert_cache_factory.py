"""
Factory for creating expert cache instances.

This module provides factory methods for creating different types of expert
cache implementations with appropriate configuration and dependencies.
"""

from typing import Optional, Dict
from src.config.cache_config import CacheConfig
from src.domain.cache.interfaces.expert_cache import IExpertCacheManager
from src.domain.cache.interfaces.memory_tier import IMemoryTierManager
from src.domain.manager.lru_expert_cache import LRUExpertCacheManager
from src.domain.manager.direct_vram_cache import DirectVRAMExpertCache
from src.domain.manager.memory_tier import SetBasedMemoryTierManager
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
        "direct_vram": DirectVRAMExpertCache,
    }

    @classmethod
    def create_lru_cache(
        cls,
        model_type: ModelType,
        config: Optional[CacheConfig] = None,
        tier_manager: Optional[IMemoryTierManager] = None,
    ) -> IExpertCacheManager:
        """
        Create an LRU-based expert cache.

        Args:
            model_type: Type of model for configuration
            config: Cache configuration (optional)
            tier_manager: Memory tier manager (optional)
        Returns:
            LRU expert cache manager
        """
        # Use provided config or create default
        if config is None:
            config = CacheConfig.for_model(model_type)

        # Use provided tier manager or create default
        if tier_manager is None:
            tier_manager = SetBasedMemoryTierManager()

        # Create LRU cache with configuration
        return LRUExpertCacheManager(
            model_type=model_type,
            memory_tier_manager=tier_manager,
            max_vram_experts=config.max_vram_experts,
            max_ram_experts=config.max_ram_experts,
        )

    @classmethod
    def create_direct_vram_cache(
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
        return DirectVRAMExpertCache(model_type=model_type)
