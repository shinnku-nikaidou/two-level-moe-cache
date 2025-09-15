"""
Factory for creating expert cache instances.

This module provides factory methods for creating different types of expert
cache implementations with appropriate configuration and dependencies.
"""

from typing import Optional, Dict, Any
from ...config.cache_config import CacheConfig
from ...domain.cache.interfaces.expert_cache import IExpertCache
from ...domain.cache.interfaces.memory_tier import IMemoryTierManager
from ...domain.manager.lru_expert_cache import LRUExpertCacheManager
from ...domain.manager.direct_vram_cache import DirectVRAMExpertCache
from ...domain.manager.memory_tier import SetBasedMemoryTierManager
from ...domain import ModelType


class ExpertCacheFactory:
    """
    Factory for creating expert cache instances with different strategies.

    This factory encapsulates the creation logic for expert cache systems,
    allowing easy switching between different implementations and configurations.
    """

    # Registry of available cache implementations
    _cache_implementations: Dict[str, type] = {
        "lru": LRUExpertCacheManager,
        "direct_vram": DirectVRAMExpertCache,  # Simple use-and-delete cache
        # Future implementations can be added here:
        # "lfu": LFUExpertCacheManager,
        # "adaptive": AdaptiveExpertCacheManager,
    }

    # Registry of available memory tier managers
    _tier_manager_implementations: Dict[str, type] = {
        "set_based": SetBasedMemoryTierManager,
        # Future implementations:
        # "priority_queue": PriorityQueueMemoryTierManager,
        # "weighted": WeightedMemoryTierManager,
    }

    @classmethod
    def create_lru_cache(
        cls,
        model_type: ModelType,
        config: Optional[CacheConfig] = None,
        tier_manager: Optional[IMemoryTierManager] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ) -> IExpertCache:
        """
        Create an LRU-based expert cache.

        Args:
            model_type: Type of model for configuration
            config: Cache configuration (optional)
            tier_manager: Memory tier manager (optional)
            checkpoint_path: Path to checkpoint directory for adapter creation
            **kwargs: Additional arguments

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
        # Note: checkpoint_path is not passed to LRUExpertCacheManager directly
        # It will be used by the adapter when needed
        return LRUExpertCacheManager(
            model_type=model_type,
            memory_tier_manager=tier_manager,
            max_vram_experts=config.max_vram_experts,
            max_ram_experts=config.max_ram_experts,
            # Don't pass checkpoint_path to the cache manager
        )

    @classmethod
    def create_direct_vram_cache(
        cls,
        model_type: ModelType,
        **kwargs,
    ) -> IExpertCache:
        """
        Create a direct VRAM cache for use-and-delete strategy.

        This creates the simplest possible cache that loads experts directly
        to VRAM and unloads them immediately after use. No capacity management,
        no LRU tracking - maximum simplicity for maximum performance.

        Args:
            model_type: Type of model for expert loading
            **kwargs: Additional arguments (ignored for simplicity)

        Returns:
            Direct VRAM expert cache
        """
        return DirectVRAMExpertCache(model_type=model_type)

    @classmethod
    def create_cache(
        cls,
        cache_type: str,
        model_type: ModelType,
        config: Optional[CacheConfig] = None,
        tier_manager_type: str = "set_based",
        **kwargs,
    ) -> IExpertCache:
        """
        Create an expert cache of specified type.

        Args:
            cache_type: Type of cache to create ("lru", etc.)
            model_type: Model type for expert loading
            config: Cache configuration (creates default if None)
            tier_manager_type: Type of memory tier manager to use
            **kwargs: Additional parameters for cache creation

        Returns:
            Configured expert cache instance

        Raises:
            ValueError: If cache_type or tier_manager_type is not supported
        """
        # Validate cache type
        if cache_type not in cls._cache_implementations:
            available = list(cls._cache_implementations.keys())
            raise ValueError(
                f"Unsupported cache type: {cache_type}. Available: {available}"
            )

        # Validate tier manager type
        if tier_manager_type not in cls._tier_manager_implementations:
            available = list(cls._tier_manager_implementations.keys())
            raise ValueError(
                f"Unsupported tier manager type: {tier_manager_type}. Available: {available}"
            )

        # Create configuration if not provided
        if config is None:
            config = CacheConfig.for_model(model_type)

        # Create tier manager
        tier_manager_class = cls._tier_manager_implementations[tier_manager_type]
        tier_manager = tier_manager_class()

        # Create cache
        cache_class = cls._cache_implementations[cache_type]

        # Handle different cache constructors
        if cache_type == "lru":
            return cache_class(
                model_type=model_type,
                memory_tier_manager=tier_manager,
                max_vram_experts=config.max_vram_experts,
                max_ram_experts=config.max_ram_experts,
                **kwargs,
            )
        elif cache_type == "direct_vram":
            # DirectVRAM cache doesn't need tier manager or capacity limits
            return cache_class(model_type=model_type, **kwargs)
        else:
            # Generic creation - may need adjustment for future implementations
            return cache_class(
                model_type=model_type,
                config=config,
                memory_tier_manager=tier_manager,
                **kwargs,
            )

    @classmethod
    def create_from_config(cls, config: CacheConfig, **kwargs) -> IExpertCache:
        """
        Create expert cache from configuration object.

        Args:
            config: Complete cache configuration
            **kwargs: Additional parameters to override config

        Returns:
            Configured expert cache instance
        """
        # Default to LRU cache for now
        cache_type = kwargs.pop("cache_type", "lru")
        tier_manager_type = kwargs.pop("tier_manager_type", "set_based")

        return cls.create_cache(
            cache_type=cache_type,
            model_type=config.model_type,
            config=config,
            tier_manager_type=tier_manager_type,
            **kwargs,
        )

    @classmethod
    def get_available_implementations(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available cache implementations.

        Returns:
            Dictionary with implementation details
        """
        return {
            "cache_types": {
                name: {
                    "class": impl.__name__,
                    "description": (
                        impl.__doc__.split("\n")[1].strip()
                        if impl.__doc__
                        else "No description"
                    ),
                }
                for name, impl in cls._cache_implementations.items()
            },
            "tier_manager_types": {
                name: {
                    "class": impl.__name__,
                    "description": (
                        impl.__doc__.split("\n")[1].strip()
                        if impl.__doc__
                        else "No description"
                    ),
                }
                for name, impl in cls._tier_manager_implementations.items()
            },
        }

    @classmethod
    def register_cache_implementation(cls, name: str, implementation: type) -> None:
        """
        Register a new cache implementation.

        Args:
            name: Name for the implementation
            implementation: Cache class implementing IExpertCache
        """
        if not issubclass(implementation, IExpertCache):
            raise ValueError(f"Implementation must inherit from IExpertCache")

        cls._cache_implementations[name] = implementation

    @classmethod
    def register_tier_manager_implementation(
        cls, name: str, implementation: type
    ) -> None:
        """
        Register a new memory tier manager implementation.

        Args:
            name: Name for the implementation
            implementation: Manager class implementing IMemoryTierManager
        """
        if not issubclass(implementation, IMemoryTierManager):
            raise ValueError(f"Implementation must inherit from IMemoryTierManager")

        cls._tier_manager_implementations[name] = implementation
