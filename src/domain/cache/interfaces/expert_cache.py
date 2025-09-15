"""
Abstract interface for expert caching system.

This module defines the contract for caching and managing expert weights
with automatic memory management and tier coordination.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from ..entities.expert import Expert
from ..entities.types import ExpertKey


class IExpertCacheManager(ABC):
    """
    Abstract interface for expert weight caching system.

    This interface defines operations for storing, retrieving, and managing
    expert weights with automatic memory tier coordination and eviction policies.
    """

    @abstractmethod
    def get(self, key: ExpertKey) -> Expert:
        """
        Retrieve a single expert by key, loading if necessary.

        Args:
            key: Expert identifier

        Returns:
            Expert instance with loaded weights

        Raises:
            KeyError: If expert cannot be loaded
        """
        pass

    @abstractmethod
    def get_batch(self, keys: List[ExpertKey]) -> Dict[ExpertKey, Expert]:
        """
        Retrieve multiple experts efficiently in batch.

        Args:
            keys: List of expert identifiers

        Returns:
            Dictionary mapping keys to expert instances

        Raises:
            KeyError: If any expert cannot be loaded
        """
        pass

    @abstractmethod
    def put(self, key: ExpertKey, expert: Expert) -> None:
        """
        Store an expert in the cache.

        Args:
            key: Expert identifier
            expert: Expert instance to store
        """
        pass

    @abstractmethod
    def evict(self, key: ExpertKey) -> bool:
        """
        Remove an expert from the cache.

        Args:
            key: Expert identifier to evict

        Returns:
            True if expert was evicted, False if not found
        """
        pass

    @abstractmethod
    def evict_batch(self, keys: List[ExpertKey]) -> int:
        """
        Remove multiple experts from the cache.

        Args:
            keys: List of expert identifiers to evict

        Returns:
            Number of experts actually evicted
        """
        pass

    def unload_batch(self, keys: List[ExpertKey]) -> int:
        """
        Immediate unloading for use-and-delete strategy.

        Default implementation delegates to evict_batch, but specialized
        caches can override for immediate memory cleanup.

        Args:
            keys: List of expert identifiers to unload

        Returns:
            Number of experts actually unloaded
        """
        return self.evict_batch(keys)

    @abstractmethod
    def contains(self, key: ExpertKey) -> bool:
        """
        Check if an expert is currently cached.

        Args:
            key: Expert identifier to check

        Returns:
            True if expert is cached, False otherwise
        """
        pass

    @abstractmethod
    def get_cached_keys(self) -> List[ExpertKey]:
        """
        Get all expert keys currently cached.

        Returns:
            List of all cached expert keys
        """
        pass

    @abstractmethod
    def get_cache_size(self) -> int:
        """
        Get the current number of cached experts.

        Returns:
            Number of experts in cache
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all experts from the cache.
        """
        pass

    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary containing cache statistics like hit rate, eviction count, etc.
        """
        pass
