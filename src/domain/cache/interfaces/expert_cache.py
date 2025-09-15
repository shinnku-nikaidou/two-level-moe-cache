"""
Abstract interface for expert caching system.

This module defines the contract for caching and managing expert weights
with automatic memory management and tier coordination.
"""

from abc import ABC, abstractmethod
from typing import List
from ..entities.expert import Expert
from ..entities.types import ExpertKey


class IExpertCacheManager(ABC):
    """
    High-level interface for expert weight retrieval system.

    This interface focuses on the primary use case: efficiently retrieving
    expert weights for inference. Internal cache management details like
    eviction policies and memory tier coordination are implementation-specific.
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
    def get_batch(self, keys: List[ExpertKey]) -> List[Expert]:
        """
        Retrieve multiple experts efficiently in batch.

        Args:
            keys: List of expert identifiers

        Returns:
            List of expert instances in the same order as keys

        Raises:
            KeyError: If any expert cannot be loaded
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all experts from the cache.
        
        Useful for cleanup at end of generation or error recovery.
        """
        pass
