"""
Direct RAM expert cache implementation with persistent caching.

This module provides a "warmup cache" strategy where experts are loaded
to RAM once and kept there permanently. After a few warmup rounds, most
experts will be in RAM providing fast access with no loading delays.
RAM is assumed to be "unlimited" - no eviction, just keep accumulating.
"""

from typing import Dict, List
from src.domain.cache.interfaces.expert_cache import IExpertCacheManager
from src.domain.cache.entities.expert import Expert
from src.domain.cache.entities.types import ExpertKey
from src.domain import ModelType


class DirectRAMExpertCacheManager(IExpertCacheManager):
    """
    Direct RAM persistent cache with no eviction.

    This cache implementation follows the "warmup and persist" principle:
    1. Check if expert is already in RAM cache
    2. If not, load from NVME to RAM and cache it
    3. Keep all experts in RAM indefinitely (no cleanup)
    4. After warmup rounds, most experts will be RAM-resident
    5. Provides fast access once warmed up

    Perfect for scenarios with sufficient RAM and repeated expert usage.
    """

    def __init__(self, model_type: ModelType):
        """
        Initialize direct RAM cache.

        Args:
            model_type: Model type for creating Expert instances
        """
        # Call ABC constructor which pre-creates all experts
        super().__init__(model_type)

        # Only track which experts have been loaded to RAM
        self._loaded_keys: set[ExpertKey] = set()

    def get(self, key: ExpertKey) -> Expert:
        """
        Get a single expert from RAM cache or load if not cached.

        Args:
            key: Expert identifier

        Returns:
            Expert instance with data loaded in RAM

        Raises:
            KeyError: If expert cannot be loaded
        """
        # Get pre-created expert instance (no duplicate creation!)
        expert = self._get_expert(key)

        # Load to RAM if not already loaded
        if key not in self._loaded_keys:
            expert.nvme_to_ram()
            self._loaded_keys.add(key)

        return expert

    def get_batch(self, keys: List[ExpertKey]) -> List[Expert]:
        """
        Get multiple experts from RAM cache, loading any missing ones.

        This implements persistent caching - no cleanup between calls.
        Experts accumulate in RAM over time (warmup effect).

        Args:
            keys: List of expert identifiers

        Returns:
            List of expert instances (all in RAM) in the same order as keys

        Raises:
            KeyError: If any expert cannot be loaded
        """
        result = []

        # Process each expert: use cached or load fresh
        for key in keys:
            expert = self.get(key)  # Reuse single-expert logic
            result.append(expert)

        return result

    def clear(self) -> None:
        """
        Clear all experts from RAM cache.

        Note: This defeats the warmup purpose, use sparingly!
        """
        # Unload all loaded experts
        for key in self._loaded_keys:
            expert = self._experts[key]  # Use pre-created instance
            expert.unload()
        self._loaded_keys.clear()

    def next(self) -> None:
        """No-op for persistent cache - no time-based policies."""
        pass

    def get_cache_status(self) -> Dict[str, int]:
        """
        Get cache status for monitoring warmup progress.

        Returns:
            Dictionary with cache statistics
        """
        memory_mb = 0
        for key in self._loaded_keys:
            expert = self._experts[key]
            if expert.data_ram is not None:
                memory_mb += expert.data_ram.numel() * expert.data_ram.element_size()
        memory_mb = memory_mb // (1024 * 1024)

        return {"cached_experts": len(self._loaded_keys), "memory_mb": memory_mb}
