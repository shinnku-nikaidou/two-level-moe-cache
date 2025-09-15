"""
Direct RAM expert cache implementation with persistent caching.

This module provides a "warmup cache" strategy where experts are loaded
to RAM once and kept there permanently. After a few warmup rounds, most
experts will be in RAM providing fast access with no loading delays.
RAM is assumed to be "unlimited" - no eviction, just keep accumulating.
"""

import torch
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
        self._model_type = model_type

        # Persistent cache - experts stay here once loaded
        self._cached_experts: Dict[ExpertKey, Expert] = {}

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
        # Check if already cached in RAM
        if key in self._cached_experts:
            return self._cached_experts[key]

        # Not cached - load to RAM and cache it
        expert = self._load_expert_to_ram(key)
        self._cached_experts[key] = expert
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
            if key in self._cached_experts:
                # Cache hit - use existing RAM expert
                expert = self._cached_experts[key]
            else:
                # Cache miss - load to RAM and cache
                expert = self._load_expert_to_ram(key)
                self._cached_experts[key] = expert

            result.append(expert)

        return result

    def clear(self) -> None:
        """
        Clear all experts from RAM cache.

        Note: This defeats the warmup purpose, use sparingly!
        """
        for expert in self._cached_experts.values():
            expert.unload()
        self._cached_experts.clear()

    def _load_expert_to_ram(self, key: ExpertKey) -> Expert:
        """
        Load an expert directly to RAM.

        Args:
            key: Expert identifier to load

        Returns:
            Expert instance with data loaded in RAM

        Raises:
            KeyError: If expert cannot be loaded
        """
        # Create expert instance
        expert = Expert(expert_key=key, model_type=self._model_type)

        # Load to RAM (not VRAM)
        expert.nvme_to_ram()

        return expert
