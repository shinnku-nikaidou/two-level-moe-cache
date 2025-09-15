"""
Direct VRAM expert cache implementation for immediate loading and unloading.

This module provides a simple "load-and-delete" cache strategy where experts
are loaded directly to VRAM when needed and immediately unloaded after use.
No persistence, no LRU management - maximum simplicity for maximum performance.
"""

from typing import List, Set
from src.domain.cache.interfaces.expert_cache import IExpertCacheManager
from src.domain.cache.entities.expert import Expert
from src.domain.cache.entities.types import ExpertKey
from src.domain import ModelType


class DirectNVMEExpertCacheManager(IExpertCacheManager):
    """
    Direct VRAM loading cache with immediate cleanup.

    This cache implementation follows the principle of "load when needed, delete immediately":
    1. Load experts directly to VRAM when requested
    2. Return expert tensors for immediate use
    3. Provide unload_batch() method for immediate cleanup
    4. No persistence between forward passes
    5. No capacity management or eviction logic

    This should be much faster than complex LRU caching for single-token generation.
    Uses shared expert storage from base class - no duplicate Expert instances.
    """

    def __init__(self, model_type: ModelType):
        """
        Initialize direct VRAM cache using shared expert storage.

        Args:
            model_type: Model type for creating Expert instances
        """
        # Initialize base class with shared expert storage
        super().__init__(model_type)

        # Track currently loaded expert keys (not Expert instances)
        self._loaded_expert_keys: Set[ExpertKey] = set()

    def get(self, key: ExpertKey) -> Expert:
        """
        Load a single expert directly to VRAM.

        Args:
            key: Expert identifier

        Returns:
            Expert instance with data loaded in VRAM

        Raises:
            KeyError: If expert cannot be loaded
            RuntimeError: If CUDA not available
        """
        expert = self._get_expert(key)
        expert.nvme_to_vram()
        self._loaded_expert_keys.add(key)
        return expert

    def get_batch(self, keys: List[ExpertKey]) -> List[Expert]:
        """
        Load multiple experts directly to VRAM in batch.

        This implements true "use-and-delete" strategy by automatically
        clearing all previously loaded experts before loading new ones.

        Args:
            keys: List of expert identifiers

        Returns:
            List of expert instances (all in VRAM) in the same order as keys

        Raises:
            KeyError: If any expert cannot be loaded
            RuntimeError: If CUDA not available
        """

        # AUTO-CLEANUP: Unload all previously loaded experts first
        # This ensures maximum memory efficiency with no manual management needed
        self._clear_all()

        result = []

        # Load each expert directly to VRAM in order
        for key in keys:
            expert = self._get_expert(key)  # Get pre-created expert
            expert.nvme_to_vram()  # Load to VRAM
            self._loaded_expert_keys.add(key)  # Track as loaded
            result.append(expert)

        return result

    def clear(self) -> None:
        """
        Clear all experts from VRAM.
        """
        self._clear_all()

    def _clear_all(self) -> int:
        """
        Internal method to unload all currently loaded experts.

        Returns:
            Number of experts unloaded
        """
        all_keys = list(self._loaded_expert_keys)
        return self._evict_batch(all_keys)

    def _evict(self, key: ExpertKey) -> bool:
        """
        Internal method to remove an expert from VRAM immediately.

        Args:
            key: Expert identifier to unload

        Returns:
            True if expert was unloaded, False if not found
        """
        if key not in self._loaded_expert_keys:
            return False

        expert = self._get_expert(key)
        expert.unload()  # Free VRAM memory
        self._loaded_expert_keys.remove(key)

        return True

    def _evict_batch(self, keys: List[ExpertKey]) -> int:
        """
        Internal method to remove multiple experts from VRAM immediately.

        Args:
            keys: List of expert identifiers to evict

        Returns:
            Number of experts actually evicted
        """
        unloaded_count = 0
        for key in keys:
            if self._evict(key):
                unloaded_count += 1

        return unloaded_count

    def next(self) -> None:
        pass
