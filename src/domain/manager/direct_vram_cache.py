"""
Direct VRAM expert cache implementation for immediate loading and unloading.

This module provides a simple "load-and-delete" cache strategy where experts
are loaded directly to VRAM when needed and immediately unloaded after use.
No persistence, no LRU management - maximum simplicity for maximum performance.
"""

import torch
from typing import Dict, List, Optional, Any
from ..cache.interfaces.expert_cache import IExpertCacheManager
from ..cache.entities.expert import Expert
from ..cache.entities.types import ExpertKey, MemoryTier
from ...domain import ModelType


class DirectVRAMExpertCache(IExpertCacheManager):
    """
    Direct VRAM loading cache with immediate cleanup.

    This cache implementation follows the principle of "load when needed, delete immediately":
    1. Load experts directly to VRAM when requested
    2. Return expert tensors for immediate use
    3. Provide unload_batch() method for immediate cleanup
    4. No persistence between forward passes
    5. No capacity management or eviction logic

    This should be much faster than complex LRU caching for single-token generation.
    """

    def __init__(self, model_type: ModelType):
        """
        Initialize direct VRAM cache.

        Args:
            model_type: Model type for creating Expert instances
        """
        self._model_type = model_type

        # Track currently loaded experts for cleanup
        self._loaded_experts: Dict[ExpertKey, Expert] = {}

        # Simple statistics tracking
        self._stats = {
            "loads": 0,
            "unloads": 0,
            "batch_loads": 0,
            "batch_unloads": 0,
        }

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
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - DirectVRAMCache requires GPU")

        # Always load fresh - no caching
        expert = self._load_expert_to_vram(key)
        self._loaded_experts[key] = expert
        self._stats["loads"] += 1

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
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - DirectVRAMCache requires GPU")

        # AUTO-CLEANUP: Unload all previously loaded experts first
        # This ensures maximum memory efficiency with no manual management needed
        self.unload_all()

        result = []

        # Load each expert directly to VRAM in order
        for key in keys:
            expert = self._load_expert_to_vram(key)
            self._loaded_experts[key] = expert
            result.append(expert)

        self._stats["batch_loads"] += 1
        self._stats["loads"] += len(keys)

        return result

    def put(self, key: ExpertKey, expert: Expert) -> None:
        """
        Store an expert (no-op for direct cache).

        This method exists for interface compatibility but does nothing
        since DirectVRAMCache doesn't persist experts.

        Args:
            key: Expert identifier
            expert: Expert instance (ignored)
        """
        # No-op: direct cache doesn't store experts
        pass

    def evict(self, key: ExpertKey) -> bool:
        """
        Remove an expert from VRAM immediately.

        Args:
            key: Expert identifier to unload

        Returns:
            True if expert was unloaded, False if not found
        """
        if key not in self._loaded_experts:
            return False

        expert = self._loaded_experts.pop(key)
        expert.unload()  # Free VRAM memory
        self._stats["unloads"] += 1

        return True

    def evict_batch(self, keys: List[ExpertKey]) -> int:
        """
        Remove multiple experts from VRAM immediately.

        Args:
            keys: List of expert identifiers to unload

        Returns:
            Number of experts actually unloaded
        """
        unloaded_count = 0
        for key in keys:
            if self.evict(key):
                unloaded_count += 1

        if unloaded_count > 0:
            self._stats["batch_unloads"] += 1

        return unloaded_count

    def unload_batch(self, keys: List[ExpertKey]) -> int:
        """
        Immediate cleanup method for batch unloading after forward pass.

        This is the key method for "use-and-delete" strategy.
        Call this immediately after using experts in forward pass.

        Args:
            keys: List of expert identifiers to unload

        Returns:
            Number of experts actually unloaded
        """
        return self.evict_batch(keys)

    def unload_all(self) -> int:
        """
        Unload all currently loaded experts.

        Useful for cleanup at end of generation or error handling.

        Returns:
            Number of experts unloaded
        """
        all_keys = list(self._loaded_experts.keys())
        return self.unload_batch(all_keys)

    def contains(self, key: ExpertKey) -> bool:
        """
        Check if an expert is currently loaded in VRAM.

        Args:
            key: Expert identifier to check

        Returns:
            True if expert is currently loaded in VRAM
        """
        return key in self._loaded_experts

    def get_cached_keys(self) -> List[ExpertKey]:
        """
        Get all expert keys currently loaded in VRAM.

        Returns:
            List of all currently loaded expert keys
        """
        return list(self._loaded_experts.keys())

    def get_cache_size(self) -> int:
        """
        Get the current number of loaded experts.

        Returns:
            Number of experts currently in VRAM
        """
        return len(self._loaded_experts)

    def clear(self) -> None:
        """
        Clear all experts from VRAM.
        """
        self.unload_all()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary containing cache statistics
        """
        return {
            **self._stats,
            "currently_loaded": len(self._loaded_experts),
            "memory_usage_mb": self._estimate_memory_usage(),
        }

    def _load_expert_to_vram(self, key: ExpertKey) -> Expert:
        """
        Load an expert directly to VRAM.

        Args:
            key: Expert identifier to load

        Returns:
            Expert instance with data loaded in VRAM

        Raises:
            KeyError: If expert cannot be loaded
            RuntimeError: If CUDA not available or loading fails
        """
        # Create expert instance
        expert = Expert(expert_key=key, model_type=self._model_type)

        # Load directly to VRAM (bypass RAM completely)
        expert.load_from_nvme_to_vram()

        return expert

    def _estimate_memory_usage(self) -> float:
        """
        Estimate current VRAM usage in MB.

        Returns:
            Estimated memory usage in MB
        """
        total_bytes = 0
        for expert in self._loaded_experts.values():
            if expert.data is not None:
                total_bytes += expert.data.numel() * expert.data.element_size()

        return total_bytes / (1024 * 1024)  # Convert to MB
