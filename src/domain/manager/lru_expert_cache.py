"""
LRU-based expert cache manager implementation.

This module provides an LRU (Least Recently Used) cache implementation for
managing expert weights with automatic memory tier coordination and eviction.
"""

import torch
from collections import OrderedDict
from typing import Dict, List, Optional
from ..cache.interfaces.expert_cache import IExpertCacheManager
from ..cache.interfaces.memory_tier import IMemoryTierManager
from ..cache.entities.expert import Expert
from ..cache.entities.types import ExpertKey, MemoryTier
from .. import ModelType
from .memory_tier import SetBasedMemoryTierManager


class LRUExpertCacheManager(IExpertCacheManager):
    """
    LRU-based expert cache manager with automatic tier coordination.

    This implementation maintains an LRU cache of expert instances while
    coordinating with a memory tier manager to track expert positions
    across VRAM, RAM, and DISK storage tiers.
    """

    def __init__(
        self,
        model_type: ModelType,
        memory_tier_manager: Optional[IMemoryTierManager] = None,
        max_vram_experts: int = 8,
        max_ram_experts: int = 32,
    ):
        """
        Initialize LRU expert cache manager.

        Args:
            model_type: Model type for creating Expert instances
            memory_tier_manager: Memory tier manager instance (creates default if None)
            max_vram_experts: Maximum experts to keep in VRAM before eviction
            max_ram_experts: Maximum experts to keep in RAM before eviction to disk
        """
        self._experts: OrderedDict[ExpertKey, Expert] = OrderedDict()
        self._tier_manager = memory_tier_manager or SetBasedMemoryTierManager()
        self._max_vram_experts = max_vram_experts
        self._max_ram_experts = max_ram_experts
        self._model_type = model_type

        # Statistics tracking
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "loads": 0,
            "tier_migrations": 0,
        }

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
        if key in self._experts:
            # Cache hit: move to end (most recently used)
            expert = self._experts.pop(key)
            self._experts[key] = expert
            self._stats["hits"] += 1

            # Ensure expert is in appropriate tier
            self._ensure_expert_tier(key, expert)
            return expert

        # Cache miss: load expert
        self._stats["misses"] += 1
        expert = self._load_expert(key)
        self.put(key, expert)
        return expert

    def get_batch(self, keys: List[ExpertKey]) -> List[Expert]:
        """
        Retrieve multiple experts efficiently in batch.

        All experts in the batch will be promoted to the same memory tier (preferring VRAM)
        to ensure device consistency for tensor operations.

        Args:
            keys: List of expert identifiers

        Returns:
            List of expert instances in the same order as keys

        Raises:
            KeyError: If any expert cannot be loaded
        """
        result = []
        experts_dict = {}
        missing_keys = []

        # Collect cached experts and identify missing ones
        for key in keys:
            if key in self._experts:
                expert = self._experts.pop(key)
                self._experts[key] = expert  # Move to end (LRU)

                # Store in dict for device consistency check
                experts_dict[key] = expert

                self._stats["hits"] += 1
                self._ensure_expert_tier(key, expert)
            else:
                missing_keys.append(key)
                self._stats["misses"] += 1

        # Load missing experts in batch
        for key in missing_keys:
            expert = self._load_expert(key)
            self.put(key, expert)
            experts_dict[key] = expert

        # IMPORTANT: Ensure all experts in batch are on same device for consistency
        self._ensure_batch_device_consistency(experts_dict)

        # Build result list in the same order as input keys
        for key in keys:
            result.append(experts_dict[key])

        return result

    def put(self, key: ExpertKey, expert: Expert) -> None:
        """
        Store an expert in the cache with LRU eviction.

        Args:
            key: Expert identifier
            expert: Expert instance to store
        """
        # Remove if already exists (for re-insertion at end)
        if key in self._experts:
            del self._experts[key]

        # Add expert to cache
        self._experts[key] = expert

        # Update tier tracking
        current_tier = expert.current_tier
        if current_tier:
            self._tier_manager.add_to_tier(current_tier, key)

        # Trigger eviction if necessary
        self._enforce_capacity_limits()

    def evict(self, key: ExpertKey) -> bool:
        """
        Remove an expert from the cache.

        Args:
            key: Expert identifier to evict

        Returns:
            True if expert was evicted, False if not found
        """
        if key not in self._experts:
            return False

        expert = self._experts.pop(key)

        # Remove from tier tracking
        current_tier = self._tier_manager.get_tier(key)
        if current_tier:
            self._tier_manager.remove_from_tier(current_tier, key)

        # IMPORTANT: Do NOT unload the expert data here!
        # The expert instance might still be referenced elsewhere.
        # Just remove it from the cache and let garbage collection handle it.
        # Add to DISK tier tracking for consistency
        self._tier_manager.add_to_tier(MemoryTier.DISK, key)

        self._stats["evictions"] += 1
        return True

    def evict_batch(self, keys: List[ExpertKey]) -> int:
        """
        Remove multiple experts from the cache.

        Args:
            keys: List of expert identifiers to evict

        Returns:
            Number of experts actually evicted
        """
        evicted_count = 0
        for key in keys:
            if self.evict(key):
                evicted_count += 1
        return evicted_count

    def contains(self, key: ExpertKey) -> bool:
        """
        Check if an expert is currently cached.

        Args:
            key: Expert identifier to check

        Returns:
            True if expert is cached, False otherwise
        """
        return key in self._experts

    def get_cached_keys(self) -> List[ExpertKey]:
        """
        Get all expert keys currently cached.

        Returns:
            List of all cached expert keys in LRU order
        """
        return list(self._experts.keys())

    def get_cache_size(self) -> int:
        """
        Get the current number of cached experts.

        Returns:
            Number of experts in cache
        """
        return len(self._experts)

    def clear(self) -> None:
        """
        Clear all experts from the cache.
        """
        # Move all experts to disk
        for key, expert in self._experts.items():
            if expert.current_tier != MemoryTier.DISK:
                expert.unload()  # Move to disk by unloading

        # Clear cache and tier tracking
        self._experts.clear()
        self._tier_manager.clear_tier(MemoryTier.VRAM)
        self._tier_manager.clear_tier(MemoryTier.RAM)

    def _load_expert(self, key: ExpertKey) -> Expert:
        """
        Load an expert from storage.

        Args:
            key: Expert identifier to load

        Returns:
            Loaded expert instance

        Raises:
            KeyError: If expert cannot be loaded
        """
        # Create expert instance
        expert = Expert(expert_key=key, model_type=self._model_type)

        # Load to VRAM if space available, otherwise RAM
        vram_count = self._tier_manager.get_tier_size(MemoryTier.VRAM)
        if vram_count < self._max_vram_experts and torch.cuda.is_available():
            expert.load_from_nvme_to_vram()  # Load directly to VRAM
            self._tier_manager.add_to_tier(MemoryTier.VRAM, key)
        else:
            expert.load_from_nvme_to_ram()  # Load to RAM
            self._tier_manager.add_to_tier(MemoryTier.RAM, key)

        self._stats["loads"] += 1
        return expert

    def _ensure_expert_tier(self, key: ExpertKey, expert: Expert) -> None:
        """
        Ensure expert is promoted to appropriate memory tier.

        Args:
            key: Expert identifier
            expert: Expert instance to check/promote
        """
        current_tier = expert.current_tier
        vram_count = self._tier_manager.get_tier_size(MemoryTier.VRAM)

        # Promote to VRAM if there's space and not already there
        if current_tier != MemoryTier.VRAM and vram_count < self._max_vram_experts:
            if current_tier:
                self._tier_manager.move_between_tiers(
                    key, current_tier, MemoryTier.VRAM
                )
            else:
                self._tier_manager.add_to_tier(MemoryTier.VRAM, key)

            expert.move_to_vram()  # Promote to VRAM
            self._stats["tier_migrations"] += 1

    def _enforce_capacity_limits(self) -> None:
        """
        Enforce cache capacity limits with LRU eviction.
        """
        # Get current tier counts
        vram_count = self._tier_manager.get_tier_size(MemoryTier.VRAM)
        ram_count = self._tier_manager.get_tier_size(MemoryTier.RAM)

        # Evict from VRAM to RAM if over limit
        if vram_count > self._max_vram_experts:
            experts_to_demote = vram_count - self._max_vram_experts
            vram_experts = self._tier_manager.get_experts_in_tier(MemoryTier.VRAM)

            # Find least recently used experts in VRAM
            lru_experts = []
            for key in reversed(list(self._experts.keys())):  # LRU order
                if key in vram_experts and len(lru_experts) < experts_to_demote:
                    lru_experts.append(key)

            # Move to RAM
            for key in lru_experts:
                if key in self._experts:
                    expert = self._experts[key]
                    expert.move_to_ram()
                    self._tier_manager.move_between_tiers(
                        key, MemoryTier.VRAM, MemoryTier.RAM
                    )
                    self._stats["tier_migrations"] += 1

        # Evict from RAM to DISK if over limit
        if ram_count > self._max_ram_experts:
            experts_to_evict = ram_count - self._max_ram_experts
            ram_experts = self._tier_manager.get_experts_in_tier(MemoryTier.RAM)

            # Find least recently used experts in RAM
            lru_experts = []
            for key in reversed(list(self._experts.keys())):  # LRU order
                if key in ram_experts and len(lru_experts) < experts_to_evict:
                    lru_experts.append(key)

            # Move to disk and remove from cache
            for key in lru_experts:
                self.evict(key)

    def _ensure_batch_device_consistency(
        self, expert_dict: Dict[ExpertKey, Expert]
    ) -> None:
        """
        Ensure all experts in a batch are on the same device.

        This is critical for tensor operations - if experts are on different devices
        (some in VRAM, some in RAM), tensor operations will fail.

        Args:
            expert_dict: Dictionary of experts to check and fix device consistency
        """
        if not expert_dict:
            return

        # Determine target device - prefer CUDA if available
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        target_tier = MemoryTier.VRAM if torch.cuda.is_available() else MemoryTier.RAM

        # Check and fix device for each expert
        for key, expert in expert_dict.items():
            if expert.data is None:
                continue  # Skip unloaded experts

            current_device_str = str(expert.data.device)

            # If expert is not on target device, promote it
            if current_device_str != target_device:
                if (
                    target_tier == MemoryTier.VRAM
                    and expert.current_tier != MemoryTier.VRAM
                ):
                    # Try to promote to VRAM
                    try:
                        expert.move_to_vram()

                        # Update tier tracking
                        old_tier = self._tier_manager.get_tier(key)
                        if old_tier:
                            self._tier_manager.move_between_tiers(
                                key, old_tier, MemoryTier.VRAM
                            )
                        else:
                            self._tier_manager.add_to_tier(MemoryTier.VRAM, key)

                        self._stats["tier_migrations"] += 1

                    except RuntimeError:
                        # VRAM promotion failed, ensure at least on CPU
                        if expert.current_tier != MemoryTier.RAM:
                            expert.move_to_ram()

                elif (
                    target_tier == MemoryTier.RAM
                    and expert.current_tier != MemoryTier.RAM
                ):
                    expert.move_to_ram()

                    # Update tier tracking
                    old_tier = self._tier_manager.get_tier(key)
                    if old_tier:
                        self._tier_manager.move_between_tiers(
                            key, old_tier, MemoryTier.RAM
                        )
                    else:
                        self._tier_manager.add_to_tier(MemoryTier.RAM, key)
