"""
Set-based memory tier manager implementation.

This module provides a simple and efficient implementation of memory tier
management using Python sets to track expert positions across VRAM, RAM, and DISK.
"""

from typing import Optional, Set, List
from ..cache.interfaces.memory_tier import IMemoryTierManager
from ..cache.entities.types import ExpertKey, MemoryTier


class SetBasedMemoryTierManager(IMemoryTierManager):
    """
    Memory tier manager using set-based tracking.

    This implementation uses three separate sets to track expert positions
    across VRAM, RAM, and DISK tiers. It provides O(1) operations for most
    common use cases like checking tier membership and moving between tiers.
    """

    def __init__(self):
        """Initialize empty tier tracking sets."""
        self._vram_experts: Set[ExpertKey] = set()
        self._ram_experts: Set[ExpertKey] = set()
        self._disk_experts: Set[ExpertKey] = set()

        # Mapping for efficient tier lookup
        self._tier_sets = {
            MemoryTier.VRAM: self._vram_experts,
            MemoryTier.RAM: self._ram_experts,
            MemoryTier.DISK: self._disk_experts,
        }

    def add_to_tier(self, tier: MemoryTier, key: ExpertKey) -> None:
        """
        Add an expert key to a specific memory tier.

        Args:
            tier: Target memory tier (VRAM, RAM, or DISK)
            key: Expert identifier to add
        """
        self._tier_sets[tier].add(key)

    def remove_from_tier(self, tier: MemoryTier, key: ExpertKey) -> None:
        """
        Remove an expert key from a specific memory tier.

        Args:
            tier: Source memory tier to remove from
            key: Expert identifier to remove
        """
        self._tier_sets[tier].discard(key)  # discard() won't raise if key not present

    def move_between_tiers(
        self, key: ExpertKey, from_tier: MemoryTier, to_tier: MemoryTier
    ) -> None:
        """
        Move an expert key between memory tiers atomically.

        Args:
            key: Expert identifier to move
            from_tier: Source memory tier
            to_tier: Destination memory tier
        """
        # Remove from source tier (if present)
        self._tier_sets[from_tier].discard(key)
        # Add to destination tier
        self._tier_sets[to_tier].add(key)

    def get_tier(self, key: ExpertKey) -> Optional[MemoryTier]:
        """
        Get the current memory tier of an expert.

        Args:
            key: Expert identifier to query

        Returns:
            Current memory tier or None if not tracked
        """
        for tier, expert_set in self._tier_sets.items():
            if key in expert_set:
                return tier
        return None

    def get_experts_in_tier(self, tier: MemoryTier) -> Set[ExpertKey]:
        """
        Get all expert keys currently in a specific tier.

        Args:
            tier: Memory tier to query

        Returns:
            Set of expert keys in the specified tier (copy for safety)
        """
        return self._tier_sets[tier].copy()

    def get_tier_size(self, tier: MemoryTier) -> int:
        """
        Get the number of experts currently in a tier.

        Args:
            tier: Memory tier to query

        Returns:
            Number of experts in the tier
        """
        return len(self._tier_sets[tier])

    def clear_tier(self, tier: MemoryTier) -> None:
        """
        Remove all expert keys from a specific tier.

        Args:
            tier: Memory tier to clear
        """
        self._tier_sets[tier].clear()

    def get_all_tracked_experts(self) -> List[ExpertKey]:
        """
        Get all expert keys currently tracked across all tiers.

        Returns:
            List of all tracked expert keys
        """
        all_experts = set()
        for expert_set in self._tier_sets.values():
            all_experts.update(expert_set)
        return list(all_experts)

    def get_tier_summary(self) -> dict:
        """
        Get a summary of current tier usage.

        Returns:
            Dictionary with tier sizes and total count
        """
        return {
            "vram_count": len(self._vram_experts),
            "ram_count": len(self._ram_experts),
            "disk_count": len(self._disk_experts),
            "total_tracked": len(self.get_all_tracked_experts()),
        }
