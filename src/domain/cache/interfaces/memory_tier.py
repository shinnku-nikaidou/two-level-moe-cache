"""
Abstract interface for memory tier management.

This module defines the contract for managing expert weights across
different memory tiers (VRAM, RAM, DISK).
"""

from abc import ABC, abstractmethod
from typing import Optional, Set, List
from ..entities.types import ExpertKey, MemoryTier


class IMemoryTierManager(ABC):
    """
    Abstract interface for managing expert weights across memory tiers.

    This interface defines operations for tracking expert positions
    across VRAM, RAM, and DISK storage tiers without managing the
    actual expert data.
    """

    @abstractmethod
    def add_to_tier(self, tier: MemoryTier, key: ExpertKey) -> None:
        """
        Add an expert key to a specific memory tier.

        Args:
            tier: Target memory tier (VRAM, RAM, or DISK)
            key: Expert identifier to add
        """
        pass

    @abstractmethod
    def remove_from_tier(self, tier: MemoryTier, key: ExpertKey) -> None:
        """
        Remove an expert key from a specific memory tier.

        Args:
            tier: Source memory tier to remove from
            key: Expert identifier to remove
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_tier(self, key: ExpertKey) -> Optional[MemoryTier]:
        """
        Get the current memory tier of an expert.

        Args:
            key: Expert identifier to query

        Returns:
            Current memory tier or None if not tracked
        """
        pass

    @abstractmethod
    def get_experts_in_tier(self, tier: MemoryTier) -> Set[ExpertKey]:
        """
        Get all expert keys currently in a specific tier.

        Args:
            tier: Memory tier to query

        Returns:
            Set of expert keys in the specified tier
        """
        pass

    @abstractmethod
    def get_tier_size(self, tier: MemoryTier) -> int:
        """
        Get the number of experts currently in a tier.

        Args:
            tier: Memory tier to query

        Returns:
            Number of experts in the tier
        """
        pass

    @abstractmethod
    def clear_tier(self, tier: MemoryTier) -> None:
        """
        Remove all expert keys from a specific tier.

        Args:
            tier: Memory tier to clear
        """
        pass

    @abstractmethod
    def get_all_tracked_experts(self) -> List[ExpertKey]:
        """
        Get all expert keys currently tracked across all tiers.

        Returns:
            List of all tracked expert keys
        """
        pass
