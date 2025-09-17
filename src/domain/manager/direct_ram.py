"""
Direct RAM expert cache implementation with full pre-warming.

This module provides a true "warmup cache" strategy where ALL experts are
loaded to RAM during initialization, then get()/get_batch() operations
copy from RAM to VRAM for computation. This maximizes access speed by
eliminating all disk I/O during inference.
"""

from typing import List, Set
from src.domain.cache.interfaces.expert_cache import IExpertCacheManager
from src.domain.cache.entities.expert import Expert
from src.common.types import ExpertKey
from src.domain import ModelType


class DirectRAMExpertCacheManager(IExpertCacheManager):
    """
    Direct RAM cache with full expert pre-warming.

    Architecture:
    1. __init__(): Pre-load ALL experts to RAM (warmup phase)
    2. get()/get_batch(): Copy from RAM to VRAM and return VRAM experts
    3. RAM serves as permanent cache tier, VRAM as compute tier
    4. Zero disk I/O during inference - maximum performance

    Memory Usage:
    - RAM: ALL experts loaded permanently (cache tier)
    - VRAM: Active experts copied from RAM (compute tier)
    - Total = RAM usage + VRAM usage (double storage during computation)

    Perfect for scenarios with abundant RAM and performance-critical inference.
    """

    def __init__(self, model_type: ModelType):
        """
        Initialize DirectRAM cache with full expert pre-warming.

        This is where the magic happens - ALL experts are loaded to RAM
        during initialization to eliminate any disk I/O during inference.

        Args:
            model_type: Model type for creating Expert instances

        Note:
            This initialization will take time and use significant RAM,
            but provides maximum inference performance.
        """
        # Call base constructor to pre-create all Expert instances
        super().__init__(model_type)

        # Track experts currently in VRAM (for batch eviction)
        self._vram_expert_keys: Set[ExpertKey] = set()

        # 🔥 CORE FEATURE: Pre-warm ALL experts to RAM during init
        print(f"🔥 Pre-warming {len(self._experts)} experts to RAM...")
        self._warmup_all_experts_to_ram()
        print(
            f"✅ DirectRAM cache ready with {self.get_loaded_expert_count()} experts in RAM"
        )

    def _warmup_all_experts_to_ram(self) -> None:
        """
        Load all expert weights from disk to RAM during initialization.

        This is the core of the DirectRAM strategy - frontload all I/O
        so that inference operations are purely memory-to-memory copies.
        """
        for expert_key, expert in self._experts.items():
            try:
                expert.nvme_to_ram()  # Load from disk to RAM
            except Exception as e:
                print(f"⚠️ Warning: Failed to load {expert_key} to RAM: {e}")
                continue

    def get(self, key: ExpertKey) -> Expert:
        """
        Get expert from RAM cache, copying to VRAM for computation.

        Since all experts are pre-loaded to RAM, this operation is purely
        a RAM-to-VRAM copy with no disk I/O.

        Args:
            key: Expert identifier

        Returns:
            Expert instance with data in VRAM (and still in RAM as cache)

        Raises:
            KeyError: If expert key is not valid for this model
            RuntimeError: If expert not in RAM (should never happen)
        """
        # Get pre-created and pre-loaded expert
        expert = self._get_expert(key)

        # Verify expert is in RAM (should always be true after warmup)
        if not expert.is_loaded:
            raise RuntimeError(f"Expert {key} not in RAM - warmup failed?")

        # Copy from RAM to VRAM for computation
        # This creates VRAM copy while keeping RAM copy as cache
        expert.ram_to_vram()

        # Track this expert as having a VRAM copy
        self._vram_expert_keys.add(key)

        return expert

    def get_batch(self, keys: List[ExpertKey]) -> List[Expert]:
        """
        Get multiple experts from RAM cache, copying all to VRAM.

        Since all experts are pre-loaded to RAM, this performs efficient
        batch RAM-to-VRAM copying with no disk I/O.

        Args:
            keys: List of expert identifiers

        Returns:
            List of expert instances (all with VRAM copies) in same order as keys

        Raises:
            KeyError: If any expert key is not valid for this model
            RuntimeError: If any expert not in RAM (should never happen)
        """
        result = []

        # Process each expert with RAM-to-VRAM copy
        # Note: VRAM tracking happens automatically via get() calls
        for key in keys:
            expert = self.get(key)  # Reuse single-expert logic (includes VRAM tracking)
            result.append(expert)

        return result

    def clear(self) -> None:
        """
        Clear all VRAM copies while keeping RAM cache intact.

        This allows freeing VRAM memory while preserving the RAM cache
        for future use. The experts remain "warmed up" in RAM.
        """
        # Clear VRAM copies but keep RAM cache
        for expert in self._experts.values():
            if expert.is_in_vram:
                expert.vram_to_ram()  # Remove VRAM, keep RAM

        # Clear VRAM tracking set
        self._vram_expert_keys.clear()

    def next(self) -> None:
        """
        Batch evict all VRAM copies while keeping RAM cache intact.

        This clears all VRAM copies to free GPU memory while preserving
        the pre-warmed RAM cache for future use. This is the key method
        for preventing VRAM memory leaks in the DirectRAM strategy.
        """
        if not self._vram_expert_keys:
            return  # No VRAM experts to evict

        # Batch evict all VRAM copies while keeping RAM copies
        for key in list(self._vram_expert_keys):  # Copy to avoid iteration issues
            expert = self._get_expert(key)
            if expert.is_in_vram:
                expert.vram_to_ram()  # Remove VRAM copy, keep RAM copy

        # Clear the tracking set
        self._vram_expert_keys.clear()
