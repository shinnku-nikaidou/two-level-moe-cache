"""
Core entities for expert weight caching system.
"""

from typing import Optional
import torch

from src.domain import ModelType
from src.adapters.expert.factory import AdapterFactory
from src.adapters.expert.base import ExpertAdapter
from src.config import TORCH_VRAM_DEVICE, TORCH_RAM_DEVICE
from src.common.types import ExpertKey, MemoryTier


class Expert:
    """
    Encapsulates expert weight data with two-level cache architecture.

    This class implements the two-level cache design from the paper:
    - data_ram: Cache layer - stores experts for fast access
    - data_vram: Compute layer - REQUIRED for inference, where actual computation happens
    - Cache flow: DISK -> RAM (cache) -> VRAM (compute) -> computation
    - Invariant: For inference, data must be in VRAM (device=TORCH_VRAM_DEVICE)
    - RAM serves as intermediate cache to avoid repeated DISK loads
    """

    def __init__(
        self,
        expert_key: ExpertKey,
        model_type: ModelType,
    ):
        """
        Initialize expert weight data.

        Args:
            expert_key: Unique identifier for this expert weight
            model_type: The model type this expert belongs to
        """
        self.expert_key = expert_key

        # Dual storage design
        self.data_ram: Optional[torch.Tensor] = None  # RAM copy
        self.data_vram: Optional[torch.Tensor] = None  # VRAM copy

        # Create adapter using factory with model type only
        self.adapter: ExpertAdapter = AdapterFactory.create_adapter(model_type)

    @property
    def current_tier(self) -> MemoryTier:
        """
        Determine current memory tier based on actual data presence.

        Logic:
        - If data_vram exists: VRAM tier
        - Elif data_ram exists: RAM tier
        - Else: DISK tier
        """
        if self.data_vram is not None:
            return MemoryTier.VRAM
        elif self.data_ram is not None:
            return MemoryTier.RAM
        else:
            return MemoryTier.DISK

    @property
    def is_loaded(self) -> bool:
        """Check if expert data is loaded in memory (RAM or VRAM)."""
        return self.current_tier != MemoryTier.DISK

    @property
    def is_in_vram(self) -> bool:
        """Check if expert data is available in VRAM."""
        return self.current_tier == MemoryTier.VRAM

    def memory_usage(self) -> int:
        """
        Calculate total memory usage in bytes (RAM + VRAM).

        Returns:
            int: Total memory usage in bytes, 0 if not loaded
        """
        total_usage = 0

        if self.data_ram is not None:
            total_usage += self.data_ram.numel() * self.data_ram.element_size()

        return total_usage

    def ram_to_vram(self) -> None:
        """
        Move expert data to VRAM (creating VRAM copy).

        Ensures RAM copy exists first, then creates VRAM copy.
        """
        # Must have RAM copy first
        assert self.data_ram is not None, "Cannot move to VRAM: no RAM copy available"

        # Create VRAM copy from RAM
        self.data_vram = self.data_ram.to(TORCH_VRAM_DEVICE)

    def vram_to_ram(self) -> None:
        """
        Move expert data to RAM only (remove VRAM copy).

        Keeps RAM copy, removes VRAM copy.
        """
        assert self.data_ram is not None
        self.data_vram = None

    def unload(self) -> None:
        """Unload expert data from all memory tiers."""
        self.data_ram = None
        self.data_vram = None

    def nvme_to_ram(self) -> None:
        """
        Load expert data from NVMe/disk to RAM cache layer.

        This populates the cache layer for later VRAM promotion.
        """
        if self.is_loaded or self.current_tier != MemoryTier.DISK:
            return

        # Load tensor using adapter
        tensor = self.adapter.load_expert_tensor(self.expert_key)

        # Store data in RAM
        self.data_ram = tensor.to(TORCH_RAM_DEVICE)

    def nvme_to_vram(self) -> None:
        """
        Load expert data from NVMe/disk directly to VRAM compute layer.

        For direct computation without intermediate RAM caching.
        Also creates RAM copy to maintain cache invariant.
        """
        if self.is_loaded or self.current_tier != MemoryTier.DISK:
            return

        # Load tensor using adapter
        tensor = self.adapter.load_expert_tensor(self.expert_key)

        # Create both RAM cache and VRAM compute copies
        self.data_ram = tensor.to(TORCH_RAM_DEVICE)
        self.data_vram = tensor.to(TORCH_VRAM_DEVICE)

    def promote_to_upper_tier(self) -> None:
        """
        Promote expert to a higher tier (DISK -> RAM -> VRAM).

        Raises:
            RuntimeError: If already at highest tier or cannot promote
        """
        if self.current_tier == MemoryTier.DISK:
            self.nvme_to_ram()
        elif self.current_tier == MemoryTier.RAM:
            self.ram_to_vram()
        else:
            raise RuntimeError(f"Cannot promote from tier {self.current_tier}")

    def demote_to_lower_tier(self) -> None:
        """
        Demote expert to a lower tier (VRAM -> RAM -> DISK).

        Raises:
            RuntimeError: If already at lowest tier
        """
        if self.current_tier == MemoryTier.VRAM:
            self.vram_to_ram()
        elif self.current_tier == MemoryTier.RAM:
            self.unload()
        else:
            raise RuntimeError(f"Cannot demote from tier {self.current_tier}")

    def __repr__(self) -> str:
        """String representation for debugging."""
        loaded_status = "loaded" if self.is_loaded else "unloaded"
        status = f"@{self.current_tier.name}"
        usage = f"{self.memory_usage() / (1024**2):.1f}MB" if self.is_loaded else "0MB"
        return f"Expert({self.expert_key}, {loaded_status}, " f"{status}, {usage})"
