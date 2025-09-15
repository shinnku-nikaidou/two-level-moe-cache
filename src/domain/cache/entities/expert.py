"""
Core entities for expert weight caching system.
"""

from typing import Optional, Union
import torch

from src.domain import ModelType
from src.adapters.expert.factory import AdapterFactory
from src.adapters.expert.base import ExpertAdapter
from .types import ExpertKey, MemoryTier


class Expert:
    """
    Encapsulates expert weight data with tier-aware storage management.

    This class manages a single expert's weight data across different memory tiers
    (VRAM/RAM/DISK) and provides operations for tier migration and device management.
    """

    def __init__(
        self,
        expert_key: ExpertKey,
        model_type: ModelType,
        current_tier: MemoryTier = MemoryTier.DISK,
    ):
        """
        Initialize expert weight data.

        Args:
            expert_key: Unique identifier for this expert weight
            model_type: The model type this expert belongs to
            current_tier: Current memory tier where data is stored (default: DISK)
        """
        self.expert_key = expert_key
        self.current_tier = current_tier
        self.data: Optional[torch.Tensor] = None
        self.device: Optional[torch.device] = None
        self.adapter: ExpertAdapter = AdapterFactory.create_adapter(model_type)

    @property
    def is_loaded(self) -> bool:
        """Check if expert data is loaded in memory (RAM or VRAM)."""
        return self.data is not None and self.current_tier != MemoryTier.DISK

    def memory_usage(self) -> int:
        """
        Calculate memory usage in bytes.

        Returns:
            int: Memory usage in bytes, 0 if not loaded
        """
        if not self.is_loaded or self.data is None:
            return 0
        return self.data.numel() * self.data.element_size()

    def move_to_vram(self, device: Optional[Union[str, torch.device]] = None) -> None:
        """
        Move expert data to VRAM.

        Args:
            device: CUDA device to move to (defaults to self.device or "cuda")

        Raises:
            RuntimeError: If CUDA not available or data not loaded
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        if not self.is_loaded or self.data is None:
            raise RuntimeError("Cannot move unloaded expert to VRAM")

        # Determine target CUDA device
        if device is not None:
            target_device = device
        elif self.device is not None and "cuda" in str(self.device):
            target_device = self.device
        else:
            target_device = "cuda"

        self.data = self.data.to(target_device)
        self.device = torch.device(target_device)
        self.current_tier = MemoryTier.VRAM

    def move_to_ram(self) -> None:
        """Move expert data to RAM (CPU)."""
        if not self.is_loaded or self.data is None:
            raise RuntimeError("Cannot move unloaded expert to RAM")

        self.data = self.data.to("cpu")
        self.device = torch.device("cpu")
        self.current_tier = MemoryTier.RAM

    def unload(self) -> None:
        """Unload expert data from memory or gpu"""
        self.data = None
        self.current_tier = MemoryTier.DISK
        # Note: Keep device preference for future loads

    def load_from_nvme_to_ram(self) -> None:
        """
        Load expert data from NVMe/disk to RAM.

        Raises:
            ValueError: If the expert_key is not supported by the adapter
        """
        if self.is_loaded:
            return  # Already loaded, no need to load again

        if self.current_tier != MemoryTier.DISK:
            return  # Not on disk, cannot load from NVMe

        # Load tensor using adapter
        tensor = self.adapter.load_expert_tensor(self.expert_key)

        # Place on CPU (RAM)
        self.data = tensor.to("cpu")
        self.current_tier = MemoryTier.RAM

    def load_from_nvme_to_vram(self) -> None:
        """
        Load expert data from NVMe/disk directly to VRAM.

        Raises:
            RuntimeError: If CUDA not available or expert is already loaded
            ValueError: If the expert_key is not supported by the adapter
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available, cannot load to VRAM")

        if self.is_loaded:
            return  # Already loaded, no need to load again

        if self.current_tier != MemoryTier.DISK:
            return  # Not on disk, cannot load from NVMe

        # Load tensor using adapter
        tensor = self.adapter.load_expert_tensor(self.expert_key)

        # Place on CUDA device (VRAM)
        cuda_device = "cuda" if self.device is None else self.device
        self.data = tensor.to(cuda_device)
        self.device = torch.device(cuda_device)
        self.current_tier = MemoryTier.VRAM

    def promote_to_upper_tier(self) -> None:
        """
        Promote expert to a higher tier (DISK -> RAM -> VRAM).

        Raises:
            RuntimeError: If already at highest tier or cannot promote
        """
        if self.current_tier == MemoryTier.DISK:
            self.load_from_nvme_to_ram()
        elif self.current_tier == MemoryTier.RAM:
            self.move_to_vram()
        else:
            raise RuntimeError(f"Cannot promote from tier {self.current_tier}")

    def demote_to_lower_tier(self) -> None:
        """
        Demote expert to a lower tier (VRAM -> RAM -> DISK).

        Raises:
            RuntimeError: If already at lowest tier
        """
        if self.current_tier == MemoryTier.VRAM:
            self.move_to_ram()
        elif self.current_tier == MemoryTier.RAM:
            self.unload()
        else:
            raise RuntimeError(f"Cannot demote from tier {self.current_tier}")

    def __repr__(self) -> str:
        """String representation for debugging."""
        loaded_status = "loaded" if self.is_loaded else "unloaded"
        status = f"@{self.current_tier.name}"
        if self.device:
            status += f"({self.device})"

        usage = f"{self.memory_usage() / (1024**2):.1f}MB" if self.is_loaded else "0MB"

        return f"Expert({self.expert_key}, {loaded_status}, " f"{status}, {usage})"
