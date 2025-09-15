"""
Core entities for expert weight caching system.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import torch


class MemoryTier(Enum):
    """Memory storage tiers for expert weights."""

    VRAM = 0  # GPU memory - fastest access
    RAM = 1  # CPU memory - medium access speed
    DISK = 2  # Disk storage - slowest but largest capacity


@dataclass(frozen=True)
class ExpertKey:
    """Unique identifier for expert weights."""

    layer_idx: int  # 0-23 (24 layers)
    expert_id: int  # 0-31 (32 experts)
    param_type: str  # "mlp1_weight" | "mlp1_bias" | "mlp2_weight" | "mlp2_bias"

    def __hash__(self):
        return hash((self.layer_idx, self.expert_id, self.param_type))

    def __str__(self):
        return f"L{self.layer_idx}_E{self.expert_id}_{self.param_type}"


class Expert:
    """
    Encapsulates expert weight data with tier-aware storage management.

    This class manages a single expert's weight data across different memory tiers
    (VRAM/RAM/DISK) and provides operations for tier migration and device management.
    """

    def __init__(
        self,
        expert_key: ExpertKey,
        current_tier: MemoryTier = MemoryTier.DISK,
        data: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize expert weight data.

        Args:
            expert_key: Unique identifier for this expert weight
            current_tier: Current memory tier where data is stored
            data: The actual tensor data (None means not loaded)
            device: Device where the tensor should be placed when loaded
        """
        self.expert_key = expert_key
        self.current_tier = current_tier
        self._data: Optional[torch.Tensor] = data
        self.device = torch.device(device) if device else None
        self.last_access_time = time.time()
        self.access_count = 0

    @property
    def data(self) -> Optional[torch.Tensor]:
        """Get the tensor data if loaded in memory."""
        return self._data

    @data.setter
    def data(self, tensor: Optional[torch.Tensor]):
        """Set the tensor data and update access time."""
        self._data = tensor
        self.last_access_time = time.time()
        if tensor is not None:
            self.access_count += 1

    @property
    def is_loaded(self) -> bool:
        """Check if the tensor data is currently loaded in memory."""
        return self._data is not None

    def memory_usage(self) -> int:
        """Get memory usage in bytes."""
        if self._data is None:
            return 0
        return self._data.numel() * self._data.element_size()

    def promote_to(self, target_tier: MemoryTier) -> None:
        """
        Promote this expert to a higher tier (faster access).

        Args:
            target_tier: Target memory tier to promote to
        """
        if target_tier.value < self.current_tier.value:  # VRAM < RAM < DISK
            self.current_tier = target_tier

    def demote_to(self, target_tier: MemoryTier) -> None:
        """
        Demote this expert to a lower tier (slower access).

        Args:
            target_tier: Target memory tier to demote to
        """
        if target_tier.value > self.current_tier.value:
            self.current_tier = target_tier

    def move_to_device(self, device: Union[str, torch.device]) -> None:
        """
        Move the tensor data to a specific device.

        Args:
            device: Target device (e.g., 'cuda', 'cpu')
        """
        if self._data is not None:
            self.device = torch.device(device)
            self._data = self._data.to(device)

    def unload(self) -> None:
        """Unload the tensor data from memory."""
        self._data = None

    def touch(self) -> None:
        """Update access time and increment access count."""
        self.last_access_time = time.time()
        self.access_count += 1

    def __repr__(self):
        status = f"@{self.current_tier.value}"
        if self.is_loaded:
            status += f" loaded({self._data.shape})"  # pyright: ignore[reportOptionalMemberAccess]
        else:
            status += " unloaded"
        return f"Expert[{self.expert_key}] {status} (accessed: {self.access_count})"
