"""
Core types for expert weight caching system.
"""

from dataclasses import dataclass
from enum import Enum


class MemoryTier(Enum):
    """
    Represents different memory tiers for expert storage.

    Lower values indicate faster access tiers.
    """

    VRAM = 0  # GPU memory - fastest, most limited
    RAM = 1  # System memory - fast, moderate capacity
    DISK = 2  # NVMe/SSD storage - slower, largest capacity


@dataclass(frozen=True)
class ExpertKey:
    """
    Unique identifier for an expert weight tensor.

    Combines layer index, expert ID within that layer, and parameter type
    to uniquely identify any expert weight in the model.
    """

    layer_idx: int  # Transformer layer index (0-based)
    expert_id: int  # Expert ID within the layer (0-based)
    param_type: str  # Parameter type (e.g., "mlp1_weight", "mlp2_bias")

    def __str__(self) -> str:
        """String representation for logging/debugging."""
        return f"L{self.layer_idx}_E{self.expert_id}_{self.param_type}"

    def __post_init__(self):
        """Validate key parameters."""
        if self.layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {self.layer_idx}")
        if self.expert_id < 0:
            raise ValueError(f"expert_id must be non-negative, got {self.expert_id}")
        if not self.param_type:
            raise ValueError("param_type cannot be empty")
