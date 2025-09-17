"""
Core types for the two-level MOE cache system.

This module contains fundamental type definitions that are used across
the entire codebase. It should have minimal dependencies to avoid
circular imports.
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


class ExpertParamType(Enum):
    """
    Represents different parameter types for MoE expert weights.

    Each expert has these parameter types for its MLP layers.
    """

    MLP1_WEIGHT = "mlp1_weight"  # First MLP layer weights (up projection)
    MLP1_BIAS = "mlp1_bias"  # First MLP layer bias
    MLP2_WEIGHT = "mlp2_weight"  # Second MLP layer weights (down projection)
    MLP2_BIAS = "mlp2_bias"  # Second MLP layer bias


@dataclass(frozen=True)
class ExpertKey:
    """
    Unique identifier for an expert weight tensor.

    Combines layer index, expert ID within that layer, and parameter type
    to uniquely identify any expert weight in the model.
    """

    layer_idx: int  # Transformer layer index (0-based)
    expert_id: int  # Expert ID within the layer (0-based)
    param_type: ExpertParamType  # Parameter type enum

    def __str__(self) -> str:
        """String representation for logging/debugging."""
        return f"L{self.layer_idx}_E{self.expert_id}_{self.param_type.value}"

    def __post_init__(self):
        """Validate key parameters."""
        if self.layer_idx < 0:
            raise ValueError(f"layer_idx must be non-negative, got {self.layer_idx}")
        if self.expert_id < 0:
            raise ValueError(f"expert_id must be non-negative, got {self.expert_id}")
        if not isinstance(self.param_type, ExpertParamType):
            raise ValueError(
                f"param_type must be ExpertParamType, got {type(self.param_type)}"
            )
