"""
Type stubs for Rust core library exports.

This file provides type hints for Pylance/mypy to understand the Rust-exported classes.
"""

from typing import Dict, List, Optional, Tuple
from enum import IntEnum

class MemoryTier(IntEnum):
    """Memory tier enumeration for expert storage locations."""

    VRAM = 0  # GPU memory - fastest, most limited
    RAM = 1  # System memory - fast, moderate capacity
    DISK = 2  # NVMe/SSD storage - slower, largest capacity

class ExpertParamType(IntEnum):
    """Expert parameter type enumeration."""

    MLP1_WEIGHT = 0
    MLP1_BIAS = 1
    MLP2_WEIGHT = 2
    MLP2_BIAS = 3

class ExpertKey:
    """Expert identifier combining layer, expert ID, and parameter type."""

    def __init__(
        self, layer_idx: int, expert_id: int, param_type: ExpertParamType
    ) -> None: ...
    @property
    def layer_idx(self) -> int: ...
    @property
    def expert_id(self) -> int: ...
    @property
    def param_type(self) -> ExpertParamType: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class ExpertRef:
    """Reference to an expert with memory tier and size information."""

    def __init__(self, key: ExpertKey) -> None: ...
    @property
    def key(self) -> ExpertKey: ...
    @property
    def tier(self) -> Optional[MemoryTier]: ...
    @property
    def size(self) -> int: ...
    def set_tier(self, tier: Optional[MemoryTier]) -> None: ...
    def set_size(self, size: int) -> None: ...

class WatermarkConfig:
    """Configuration for watermark-based caching algorithm."""

    def __init__(
        self,
        vram_capacity: int,
        ram_capacity: int,
        vram_learning_rate: float = 0.01,
        ram_learning_rate: float = 0.01,
        fusion_eta: float = 0.3,
        reuse_decay_gamma: float = 0.1,
    ) -> None: ...
    @property
    def vram_capacity(self) -> int: ...
    @property
    def ram_capacity(self) -> int: ...
    @property
    def vram_learning_rate(self) -> float: ...
    @property
    def ram_learning_rate(self) -> float: ...
    def validate(self) -> None: ...

class TwoTireWmExpertCacheManager:
    """Two-tier watermark-based expert cache manager."""

    def __init__(
        self,
        model_type: str,  # ModelType string
        total_layers: int,
        vram_capacity: int,  # Capacity in bytes
        ram_capacity: int,   # Capacity in bytes
    ) -> None: ...
    
    def get(self, expert_key: ExpertKey) -> ExpertRef:
        """Get a single expert by key, loading if necessary."""
        ...

    def update_activations(self, activated_experts: List[int]) -> None:
        """Update with new layer activations."""
        ...

    def step_forward(self) -> None:
        """Advance to next time step."""
        ...

    def get_stats(self) -> Dict[str, float]:
        """Get statistics for analysis."""
        ...
