"""
Type stubs for Rust core library exports.

This file provides type hints for Pylance/mypy to understand the Rust-exported classes.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
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
        model_type: str,  # Now accepts string instead of ModelType enum
        config: WatermarkConfig,
        total_layers: int,
    ) -> None: ...
    def get(self, expert_key: ExpertKey) -> ExpertRef:
        """Get a single expert by key, loading if necessary."""
        ...

    def get_batch(self, expert_keys: List[ExpertKey]) -> List[ExpertRef]:
        """Get multiple experts efficiently in batch."""
        ...

    def clear(self) -> None:
        """Clear all cache state."""
        ...

    def next(self) -> None:
        """Advance to next time step."""
        ...

    def update_fused_predictions(self, predictions: Dict[str, float]) -> None:
        """Update fused predictions from external policy components."""
        ...

    def get_watermarks(self) -> Tuple[float, float]:
        """Get current watermark values (vram_watermark, ram_watermark)."""
        ...

    def get_capacity_usage(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get current capacity usage ((vram_used, vram_capacity), (ram_used, ram_capacity))."""
        ...

    def get_stats(self) -> Dict[str, float]:
        """Get statistics for analysis."""
        ...

# Legacy exports for backwards compatibility
def add(left: int, right: int) -> int:
    """Add two integers."""
    ...

def get_version() -> str:
    """Get library version."""
    ...

class PyCoreCache:
    """Legacy cache class."""

    def __init__(self) -> None: ...
    def get(self, key: str) -> Optional[str]: ...
    def set(self, key: str, value: str) -> None: ...
