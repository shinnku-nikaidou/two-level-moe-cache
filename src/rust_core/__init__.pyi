"""
Type stubs for Rust core library exports.

This file provides type hints for Pylance/mypy to understand the Rust-exported classes.
Updated during policy layer integration to include watermark debugging methods.
"""

from typing import Dict, List, Optional
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

class WatermarkConfig:
    """Configuration for watermark-based caching algorithm."""

    def __init__(
        self,
        vram_capacity: int,
        ram_capacity: int,
        vram_learning_rate: Optional[float] = None,
        ram_learning_rate: Optional[float] = None,
        fusion_eta: Optional[float] = None,
        reuse_decay_gamma: Optional[float] = None,
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

class ExpertStatus:
    """Expert status information."""

    def __init__(self, expert_key: ExpertKey, current_tier: int) -> None: ...
    @property
    def expert_key(self) -> ExpertKey: ...
    @property
    def current_tier(self) -> int: ...  # 0=VRAM, 1=RAM, 2=DISK
    def __repr__(self) -> str: ...

class TwoTireWmExpertCacheManager:
    """Two-tier watermark-based expert cache manager."""

    def __init__(
        self,
        model_type: str,  # ModelType string
        vram_capacity: int,  # Capacity in bytes
        ram_capacity: int,  # Capacity in bytes
    ) -> None: ...
    def get(self, expert_key: ExpertKey) -> None:
        """Get a single expert by key, loading if necessary."""
        ...

    def update_activations(self, activated_experts: List[int]) -> None:
        """Update with new layer activations."""
        ...

    def step_forward(self) -> None:
        """Advance to next time step."""
        ...
    # New methods added during integration
    def current_time(self) -> int:
        """Get current time step."""
        ...

    def current_layer(self) -> int:
        """Get current layer index."""
        ...

    def total_layers(self) -> int:
        """Get total number of layers."""
        ...

    def get_watermarks(self) -> tuple[float, float]:
        """Get current watermark values (VRAM, RAM) for debugging."""
        ...

    def get_memory_usage(self) -> tuple[int, int]:
        """Get current memory usage (VRAM bytes, RAM bytes) for debugging."""
        ...

    def experts_status(self) -> List[ExpertStatus]:
        """Get status of all tracked experts."""
        ...
