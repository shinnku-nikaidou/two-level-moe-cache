"""
Type stubs for Rust core library exports.

This file provides type hints for Pylance/mypy to understand the Rust-exported classes.
Updated during policy layer integration to include watermark debugging methods.
All Rust PyClass exports now have "Rust" prefix for clear namespace separation.
"""

from typing import List, Optional
from enum import IntEnum

class RustMemoryTier(IntEnum):
    """Memory tier enumeration for expert storage locations."""

    VRAM = 0  # GPU memory - fastest, most limited
    RAM = 1  # System memory - fast, moderate capacity
    DISK = 2  # NVMe/SSD storage - slower, largest capacity

class RustExpertParamType(IntEnum):
    """Expert parameter type enumeration."""

    MLP1_WEIGHT = 0
    MLP1_BIAS = 1
    MLP2_WEIGHT = 2
    MLP2_BIAS = 3

class RustExpertKey:
    """Expert identifier combining layer, expert ID, and parameter type."""

    def __init__(
        self, layer_idx: int, expert_id: int, param_type: RustExpertParamType
    ) -> None: ...
    @property
    def layer_idx(self) -> int: ...
    @property
    def expert_id(self) -> int: ...
    @property
    def param_type(self) -> RustExpertParamType: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class RustWatermarkConfig:
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

class RustExpertStatus:
    """Expert status information."""

    def __init__(self, expert_key: RustExpertKey, current_tier: int) -> None: ...
    @property
    def expert_key(self) -> RustExpertKey: ...
    @property
    def current_tier(self) -> int: ...  # 0=VRAM, 1=RAM, 2=DISK
    def __repr__(self) -> str: ...

class RustTwoTireWmExpertCacheManager:
    """Two-tier watermark-based expert cache manager."""

    def __init__(
        self,
        model_type: str,  # ModelType string
        vram_capacity: int,  # Capacity in bytes
        ram_capacity: int,  # Capacity in bytes
    ) -> None: ...
    def get(self, expert_key: RustExpertKey) -> None:
        """Get a single expert by key, loading if necessary."""
        ...

    def update_activations(self, activated_experts: List[int]) -> None:
        """Update with new layer activations."""
        ...

    def step_forward(self) -> None:
        """Advance to next time step."""
        ...

    def experts_status(self) -> List[RustExpertStatus]:
        """Get status of all tracked experts."""
        ...
