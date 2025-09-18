"""
Manager domain for coordinating expert weight management.

This module provides concrete implementations of management interfaces
for expert caching with different strategies:
- LRUExpertCacheManager: LRU-based caching with integrated memory tier management
- DirectNVMEExpertCacheManager: Direct VRAM loading with immediate cleanup
- DirectRAMExpertCacheManager: Full pre-warming to RAM with fast VRAM copying
- TwoTierWmExpertCacheManager: Two-tier watermark-based caching with adaptive thresholds
"""

from enum import Enum
from .lru import LRUExpertCacheManager
from .direct_nvme import DirectNVMEExpertCacheManager
from .direct_ram import DirectRAMExpertCacheManager
from .two_tier_wm import TwoTierWmExpertCacheManager


class CacheManagerType(Enum):
    """Enumeration of available expert cache manager types."""

    LRU = "lru"
    DIRECT_NVME = "direct_nvme"
    DIRECT_RAM = "direct_ram"
    TWO_TIER_WM = "two_tier_wm"


__all__ = [
    "CacheManagerType",
    "LRUExpertCacheManager",
    "DirectNVMEExpertCacheManager",
    "DirectRAMExpertCacheManager",
    "TwoTierWmExpertCacheManager",
]
