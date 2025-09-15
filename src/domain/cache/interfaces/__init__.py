"""
Abstract interfaces for expert caching system.

This module provides abstract base classes that define the contracts
for expert caching and memory tier management implementations.
"""

from .expert_cache import IExpertCacheManager
from .memory_tier import IMemoryTierManager

__all__ = [
    "IExpertCacheManager",
    "IMemoryTierManager",
]
