"""
Abstract interfaces for expert caching system.

This module provides abstract base classes that define the contracts
for expert caching implementations.
"""

from .expert_cache import IExpertCacheManager

__all__ = [
    "IExpertCacheManager",
]
