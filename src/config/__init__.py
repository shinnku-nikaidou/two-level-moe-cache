"""
Configuration management for expert caching system.
"""

import torch
from .cache_config import CacheConfig

# Global device configuration
TORCH_VRAM_DEVICE = torch.device("cuda")
TORCH_RAM_DEVICE = torch.device("cpu")

__all__ = [
    "CacheConfig",
    "TORCH_VRAM_DEVICE",
    "TORCH_RAM_DEVICE",
]
