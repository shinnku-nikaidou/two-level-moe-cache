"""
Domain layer for GPT-OSS with memory-efficient MoE implementations.

This module provides optimized versions of GPT-OSS components with
on-demand loading capabilities to reduce memory usage.
"""

# Domain implementations
from .moe import LazyMLPBlock, LazyExpertTensor
from .model import LazyTransformer, LazyTransformerBlock, LazyTokenGenerator

# Re-export essential boilerplate components
from ...boilerplate.gpt_oss.model import (
    ModelConfig,
    RMSNorm,
    AttentionBlock,
    TransformerBlock,
    TokenGenerator,
    swiglu,
)

# Re-export utilities
from ...boilerplate.gpt_oss.weights import Checkpoint

__all__ = [
    # Domain-specific implementations
    "LazyMLPBlock",
    "LazyExpertTensor",
    "LazyTransformer",
    "LazyTransformerBlock",
    "LazyTokenGenerator",
    # Re-exported boilerplate components
    "ModelConfig",
    "RMSNorm",
    "AttentionBlock",
    "TransformerBlock",
    "TokenGenerator",
    "swiglu",
    "Checkpoint",
]
