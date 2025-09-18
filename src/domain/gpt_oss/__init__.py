"""
Domain layer for GPT-OSS with memory-efficient MoE implementations.

This module provides optimized versions of GPT-OSS components with
on-demand loading capabilities to reduce memory usage.
"""

# Re-export essential boilerplate components
from ...boilerplate.gpt_oss.model import (AttentionBlock, ModelConfig, RMSNorm,
                                          TokenGenerator, TransformerBlock,
                                          swiglu)
# Re-export utilities
from ...boilerplate.gpt_oss.weights import Checkpoint
from .model import LazyTokenGenerator, LazyTransformer, LazyTransformerBlock
# Domain implementations
from .moe import LazyExpertTensor, LazyMLPBlock

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
