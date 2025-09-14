"""Triton-optimized implementations for GPT-OSS model components.

This module contains high-performance implementations of:
- FlashAttention with learned sinks and sliding window support
- MoE (Mixture of Experts) with quantization
- Complete Transformer model with CUDA Graph optimization
"""

from .attention import attention, attention_ref
from .moe import quantize_mx4, moe, swiglu
from .model import (
    RotaryEmbedding, 
    Cache, 
    AttentionBlock, 
    MLPBlock, 
    TransformerBlock, 
    Transformer, 
    TokenGenerator
)

__all__ = [
    # Attention
    "attention",
    "attention_ref",
    
    # MoE
    "quantize_mx4",
    "moe", 
    "swiglu",
    
    # Model components
    "RotaryEmbedding",
    "Cache",
    "AttentionBlock",
    "MLPBlock", 
    "TransformerBlock",
    "Transformer",
    "TokenGenerator",
]