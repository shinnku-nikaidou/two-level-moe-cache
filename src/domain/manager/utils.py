"""
Utility functions for manager modules.

This module provides common utility functions for converting between
Python and Rust types used in cache managers.
"""

from src.common.types import ExpertKey, ExpertParamType
from src.domain import ModelType

# Import the Rust types
from rust_core import ExpertKey as RustExpertKey
from rust_core import ExpertParamType as RustExpertParamType


def python_to_rust_key(key: ExpertKey) -> RustExpertKey:
    """Convert Python ExpertKey to Rust ExpertKey."""
    return RustExpertKey(
        layer_idx=key.layer_idx,
        expert_id=key.expert_id,
        param_type=python_to_rust_param_type(key.param_type),
    )


def python_to_rust_param_type(param_type: ExpertParamType) -> RustExpertParamType:
    """Convert Python ExpertParamType to Rust ExpertParamType."""
    mapping = {
        ExpertParamType.MLP1_WEIGHT: RustExpertParamType.MLP1_WEIGHT,
        ExpertParamType.MLP1_BIAS: RustExpertParamType.MLP1_BIAS,
        ExpertParamType.MLP2_WEIGHT: RustExpertParamType.MLP2_WEIGHT,
        ExpertParamType.MLP2_BIAS: RustExpertParamType.MLP2_BIAS,
    }
    return mapping[param_type]


def rust_model_type(model_type: ModelType) -> str:
    """Convert Python ModelType to Rust-compatible string."""
    mapping = {
        ModelType.GPT_OSS_20B: "gpt-oss-20b",
        ModelType.GPT_OSS_120B: "gpt-oss-120b",
        ModelType.PHI_TINY_MOE: "phi-tiny-moe",
    }
    return mapping[model_type]


def expert_key_to_string(key: ExpertKey) -> str:
    """Convert ExpertKey to string format expected by Rust."""
    return f"L{key.layer_idx}_E{key.expert_id}_{key.param_type}"
