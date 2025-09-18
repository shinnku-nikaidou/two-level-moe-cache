"""
Configuration utilities for model checkpoint path resolution.
"""

import os

from ...domain import ModelType


def get_checkpoint_path(model_type: ModelType) -> str:
    """
    Get the checkpoint path for a given model type.

    Args:
        model_type: The model type to get the checkpoint path for

    Returns:
        str: The path to the model checkpoint directory

    Raises:
        ValueError: If the model type is not supported or path doesn't exist
    """
    # Base models directory
    base_dir = "data/models"

    # Map model types to their directory names
    model_path_map = {
        ModelType.GPT_OSS_20B: "gpt-oss-20b/original/",
        ModelType.GPT_OSS_120B: "gpt-oss-120b/original/",
        ModelType.PHI_TINY_MOE: "phi-tiny-moe/original/",
    }

    if model_type not in model_path_map:
        supported_models = list(model_path_map.keys())
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported models: {[m.value for m in supported_models]}"
        )

    checkpoint_path = os.path.join(base_dir, model_path_map[model_type])

    # Verify the path exists
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    return checkpoint_path


__all__ = ["get_checkpoint_path"]
