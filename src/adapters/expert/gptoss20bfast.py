"""
GPT-OSS 20B Fast Expert Adapter - Precomputed weights support.

This adapter prioritizes loading from precomputed FP16/BF16 weights for zero
decoding overhead, with fallback to original MXFP4 decoding for compatibility.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any

import torch
from safetensors.torch import load_file

from src.boilerplate.gpt_oss.weights import Checkpoint
from src.common.types import ExpertKey, ExpertParamType
from src.config.util import get_checkpoint_path
from src.domain import ModelType

from .base import ExpertAdapter


class GPTOSS20bFastExpertAdapter(ExpertAdapter):
    """
    GPT-OSS 20B Fast Expert Adapter with precomputed weight support.

    Loading strategy:
    1. Prioritize precomputed FP16/BF16 weights from individual files (zero decoding overhead)
    2. Fallback to original MXFP4 decoding if precomputed weights are not available
    3. True on-demand loading - no caching overhead
    """

    def __init__(self, model_type: ModelType):
        """Initialize fast adapter for GPT-OSS-20B."""

        if model_type != ModelType.GPT_OSS_20B:
            raise ValueError(
                f"This adapter only supports GPT-OSS-20B, received: {model_type}"
            )

        self.model_type = model_type

        # Setup paths
        self.checkpoint_path = Path(get_checkpoint_path(model_type))
        self.precomputed_dir = self.checkpoint_path / "precomputed"
        # print(f"GPT-OSS-20B Fast Adapter initialized")

    def load_expert_tensor(self, expert_key: ExpertKey) -> torch.Tensor:
        """
        Load expert weight tensor (prioritize precomputed, fallback to MXFP4).

        Args:
            expert_key: Expert identifier

        Returns:
            Expert weight tensor ready for computation
        """

        self.validate_expert_key(expert_key)

        try:
            tensor = self._load_from_precomputed(expert_key)
            return tensor
        except Exception as e:
            print(f"Warning: Precomputed load failed for {expert_key}: {e}")
            raise e

    def _get_expert_file_path(
        self, layer_idx: int, expert_id: int, param_type: ExpertParamType
    ) -> Path:
        """Get file path for a precomputed expert tensor."""
        layer_dir = self.precomputed_dir / f"layer_{layer_idx:02d}"
        filename = f"expert_{expert_id:02d}_{param_type.value}.safetensors"
        return layer_dir / filename

    def _load_from_precomputed(self, expert_key: ExpertKey) -> torch.Tensor:
        """Load from precomputed individual files (zero decoding overhead)."""

        expert_file = self._get_expert_file_path(
            expert_key.layer_idx, expert_key.expert_id, expert_key.param_type
        )

        if not expert_file.exists():
            raise FileNotFoundError(f"Precomputed expert file not found: {expert_file}")

        # Load individual safetensors file
        data = load_file(expert_file, device="cpu")
        return data["tensor"].clone()


    def is_supported(self, expert_key: ExpertKey) -> bool:
        """Check if expert key is supported (GPT-OSS-20B: 24 layers, 32 experts)."""

        valid_layers = 0 <= expert_key.layer_idx <= 23  # 24 layers
        valid_experts = 0 <= expert_key.expert_id <= 31  # 32 experts
        valid_param_types = expert_key.param_type in [
            ExpertParamType.MLP1_WEIGHT,
            ExpertParamType.MLP1_BIAS,
            ExpertParamType.MLP2_WEIGHT,
            ExpertParamType.MLP2_BIAS,
        ]
        return valid_layers and valid_experts and valid_param_types
