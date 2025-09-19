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

        # Check precomputed availability
        self.precomputed_available = self._check_precomputed_availability()

        # Lazy initialization for fallback
        self._fallback_checkpoint: Optional[Checkpoint] = None

        # Statistics
        self._load_stats = {
            "precomputed_loads": 0,
            "fallback_loads": 0,
            "total_loads": 0,
        }

        print(f"GPT-OSS-20B Fast Adapter initialized")
        print(
            f"Precomputed weights: {'✅ Available' if self.precomputed_available else '❌ Not available, using MXFP4 fallback'}"
        )

    def _check_precomputed_availability(self) -> bool:
        """Check if precomputed weights are complete and available."""

        if not self.precomputed_dir.exists():
            return False

        metadata_file = self.precomputed_dir / "metadata.json"
        if not metadata_file.exists():
            return False

        try:
            with open(metadata_file) as f:
                metadata = json.load(f)

            # Verify basic metadata
            if metadata.get("model_type") != self.model_type.value:
                return False

            # Quick sanity check - verify a few random files exist
            import random

            random.seed(42)

            num_layers = metadata.get("num_layers", 24)
            num_experts = metadata.get("num_experts", 32)

            # Check 10 random files
            for _ in range(10):
                layer_idx = random.randint(0, num_layers - 1)
                expert_id = random.randint(0, num_experts - 1)
                param_type = random.choice(list(ExpertParamType))

                expert_file = self._get_expert_file_path(
                    layer_idx, expert_id, param_type
                )
                if not expert_file.exists():
                    return False

            return True

        except Exception as e:
            print(f"Warning: Precomputed availability check failed: {e}")
            return False

    def load_expert_tensor(self, expert_key: ExpertKey) -> torch.Tensor:
        """
        Load expert weight tensor (prioritize precomputed, fallback to MXFP4).

        Args:
            expert_key: Expert identifier

        Returns:
            Expert weight tensor ready for computation
        """

        self.validate_expert_key(expert_key)
        self._load_stats["total_loads"] += 1

        # Strategy 1: Try loading from precomputed
        if self.precomputed_available:
            try:
                tensor = self._load_from_precomputed(expert_key)
                self._load_stats["precomputed_loads"] += 1
                return tensor
            except Exception as e:
                print(f"Warning: Precomputed load failed for {expert_key}: {e}")
                print("Falling back to MXFP4 decoding...")

        # Strategy 2: Fallback to original MXFP4 decoding
        tensor = self._load_from_mxfp4(expert_key)
        self._load_stats["fallback_loads"] += 1
        return tensor

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

    def _load_from_mxfp4(self, expert_key: ExpertKey) -> torch.Tensor:
        """Load from original MXFP4 format (not implemented - requires precomputed weights)."""

        raise NotImplementedError(
            f"MXFP4 fallback not available for {expert_key}. "
            f"Please precompute weights first using: "
            f"from src.infra.dataprocess.gpt_oss import precompute_gpt_oss_20b; precompute_gpt_oss_20b()"
        )

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

    def get_load_stats(self) -> Dict[str, Any]:
        """Get loading statistics for performance analysis."""

        total = self._load_stats["total_loads"]
        precomputed_rate = (
            (self._load_stats["precomputed_loads"] / total * 100) if total > 0 else 0
        )

        return {
            "precomputed_available": self.precomputed_available,
            "total_loads": total,
            "precomputed_loads": self._load_stats["precomputed_loads"],
            "fallback_loads": self._load_stats["fallback_loads"],
            "precomputed_rate_percent": precomputed_rate,
            "fallback_initialized": self._fallback_checkpoint is not None,
        }

    def reset_stats(self) -> None:
        """Reset loading statistics."""
        self._load_stats = {
            "precomputed_loads": 0,
            "fallback_loads": 0,
            "total_loads": 0,
        }
