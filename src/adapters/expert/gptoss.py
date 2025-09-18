"""
GPT-OSS model expert weight adapter.
"""

import torch

from src.boilerplate.gpt_oss.weights import Checkpoint
from src.common.types import ExpertKey, ExpertParamType
from src.config.util import get_checkpoint_path
from src.domain import ModelType

from .base import ExpertAdapter


class GPTOSSExpertAdapter(ExpertAdapter):
    """
    Expert adapter for GPT-OSS models (20B, 120B).

    Handles MXFP4 format conversion and parameter name mapping
    according to GPT-OSS checkpoint structure.
    """

    def __init__(self, model_type: ModelType):
        """
        Initialize the GPT-OSS adapter.

        Args:
            model_type: Model type for automatic checkpoint path resolution
        """
        self.model_type = model_type
        self.checkpoint = Checkpoint(
            get_checkpoint_path(model_type), device=torch.device("cpu")
        )

    def load_expert_tensor(self, expert_key: ExpertKey) -> torch.Tensor:
        """
        Load a specific expert tensor from GPT-OSS checkpoint.

        Args:
            expert_key: The expert key identifying which tensor to load

        Returns:
            torch.Tensor: The expert weight tensor (already converted from MXFP4 if applicable)
        """
        self.validate_expert_key(expert_key)

        # Construct parameter name: block.{layer_idx}.mlp.{param_type}
        param_name = self._construct_param_name(expert_key)

        # Load full tensor using Checkpoint class (handles MXFP4 conversion automatically)
        full_tensor = self.checkpoint.get(param_name)

        if full_tensor is None:
            raise RuntimeError(f"Failed to load parameter {param_name} from checkpoint")

        # Extract the specific expert slice
        expert_tensor = self._extract_expert_slice(full_tensor, expert_key)

        return expert_tensor

    def is_supported(self, expert_key: ExpertKey) -> bool:
        """
        Check if the expert key is valid for GPT-OSS models.

        Args:
            expert_key: The expert key to validate

        Returns:
            bool: True if supported
        """
        # GPT-OSS-20B: 24 layers, 32 experts per layer
        # GPT-OSS-120B: 40 layers, 64 experts per layer (based on typical scaling)
        valid_layers = (
            expert_key.layer_idx >= 0 and expert_key.layer_idx <= 39
        )  # Support both 20B and 120B
        valid_experts = (
            expert_key.expert_id >= 0 and expert_key.expert_id <= 63
        )  # Support both 32 and 64 experts
        valid_param_types = expert_key.param_type in [
            ExpertParamType.MLP1_WEIGHT,
            ExpertParamType.MLP1_BIAS,
            ExpertParamType.MLP2_WEIGHT,
            ExpertParamType.MLP2_BIAS,
        ]

        return valid_layers and valid_experts and valid_param_types

    def _construct_param_name(self, expert_key: ExpertKey) -> str:
        return f"block.{expert_key.layer_idx}.mlp.{expert_key.param_type.value}"

    def _extract_expert_slice(
        self, full_tensor: torch.Tensor, expert_key: ExpertKey
    ) -> torch.Tensor:
        expert_id = expert_key.expert_id

        if "weight" in expert_key.param_type.value:
            return full_tensor[expert_id, ...].clone()
        else:
            return full_tensor[expert_id, :].clone()
