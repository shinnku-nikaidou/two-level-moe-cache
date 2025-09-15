"""
GPT-OSS model expert weight adapter.
"""

import os
from typing import Optional
import torch
from src.boilerplate.gpt_oss.weights import Checkpoint
from src.domain.cache.entities.types import ExpertKey
from .base import ExpertAdapter


class GPTOSSExpertAdapter(ExpertAdapter):
    """
    Expert adapter for GPT-OSS models (20B, 120B).

    Handles MXFP4 format conversion and parameter name mapping
    according to GPT-OSS checkpoint structure.
    """

    def __init__(self, checkpoint_path: str):
        """
        Initialize the GPT-OSS adapter.

        Args:
            checkpoint_path: Path to the checkpoint directory (e.g., "data/models/gpt-oss-20b/original/")
        """
        self.checkpoint_path = checkpoint_path
        self._checkpoint: Optional[Checkpoint] = None

    @property
    def checkpoint(self) -> Checkpoint:
        """Lazy load the checkpoint."""
        if self._checkpoint is None:
            # Checkpoint class expects directory path, not file path
            # Use CPU device by default for loading, will move to target device later
            self._checkpoint = Checkpoint(
                self.checkpoint_path, device=torch.device("cpu")
            )
        return self._checkpoint

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
            "mlp1_weight",
            "mlp1_bias",
            "mlp2_weight",
            "mlp2_bias",
        ]

        return valid_layers and valid_experts and valid_param_types

    def _construct_param_name(self, expert_key: ExpertKey) -> str:
        """
        Construct checkpoint parameter name from expert key.

        Args:
            expert_key: The expert key

        Returns:
            str: The parameter name (e.g., "block.5.mlp.mlp1_weight")
        """
        return f"block.{expert_key.layer_idx}.mlp.{expert_key.param_type}"

    def _extract_expert_slice(
        self, full_tensor: torch.Tensor, expert_key: ExpertKey
    ) -> torch.Tensor:
        """
        Extract a specific expert from the full tensor.

        Args:
            full_tensor: The complete tensor containing all experts
            expert_key: The expert key identifying which slice to extract

        Returns:
            torch.Tensor: The tensor slice for the specific expert
        """
        expert_id = expert_key.expert_id

        if "weight" in expert_key.param_type:
            # For weight matrices, first dimension is expert_id
            # Shape: [num_experts, ...] -> [...]
            return full_tensor[expert_id, ...]
        else:
            # For bias vectors, first dimension is expert_id
            # Shape: [num_experts, hidden_size] -> [hidden_size]
            return full_tensor[expert_id, :]
