"""
Expert weight loading adapters.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch

from src.common.types import ExpertKey


class ExpertAdapter(ABC):
    """
    Abstract base class for loading expert weights from disk.

    Different model architectures require different loading strategies
    (MXFP4 format, parameter naming conventions, etc).
    """

    @abstractmethod
    def load_expert_tensor(self, expert_key: ExpertKey) -> torch.Tensor:
        """
        Load a specific expert tensor from disk.

        Args:
            expert_key: Unique identifier for the expert weight

        Returns:
            torch.Tensor: The loaded expert weight tensor

        Raises:
            ValueError: If the expert_key is not supported by this adapter
            FileNotFoundError: If the checkpoint files are not found
        """
        pass

    @abstractmethod
    def is_supported(self, expert_key: ExpertKey) -> bool:
        """
        Check if this adapter supports the given expert key.

        Args:
            expert_key: The expert key to check

        Returns:
            bool: True if supported, False otherwise
        """
        pass

    def validate_expert_key(self, expert_key: ExpertKey) -> None:
        """
        Validate that the expert key is supported.

        Args:
            expert_key: The expert key to validate

        Raises:
            ValueError: If the expert key is not supported
        """
        if not self.is_supported(expert_key):
            raise ValueError(
                f"Expert key {expert_key} is not supported by {self.__class__.__name__}"
            )
