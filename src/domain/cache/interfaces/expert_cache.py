"""
Abstract interface for expert caching system.

This module defines the contract for caching and managing expert weights
with automatic memory management and tier coordination.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, List
from ..entities.expert import Expert
from ..entities.types import ExpertKey, ExpertParamType
from src.domain import ModelType
from src.boilerplate.gpt_oss.model import ModelConfig
from src.config.util import get_checkpoint_path


class IExpertCacheManager(ABC):
    """
    High-level interface for expert weight retrieval system.

    This base class provides common expert storage and management,
    while subclasses implement specific caching policies and strategies.
    """

    def __init__(self, model_type: ModelType):
        """
        Initialize expert cache manager with pre-created expert instances.

        Args:
            model_type: Model type for creating Expert instances
        """
        super().__init__()
        self._model_type = model_type

        # Load model configuration from actual config.json file
        self._config = self._load_model_config(model_type)

        # Pre-create all possible Expert instances based on model configuration
        self._experts: Dict[ExpertKey, Expert] = self._create_all_experts()

    def _load_model_config(self, model_type: ModelType) -> ModelConfig:
        """
        Load model configuration from config.json file.

        Args:
            model_type: Model type for checkpoint path resolution

        Returns:
            ModelConfig loaded from the actual model's config.json
        """
        checkpoint_dir = get_checkpoint_path(model_type)
        config_path = os.path.join(checkpoint_dir, "config.json")

        if not os.path.exists(config_path):
            print(f"⚠️ Config file not found at {config_path}, using defaults")
            return ModelConfig()

        try:
            with open(config_path, "r") as f:
                json_config = json.load(f)
                return ModelConfig(**json_config)
        except Exception as e:
            print(f"⚠️ Failed to load config from {config_path}: {e}, using defaults")
            return ModelConfig()

    def _create_all_experts(self) -> Dict[ExpertKey, Expert]:
        """
        Pre-create all Expert instances based on model configuration.

        This eliminates duplicate Expert creation across different cache implementations.

        Returns:
            Dictionary mapping ExpertKey to Expert instances
        """
        experts = {}

        # All parameter types for each expert
        param_types = [
            ExpertParamType.MLP1_WEIGHT,
            ExpertParamType.MLP1_BIAS,
            ExpertParamType.MLP2_WEIGHT,
            ExpertParamType.MLP2_BIAS,
        ]

        # Generate all possible expert keys
        for layer_idx in range(self._config.num_hidden_layers):
            for expert_id in range(self._config.num_experts):
                for param_type in param_types:
                    key = ExpertKey(
                        layer_idx=layer_idx, expert_id=expert_id, param_type=param_type
                    )
                    # Create Expert instance (unloaded state)
                    experts[key] = Expert(expert_key=key, model_type=self._model_type)

        return experts

    def _get_expert(self, key: ExpertKey) -> Expert:
        """
        Get pre-created Expert instance by key.

        Args:
            key: Expert identifier

        Returns:
            Expert instance

        Raises:
            KeyError: If expert key is not valid for this model
        """
        if key not in self._experts:
            raise KeyError(f"Expert key not found: {key}")
        return self._experts[key]

    def get_total_expert_count(self) -> int:
        """Get total number of expert instances managed by this cache."""
        return len(self._experts)

    def get_loaded_expert_count(self) -> int:
        """Get number of experts currently loaded in memory."""
        return sum(1 for expert in self._experts.values() if expert.is_loaded)

    def get_vram_expert_count(self) -> int:
        """Get number of experts currently in VRAM."""
        return sum(1 for expert in self._experts.values() if expert.is_in_vram)

    @abstractmethod
    def get(self, key: ExpertKey) -> Expert:
        """
        Retrieve a single expert by key, loading if necessary.

        Args:
            key: Expert identifier

        Returns:
            Expert instance with loaded weights

        Raises:
            KeyError: If expert cannot be loaded
        """
        self._get_expert(key)

    @abstractmethod
    def get_batch(self, keys: List[ExpertKey]) -> List[Expert]:
        """
        Retrieve multiple experts efficiently in batch.

        Args:
            keys: List of expert identifiers

        Returns:
            List of expert instances in the same order as keys

        Raises:
            KeyError: If any expert cannot be loaded
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all experts from the cache.

        Useful for cleanup at end of generation or error recovery.
        """
        pass

    @abstractmethod
    def next(self) -> None:
        """
        Advance internal state for time-based policies.

        This can be used to implement time-based eviction or refresh
        strategies in cache implementations that support it.
        """
        pass
