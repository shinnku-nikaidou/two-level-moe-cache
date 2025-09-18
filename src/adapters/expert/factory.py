"""
Factory for creating expert adapters based on model type.
"""

from typing import Dict, Type

from src.domain import ModelType

from .base import ExpertAdapter
from .gptoss import GPTOSSExpertAdapter


class AdapterFactory:
    """Factory for creating appropriate expert adapters."""

    # Registry mapping model types to adapter classes
    _ADAPTER_REGISTRY: Dict[ModelType, Type[ExpertAdapter]] = {
        ModelType.GPT_OSS_20B: GPTOSSExpertAdapter,
        ModelType.GPT_OSS_120B: GPTOSSExpertAdapter,
    }

    @classmethod
    def create_adapter(cls, model_type: ModelType) -> ExpertAdapter:
        """
        Create an expert adapter for the specified model type.

        Args:
            model_type: The type of model to create an adapter for

        Returns:
            ExpertAdapter: An adapter instance for the specified model

        Raises:
            ValueError: If the model type is not supported
        """
        if model_type not in cls._ADAPTER_REGISTRY:
            supported_models = list(cls._ADAPTER_REGISTRY.keys())
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported models: {[m.value for m in supported_models]}"
            )

        adapter_class = cls._ADAPTER_REGISTRY[model_type]
        return adapter_class(model_type)

    @classmethod
    def list_supported_models(cls) -> list[ModelType]:
        """Get list of all supported model types."""
        return list(cls._ADAPTER_REGISTRY.keys())

    @classmethod
    def register_adapter(
        cls, model_type: ModelType, adapter_class: Type[ExpertAdapter]
    ) -> None:
        """
        Register a new adapter class for a model type.

        Args:
            model_type: The model type to register
            adapter_class: The adapter class to associate with this model type
        """
        cls._ADAPTER_REGISTRY[model_type] = adapter_class
