"""
Integration layer for memory-efficient GPT-OSS model implementations.

This module provides integrated model classes that combine boilerplate
components with domain-specific optimizations like expert caching.
"""

import os
import json
import torch

from ...boilerplate.gpt_oss.model import (
    ModelConfig,
    AttentionBlock,
    RMSNorm,
)
from ...boilerplate.gpt_oss.weights import Checkpoint
from ..cache.interfaces.expert_cache import IExpertCache
from ...services.cache import ExpertCacheFactory
from ...config.util import get_checkpoint_path
from ...domain import ModelType
from .moe import LazyMLPBlock


class LazyTransformerBlock(torch.nn.Module):
    """
    Transformer block using LazyMLPBlock for memory-efficient expert loading.

    Uses expert cache system for automatic memory management across
    VRAM, RAM, and DISK tiers.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        expert_cache: IExpertCache,
        device: torch.device | None = None,
    ):
        """
        Initialize lazy transformer block.

        Args:
            config: Model configuration
            layer_idx: Layer index
            expert_cache: Expert cache instance for loading weights
            device: Target device for components
        """
        super().__init__()
        self.layer_idx = layer_idx

        # Use regular attention block with specified device
        self.attn = AttentionBlock(config, layer_idx, device=device)

        # Use lazy MLP block with expert cache for memory efficiency
        self.mlp = LazyMLPBlock(config, expert_cache, layer_idx, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention and lazy MLP blocks."""
        x = self.attn(x)
        x = self.mlp(x)
        return x


class LazyTransformer(torch.nn.Module):
    """
    Memory-efficient Transformer using expert cache system for MoE weights.

    This model uses the expert caching system for automatic memory management
    across VRAM, RAM, and DISK tiers, significantly reducing memory usage.
    """

    def __init__(
        self,
        config: ModelConfig,
        model_type: ModelType,
        device: torch.device | None = None,
    ):
        """
        Initialize lazy transformer with expert cache system.

        Args:
            config: Model configuration
            model_type: Model type for automatic checkpoint path resolution
            device: Target device for components
        """
        super().__init__()
        self.config = config

        # Create expert cache using model_type (no external dependencies needed)
        # Create simple DirectVRAM cache for maximum performance
        # This implements "use-and-delete" strategy - no complex LRU management
        self.expert_cache = ExpertCacheFactory.create_direct_vram_cache(
            model_type=model_type,
        )

        # Standard components with specified device
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=torch.bfloat16, device=device
        )

        # Use lazy transformer blocks with expert cache
        self.block = torch.nn.ModuleList(
            [
                LazyTransformerBlock(
                    config, layer_idx, self.expert_cache, device=device
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            dtype=torch.bfloat16,
            bias=False,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the lazy transformer."""
        # Components should already be on the correct device from initialization
        x = self.embedding(x)
        for block in self.block:
            x = block(x)
        x = self.norm(x)
        x = self.unembedding(x)
        return x

    @staticmethod
    def from_model_type(
        model_type: ModelType, device: str | torch.device
    ) -> "LazyTransformer":
        """
        Load transformer from model type with lazy loading.

        Args:
            model_type: Model type for automatic checkpoint path and config resolution
            device: Target device

        Returns:
            LazyTransformer with lazy loading enabled
        """
        if not isinstance(device, torch.device):
            device = torch.device(device)

        # Auto-resolve checkpoint directory from model type

        checkpoint_dir = get_checkpoint_path(model_type)

        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            model_config = ModelConfig(**json_config)

        model = LazyTransformer(
            config=model_config, model_type=model_type, device=device
        )
        model.eval()

        checkpoint = Checkpoint(checkpoint_dir, device)

        for name, param in model.named_parameters():
            # Skip MLP expert weights (they'll be loaded lazily)
            if (
                "mlp.mlp1_weight" in name
                or "mlp.mlp1_bias" in name
                or "mlp.mlp2_weight" in name
                or "mlp.mlp2_bias" in name
            ):
                continue

            # Load other parameters normally
            try:
                loaded_tensor = checkpoint.get(name)
                param.data.copy_(loaded_tensor)

            except Exception as e:
                print(f"Skipping parameter {name}: {e}")

        return model


class LazyTokenGenerator:
    """
    Token generator using LazyTransformer for memory-efficient inference.
    """

    @torch.inference_mode()
    def __init__(self, model_type: ModelType, device: torch.device):
        """
        Initialize lazy token generator.

        Args:
            model_type: Model type for automatic checkpoint resolution
            device: Target device for inference
        """
        self.device = device
        self.model = LazyTransformer.from_model_type(model_type, device=device)
        # Move model to device after creation
        self.model = self.model.to(device)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: list[int],
        stop_tokens: list[int],
        temperature: float = 1.0,
        max_tokens: int = 0,
        return_logprobs: bool = False,
    ):
        """
        Generate tokens using the lazy-loaded model.

        Args:
            prompt_tokens: Initial token sequence
            stop_tokens: Tokens that trigger generation stop
            temperature: Sampling temperature (0.0 for greedy)
            max_tokens: Maximum tokens to generate (0 for unlimited)
            return_logprobs: Whether to return log probabilities

        Yields:
            Generated tokens (and log probabilities if requested)
        """
        tokens = list(prompt_tokens)
        num_generated_tokens = 0

        while max_tokens == 0 or num_generated_tokens < max_tokens:
            logits = self.model(
                torch.as_tensor(tokens, dtype=torch.int32, device=self.device)
            )[-1]

            if temperature == 0.0:
                predicted_token = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = int(torch.multinomial(probs, num_samples=1).item())

            tokens.append(predicted_token)
            num_generated_tokens += 1

            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                break
