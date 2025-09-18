"""
Integration layer for memory-efficient GPT-OSS model implementations.

This module provides integrated model classes that combine boilerplate
components with domain-specific optimizations like expert caching.
"""

import json
import os
from typing import Optional

import torch

from src.boilerplate.gpt_oss.model import AttentionBlock, ModelConfig, RMSNorm
from src.boilerplate.gpt_oss.weights import Checkpoint
from src.config import TORCH_VRAM_DEVICE
from src.config.util import get_checkpoint_path
from src.domain import ModelType
from src.domain.cache.interfaces.expert_cache import IExpertCacheManager
from src.domain.manager import CacheManagerType
from src.services.cache import ExpertCacheFactory

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
        expert_cache_manager: IExpertCacheManager,
    ):
        """
        Initialize lazy transformer block.

        Args:
            config: Model configuration
            layer_idx: Layer index
            expert_cache: Expert cache instance for loading weights
        """
        super().__init__()
        self.layer_idx = layer_idx

        # Use regular attention block with global device config
        self.attn = AttentionBlock(config, layer_idx, device=TORCH_VRAM_DEVICE)

        # Use lazy MLP block with expert cache for memory efficiency
        self.mlp = LazyMLPBlock(config, expert_cache_manager, layer_idx)

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
        cache_manager_type: CacheManagerType = CacheManagerType.DIRECT_RAM,
    ):
        """
        Initialize lazy transformer with expert cache system.

        Args:
            config: Model configuration
            model_type: Model type for automatic checkpoint path resolution
            cache_manager_type: Type of cache manager to use (default: DIRECT_RAM)
        """
        super().__init__()
        self.config = config
        
        # Create appropriate cache manager based on type
        match cache_manager_type:
            case CacheManagerType.LRU:
                self.expert_cache_manager = ExpertCacheFactory.create_lru_cache_manager(
                    model_type=model_type
                )
            case CacheManagerType.DIRECT_NVME:
                self.expert_cache_manager = ExpertCacheFactory.create_direct_nvme_cache_manager(
                    model_type=model_type
                )
            case CacheManagerType.DIRECT_RAM:
                self.expert_cache_manager = ExpertCacheFactory.create_direct_ram_cache_manager(
                    model_type=model_type
                )
            case CacheManagerType.TWO_TIER_WM:
                self.expert_cache_manager = ExpertCacheFactory.create_two_tier_wm_cache_manager(
                    model_type=model_type
                )
            case _:
                raise ValueError(f"Unsupported cache manager type: {cache_manager_type}")

        # Standard components with global device config
        self.embedding = torch.nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=torch.bfloat16,
            device=TORCH_VRAM_DEVICE,
        )

        # Use lazy transformer blocks with expert cache
        self.block = torch.nn.ModuleList(
            [
                LazyTransformerBlock(config, layer_idx, self.expert_cache_manager)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, device=TORCH_VRAM_DEVICE)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            dtype=torch.bfloat16,
            bias=False,
            device=TORCH_VRAM_DEVICE,
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
        model_type: ModelType, 
        cache_manager_type: CacheManagerType = CacheManagerType.DIRECT_RAM
    ) -> "LazyTransformer":
        """
        Load transformer from model type with lazy loading.

        Args:
            model_type: Model type for automatic checkpoint path and config resolution
            cache_manager_type: Type of cache manager to use (default: DIRECT_RAM)

        Returns:
            LazyTransformer with lazy loading enabled
        """
        # Auto-resolve checkpoint directory from model type
        checkpoint_dir = get_checkpoint_path(model_type)

        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            model_config = ModelConfig(**json_config)

        model = LazyTransformer(
            config=model_config, 
            model_type=model_type,
            cache_manager_type=cache_manager_type
        )
        model.eval()

        checkpoint = Checkpoint(checkpoint_dir, TORCH_VRAM_DEVICE)

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
        self.model = LazyTransformer.from_model_type(model_type)
        # Move model to device after creation (components already on TORCH_VRAM_DEVICE)
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
