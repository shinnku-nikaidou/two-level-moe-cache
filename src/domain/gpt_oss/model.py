"""
Integration layer for memory-efficient GPT-OSS model implementations.

This module provides integrated model classes that combine boilerplate
components with domain-specific optimizations like expert caching.
"""

import os
import json
import torch
import torch.distributed as dist

from ...boilerplate.gpt_oss.model import (
    ModelConfig,
    AttentionBlock,
    RMSNorm,
)
from ...boilerplate.gpt_oss.weights import Checkpoint
from ..cache.interfaces.expert_cache import IExpertCache
from ...services.cache import ExpertCacheFactory
from ...config import CacheConfig
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
        checkpoint_path: str | None = None,
        expert_cache: IExpertCache | None = None,
        device: torch.device | None = None,
    ):
        """
        Initialize lazy transformer with expert cache system.

        Args:
            config: Model configuration
            checkpoint_path: Path to checkpoint directory (used if expert_cache is None)
            expert_cache: Pre-configured expert cache (creates default if None)
            device: Target device for components
        """
        super().__init__()
        self.config = config

        # Create expert cache if not provided
        if expert_cache is None:
            if checkpoint_path is None:
                raise ValueError(
                    "Either expert_cache or checkpoint_path must be provided"
                )

            # Ensure we have directory path for Checkpoint class
            checkpoint_dir = self._ensure_checkpoint_dir(checkpoint_path)

            # Auto-detect model type from checkpoint path
            model_type = self._detect_model_type(checkpoint_dir)

            # Create cache configuration optimized for the model
            cache_config = CacheConfig.for_model(model_type)

            # Create expert cache using factory
            self.expert_cache = ExpertCacheFactory.create_lru_cache(
                model_type=model_type,
                config=cache_config,
                checkpoint_path=checkpoint_dir,  # Pass directory path
            )
        else:
            self.expert_cache = expert_cache

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

    def _detect_model_type(self, checkpoint_path: str) -> ModelType:
        """
        Detect model type from checkpoint path.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Detected model type
        """
        checkpoint_path_lower = checkpoint_path.lower()

        if "gpt-oss-20b" in checkpoint_path_lower:
            return ModelType.GPT_OSS_20B
        elif "gpt-oss-120b" in checkpoint_path_lower:
            return ModelType.GPT_OSS_120B
        elif "phi-tiny-moe" in checkpoint_path_lower:
            return ModelType.PHI_TINY_MOE
        else:
            # Default to 20B model if cannot detect
            return ModelType.GPT_OSS_20B

    def _ensure_checkpoint_dir(self, checkpoint_path: str) -> str:
        """Ensure checkpoint_path points to directory, not file"""
        if checkpoint_path.endswith(".safetensors"):
            # If it's a file path, use its directory
            return os.path.dirname(checkpoint_path)
        return checkpoint_path

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
    def from_checkpoint(path: str, device: str | torch.device) -> "LazyTransformer":
        """
        Load transformer from checkpoint with lazy loading.

        Args:
            path: Path to checkpoint directory or file
            device: Target device

        Returns:
            LazyTransformer with lazy loading enabled
        """
        if not isinstance(device, torch.device):
            device = torch.device(device)

        # Ensure we have directory path for Checkpoint class
        checkpoint_dir = path
        if path.endswith(".safetensors"):
            checkpoint_dir = os.path.dirname(path)

        # Load configuration from checkpoint path (expects original subdirectory or direct path)
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)

            # Use direct mapping since original config matches ModelConfig fields
            model_config = ModelConfig(**json_config)

        # Create lazy transformer with specified device
        model = LazyTransformer(
            config=model_config, checkpoint_path=checkpoint_dir, device=device
        )
        model.eval()

        # Load non-expert weights normally - use correct device
        checkpoint = Checkpoint(checkpoint_dir, device)

        my_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

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

                # Apply world_size sharding if needed (but not for single node CPU)
                if world_size > 1:
                    per_rank_intermediate_size = (
                        model_config.intermediate_size // world_size
                    )
                    if "mlp1" in name:  # Non-expert MLP1 parameters (if any)
                        loaded_tensor = loaded_tensor[
                            :,
                            my_rank
                            * 2
                            * per_rank_intermediate_size : (my_rank + 1)
                            * 2
                            * per_rank_intermediate_size,
                            ...,
                        ]
                    elif "mlp2_weight" in name:  # Non-expert MLP2 weights (if any)
                        loaded_tensor = loaded_tensor[
                            ...,
                            my_rank
                            * per_rank_intermediate_size : (my_rank + 1)
                            * per_rank_intermediate_size,
                        ]

                # Ensure loaded tensor is on the same device as parameter
                if loaded_tensor.device != param.device:
                    loaded_tensor = loaded_tensor.to(param.device)

                param.data.copy_(loaded_tensor)

            except Exception as e:
                # Skip missing parameters - don't print for expected skips
                if "No CUDA GPUs are available" not in str(e):
                    print(f"Skipping parameter {name}: {e}")

        return model


class LazyTokenGenerator:
    """
    Token generator using LazyTransformer for memory-efficient inference.
    """

    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device):
        """
        Initialize lazy token generator.

        Args:
            checkpoint: Path to checkpoint directory
            device: Target device for inference
        """
        self.device = device
        self.model = LazyTransformer.from_checkpoint(checkpoint, device=device)
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
