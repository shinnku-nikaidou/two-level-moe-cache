"""
Integration layer for memory-efficient GPT-OSS model implementations.

This module provides integrated model classes that combine boilerplate
components with domain-specific optimizations like lazy loading.
"""

import os
import json
import torch
import torch.distributed as dist

from ...boilerplate.gpt_oss.model import (
    Transformer as BaseTransformer,
    TransformerBlock as BaseTransformerBlock,
    ModelConfig,
    AttentionBlock,
    RMSNorm,
)
from ...boilerplate.gpt_oss.weights import Checkpoint
from .moe import LazyMLPBlock


class LazyTransformerBlock(torch.nn.Module):
    """
    Transformer block using LazyMLPBlock for memory-efficient expert loading.

    Replaces the standard MLP block with a lazy-loading version that only
    loads expert weights when needed.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        checkpoint_path: str,
        device: torch.device | None = None,
    ):
        """
        Initialize lazy transformer block.

        Args:
            config: Model configuration
            layer_idx: Layer index
            checkpoint_path: Path to checkpoint directory
            device: Target device for computations
        """
        super().__init__()
        self.layer_idx = layer_idx

        # Use regular attention block (no memory optimization needed)
        self.attn = AttentionBlock(config, layer_idx, device)

        # Use lazy MLP block for memory efficiency
        self.mlp = LazyMLPBlock(config, checkpoint_path, layer_idx, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention and lazy MLP blocks."""
        x = self.attn(x)
        x = self.mlp(x)
        return x


class LazyTransformer(torch.nn.Module):
    """
    Memory-efficient Transformer using lazy loading for MoE expert weights.

    This model loads expert weights on-demand, significantly reducing
    GPU memory usage compared to the standard implementation.
    """

    def __init__(
        self,
        config: ModelConfig,
        checkpoint_path: str | None = None,
        device: torch.device | None = None,
    ):
        """
        Initialize lazy transformer.

        Args:
            config: Model configuration
            checkpoint_path: Path to checkpoint directory (required for lazy loading)
            device: Target device for computations
        """
        super().__init__()
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device("cuda")

        # Standard components (no lazy loading needed)
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )

        # Lazy transformer blocks
        if checkpoint_path:
            self.block = torch.nn.ModuleList(
                [
                    LazyTransformerBlock(config, layer_idx, checkpoint_path, device)
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )
        else:
            # Fallback to regular blocks if no checkpoint path provided
            self.block = torch.nn.ModuleList(
                [
                    BaseTransformerBlock(config, layer_idx, device)
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )

        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the lazy transformer."""
        x = self.embedding(x)
        for block in self.block:
            x = block(x)
        x = self.norm(x)
        x = self.unembedding(x)
        return x

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda", lazy_loading: bool = True
    ) -> "LazyTransformer | BaseTransformer":
        """
        Load transformer from checkpoint with optional lazy loading.

        Args:
            path: Path to checkpoint directory
            device: Target device
            lazy_loading: Whether to use lazy loading for expert weights

        Returns:
            LazyTransformer if lazy_loading=True, otherwise BaseTransformer
        """
        if not isinstance(device, torch.device):
            device = torch.device(device)

        # Load configuration
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)

            # Map from actual config to our ModelConfig
            model_config = ModelConfig(
                num_hidden_layers=json_config.get("num_hidden_layers", 24),
                num_experts=json_config.get(
                    "num_local_experts", 32
                ),  # Note: different name
                experts_per_token=json_config.get("experts_per_token", 4),
                vocab_size=json_config.get("vocab_size", 201088),
                hidden_size=json_config.get("hidden_size", 2880),
                intermediate_size=json_config.get("intermediate_size", 2880),
                swiglu_limit=json_config.get("swiglu_limit", 7.0),
                head_dim=json_config.get("head_dim", 64),
                num_attention_heads=json_config.get("num_attention_heads", 64),
                num_key_value_heads=json_config.get("num_key_value_heads", 8),
                sliding_window=json_config.get("sliding_window", 128),
                initial_context_length=json_config.get("max_position_embeddings", 4096),
                rope_theta=json_config.get("rope_theta", 150000.0),
                # Handle rope_scaling if present
                rope_scaling_factor=(
                    json_config.get("rope_scaling", {}).get("factor", 32.0)
                    if "rope_scaling" in json_config
                    else 32.0
                ),
                rope_ntk_alpha=(
                    json_config.get("rope_scaling", {}).get("beta_slow", 1.0)
                    if "rope_scaling" in json_config
                    else 1.0
                ),
                rope_ntk_beta=(
                    json_config.get("rope_scaling", {}).get("beta_fast", 32.0)
                    if "rope_scaling" in json_config
                    else 32.0
                ),
            )

        if not lazy_loading:
            # Use original implementation
            return BaseTransformer.from_checkpoint(path, device)

        # Create lazy transformer
        model = LazyTransformer(
            config=model_config, checkpoint_path=path, device=device
        )
        model.eval()

        # Load non-expert weights normally
        checkpoint = Checkpoint(path, device)

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

                # Apply world_size sharding if needed
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

                param.data.copy_(loaded_tensor)

            except Exception as e:
                # Skip missing parameters (likely expert weights)
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
        self.model = LazyTransformer.from_checkpoint(
            checkpoint, device=device, lazy_loading=True
        )

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
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()

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
