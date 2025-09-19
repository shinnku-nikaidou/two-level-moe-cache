"""
Token generation utilities for GPT-OSS models.

This module provides token generation classes that work with different
GPT-OSS model implementations, including lazy-loading variants.
"""

import torch

from src.domain import ModelType
from src.domain.manager import CacheManagerType

from .model import LazyTransformer


class LazyTokenGenerator:
    """
    Token generator using LazyTransformer for memory-efficient inference.

    This generator leverages the expert cache system to minimize memory usage
    during token generation, making it suitable for large models on
    resource-constrained hardware.
    """

    @torch.inference_mode()
    def __init__(
        self,
        model_type: ModelType,
        device: torch.device,
        cache_manager_type: CacheManagerType,
    ):
        """
        Initialize lazy token generator.

        Args:
            model_type: Model type for automatic checkpoint resolution
            device: Target device for inference
            cache_manager_type: Type of cache manager to use
        """
        self.device = device
        self.model = LazyTransformer.from_model_type(model_type, cache_manager_type)
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
