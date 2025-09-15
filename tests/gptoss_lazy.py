#!/usr/bin/env python3
"""
Test for LazyTransformer with lazy MoE expert loading
"""
import os
import sys
import torch
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.domain.gpt_oss.model import LazyTransformer
from src.domain import ModelType
from src.boilerplate.gpt_oss.tokenizer import get_tokenizer

MODEL_PATH = os.path.join(project_root, "data", "models", "gpt-oss-20b", "original")


class LazyTokenGenerator:
    """LazyTransformer version of TokenGenerator"""

    @torch.inference_mode()
    def __init__(self, model_type: ModelType, device: torch.device):
        self.device = device
        self.model = self._load_lazy_model(model_type)

    def _load_lazy_model(self, model_type: ModelType) -> LazyTransformer:
        """Load LazyTransformer with lazy expert loading"""
        model = LazyTransformer.from_model_type(model_type)
        model.eval()
        return model

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: list[int],
        stop_tokens: list[int],
        temperature: float = 1.0,
        max_tokens: int = 0,
        return_logprobs: bool = False,
    ):
        tokens = list(prompt_tokens)
        num_generated_tokens = 0
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            # Only pass the last context_length tokens to avoid memory issues
            context_length = 4096
            input_tokens = (
                tokens[-context_length:] if len(tokens) > context_length else tokens
            )

            # Use full context like the working gptoss_cpu_boil.py
            logits = self.model(
                torch.as_tensor(input_tokens, dtype=torch.int32, device=self.device)
            )[
                -1
            ]  # Take logits for the last position

            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()

            tokens.append(predicted_token)  # pyright: ignore[reportArgumentType]
            num_generated_tokens += 1

            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[
                    predicted_token
                ].item()  # pyright: ignore[reportArgumentType]
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                break


def test_basic_functionality():
    """Test basic LazyTransformer functionality without generation loop"""
    print("Basic Functionality Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load LazyTransformer (device is configured globally)
    model = LazyTransformer.from_model_type(ModelType.GPT_OSS_20B)
    print("LazyTransformer loaded successfully")

    # Get tokenizer
    tokenizer = get_tokenizer()

    # Test simple forward pass
    prompt = "Hello"
    tokens = tokenizer.encode(prompt)
    test_input = torch.as_tensor(tokens, dtype=torch.int32, device=device)

    print(f"\nTesting forward pass:")
    print(f"Input: '{prompt}' -> tokens: {tokens}")
    print(f"Input tensor: {test_input.shape}, dtype: {test_input.dtype}")

    try:
        with torch.no_grad():
            output = model(test_input)

        print(f"✅ Forward pass successful!")
        print(f"Output shape: {output.shape}, dtype: {output.dtype}")

        # Get top predictions
        logits = output[0]  # First (and only) token's logits
        top_5_probs, top_5_indices = torch.topk(torch.softmax(logits, dim=-1), 5)

        print(f"\nTop 5 predictions:")
        for i in range(5):
            token_id = top_5_indices[i].item()
            prob = top_5_probs[i].item()
            token_text = tokenizer.decode(
                [token_id]  # pyright: ignore[reportArgumentType]
            )
            print(f"  {i+1}. {repr(token_text)} (prob: {prob:.4f})")

        return True

    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_lazy_generation():
    """Test LazyTransformer generation"""
    print("Lazy MoE TransformerTest")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load LazyTransformer
    generator = LazyTokenGenerator(ModelType.GPT_OSS_20B, device=device)

    # Get tokenizer
    tokenizer = get_tokenizer()

    # Test simple prompts
    prompts = ["Hello", "The capital of France is", "114514, ", "rust lang"]

    for prompt in prompts:
        print(f"\nInput: '{prompt}'")
        tokens = tokenizer.encode(prompt)
        print(f"Token count: {len(tokens)}")

        temp = 0.7
        generated_tokens = []

        try:
            # Generate just a few tokens to test
            for i, token_or_tuple in enumerate(
                generator.generate(
                    tokens,
                    stop_tokens=[tokenizer.eot_token],
                    temperature=temp,
                    max_tokens=30,  # Only generate 3 tokens to test
                    return_logprobs=(temp > 0),
                )
            ):
                token, logprob = (
                    token_or_tuple  # pyright: ignore[reportGeneralTypeIssues]
                )
                token_text = tokenizer.decode(
                    [token]  # pyright: ignore[reportArgumentType]
                )
                print(f"  Token {i+1}: {repr(token_text)} (logprob: {logprob:.3f})")
                generated_tokens.append(token)

                if i >= 5:  # Stop after 3 tokens
                    break

            # Decode full text
            full_text = tokenizer.decode(tokens + generated_tokens)
            generated_part = tokenizer.decode(generated_tokens)

            print(f"Full text: {full_text}")
            print(f"Generated part: {repr(generated_part)}")

        except Exception as e:
            print(f"Generation failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    print("LazyTransformer with Lazy MoE Expert Loading Test")
    print(f"Model path: {MODEL_PATH}")

    try:
        # First test basic functionality
        if test_basic_functionality():
            print("\n" + "=" * 60)
            # Then test generation if basic test passes
            test_lazy_generation()

    except Exception as e:
        print(f"\nTest failed: {e}")

        traceback.print_exc()
