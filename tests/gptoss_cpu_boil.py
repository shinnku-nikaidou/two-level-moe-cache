#!/usr/bin/env python3
"""
Fixed CPU test for gpt-oss compatibility
"""
import json
import os
import sys

import torch

# Ensure CUDA is not available
os.environ["CUDA_VISIBLE_DEVICES"] = ""

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.boilerplate.gpt_oss.model import ModelConfig, Transformer
from src.boilerplate.gpt_oss.tokenizer import get_tokenizer
from src.boilerplate.gpt_oss.weights import Checkpoint

# Use relative path to model
MODEL_PATH = os.path.join(project_root, "data", "models", "gpt-oss-20b", "original")


class CPUTokenGenerator:
    """Fixed CPU version of TokenGenerator"""

    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device):
        self.device = device
        self.model = self._load_model_cpu(checkpoint)

    def _load_model_cpu(self, path: str) -> Transformer:
        """Load model on CPU, bypassing distributed logic"""
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = ModelConfig(**json_config)

        model = Transformer(
            config=config,
            device=self.device,
        )
        model.eval()

        # Load weights directly without distributed logic
        checkpoint = Checkpoint(path, self.device)

        for name, param in model.named_parameters():
            try:
                loaded_tensor = checkpoint.get(name)
                param.data.copy_(loaded_tensor)
            except Exception as e:
                print(f"Loading failed: {name} - {e}")
                continue

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

            logits = self.model(
                torch.as_tensor(input_tokens, dtype=torch.int32, device=self.device)
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


def test_cpu_generation():
    """Test CPU version generation"""
    print("CPU TokenGenerator Test")
    print("=" * 60)

    device = torch.device("cpu")
    print(f"Using device: {device}")

    generator = CPUTokenGenerator(MODEL_PATH, device=device)

    # Get tokenizer
    tokenizer = get_tokenizer()

    # Test simple prompts
    prompts = [
        "Hello",
        "The capital of France is",
        "2 + 2 =",
    ]

    for prompt in prompts:
        print(f"\nInput: '{prompt}'")
        tokens = tokenizer.encode(prompt)
        print(f"Token count: {len(tokens)}")

        temp = 0.7
        generated_tokens = []

        try:
            # Generate 5 tokens
            for i, token_or_tuple in enumerate(
                generator.generate(
                    tokens,
                    stop_tokens=[tokenizer.eot_token],
                    temperature=temp,
                    max_tokens=20,
                    return_logprobs=(temp > 0),
                )
            ):
                token, logprob = token_or_tuple
                token_text = tokenizer.decode([token])
                print(f"  Token {i+1}: {repr(token_text)} (logprob: {logprob:.3f})")
                generated_tokens.append(token)

                if i >= 5:
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

        print("-" * 40)


if __name__ == "__main__":
    print("Fixed GPT-OSS CPU Test")
    print(f"Model path: {MODEL_PATH}")

    try:
        test_cpu_generation()
        print("\nTest completed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
