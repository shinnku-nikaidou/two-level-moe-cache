#!/usr/bin/env python3
"""
Test for LazyTransformer with lazy MoE expert loading
"""
import os
import sys
import traceback
from datetime import datetime
import gc
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.boilerplate.gpt_oss.tokenizer import get_tokenizer
from src.domain import ModelType
from src.domain.gpt_oss.generator import LazyTokenGenerator
from src.domain.manager import CacheManagerType

MODEL_PATH = os.path.join(project_root, "data", "models", "gpt-oss-20b", "original")


def test_lazy_generation():
    """Test LazyTransformer generation"""
    print("Lazy MoE TransformerTest")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load LazyTransformer
        # Load LazyTransformer with TWO_TIER_WM cache
    generator = LazyTokenGenerator(
        ModelType.GPT_OSS_20B,
        device=device,
        cache_manager_type=CacheManagerType.DIRECT_NVME,
    )

    # Get tokenizer
    tokenizer = get_tokenizer()

    # Test simple prompts
    prompts = ["Hello", "The capital of France is", "large language model mixture of expert", "rust lang"]

    for prompt in prompts:
        print(f"\nInput: '{prompt}'")
        tokens = tokenizer.encode(prompt)
        print(f"Token count: {len(tokens)}")

        temp = 0.7
        generated_tokens = []

        try:
            # Generate just a few tokens to test
            for i, result in enumerate(
                generator.generate(
                    tokens,
                    stop_tokens=[tokenizer.eot_token],
                    temperature=temp,
                    max_tokens=200,  # Only generate 200 tokens to test
                    return_logprobs=(temp > 0),
                )
            ):
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                if temp > 0 and isinstance(result, tuple):
                    token, logprob = result
                    assert isinstance(token, int), f"Expected int, got {type(token)}"
                    token_text = tokenizer.decode([token])
                    
                    print(f"  Token {i+1}: {repr(token_text)} (logprob: {logprob:.3f}) {now}")
                    generated_tokens.append(token)
                else:
                    token = result
                    assert isinstance(token, int), f"Expected int, got {type(token)}"
                    token_text = tokenizer.decode([token])
                    print(f"  Token {i+1}: {repr(token_text)}  {now}")
                    generated_tokens.append(token)
                gc.collect()

                if i >= 50:  # Stop after 50 tokens
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
        test_lazy_generation()
    except Exception as e:
        print(f"\nTest failed: {e}")
        traceback.print_exc()
