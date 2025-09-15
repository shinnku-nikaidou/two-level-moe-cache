#!/usr/bin/env python3
"""
Test LazyTransformer initialization and expert cache integration
"""
import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.domain.gpt_oss.model import LazyTransformer
from src.domain import ModelType
from src.services.cache.expert_cache_factory import ExpertCacheFactory
from src.config.cache_config import CacheConfig

MODEL_PATH = os.path.join(project_root, "data", "models", "gpt-oss-20b", "original")


def test_lazy_transformer_creation():
    """Test LazyTransformer creation step by step"""
    print("=" * 60)
    print("Test: LazyTransformer Creation and Initialization")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Step 1: Create LazyTransformer
        print("\nStep 1: Creating LazyTransformer...")
        model = LazyTransformer.from_checkpoint(MODEL_PATH, device=device)
        print(f"✅ LazyTransformer created successfully")
        print(f"  Model type: {type(model)}")
        print(f"  Device: {model.device}")
        print(f"  Expert cache type: {type(model.expert_cache)}")

        # Step 2: Inspect model structure
        print(f"\nStep 2: Model structure inspection...")
        print(f"  Number of blocks: {len(model.block)}")

        # Inspect first block
        first_block = model.block[0]
        print(f"  First block type: {type(first_block)}")
        print(f"  First block MLP type: {type(first_block.mlp)}")

        # Inspect MLP expert tensors (if it's LazyMLPBlock)
        mlp = first_block.mlp
        print(f"  MLP type: {type(mlp)}")
        if hasattr(mlp, "mlp1_weight"):
            mlp1_weight = getattr(mlp, "mlp1_weight")
            mlp1_bias = getattr(mlp, "mlp1_bias")
            print(f"  MLP mlp1_weight type: {type(mlp1_weight)}")
            print(f"  MLP mlp1_bias type: {type(mlp1_bias)}")
            if hasattr(mlp1_weight, "expert_cache"):
                expert_cache = getattr(mlp1_weight, "expert_cache")
                print(f"  MLP expert cache: {expert_cache is not None}")
        else:
            print(f"  MLP doesn't have mlp1_weight attribute (regular MLP, not Lazy)")

        return True

    except Exception as e:
        print(f"❌ Error creating LazyTransformer: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_lazy_transformer_single_forward():
    """Test single token forward pass"""
    print("\n" + "=" * 60)
    print("Test: LazyTransformer Single Forward Pass")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Create model
        model = LazyTransformer.from_checkpoint(MODEL_PATH, device=device)

        # Prepare single token input
        test_token = 13225  # "Hello" token
        test_input = torch.tensor([test_token], dtype=torch.int32, device=device)

        print(f"Input token: {test_token}")
        print(
            f"Input tensor: {test_input.shape}, dtype: {test_input.dtype}, device: {test_input.device}"
        )

        # Forward pass
        print("\nPerforming forward pass...")
        with torch.no_grad():
            output = model(test_input)

        print(f"✅ Forward pass successful!")
        print(
            f"Output shape: {output.shape}, dtype: {output.dtype}, device: {output.device}"
        )

        # Check output values
        logits = output[0]  # First token logits
        top_5_probs, top_5_indices = torch.topk(torch.softmax(logits, dim=-1), 5)

        print(f"\nTop 5 predictions:")
        for i in range(5):
            token_id = top_5_indices[i].item()
            prob = top_5_probs[i].item()
            print(f"  {i+1}. Token {token_id} (prob: {prob:.4f})")

        return True

    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_expert_cache_usage_during_forward():
    """Test expert cache usage patterns during forward pass"""
    print("\n" + "=" * 60)
    print("Test: Expert Cache Usage During Forward")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Create model
        model = LazyTransformer.from_checkpoint(MODEL_PATH, device=device)

        # Get cache statistics before (if available)
        cache = model.expert_cache
        stats_before = None
        if hasattr(cache, "get_stats"):
            stats_before = getattr(cache, "get_stats")()
            print(f"Cache stats before: {stats_before}")
        else:
            print("Cache doesn't have get_stats method")

        # Prepare input
        test_input = torch.tensor([13225], dtype=torch.int32, device=device)

        # Forward pass with detailed logging
        print(f"\nStarting forward pass with detailed expert loading...")

        with torch.no_grad():
            # Disable debug prints for cleaner output
            # Let's temporarily remove debug prints
            output = model(test_input)

        # Get cache statistics after (if available)
        stats_after = None
        if hasattr(cache, "get_stats"):
            stats_after = getattr(cache, "get_stats")()
            print(f"Cache stats after: {stats_after}")

        print(f"✅ Forward pass completed successfully!")

        if stats_before is not None and stats_after is not None:
            print(f"Cache hits: {stats_after['hits'] - stats_before['hits']}")
            print(f"Cache misses: {stats_after['misses'] - stats_before['misses']}")
            print(f"Cache loads: {stats_after['loads'] - stats_before['loads']}")
        else:
            print("Cache statistics not available")

        return True

    except Exception as e:
        print(f"❌ Error during cache usage test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_expert_cache_direct_access():
    """Test accessing the same expert cache directly"""
    print("\n" + "=" * 60)
    print("Test: Direct Expert Cache Access")
    print("=" * 60)

    try:
        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LazyTransformer.from_checkpoint(MODEL_PATH, device=device)

        # Get the expert cache from the model
        cache = model.expert_cache

        # Try to access some experts that would be used in forward pass
        from src.domain.cache.entities.expert import ExpertKey

        test_keys = [
            ExpertKey(layer_idx=0, expert_id=1, param_type="mlp1_weight"),
            ExpertKey(layer_idx=0, expert_id=1, param_type="mlp1_bias"),
            ExpertKey(layer_idx=0, expert_id=2, param_type="mlp1_weight"),
            ExpertKey(layer_idx=0, expert_id=2, param_type="mlp1_bias"),
        ]

        print(f"Testing direct cache access for {len(test_keys)} experts...")

        expert_dict = cache.get_batch(test_keys)

        print(f"Retrieved {len(expert_dict)} experts from cache:")
        all_valid = True

        for key, expert in expert_dict.items():
            print(f"  {key}:")
            if expert is None:
                print(f"    ❌ Expert is None!")
                all_valid = False
            elif expert.data is None:
                print(f"    ❌ Expert data is None!")
                all_valid = False
            else:
                print(
                    f"    ✅ OK: shape {expert.data.shape}, device {expert.data.device}"
                )

        if all_valid:
            print("\n✅ All experts loaded successfully through model's cache!")
        else:
            print("\n❌ Some experts failed to load through model's cache!")

        return all_valid

    except Exception as e:
        print(f"❌ Error in direct cache access: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("LazyTransformer Integration Testing")
    print(f"Model path: {MODEL_PATH}")

    tests = [
        ("LazyTransformer Creation", test_lazy_transformer_creation),
        ("Direct Cache Access", test_expert_cache_direct_access),
        ("Cache Usage During Forward", test_expert_cache_usage_during_forward),
        ("Single Forward Pass", test_lazy_transformer_single_forward),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")

        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nTest {test_name} crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All integration tests passed!")
    else:
        print("\n💥 Some integration tests failed.")
