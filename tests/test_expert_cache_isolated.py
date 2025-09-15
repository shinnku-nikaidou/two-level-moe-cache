#!/usr/bin/env python3
"""
Isolated test for Expert Cache system to debug loading issues
"""
import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.domain.cache.entities.expert import Expert, ExpertKey
from src.domain import ModelType
from src.services.cache.expert_cache_factory import ExpertCacheFactory
from src.config.cache_config import CacheConfig

MODEL_PATH = os.path.join(project_root, "data", "models", "gpt-oss-20b", "original")


def test_single_expert_loading():
    """Test loading a single expert directly"""
    print("=" * 60)
    print("Test 1: Direct Expert Loading")
    print("=" * 60)

    # Create a single expert key
    expert_key = ExpertKey(layer_idx=0, expert_id=1, param_type="mlp1_weight")

    print(f"Testing expert key: {expert_key}")

    # Create expert instance directly
    expert = Expert(expert_key=expert_key, model_type=ModelType.GPT_OSS_20B)

    print(f"Expert created, current_tier: {expert.current_tier}")
    print(f"Expert data is None: {expert.data is None}")

    try:
        # Load expert data
        expert.load_from_nvme_to_ram()

        print(f"After loading:")
        print(f"  current_tier: {expert.current_tier}")
        print(f"  data is None: {expert.data is None}")
        if expert.data is not None:
            print(f"  data shape: {expert.data.shape}")
            print(f"  data device: {expert.data.device}")
            print(f"  data dtype: {expert.data.dtype}")

        return expert.data is not None

    except Exception as e:
        print(f"Error loading expert: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_expert_cache_single():
    """Test loading expert through cache manager"""
    print("\n" + "=" * 60)
    print("Test 2: Expert Cache Manager Loading")
    print("=" * 60)

    # Create cache
    cache_config = CacheConfig.for_model(ModelType.GPT_OSS_20B)
    expert_cache = ExpertCacheFactory.create_lru_cache(
        model_type=ModelType.GPT_OSS_20B,
        config=cache_config,
        checkpoint_path=MODEL_PATH,
    )

    print(f"Expert cache created: {type(expert_cache)}")

    # Create expert key
    expert_key = ExpertKey(layer_idx=0, expert_id=1, param_type="mlp1_weight")

    print(f"Testing expert key: {expert_key}")

    try:
        # Get expert from cache
        expert = expert_cache.get(expert_key)

        print(f"Expert retrieved from cache:")
        print(f"  expert is None: {expert is None}")
        if expert is not None:
            print(f"  current_tier: {expert.current_tier}")
            print(f"  data is None: {expert.data is None}")
            if expert.data is not None:
                print(f"  data shape: {expert.data.shape}")
                print(f"  data device: {expert.data.device}")
                print(f"  data dtype: {expert.data.dtype}")

        return expert is not None and expert.data is not None

    except Exception as e:
        print(f"Error loading expert through cache: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_expert_cache_batch():
    """Test batch loading through cache manager"""
    print("\n" + "=" * 60)
    print("Test 3: Expert Cache Batch Loading")
    print("=" * 60)

    # Create cache
    cache_config = CacheConfig.for_model(ModelType.GPT_OSS_20B)
    expert_cache = ExpertCacheFactory.create_lru_cache(
        model_type=ModelType.GPT_OSS_20B,
        config=cache_config,
        checkpoint_path=MODEL_PATH,
    )

    # Create multiple expert keys
    expert_keys = [
        ExpertKey(layer_idx=0, expert_id=1, param_type="mlp1_weight"),
        ExpertKey(layer_idx=0, expert_id=2, param_type="mlp1_weight"),
        ExpertKey(layer_idx=0, expert_id=1, param_type="mlp1_bias"),
    ]

    print(f"Testing batch loading for {len(expert_keys)} experts")
    for key in expert_keys:
        print(f"  - {key}")

    try:
        # Get experts in batch
        expert_dict = expert_cache.get_batch(expert_keys)

        print(f"\nBatch loaded {len(expert_dict)} experts:")
        all_loaded = True

        for key, expert in expert_dict.items():
            print(f"  {key}:")
            print(f"    expert is None: {expert is None}")
            if expert is not None:
                print(f"    current_tier: {expert.current_tier}")
                print(f"    data is None: {expert.data is None}")
                if expert.data is not None:
                    print(f"    data shape: {expert.data.shape}")
                    print(f"    data device: {expert.data.device}")
                else:
                    all_loaded = False
            else:
                all_loaded = False

        return all_loaded

    except Exception as e:
        print(f"Error in batch loading: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_lazy_tensor_simulation():
    """Simulate the LazyExpertTensor loading pattern"""
    print("\n" + "=" * 60)
    print("Test 4: LazyExpertTensor Pattern Simulation")
    print("=" * 60)

    # Create cache
    cache_config = CacheConfig.for_model(ModelType.GPT_OSS_20B)
    expert_cache = ExpertCacheFactory.create_lru_cache(
        model_type=ModelType.GPT_OSS_20B,
        config=cache_config,
        checkpoint_path=MODEL_PATH,
    )

    # Simulate expert indices from routing (like in real usage)
    expert_indices = torch.tensor([1, 2, 5, 1])  # Some experts, with duplicates
    layer_idx = 0
    param_type = "mlp1_weight"

    print(f"Simulating loading for layer {layer_idx}, param {param_type}")
    print(f"Expert indices: {expert_indices.tolist()}")

    # Create required keys (similar to LazyExpertTensor)
    required_keys = []
    for idx in expert_indices.cpu().tolist():
        key = ExpertKey(layer_idx=layer_idx, expert_id=idx, param_type=param_type)
        required_keys.append(key)

    # Remove duplicates while preserving order
    seen = set()
    unique_keys = []
    for key in required_keys:
        if key not in seen:
            unique_keys.append(key)
            seen.add(key)

    print(f"Unique keys to load: {len(unique_keys)}")
    for key in unique_keys:
        print(f"  - {key}")

    try:
        # Load experts from cache in batch (like LazyExpertTensor does)
        expert_dict = expert_cache.get_batch(unique_keys)

        print(f"\nLoaded {len(expert_dict)} experts from cache")

        # Check each expert (similar to LazyExpertTensor validation)
        all_valid = True
        for key, expert in expert_dict.items():
            print(f"Checking {key}:")
            if expert.data is None:
                print(f"  ERROR: Expert {key} data is None after loading!")
                all_valid = False
            else:
                print(f"  OK: data shape {expert.data.shape}")

        if all_valid:
            print("\n✅ All experts loaded successfully!")
            # Try to build tensor mapping like LazyExpertTensor
            expert_tensors = {}
            for key, expert in expert_dict.items():
                expert_tensors[key.expert_id] = expert.data

            print(f"Built tensor mapping for expert IDs: {list(expert_tensors.keys())}")
            return True
        else:
            print("\n❌ Some experts failed to load!")
            return False

    except Exception as e:
        print(f"Error in simulation: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Expert Cache Isolated Testing")
    print(f"Model path: {MODEL_PATH}")

    tests = [
        ("Direct Expert Loading", test_single_expert_loading),
        ("Cache Single Loading", test_expert_cache_single),
        ("Cache Batch Loading", test_expert_cache_batch),
        ("LazyTensor Simulation", test_lazy_tensor_simulation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nTest {test_name} crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed! Expert cache system is working correctly.")
    else:
        print("\n💥 Some tests failed. Need to debug the expert cache system.")
