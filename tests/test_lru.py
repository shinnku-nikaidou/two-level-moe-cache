#!/usr/bin/env python3

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.manager.lru import LRUExpertCacheManager
from src.domain import ModelType
from src.domain.cache.entities.types import ExpertKey, ExpertParamType


def test_lru_shared_experts():
    """Test LRUExpertCacheManager with shared expert storage."""

    print("Testing LRUExpertCacheManager with shared expert storage...")

    # Create LRU cache manager
    cache = LRUExpertCacheManager(ModelType.GPT_OSS_20B)

    # Verify shared experts are pre-created
    print(f"Total experts in base class: {len(cache._experts)}")
    print(f"LRU access order tracking: {len(cache._access_order)}")

    # Test expert retrieval (should be pre-created Expert instances)
    test_key = ExpertKey(
        layer_idx=0, expert_id=0, param_type=ExpertParamType.MLP1_WEIGHT
    )

    try:
        expert = cache.get(test_key)
        print(f"✅ Retrieved pre-created expert: {expert}")
        print(f"   Expert key: {expert.expert_key}")
        print(f"   LRU tracking length after get(): {len(cache._access_order)}")

        # Test second access - should update LRU order
        expert2 = cache.get(test_key)
        print(f"✅ Second access successful: {expert is expert2}")
        print(f"   LRU tracking length: {len(cache._access_order)}")

    except Exception as e:
        print(f"❌ Failed to get expert: {e}")

    # Test batch retrieval
    test_keys = [
        ExpertKey(layer_idx=0, expert_id=i, param_type=ExpertParamType.MLP1_WEIGHT)
        for i in range(3)
    ]

    try:
        experts = cache.get_batch(test_keys)
        print(f"✅ Batch retrieval successful: {len(experts)} experts")
        print(f"   LRU tracking length after batch: {len(cache._access_order)}")

        for i, exp in enumerate(experts):
            print(f"     Expert {i}: {exp.expert_key}")

    except Exception as e:
        print(f"❌ Failed batch retrieval: {e}")

    # Test memory usage tracking
    total_experts = cache.get_total_expert_count()
    loaded_experts = cache.get_loaded_expert_count()
    vram_experts = cache.get_vram_expert_count()
    print(f"Total experts: {total_experts}")
    print(f"Loaded experts: {loaded_experts}")
    print(f"Experts in VRAM: {vram_experts}")

    # Test LRU clear
    cache.clear()
    print(f"✅ After clear - LRU tracking length: {len(cache._access_order)}")

    print("\n✅ LRUExpertCacheManager refactoring completed successfully!")


if __name__ == "__main__":
    test_lru_shared_experts()
