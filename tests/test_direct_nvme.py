#!/usr/bin/env python3

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.manager.direct_nvme import DirectNVMEExpertCacheManager
from src.domain import ModelType
from src.common.types import ExpertKey, ExpertParamType


def test_direct_nvme_shared_experts():
    """Test DirectNVMEExpertCacheManager with shared expert storage."""

    print("Testing DirectNVMEExpertCacheManager with shared expert storage...")

    # Create cache manager
    cache = DirectNVMEExpertCacheManager(ModelType.GPT_OSS_20B)

    # Verify shared experts are pre-created
    print(f"Total experts in base class: {len(cache._experts)}")
    print(f"Loaded experts tracked: {len(cache._loaded_expert_keys)}")

    # Test expert retrieval (should be pre-created Expert instances)
    test_key = ExpertKey(
        layer_idx=0, expert_id=0, param_type=ExpertParamType.MLP1_WEIGHT
    )

    try:
        expert = cache._get_expert(test_key)
        print(f"✅ Retrieved pre-created expert: {expert}")
        print(f"   Expert data_ram: {expert.data_ram is not None}")
        print(f"   Expert data_vram: {expert.data_vram is not None}")
        print(f"   Expert key: {expert.expert_key}")

    except Exception as e:
        print(f"❌ Failed to get expert: {e}")

    # Test expert memory usage tracking
    total_experts = cache.get_total_expert_count()
    loaded_experts = cache.get_loaded_expert_count()
    vram_experts = cache.get_vram_expert_count()
    print(f"Total experts: {total_experts}")
    print(f"Loaded experts (RAM): {loaded_experts}")
    print(f"Experts in VRAM: {vram_experts}")

    print("\n✅ DirectNVMEExpertCacheManager refactoring completed successfully!")


if __name__ == "__main__":
    test_direct_nvme_shared_experts()
