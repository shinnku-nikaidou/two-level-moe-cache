#!/usr/bin/env python3
"""
Test DirectRAM expert cache manager with full pre-warming strategy.

This test validates that DirectRAM correctly:
1. Pre-loads ALL experts to RAM during initialization
2. Returns VRAM experts from get() and get_batch() operations
3. Maintains RAM cache while providing VRAM computation copies
"""

import sys
import os
from src.common.types import ExpertKey, ExpertParamType

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.manager.direct_ram import DirectRAMExpertCacheManager
from src.domain import ModelType

# Global cache instance - shared across all tests to avoid multiple pre-warming
_shared_cache = None


def get_shared_cache():
    """Get or create shared DirectRAM cache instance."""
    global _shared_cache
    if _shared_cache is None:
        print(
            "🔥 Creating shared DirectRAM cache (this will take time for pre-warming)..."
        )
        _shared_cache = DirectRAMExpertCacheManager(ModelType.GPT_OSS_20B)
        print("✅ Shared cache created and pre-warmed!")
    return _shared_cache


def test_direct_ram_prewarming():
    """Test DirectRAM pre-warming behavior during initialization."""

    # Use shared cache instance
    cache = get_shared_cache()

    # Verify all experts are pre-loaded to RAM
    total_experts = cache.get_total_expert_count()
    loaded_experts = cache.get_loaded_expert_count()
    vram_experts = cache.get_vram_expert_count()

    print(f"\n📊 Post-initialization statistics:")
    print(f"   Total experts: {total_experts}")
    print(f"   Experts loaded in RAM: {loaded_experts}")
    print(f"   Experts in VRAM: {vram_experts}")

    # Verify pre-warming worked
    if loaded_experts == total_experts:
        print("✅ SUCCESS: All experts pre-loaded to RAM!")
    else:
        print(f"❌ FAILED: Only {loaded_experts}/{total_experts} experts in RAM")
        return False

    if vram_experts == 0:
        print("✅ SUCCESS: No VRAM usage during init (as expected)")
    else:
        print(f"⚠️  WARNING: {vram_experts} experts in VRAM after init")

    return True


def test_direct_ram_get_behavior():
    """Test that get() returns VRAM experts from RAM cache."""

    print("\n" + "=" * 60)
    print("🚀 Testing DirectRAM get() Behavior")
    print("=" * 60)

    cache = get_shared_cache()

    # Test single expert retrieval
    test_key = ExpertKey(
        layer_idx=0, expert_id=0, param_type=ExpertParamType.MLP1_WEIGHT
    )

    print(f"\nTesting get() for expert: {test_key}")

    try:
        expert = cache.get(test_key)

        print(f"✅ Retrieved expert: {expert}")
        print(f"   Current tier: {expert.current_tier}")
        print(f"   Is in VRAM: {expert.is_in_vram}")
        print(f"   Data RAM exists: {expert.data_ram is not None}")
        print(f"   Data VRAM exists: {expert.data_vram is not None}")

        # Verify expert is in VRAM after get()
        if expert.is_in_vram:
            print("✅ SUCCESS: Expert is in VRAM after get()")
        else:
            print("❌ FAILED: Expert not in VRAM after get()")
            return False

        # Verify RAM copy still exists (cache preservation)
        if expert.data_ram is not None:
            print("✅ SUCCESS: RAM cache preserved")
        else:
            print("❌ FAILED: RAM cache was lost")
            return False

        return True

    except Exception as e:
        print(f"❌ FAILED: get() threw exception: {e}")
        return False


def test_direct_ram_batch_behavior():
    """Test that get_batch() returns VRAM experts efficiently."""

    print("\n" + "=" * 60)
    print("📦 Testing DirectRAM get_batch() Behavior")
    print("=" * 60)

    cache = get_shared_cache()

    # Test batch retrieval
    test_keys = [
        ExpertKey(layer_idx=0, expert_id=i, param_type=ExpertParamType.MLP1_WEIGHT)
        for i in range(3)
    ]

    print(f"\nTesting get_batch() for {len(test_keys)} experts")

    try:
        experts = cache.get_batch(test_keys)

        print(f"✅ Retrieved {len(experts)} experts in batch")

        # Verify all experts are in VRAM
        vram_count = 0
        ram_count = 0
        for i, expert in enumerate(experts):
            print(
                f"   Expert {i}: tier={expert.current_tier}, "
                f"RAM={expert.data_ram is not None}, "
                f"VRAM={expert.data_vram is not None}"
            )

            if expert.is_in_vram:
                vram_count += 1
            if expert.data_ram is not None:
                ram_count += 1

        if vram_count == len(experts):
            print(f"✅ SUCCESS: All {vram_count} experts in VRAM")
        else:
            print(f"❌ FAILED: Only {vram_count}/{len(experts)} experts in VRAM")
            return False

        if ram_count == len(experts):
            print(f"✅ SUCCESS: All {ram_count} experts have RAM cache")
        else:
            print(f"❌ FAILED: Only {ram_count}/{len(experts)} experts have RAM cache")
            return False

        return True

    except Exception as e:
        print(f"❌ FAILED: get_batch() threw exception: {e}")
        return False


def test_direct_ram_memory_efficiency():
    """Test memory usage patterns and clear() behavior."""

    print("\n" + "=" * 60)
    print("🧠 Testing DirectRAM Memory Management")
    print("=" * 60)

    cache = get_shared_cache()

    # Get initial state
    initial_vram = cache.get_vram_expert_count()
    initial_ram = cache.get_loaded_expert_count()

    print(f"\nInitial state: RAM={initial_ram}, VRAM={initial_vram}")

    # Get a few experts to VRAM
    test_keys = [
        ExpertKey(layer_idx=0, expert_id=i, param_type=ExpertParamType.MLP1_WEIGHT)
        for i in range(5)
    ]

    experts = cache.get_batch(test_keys)
    after_get_vram = cache.get_vram_expert_count()
    after_get_ram = cache.get_loaded_expert_count()

    print(f"After get_batch(): RAM={after_get_ram}, VRAM={after_get_vram}")

    # Test clear() - should remove VRAM but keep RAM
    cache.clear()
    after_clear_vram = cache.get_vram_expert_count()
    after_clear_ram = cache.get_loaded_expert_count()

    print(f"After clear(): RAM={after_clear_ram}, VRAM={after_clear_vram}")

    # Verify behavior
    success = True
    if after_clear_ram == initial_ram:
        print("✅ SUCCESS: RAM cache preserved after clear()")
    else:
        print(f"❌ FAILED: RAM cache changed: {initial_ram} -> {after_clear_ram}")
        success = False

    if after_clear_vram < after_get_vram:
        print("✅ SUCCESS: VRAM usage reduced after clear()")
    else:
        print(f"❌ FAILED: VRAM not cleared: {after_get_vram} -> {after_clear_vram}")
        success = False

    return success


def run_all_tests():
    """Run all DirectRAM tests."""

    print("🧪 Starting DirectRAM Expert Cache Tests")
    print("=" * 80)

    tests = [
        ("Pre-warming Initialization", test_direct_ram_prewarming),
        ("get() Behavior", test_direct_ram_get_behavior),
        ("get_batch() Behavior", test_direct_ram_batch_behavior),
        ("Memory Management", test_direct_ram_memory_efficiency),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))

    # Final summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED - DirectRAM implementation is working correctly!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed - needs investigation")
        return False


if __name__ == "__main__":
    run_all_tests()
