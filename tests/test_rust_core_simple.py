#!/usr/bin/env python3
"""
Simple test for Rust core module to verify basic functionality.
"""


def test_rust_core_basic():
    """Test basic Rust core functionality."""
    print("=== Testing Rust Core Basic ===")

    # Import all the Rust types
    try:
        from rust_core import (
            TwoTierWmExpertCacheManager,
            WatermarkConfig,
            ExpertKey,
            ExpertParamType,
            MemoryTier,
        )
    except ImportError as e:
        print(f"❌ Failed to import Rust types: {e}")
        return False

    print("✅ All Rust types imported successfully")

    # Test MemoryTier constants
    print(f"MemoryTier.VRAM: {MemoryTier.VRAM}")
    print(f"MemoryTier.RAM: {MemoryTier.RAM}")
    print(f"MemoryTier.DISK: {MemoryTier.DISK}")

    # Test ExpertParamType constants
    print(f"ExpertParamType.MLP1_WEIGHT: {ExpertParamType.MLP1_WEIGHT}")
    print(f"ExpertParamType.MLP1_BIAS: {ExpertParamType.MLP1_BIAS}")
    print(f"ExpertParamType.MLP2_WEIGHT: {ExpertParamType.MLP2_WEIGHT}")
    print(f"ExpertParamType.MLP2_BIAS: {ExpertParamType.MLP2_BIAS}")

    # Test creating ExpertKey
    expert_key = ExpertKey(
        layer_idx=0, expert_id=1, param_type=ExpertParamType.MLP1_WEIGHT
    )
    print(f"✅ ExpertKey created: {expert_key}")

    # Test creating WatermarkConfig
    config = WatermarkConfig(
        vram_capacity=1024,
        ram_capacity=4096,
        vram_learning_rate=0.01,
        ram_learning_rate=0.01,
        fusion_eta=0.5,
        reuse_decay_gamma=0.1,
    )
    print(
        f"✅ WatermarkConfig created with VRAM: {config.vram_capacity}, RAM: {config.ram_capacity}"
    )

    # Test creating TwoTierWmExpertCacheManager
    cache_manager = TwoTierWmExpertCacheManager(
        model_type="gpt-oss-20b",
        total_layers=24,
        vram_capacity=1024 * 1024 * 1024,  # 1GB
        ram_capacity=4096 * 1024 * 1024,  # 4GB
    )
    print("✅ TwoTierWmExpertCacheManager created successfully")

    return True


def main():
    """Run all tests."""
    print("Testing Rust Core Module")
    print("=" * 40)

    try:
        if test_rust_core_basic():
            print("\n🎉 All tests passed!")
            return 0
        else:
            print("\n⚠️ Some tests failed")
            return 1
    except Exception as e:
        print(f"\n❌ Test crashed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
