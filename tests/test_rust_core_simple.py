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
            RustExpertKey,
            RustExpertParamType,
            RustMemoryTier,
            RustTwoTierWmExpertCacheManager,
        )
    except ImportError as e:
        print(f"❌ Failed to import Rust types: {e}")
        return False

    print("✅ All Rust types imported successfully")

    # Test MemoryTier constants
    print(f"MemoryTier.VRAM: {RustMemoryTier.VRAM}")
    print(f"MemoryTier.RAM: {RustMemoryTier.RAM}")
    print(f"MemoryTier.DISK: {RustMemoryTier.DISK}")

    # Test ExpertParamType constants
    print(f"ExpertParamType.MLP1_WEIGHT: {RustExpertParamType.MLP1_WEIGHT}")
    print(f"ExpertParamType.MLP1_BIAS: {RustExpertParamType.MLP1_BIAS}")
    print(f"ExpertParamType.MLP2_WEIGHT: {RustExpertParamType.MLP2_WEIGHT}")
    print(f"ExpertParamType.MLP2_BIAS: {RustExpertParamType.MLP2_BIAS}")

    # Test creating ExpertKey
    expert_key = RustExpertKey(
        layer_idx=0, expert_id=1, param_type=RustExpertParamType.MLP1_WEIGHT
    )
    print(f"✅ ExpertKey created: {expert_key}")

    # Test creating TwoTierWmExpertCacheManager
    cache_manager = RustTwoTierWmExpertCacheManager(
        model_type="gpt-oss-20b",
        vram_capacity=1024,  # 1GB in MB
        ram_capacity=4096,  # 4GB in MB
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
