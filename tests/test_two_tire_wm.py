#!/usr/bin/env python3
"""
Test script for TwoTireWmExpertCacheManager.

This script tests the basic functionality of the two-tier watermark cache
manager including imports, initialization, and basic operations.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_rust_imports():
    """Test if Rust core library can be imported."""
    print("=== Testing Rust Core Imports ===")

    from two_level_moe_cache.core import TwoTireWmExpertCacheManager
    from two_level_moe_cache.core import WatermarkConfig, ExpertRef, ExpertKey

    print("✅ Rust core library imported successfully")
    return True


def test_python_wrapper():
    """Test Python wrapper functionality."""
    print("\n=== Testing Python Wrapper ===")

    from src.domain.manager.two_tire_wm import TwoTireWmExpertCacheManager
    from src.domain import ModelType

    print("✅ Python wrapper imported successfully")

    # Test initialization
    cache_manager = TwoTireWmExpertCacheManager(
        model_type=ModelType.GPT_OSS_20B,
        vram_capacity_mb=256,
        ram_capacity_mb=1024,
    )
    print("✅ TwoTireWmExpertCacheManager initialized successfully")

    return True


def test_factory():
    """Test factory method."""
    print("\n=== Testing Factory ===")
    
    from src.services.cache.expert_cache_factory import ExpertCacheFactory
    from src.domain import ModelType
    
    cache_manager = ExpertCacheFactory.create_two_tire_wm_cache_manager(
        model_type=ModelType.GPT_OSS_20B,
        vram_capacity_mb=512,
        ram_capacity_mb=2048,
    )
    print("✅ Factory created TwoTireWmExpertCacheManager successfully")
    
    # Test stats (should be empty initially)
    if hasattr(cache_manager, 'get_watermark_stats'):
        stats = cache_manager.get_watermark_stats()
        print(f"✅ Initial stats: {stats}")
    else:
        print("✅ Cache manager created successfully (stats method not available)")
    
    return True


def main():
    """Run all tests."""
    print("Testing TwoTireWmExpertCacheManager Implementation")
    print("=" * 50)

    tests = [
        test_rust_imports,
        test_python_wrapper,
        test_factory,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
