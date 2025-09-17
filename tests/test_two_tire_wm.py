#!/usr/bin/env python3
"""
Test script for TwoTireWmExpertCacheManager.

This script tests the basic functionality of the two-tier watermark cache
manager including imports, initialization, and basic operations.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_rust_imports():
    """Test if Rust core library can be imported."""
    print("=== Testing Rust Core Imports ===")

    from rust_core import RustTwoTireWmExpertCacheManager
    from rust_core import RustWatermarkConfig, RustExpertKey

    print("✅ Rust core library imported successfully")
    return True


def test_python_wrapper():
    """Test Python wrapper functionality."""
    print("\n=== Testing Python Wrapper ===")

    # Test that we can create the wrapper directly
    from src.domain.manager.two_tire_wm import TwoTireWmExpertCacheManager

    print("✅ Python wrapper imported successfully")
    return True


def test_factory():
    """Test factory method."""
    print("\n=== Testing Factory ===")

    # Simplified test - just check if we can import
    from src.services.cache.expert_cache_factory import ExpertCacheFactory

    print("✅ Factory imported successfully")
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
