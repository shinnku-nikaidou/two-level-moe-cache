"""
Command-line interface for GPT-OSS weight precomputation.

This CLI provides tools to precompute, validate, and manage precomputed
expert weights for dramatically faster inference loading times.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from src.infra.dataprocess.gpt_oss import (
    precompute_gpt_oss_20b,
    validate_precomputed_gpt_oss_20b,
    get_precomputed_storage_stats,
    is_precomputed_available,
)


def cmd_precompute(args) -> int:
    """Precompute expert weights command."""

    print("🚀 Starting GPT-OSS-20B weight precomputation...")
    print(f"Target dtype: {args.dtype}")
    print(f"Force rebuild: {args.force}")

    try:
        # Parse target dtype
        if args.dtype == "bfloat16":
            target_dtype = torch.bfloat16
        elif args.dtype == "float16":
            target_dtype = torch.float16
        else:
            print(f"❌ Unsupported dtype: {args.dtype}")
            return 1

        # Start precomputation
        start_time = time.time()
        precomputer = precompute_gpt_oss_20b(
            target_dtype=target_dtype, force_rebuild=args.force
        )
        end_time = time.time()

        # Show results
        stats = get_precomputed_storage_stats()
        duration = end_time - start_time

        print(f"\n✅ Precomputation completed in {duration:.1f} seconds")
        print(f"📊 Storage stats:")
        print(f"   Total files: {stats['total_files']}")
        print(f"   Total size: {stats['total_size_mb']:.1f} MB")
        print(f"   Completion rate: {stats['completion_rate']:.1%}")
        print(f"   Average file size: {stats['avg_file_size_kb']:.1f} KB")

        return 0

    except Exception as e:
        print(f"❌ Precomputation failed: {e}")
        return 1


def cmd_validate(args) -> int:
    """Validate precomputed weights command."""

    print(f"🔍 Validating precomputed weights ({args.samples} samples)...")

    try:
        if not is_precomputed_available():
            print("❌ Precomputed weights not available. Run precompute first.")
            return 1

        start_time = time.time()
        is_valid = validate_precomputed_gpt_oss_20b(num_samples=args.samples)
        end_time = time.time()

        duration = end_time - start_time
        print(
            f"\n{'✅ Validation PASSED' if is_valid else '❌ Validation FAILED'} in {duration:.1f} seconds"
        )

        return 0 if is_valid else 1

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1


def cmd_stats(args) -> int:
    """Show storage statistics command."""

    print("📊 Precomputed weight storage statistics:")

    try:
        stats = get_precomputed_storage_stats()

        if stats["status"] == "not_precomputed":
            print("❌ Precomputed weights not found. Run precompute first.")
            return 1

        print(f"\n🗂️  General Info:")
        print(f"   Status: {stats['status']}")
        print(f"   Total files: {stats['total_files']}")
        print(f"   Expected files: {stats['expected_files']}")
        print(f"   Completion rate: {stats['completion_rate']:.1%}")
        print(f"   Storage directory: {stats['precomputed_dir']}")

        print(f"\n💾 Storage Usage:")
        print(f"   Total size: {stats['total_size_mb']:.1f} MB")
        print(f"   Average file size: {stats['avg_file_size_kb']:.1f} KB")

        if args.detailed and "layer_stats" in stats:
            print(f"\n📋 Layer Breakdown:")
            for layer_name, layer_info in stats["layer_stats"].items():
                completion = layer_info["files"] / layer_info["expected_files"] * 100
                print(
                    f"   {layer_name}: {layer_info['files']}/{layer_info['expected_files']} files "
                    f"({completion:.0f}%), {layer_info['size_mb']:.1f} MB"
                )

        return 0

    except Exception as e:
        print(f"❌ Failed to get statistics: {e}")
        return 1


def cmd_clean(args) -> int:
    """Clean precomputed weights command."""

    try:
        stats = get_precomputed_storage_stats()

        if stats["status"] == "not_precomputed":
            print("ℹ️  No precomputed weights found to clean.")
            return 0

        precomputed_dir = Path(stats["precomputed_dir"])

        if not args.force:
            print(
                f"⚠️  This will delete {stats['total_files']} precomputed files ({stats['total_size_mb']:.1f} MB)"
            )
            print(f"Directory: {precomputed_dir}")

            confirm = input("Are you sure? (yes/no): ").strip().lower()
            if confirm not in ["yes", "y"]:
                print("Cancelled.")
                return 0

        # Remove all precomputed files
        import shutil

        if precomputed_dir.exists():
            shutil.rmtree(precomputed_dir)
            print(f"✅ Cleaned precomputed weights directory: {precomputed_dir}")
        else:
            print("ℹ️  Precomputed directory not found.")

        return 0

    except Exception as e:
        print(f"❌ Clean failed: {e}")
        return 1


def main():
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        description="GPT-OSS weight precomputation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Precompute all weights in bfloat16 format
  python -m src.cli.precompute precompute
  
  # Precompute with float16 format
  python -m src.cli.precompute precompute --dtype float16
  
  # Force rebuild existing weights
  python -m src.cli.precompute precompute --force
  
  # Validate precomputed weights
  python -m src.cli.precompute validate
  
  # Show storage statistics
  python -m src.cli.precompute stats
  
  # Show detailed layer statistics
  python -m src.cli.precompute stats --detailed
  
  # Clean all precomputed weights
  python -m src.cli.precompute clean --force
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Precompute command
    precompute_parser = subparsers.add_parser(
        "precompute", help="Precompute expert weights"
    )
    precompute_parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Target dtype for precomputed weights (default: bfloat16)",
    )
    precompute_parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild existing precomputed weights",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate precomputed weights"
    )
    validate_parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of random samples to validate (default: 20)",
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show storage statistics")
    stats_parser.add_argument(
        "--detailed", action="store_true", help="Show detailed per-layer statistics"
    )

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean precomputed weights")
    clean_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    try:
        if args.command == "precompute":
            return cmd_precompute(args)
        elif args.command == "validate":
            return cmd_validate(args)
        elif args.command == "stats":
            return cmd_stats(args)
        elif args.command == "clean":
            return cmd_clean(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user.")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
