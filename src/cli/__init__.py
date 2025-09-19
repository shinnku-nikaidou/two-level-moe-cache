"""
Command-line interface utilities for the two-level MOE cache system.

This package provides CLI tools for:
- Precomputing expert weights for faster loading
- Validating precomputed results
- Managing storage and statistics
- Performance testing and benchmarking
"""

from .precompute import main as precompute_main

__all__ = ["precompute_main"]
