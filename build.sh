#!/bin/bash
set -e

echo "🔧 Setting up Python environment..."
uv sync
source .venv/bin/activate

echo "🦀 Building Rust core module..."
maturin develop --manifest-path libs/core/Cargo.toml

echo "📝 Setting up module re-exports..."
# Ensure the rust_core package has the correct __init__.py to re-export everything
cat > src/rust_core/__init__.py << 'EOF'
# Re-export all Rust module contents
from .rust_core import *
EOF

echo "✅ Build completed successfully!"
echo "🧪 Running tests..."
python tests/test_two_tier_wm.py
