# Makefile for two-level-moe-cache project
# This replaces the build.sh script with make-based automation

# Variables
PYTHON_ENV = .venv
RUST_MANIFEST = libs/core/Cargo.toml
RUST_MODULE_DIR = src/rust_core
INIT_PY = $(RUST_MODULE_DIR)/__init__.py

# Default target
.PHONY: all
all: build test

# Set up Python environment and install dependencies
.PHONY: setup
setup:
	@echo "🔧 Setting up Python environment..."
	uv sync
	@echo "✅ Python environment ready!"

# Build the Rust core module
.PHONY: build-rust
build-rust: setup
	@echo "🦀 Building Rust core module..."
	. $(PYTHON_ENV)/bin/activate && maturin develop --manifest-path $(RUST_MANIFEST)

# Set up module re-exports
.PHONY: setup-module
setup-module: build-rust
	@echo "📝 Setting up module re-exports..."
	@mkdir -p $(RUST_MODULE_DIR)
	@echo "# Re-export all Rust module contents" > $(INIT_PY)
	@echo "from .rust_core import *" >> $(INIT_PY)
	@echo "✅ Module re-exports configured!"

# Complete build process
.PHONY: build
build: setup-module
	@echo "✅ Build completed successfully!"

# Run tests
.PHONY: test
test: build
	@echo "🧪 Running tests..."
	. $(PYTHON_ENV)/bin/activate && python tests/test_two_tier_wm.py

# Clean up build artifacts and caches
.PHONY: clean
clean:
	@echo "🧹 Cleaning up build artifacts..."
	# Remove Python cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	# Remove Rust build artifacts
	cargo clean --manifest-path $(RUST_MANIFEST)
	# Remove target directory
	rm -rf target/
	# Remove compiled Rust module
	rm -f $(RUST_MODULE_DIR)/*.so
	# Remove generated __init__.py
	rm -f $(INIT_PY)
	# Remove virtual environment
	rm -rf $(PYTHON_ENV)
	@echo "✅ Cleanup completed!"

# Deep clean - removes everything including downloaded dependencies
.PHONY: clean-all
clean-all: clean
	@echo "🗑️  Performing deep cleanup..."
	# Remove uv cache (optional - uncomment if needed)
	# uv cache clean
	# Remove Cargo cache (optional - be careful with this)
	# cargo clean --manifest-path $(RUST_MANIFEST)
	@echo "✅ Deep cleanup completed!"

# Development helpers
.PHONY: dev
dev: build
	@echo "🚀 Development environment ready!"
	@echo "💡 Run 'source $(PYTHON_ENV)/bin/activate' to activate the Python environment"

# Quick test without full rebuild
.PHONY: test-quick
test-quick:
	@echo "⚡ Running quick tests..."
	. $(PYTHON_ENV)/bin/activate && python tests/test_two_tier_wm.py

# Format code
.PHONY: format
format:
	@echo "🎨 Formatting code..."
	. $(PYTHON_ENV)/bin/activate && black src/ tests/ --line-length 88
	cargo fmt --manifest-path $(RUST_MANIFEST)
	@echo "✅ Code formatting completed!"

# Lint code
.PHONY: lint
lint:
	@echo "🔍 Linting code..."
	. $(PYTHON_ENV)/bin/activate && flake8 src/ tests/ --max-line-length 88 --extend-ignore E203,W503
	cargo clippy --manifest-path $(RUST_MANIFEST) -- -D warnings
	@echo "✅ Code linting completed!"

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all        - Build and test (default)"
	@echo "  build      - Complete build process"
	@echo "  test       - Run tests"
	@echo "  test-quick - Run tests without rebuild"
	@echo "  setup      - Set up Python environment only"
	@echo "  dev        - Prepare development environment"
	@echo "  format     - Format code (Python + Rust)"
	@echo "  lint       - Lint code (Python + Rust)"
	@echo "  clean      - Clean build artifacts"
	@echo "  clean-all  - Deep clean including dependencies"
	@echo "  help       - Show this help message"
