# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Python Environment

**CRITICAL**: Always activate the Python virtual environment before running any Python commands:

```bash
source .venv/bin/activate
```

### Build Commands

```bash
# Build Rust workspace (debug)
cargo build

# Build Rust workspace (release)
cargo build --release

# Run tests for Rust components
cargo test

# Format Rust code
cargo fmt

# Lint Rust code
cargo clippy
```

### Python Commands

```bash
# Install Python dependencies (after activating venv)
uv sync

# Format Python code
black scripts/ src/ tests/

# Run Python tests
pytest tests/

# Download example models
python scripts/download_gptoss.py
```

### Testing Specific Components

```bash
# Test expert cache managers
python tests/test_direct_ram.py
python tests/test_direct_nvme.py
python tests/test_lru.py

# Test GPT-OSS model implementation
python tests/gptoss_cpu_boil.py
```

## Architecture Overview

This is a two-level caching system for Mixture-of-Experts (MoE) models with the following key components:

### Rust Workspace Structure

- **`libs/core/`**: Core data structures and fundamental algorithms
- **`libs/policy/`**: Caching policies and watermark management
- **`libs/backend/`**: Hardware abstraction and memory management

### Python Codebase Structure

- **`src/boilerplate/gpt_oss/`**: Reference implementation for GPT-OSS model processing
- **`src/domain/`**: Domain entities and business logic
  - `cache/`: Cache-related entities and interfaces
  - `gpt_oss/`: GPT-OSS specific model implementations
  - `manager/`: Cache manager implementations (DirectRAM, DirectNVME, LRU)
- **`src/adapters/`**: Adapter pattern implementations for expert management
- **`src/services/`**: Application services and factories
- **`src/config/`**: Configuration management

### Cache Management Architecture

The system implements three main cache manager strategies:

1. **DirectRAMExpertCacheManager**: Pre-warms all experts in RAM
2. **DirectNVMEExpertCacheManager**: Manages experts with time-based eviction
3. **LRUExpertCacheManager**: Least-recently-used eviction policy

## Critical Implementation Notes

### Model Checkpoint Handling

- Model files are located in `data/models/gpt-oss-20b/original/` (note the `original/` subdirectory)
- Use the `Checkpoint` class from `src/boilerplate/gpt_oss/weights.py` for MXFP4 parameter handling
- MLP weights are stored in MXFP4 format with `.blocks` and `.scales` tensors that are automatically combined

### GPT-OSS Model Processing

- Model expects 2D input tensors `[batch_size, hidden_size]` for single token processing
- Use `torch.int32` for token inputs, not `torch.int64`
- The boilerplate code in `src/boilerplate/gpt_oss/` is the reference implementation
- Parameter names use `block.{layer_idx}` prefix, not `model.layers.{layer_idx}`

### Expert Cache Interface

All cache managers implement `IExpertCacheManager` with methods:

- `get(expert_id)`: Retrieve expert parameters
- `put(expert_id, expert_data)`: Store expert parameters
- `next()`: Time-based eviction method for some implementations

### Development Workflow

1. Always activate Python virtual environment first
2. Test Python changes with existing test files in `tests/`
3. The `tests/gptoss_cpu_boil.py` demonstrates correct model usage patterns
4. Use `cargo clippy` and `black` for code formatting before commits
