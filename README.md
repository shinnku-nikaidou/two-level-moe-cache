# Two-Level MoE Cache

A high-performance two-level caching system for Mixture-of-Experts (MoE) models optimized for resource-constrained edge inference.

## Overview

This project implements an intelligent caching system that efficiently manages MoE expert weights across the GPU-CPU-NVMe memory hierarchy. It uses adaptive watermark-based policies to optimize expert placement and reduce latency in edge computing scenarios.

## Features

- **Two-tier caching**: Intelligent management of expert weights across VRAM and RAM
- **Adaptive watermarks**: Dynamic credit-based eviction policies that respond to memory pressure
- **Model-agnostic**: Compatible with mainstream MoE language models
- **Edge-optimized**: Designed for resource-constrained environments with limited VRAM
- **Rust implementation**: High-performance core with Python utilities

## Architecture

The system consists of several modular components:

- **Core**: Fundamental data structures and algorithms
- **Policy**: Caching policies and watermark management
- **Backend**: Hardware abstraction and memory management
- **Runtime**: Execution environment and orchestration
- **CLI**: Command-line interface for system interaction

## Getting Started

### Prerequisites

- Rust 2024 edition
- Python 3.13+
- CUDA-compatible GPU (for edge deployment)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shinnku-nikaidou/two-level-moe-cache.git
   cd two-level-moe-cache
   ```

2. Build the Rust components:

   ```bash
   cargo build --release
   ```

3. Install Python dependencies:

   ```bash
   uv sync
   ```

### Download Models

To download example MoE models for testing:

```bash
python scripts/download_gptoss.py
```

This will download GPT-OSS models to the `data/models/` directory.

### Quick Start with Precomputed Weights

For the fastest possible expert loading, use the precomputation system:

```bash
# 1. Download the model (if not already done)
python scripts/download_llm.py

# 2. Precompute all expert weights (takes time, but only done once)
python -m src.cli.precompute precompute

# 3. Verify precomputation worked
python -m src.cli.precompute validate

# 4. Test the system
python tmp/test_precomputed_system.py
```

After precomputation, expert loading will be **10-50x faster** with zero MXFP4 decoding overhead!

## Usage

### Expert Weight Precomputation System

This project includes a high-performance expert weight precomputation system that dramatically accelerates inference by eliminating MXFP4 decoding overhead.

#### Prerequisites for Precomputation

Before using the precomputation system, ensure you have downloaded the GPT-OSS-20B model:

```bash
# Download GPT-OSS-20B model to data/models/gpt-oss-20b/
python scripts/download_llm.py
```

#### Command Line Interface

The precomputation CLI provides comprehensive tools for managing expert weights:

```bash
# Precompute all expert weights (generates 3072 individual safetensors files)
python -m src.cli.precompute precompute

# Precompute with specific dtype (default: bfloat16)
python -m src.cli.precompute precompute --dtype float16

# Force rebuild existing precomputed weights
python -m src.cli.precompute precompute --force

# Validate precomputed weights accuracy
python -m src.cli.precompute validate

# Validate with custom sample size
python -m src.cli.precompute validate --samples 50

# Show storage statistics
python -m src.cli.precompute stats

# Show detailed per-layer statistics
python -m src.cli.precompute stats --detailed

# Clean all precomputed weights
python -m src.cli.precompute clean

# Clean without confirmation prompt
python -m src.cli.precompute clean --force
```

#### Fast Expert Adapter Usage

Once weights are precomputed, use the fast adapter for zero-decoding-overhead loading:

```python
from src.adapters.expert.gptoss20bfast import GPTOSS20bFastExpertAdapter
from src.common.types import ExpertKey, ExpertParamType
from src.domain import ModelType

# Initialize fast adapter (automatically detects precomputed weights)
adapter = GPTOSS20bFastExpertAdapter(ModelType.GPT_OSS_20B)

# Load individual expert weights (extremely fast)
expert_key = ExpertKey(
    layer_idx=0,
    expert_id=5, 
    param_type=ExpertParamType.MLP1_WEIGHT
)

# This loads directly from precomputed safetensors file
expert_tensor = adapter.load_expert_tensor(expert_key)

# Check loading statistics
stats = adapter.get_load_stats()
print(f"Precomputed loads: {stats['precomputed_loads']}")
print(f"Fallback loads: {stats['fallback_loads']}")
print(f"Success rate: {stats['precomputed_rate_percent']:.1f}%")
```

#### Storage Structure

Precomputed weights are organized as follows:

```text
data/models/gpt-oss-20b/original/precomputed/
├── metadata.json                     # Precomputation metadata
├── layer_00/
│   ├── expert_00_mlp1_weight.safetensors
│   ├── expert_00_mlp1_bias.safetensors
│   ├── expert_00_mlp2_weight.safetensors
│   ├── expert_00_mlp2_bias.safetensors
│   └── ... (128 files per layer)
├── layer_01/
└── ... (24 layers total)
```

**Total files**: 24 layers × 32 experts × 4 parameters = **3,072 individual safetensors files**

#### Performance Benefits

- **Loading Speed**: 10-50x faster than MXFP4 decoding
- **Memory Efficiency**: True on-demand loading with zero caching overhead
- **Storage Cost**: ~4x storage increase (MXFP4 → FP16/BF16)
- **Numerical Accuracy**: Bit-exact results compared to original MXFP4 decoding

#### Integration with Existing Cache Managers

The fast adapter works seamlessly with existing cache managers:

```python
from src.domain.manager.direct_ram import DirectRAMExpertCacheManager
from src.domain import ModelType

# Cache manager automatically uses fast adapter when available
cache_manager = DirectRAMExpertCacheManager(ModelType.GPT_OSS_20B)

# Expert loading is now dramatically faster
expert = cache_manager.get(expert_key)
```

### Legacy Command Line Interface

```bash
# Run the legacy CLI tool
cargo run --bin two-level-moe-cache-cli

# Or use the compiled binary
./target/release/two-level-moe-cache-cli
```

## Configuration

The system supports various configuration options:

- Memory tier sizes (VRAM/RAM)
- Watermark adaptation rates
- Expert prediction windows
- Model-specific parameters

See configuration examples in the `data/` directory.

## Troubleshooting

### Precomputation Issues

**Problem**: `ModuleNotFoundError` when running CLI commands

```bash
# Solution: Run from project root and ensure Python path is correct
cd /path/to/two-level-moe-cache
python -m src.cli.precompute precompute
```

**Problem**: "Precomputed weights not available" error

```bash
# Solution: Run precomputation first
python -m src.cli.precompute precompute
python -m src.cli.precompute validate  # Verify results
```

**Problem**: Out of disk space during precomputation

```bash
# Check storage requirements (~80GB for GPT-OSS-20B in BF16)
python -m src.cli.precompute stats
# Clean if needed
python -m src.cli.precompute clean
```

**Problem**: Slow precomputation performance

```bash
# Use float16 instead of bfloat16 (same storage, potentially faster)
python -m src.cli.precompute precompute --dtype float16

# Check that NVMe is being used for the model directory
df -h data/models/gpt-oss-20b/
```

### Fast Adapter Issues

**Problem**: `NotImplementedError: MXFP4 fallback not available`

```python
# Solution: This is by design. The fast adapter requires precomputed weights
# Run precomputation first:
from src.infra.dataprocess.gpt_oss import precompute_gpt_oss_20b
precompute_gpt_oss_20b()
```

**Problem**: Numerical differences between adapters

```bash
# Run validation to check accuracy
python -m src.cli.precompute validate --samples 100
```

## Development

### Project Structure

```markdown
├── cli/ # Command-line interface
├── libs/
│ ├── core/ # Core data structures
│ ├── policy/ # Caching policies
│ ├── backend/ # Hardware abstraction
│ └── runtime/ # Execution runtime
├── scripts/ # Python utilities
└── data/ # Models and configurations
```

### Building

```bash
# Debug build
cargo build

# Release build with optimizations
cargo build --release

# Run tests
cargo test
```

### Code Style

The project uses standard Rust formatting:

```bash
cargo fmt
cargo clippy
```

For Python code:

```bash
black scripts/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:
