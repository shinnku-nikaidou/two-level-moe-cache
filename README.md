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

## Usage

### Command Line Interface

```bash
# Run the CLI tool
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
