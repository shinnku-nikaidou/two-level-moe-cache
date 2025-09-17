# AI Agents Development Notes

## Python Environment Setup

### ⚠️ Virtual Environment Activation

**Before running any Python scripts or commands, make sure to activate the virtual environment:**

```bash
source .venv/bin/activate
```

**Common Issue**: If you encounter `zsh: command not found: python` or `command not found: maturin` or any other Python-related command, it means the virtual environment is not activated. Always run the activation command first.

**Correct workflow:**

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run Python scripts
python tests/gptoss_cpu_boil.py
# etc...
```

### 🔧 Rust Python Module Development

**When modifying Rust Python exports (PyO3 bindings), remember to update the type stub file:**

```bash
# After modifying libs/core/watermark_cache.rs or other Python-exposed Rust code:
# 1. Build the module
make

# 2. Update Python type stubs
# Manual update: src/rust_core/__init__.pyi 
```

**Important**: The `.pyi` file provides Python type hints for Rust-exported functions and classes. Keep it synchronized with your Rust code changes to maintain proper IDE support and type checking.

## Testing Guidelines

### 📁 Temporary Test Files

**When creating temporary test files for debugging or validation, always place them in the `tmp/` directory:**

```bash
# ✅ Correct - put temporary test files in tmp/
tmp/test_experts_status.py
tmp/debug_model_type.py
tmp/test_cleanup.py

# ❌ Wrong - don't clutter root directory
test_experts_status.py
debug_model_type.py
```

**Guidelines:**

- **Use `tmp/` for ephemeral test files** that are created during development and debugging
- **Use `tests/` for permanent test files** that are part of the test suite
- **Clean up `tmp/` regularly** - these files are meant to be temporary
- **Add `tmp/` to `.gitignore`** if it's not already there to avoid committing temporary files

### 🧪 Running Temporary Tests

```bash
# Always activate environment first
source .venv/bin/activate

# Run temporary tests from project root
python tmp/test_cleanup.py
python tmp/debug_model_type.py
```

## Model Checkpoint Structure

### Important: Model Location

⚠️ **The actual model checkpoint files are located in the `original` subdirectory:**

```text
data/models/gpt-oss-20b/original/
data/models/gpt-oss-120b/original/ # Not used now
```

**NOT** in the parent directory `data/models/gpt-oss-20b/`

### Key Findings

1. **Correct Path**: Always use `data/models/gpt-oss-20b/original/` for checkpoint loading
2. **MXFP4 Format**: MLP weights are stored in MXFP4 format with separate `.blocks` and `.scales` tensors
3. **Parameter Mapping**: Use `Checkpoint` class from `src/boilerplate/gpt_oss/weights.py` for proper MXFP4 handling
4. **No Model Subdirectory**: Unlike some HuggingFace models, parameters use `block.X` prefix, not `model.layers.X`
5. **2D Input Design**: Model expects single token processing, not sequence batching
6. **Input dtype**: Use `torch.int32` for token inputs, not `torch.int64`

### Development Guidelines

- **Trust the boilerplate**: When implementing custom versions, follow the boilerplate patterns exactly
- **Test with 2D inputs**: Always test with single token inputs `[1]` -> `[1, hidden_size]` after embedding
- **Use Checkpoint class**: Let it handle MXFP4 conversion automatically
- **Reference tests/gptoss_cpu_boil.py**: This demonstrates the correct usage pattern
- **🚫 No Re-exports**: Never use `pub use module::*;` or similar re-export patterns. Always use fully qualified paths like `module::Type` to maintain clear module boundaries and avoid namespace pollution

## Indexing Convention

### ⚠️ Critical: Documentation vs Implementation Index Differences

**The documentation (docs/Two Level Caching MOE large language model.md) uses 1-based indexing throughout, but our implementation uses 0-based indexing for consistency with standard programming conventions.**

#### Documentation (1-based)

- Layers: `ℓ ∈ {1, 2, ..., L}`
- Experts: `e ∈ {1, 2, ..., E_ℓ}`
- Time steps: `t = 1, 2, 3, ...`
- Layer-local clock: `k = 1, 2, 3, ...`
- Current layer formula: `ℓ(t) = ((t-1) mod L) + 1`
- Layer-local time: `k = ⌊(t-ℓ)/L⌋ + 1`

#### Implementation (0-based)

- Layers: `layer_id ∈ {0, 1, ..., L-1}`
- Experts: `expert_id ∈ {0, 1, ..., E_layer-1}`
- Time steps: `t = 0, 1, 2, ...`
- Layer-local clock: `k = 0, 1, 2, ...`
- Current layer formula: `layer_id = t mod L`
- Visit count: `v_ℓ(t) = ⌊t/L⌋ + (1 if t%L >= ℓ else 0)`

#### Index Conversion Guidelines

- **When reading documentation**: Subtract 1 from layer/expert indices and time values
- **When implementing**: Use 0-based indexing throughout the codebase
- **When debugging**: Be aware of this offset when comparing with documentation formulas
- **Timer module**: Already implements 0-based indexing correctly with equivalent mathematics
