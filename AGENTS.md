# AI Agents Development Notes

## Python Environment Setup

### ⚠️ Virtual Environment Activation

**Before running any Python scripts or commands, make sure to activate the virtual environment:**

```bash
source .venv/bin/activate
```

**Common Issue**: If you encounter `zsh: command not found: python`, it means the virtual environment is not activated. Always run the activation command first.

**Correct workflow:**

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run Python scripts
python tests/gptoss_cpu_boil.py
# etc...
```

## Model Checkpoint Structure

### Important: Model Location

⚠️ **The actual model checkpoint files are located in the `original` subdirectory:**

```
data/models/gpt-oss-20b/original/
data/models/gpt-oss-120b/original/ # Not used now
```

**NOT** in the parent directory `data/models/gpt-oss-20b/`

### Checkpoint Structure

The model checkpoint is stored in:

- `data/models/gpt-oss-20b/original/model.safetensors` - Single safetensor file containing all parameters

### Parameter Naming Convention

The parameters in the checkpoint follow this naming pattern:

```text
embedding.weight
unembedding.weight
block.{layer_idx}.attn.norm.scale
block.{layer_idx}.attn.out.bias
block.{layer_idx}.attn.out.weight
block.{layer_idx}.attn.qkv.bias
block.{layer_idx}.attn.qkv.weight
block.{layer_idx}.attn.sinks
block.{layer_idx}.mlp.gate.bias
block.{layer_idx}.mlp.gate.weight
block.{layer_idx}.mlp.mlp1_bias
block.{layer_idx}.mlp.mlp1_weight.blocks    # MXFP4 format
block.{layer_idx}.mlp.mlp1_weight.scales    # MXFP4 format
block.{layer_idx}.mlp.mlp2_bias
block.{layer_idx}.mlp.mlp2_weight.blocks    # MXFP4 format
block.{layer_idx}.mlp.mlp2_weight.scales    # MXFP4 format
norm.scale
```

## Important: Boilerplate Code Correctness

### ⚠️ Critical Understanding

**The `src/boilerplate/gpt_oss/` code is CORRECT and should be treated as the reference implementation.**

Key facts verified through testing:

1. **Input Dimensions**: The boilerplate code is designed for **2D input tensors** `[batch_size, hidden_size]`, representing single token processing for autoregressive generation.

2. **QKV Slicing**: In `AttentionBlock.forward()`, the QKV slicing `qkv[:, :4096]` is correct for 2D tensors:

   ```python
   # For 2D input [1, 2880] -> QKV [1, 5120] -> Q slice [1, 4096] ✅
   q = qkv[:, : self.num_attention_heads * self.head_dim]
   ```

3. **Working Test**: `tests/gptoss_cpu_boil.py` successfully demonstrates the boilerplate code works correctly with proper 2D input.

4. **Token Processing**: The model processes one token at a time during generation, not batched sequences.

### MXFP4 Parameter Handling

1. **Checkpoint Class**: Use `Checkpoint` class without `.blocks` suffix - it automatically handles MXFP4 conversion:

   ```python
   # ✅ Correct - Checkpoint handles MXFP4 automatically
   tensor = checkpoint.get("block.0.mlp.mlp1_weight")  # Returns [32, 5760, 2880], bfloat16

   # ❌ Wrong - Raw MXFP4 format
   tensor = checkpoint.get("block.0.mlp.mlp1_weight.blocks")  # Returns [32, 5760, 90, 16], uint8
   ```

2. **Automatic Conversion**: The `Checkpoint` class automatically combines `.blocks` and `.scales` into properly shaped `bfloat16` tensors.

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
