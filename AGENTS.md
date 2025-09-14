# AI Agents Development Notes

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

```
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
