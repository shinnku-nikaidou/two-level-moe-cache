# AI Agents Development Notes

## Model Checkpoint Structure

### Important: Model Location

⚠️ **The actual model checkpoint files are located in the `original` subdirectory:**

```
data/models/gpt-oss-20b/original/
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
å
### Key Findings

1. **Correct Path**: Always use `data/models/gpt-oss-20b/original/` for checkpoint loading
2. **MXFP4 Format**: MLP weights are stored in MXFP4 format with separate `.blocks` and `.scales` tensors
3. **Parameter Mapping**: Use `Checkpoint` class from `src/boilerplate/gpt_oss/weights.py` for proper MXFP4 handling
4. **No Model Subdirectory**: Unlike some HuggingFace models, parameters use `block.X` prefix, not `model.layers.X`
