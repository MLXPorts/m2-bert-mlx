# M2-BERT MLX Implementation Plan

## Current Status

### ✅ Completed
1. **einops vendored and adapted** - Core rearrange, reduce, repeat functions for MLX
2. **GEMM kernels copied** - From xLSTM project for efficient matrix operations
3. **FFT convolution kernel** - Metal-optimized FFT convolution already exists
4. **Weight download script** - Can download and convert PyTorch checkpoints
5. **Basic MLX operations** - Activations, conv ops, math ops, blockdiag ops

### 📋 TODO

#### 1. Ensure Strict MLX Compliance
- [x] einops uses MLX arrays for all operations
- [ ] Run emberlint.py on all files to check:
  - No NumPy usage (except I/O)
  - Use mx.array(5, dtype=mx.int64) instead of Python 5
  - Use mx.add(), mx.multiply() instead of +, *

#### 2. Organize Library Structure
- [x] One major function per file
- [x] Clear __init__.py exports
- [ ] Document each module with usage examples

#### 3. Model Architecture
- [ ] Convert BertEmbeddings from PyTorch to MLX
- [ ] Convert BertEncoder layers from PyTorch to MLX
- [ ] Convert MonarchMixerSequenceMixing from PyTorch to MLX
- [ ] Convert HyenaFilter from PyTorch to MLX
- [ ] Convert BlockdiagLinear from PyTorch to MLX

#### 4. Weight Loading
- [x] Basic PyTorch checkpoint → MLX conversion
- [ ] Handle state_dict structure from canonical m2
- [ ] Map weight names from PyTorch to MLX model
- [ ] Handle positional embedding expansion (for long context)

#### 5. Full Inference Pipeline
- [ ] Load pretrained weights into model
- [ ] Tokenize input text
- [ ] Run forward pass
- [ ] Get predictions/embeddings

## File Structure

```
bert/
├── download_weights.py         # Download from HuggingFace
├── test_inference.py           # Basic inference test
└── src/
    ├── mlx_ops/                # MLX operations library
    │   ├── __init__.py
    │   ├── README.md
    │   ├── einops/             # Vendored & MLX-adapted
    │   │   ├── __init__.py
    │   │   ├── ops.py
    │   │   └── backend_mlx.py
    │   ├── kernels/            # Optimized kernels
    │   │   ├── gemm_kernels.py
    │   │   ├── metal_fft_conv.py
    │   │   └── metal_fft_conv_streamed.py
    │   ├── activations.py
    │   ├── conv_ops.py
    │   ├── math_ops.py
    │   ├── blockdiag_ops.py
    │   ├── blockdiag_linear.py
    │   ├── monarch_mixer.py
    │   ├── monarch_mlp.py
    │   ├── hyena_filter.py
    │   └── weight_loading.py
    │
    ├── bert_layers_mlx.py      # MLX version of bert_layers.py
    ├── configuration_bert.py   # Config (can reuse from canonical)
    └── create_bert_mlx.py      # MLX model creation

```

## Key Conversions Needed

### From PyTorch to MLX

**torch → mx mappings:**
- `torch.Tensor` → `mx.array`
- `torch.nn.Linear` → Custom MLX implementation
- `torch.nn.LayerNorm` → `mx.fast.layer_norm` or custom
- `torch.nn.GELU` → `mx.gelu` (already in mlx_ops/activations.py)
- `torch.fft.rfft` → `mx.fft.rfft`
- `torch.einsum` → `mx.einsum`
- `rearrange` (einops) → Our vendored MLX version

**Example:**
```python
# PyTorch
out = torch.nn.functional.layer_norm(x, (hidden_size,))
out = torch.nn.functional.gelu(out)

# MLX
out = mx.fast.layer_norm(x, weight, bias, eps=1e-5)
out = mx.gelu(out)  # or our version from mlx_ops.activations
```

## Next Steps

1. Run emberlint.py to ensure strict MLX compliance
2. Convert core model architecture files from PyTorch to MLX
3. Test weight loading with real checkpoint
4. Run forward pass and verify outputs match canonical model
5. Add tests for each component

