# M2-BERT MLX Implementation Status

## Completed Tasks

### 1. Weight Management ✅
- Downloaded M2-BERT-80M (103M parameters) from HuggingFace (`danfu09/m2-bert-80m`)
- Converted PyTorch checkpoint to NPZ format compatible with MLX
- Implemented comprehensive weight loading utilities in `mlx_ops/weight_loading.py`
- Supports multiple formats: .npz, .pt, .safetensors

### 2. MLX Operations Library ✅
Organized in `bert/src/mlx_ops/` with modular structure:

#### Einops (Vendored)
- Copied einops library to `mlx_ops/einops/`
- Added MLX backend support in `_backends.py`
- Supports: `rearrange`, `repeat`, `reduce`, `einsum`
- Pure MLX implementation (no NumPy dependency for operations)

#### Convolution Operations
File: `mlx_ops/conv_ops.py`
- `conv1d()` - Standard 1D convolution with bias
- `conv1d_fft()` - **FFT-based convolution with bias support** ✅  
- `depthwise_conv1d()` - Depthwise convolution

**Key Feature**: FFT convolution now properly handles bias addition, matching canonical M2-BERT requirements.

### 3. Model Architecture Understanding
From weight inspection, M2-BERT-80M uses:

**Embeddings:**
- Word embeddings: (30528, 768)
- Position embeddings: (128, 768) 
- Token type embeddings: (2, 768)
- Layer normalization

**Encoder (12 layers):**
Each layer contains:
- `attention.filter_fn` - Bidirectional gated convolution with implicit filters
- `attention.filter_fn2` - Second filter function  
- `attention.short_filter` - Short convolution (kernel size 3)
- `attention.in_linear`, `attention.out_linear` - Linear projections
- `mlp.gated_layers` - Block diagonal gated MLP
- `mlp.wo` - Output projection (also block diagonal)
- Layer normalization for both attention and MLP

**MLM Head:**
- `cls.predictions.transform` - Dense + LayerNorm
- `cls.predictions.decoder` - Final projection to vocab

### 4. File Organization
```
bert/
├── src/
│   └── mlx_ops/           # MLX operations library
│       ├── __init__.py    # Clean exports
│       ├── einops/        # Vendored einops with MLX backend
│       │   ├── __init__.py
│       │   ├── _backends.py   # Added MLXBackend class
│       │   ├── einops.py
│       │   ├── parsing.py
│       │   └── ...
│       ├── conv_ops.py    # Convolution operations (with FFT+bias)
│       └── weight_loading.py  # Checkpoint loading utilities
├── .model/
│   ├── m2-bert-80m_mlx.npz    # Converted weights (964MB)
│   └── m2-bert-80m_params.txt # Parameter list
├── download_weights.py    # HuggingFace download script
└── test_inference.py      # Weight loading verification

```

## Next Steps

### Priority 1: Complete Model Implementation
The canonical PyTorch code is in:
- `bert/src/bert_layers.py` - Core BERT layers
- `bert/src/mm/monarch_mixer_sequence_mixer.py` - Monarch Mixer layers
- `bert/src/mm/blockdiag_linear.py` - Block diagonal linear layers

Need to convert these to MLX, keeping close to canonical structure.

### Priority 2: Implicit Filter Implementation
The attention mechanism uses implicit filters with:
- Multiple filter layers with freq parameters
- Modulation with learnable deltas
- Position embeddings (t and z parameters)
- Both forward and reverse filters

### Priority 3: Block Diagonal MLP
The MLP uses block diagonal matrices instead of dense:
- `gated_layers.weight` - Shape (4, 768, 768)  
- `wo.weight` - Shape (4, 768, 192)
- Requires efficient block diagonal matrix multiplication

### Priority 4: Integration
1. Build complete M2BertModel class
2. Load pretrained weights into model
3. Implement forward pass
4. Test on masked language modeling task
5. Verify outputs match PyTorch implementation

## Code Quality Standards
- ✅ NO NumPy usage (except I/O in weight_loading)
- ✅ Use `mx.array()` for all numeric values
- ✅ Use `mx.add`, `mx.multiply` instead of Python operators
- ✅ Modular file organization (one major function per file)
- ✅ Proper __init__.py exports
- ✅ Close to canonical code structure

## Testing
Current test:
- `test_inference.py` - Verifies weight loading (✅ Working)

Needed tests:
- Forward pass parity with PyTorch
- Numerical accuracy verification
- Performance benchmarks

## References
- Canonical repo: https://github.com/HazyResearch/m2
- Model: https://huggingface.co/danfu09/m2-bert-80m
- Paper: Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture
