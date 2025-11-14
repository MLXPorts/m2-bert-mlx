# MLX Operations Library

Organized MLX-native operations for M2-BERT, following strict MLX conventions.

## Principles

1. **No NumPy**: Except for I/O operations (loading/saving weights)
2. **Strict MLX scalars**: Use `mx.array(5, dtype=mx.int64)` instead of Python `5`
3. **MLX operators**: Use `mx.add()`, `mx.multiply()` instead of `+`, `*`
4. **One function per file**: Major operations get their own file
5. **Vendored dependencies**: einops is vendored and converted to MLX

## Structure

```
mlx_ops/
├── __init__.py                 # Main exports
├── README.md                   # This file
│
├── einops/                     # Vendored einops adapted for MLX
│   ├── __init__.py
│   ├── ops.py                  # rearrange, reduce, repeat, einsum
│   └── backend_mlx.py          # MLX backend implementation
│
├── kernels/                    # Optimized Metal/CUDA kernels
│   ├── gemm_kernels.py         # GEMM operations (from xLSTM)
│   ├── metal_fft_conv.py       # FFT convolution kernel
│   └── metal_fft_conv_streamed.py
│
├── activations.py              # Activation functions
├── conv_ops.py                 # Convolution operations
├── math_ops.py                 # Mathematical utilities
├── blockdiag_ops.py            # Block diagonal operations
├── blockdiag_linear.py         # Block diagonal linear layers
├── monarch_mixer.py            # Monarch Mixer operations
├── monarch_mlp.py              # Monarch MLP operations
├── hyena_filter.py             # Hyena filter operations
└── weight_loading.py           # PyTorch checkpoint loading
```

## Usage

```python
import mlx.core as mx
from mlx_ops import rearrange, conv1d_fft, monarch_conv
from mlx_ops import load_checkpoint

# Load weights
weights = load_checkpoint("model.npz")

# Use einops
x = mx.array(...)
y = rearrange(x, 'b c h w -> b h w c')

# Use FFT convolution with bias
output = conv1d_fft(input, kernel_freq, bias=bias)
```

## Lint Checking

Use `emberlint.py` to ensure:
- No NumPy usage in compute code
- Proper MLX scalar usage
- MLX operators instead of Python operators
