"""
MLX Operations Library for M2-BERT.

This package provides MLX-native implementations of common operations
used in M2-BERT, organized by functionality.

Modules:
    einops: MLX-compatible einops operations (rearrange, repeat, reduce)
    conv_ops: Convolution operations with bias support
    weight_loading: Utilities for loading PyTorch checkpoints
    activations: Activation functions
    math_ops: Mathematical operations
    blockdiag_ops: Block diagonal operations
    losses: Loss functions
    kernels: Optimized GEMM and FFT kernels
    
All operations use strict MLX conventions:
- No NumPy usage (except for I/O in weight loading)
- Use mx.array for all numeric values including scalars
- Use mx.add, mx.multiply, etc. instead of Python operators where possible
"""

from .blockdiag_linear import BlockdiagLinear
# Import block diagonal operations
from .blockdiag_multiply import blockdiag_multiply
# Import convolution operations
from .conv_ops import conv1d, conv1d_fft, conv1d_fft_with_bias, depthwise_conv1d
# Import einops operations (vendored and adapted for MLX)
from .einops import rearrange, repeat, reduce, einsum
from .hyena_filter import HyenaFilter
# Import GEMM kernels
from .kernels.gemm.gemm_kernels import gemm_av, gemm_at_b
# Import Monarch operations
from .monarch_mixer import MonarchMixerSequenceMixing
# Import weight loading utilities
from .weight_loading import (
    load_checkpoint,
    load_pytorch_checkpoint,
    load_safetensors_checkpoint,
    match_and_load_weights,
    print_checkpoint_info
)

# MLX provides built-in activations - use them directly from mlx.nn:
#   from mlx.nn import gelu, relu, silu (functional forms)
#   or mlx.nn.GELU(), mlx.nn.ReLU(), mlx.nn.SiLU() (module forms)

__all__ = [
    # Einops operations
    'rearrange',
    'repeat',
    'reduce',
    'einsum',

    # Convolution operations
    'conv1d',
    'conv1d_fft',
    'conv1d_fft_with_bias',
    'depthwise_conv1d',

    # Weight loading
    'load_checkpoint',
    'load_pytorch_checkpoint',
    'load_safetensors_checkpoint',
    'match_and_load_weights',
    'print_checkpoint_info',

    # Block diagonal
    'blockdiag_multiply',
    'BlockdiagLinear',

    # Monarch and Hyena
    'MonarchMixerSequenceMixing',
    'HyenaFilter',

    # GEMM kernels
    'gemm_av',
    'gemm_at_b',
]

