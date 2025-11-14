"""
MLX-adapted einops
Core tensor manipulation operations for MLX
"""

__author__ = "Alex Rogozhnikov"
__version__ = "0.8.1-mlx"

class EinopsError(RuntimeError):
    """Runtime error thrown by einops"""
    pass

__all__ = ['rearrange', 'reduce', 'repeat', 'einsum', 'EinopsError']

# Use the full einops implementation, not the simplified ops.py
from .einops import rearrange, reduce, repeat, einsum

