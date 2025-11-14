"""
MLX Backend for einops
Implements einops operations using MLX arrays
"""

import mlx.core as mx
from typing import Tuple, List, Optional

class MLXBackend:
    framework_name = "mlx"
    
    def is_appropriate_type(self, tensor):
        return isinstance(tensor, mx.array)
    
    def from_numpy(self, x):
        return mx.array(x)
    
    def to_numpy(self, x):
        return mx.array(x).__array__()
    
    def arange(self, start, stop=None):
        if stop is None:
            return mx.arange(start)
        return mx.arange(start, stop)
    
    def reduce(self, x, operation, reduced_axes):
        if operation == "min":
            return mx.min(x, axis=reduced_axes, keepdims=True)
        elif operation == "max":
            return mx.max(x, axis=reduced_axes, keepdims=True)
        elif operation == "sum":
            return mx.sum(x, axis=reduced_axes, keepdims=True)
        elif operation == "mean":
            return mx.mean(x, axis=reduced_axes, keepdims=True)
        elif operation == "prod":
            return mx.prod(x, axis=reduced_axes, keepdims=True)
        else:
            raise NotImplementedError(f"Unknown reduction operation: {operation}")
    
    def transpose(self, x, axes):
        return mx.transpose(x, axes)
    
    def reshape(self, x, shape):
        return mx.reshape(x, shape)
    
    def stack_on_zeroth_dimension(self, tensors: list):
        return mx.stack(tensors, axis=mx.array(0, dtype=mx.int32))
    
    def tile(self, x, repetitions):
        return mx.tile(x, repetitions)
    
    def add_axes(self, x, n_axes, pos2len):
        # Add new axes at specified positions
        repeats = [mx.array(1, dtype=mx.int32)] * (len(x.shape) + mx.array(n_axes, dtype=mx.int32))
        for axis_position, axis_length in pos2len.items():
            x = mx.expand_dims(x, axis=axis_position)
            repeats[axis_position] = axis_length
        return mx.tile(x, tuple(int(r) for r in repeats))
    
    def is_float_type(self, x):
        return x.dtype in [mx.float16, mx.float32, mx.bfloat16]
    
    def layers(self):
        # Placeholder for layer support if needed
        return None
    
    def shape(self, x):
        return tuple(x.shape)
    
    def __repr__(self):
        return f"<MLXBackend>"
