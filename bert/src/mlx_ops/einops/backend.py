"""
MLX Backend for einops
Implements einops operations using MLX arrays
"""

import mlx.core as mx


class MLXBackend:
    framework_name = "mlx"
    
    @staticmethod
    def _to_mx(x):
        # Convert any input to an MLX array if it's not already one
        if getattr(x, "__class__", None).__module__.startswith("mlx.core"):
            return x
        return mx.array(x)
    
    def is_appropriate_type(self, tensor):
        # MLX arrays are created via mx.array(...) and have a dedicated Array type.
        # We avoid importing private classes; instead, accept anything that exposes
        # `.shape` and can be wrapped by mx.array when needed.
        try:
            # Check for MLX Array by method presence
            return getattr(tensor, "__class__", None).__module__.startswith("mlx.core")
        except Exception:
            return False
    
    def from_numpy(self, x):
        return self._to_mx(x)
    
    def to_numpy(self, x):
        return self._to_mx(x).__array__()
    
    def arange(self, start, stop=None):
        if stop is None:
            return mx.arange(int(start))
        return mx.arange(int(start), int(stop))
    
    def reduce(self, x, operation, reduced_axes):
        x = self._to_mx(x)
        axes = tuple(int(a) for a in reduced_axes)
        if operation == "min":
            return mx.min(x, axis=axes, keepdims=True)
        elif operation == "max":
            return mx.max(x, axis=axes, keepdims=True)
        elif operation == "sum":
            return mx.sum(x, axis=axes, keepdims=True)
        elif operation == "mean":
            return mx.mean(x, axis=axes, keepdims=True)
        elif operation == "prod":
            return mx.prod(x, axis=axes, keepdims=True)
        else:
            raise NotImplementedError(f"Unknown reduction operation: {operation}")
    
    def transpose(self, x, axes):
        x = self._to_mx(x)
        return mx.transpose(x, tuple(int(a) for a in axes))
    
    def reshape(self, x, shape):
        x = self._to_mx(x)
        return mx.reshape(x, tuple(int(s) if isinstance(s, (int,)) or s == -1 else s for s in shape))
    
    def stack_on_zeroth_dimension(self, tensors: list):
        # Ensure all inputs are MLX arrays
        tensors = [self._to_mx(t) for t in tensors]
        return mx.stack(tensors, axis=0)
    
    def tile(self, x, repetitions):
        x = self._to_mx(x)
        return mx.tile(x, tuple(int(r) for r in repetitions))
    
    def add_axes(self, x, n_axes, pos2len):
        # Insert new axes and tile to required lengths
        x = self._to_mx(x)
        repeats = [1] * (len(x.shape) + int(n_axes))
        for axis_position, axis_length in pos2len.items():
            x = mx.expand_dims(x, axis=int(axis_position))
            repeats[int(axis_position)] = int(axis_length)
        return mx.tile(x, tuple(repeats))

    def add_axis(self, x, new_position):
        x = self._to_mx(x)
        return mx.expand_dims(x, axis=int(new_position))

    def concat(self, tensors, axis: int):
        tensors = [self._to_mx(t) for t in tensors]
        return mx.concatenate(tensors, axis=int(axis))
    
    def is_float_type(self, x):
        return x.dtype in [mx.float16, mx.float32, mx.bfloat16]
    
    def layers(self):
        # Placeholder for layer support if needed
        return None
    
    def shape(self, x):
        # Always report MLX array shape; convert if needed
        x = self._to_mx(x)
        return tuple(int(s) for s in x.shape)

    def __repr__(self):
        return f"<MLXBackend>"

    def einsum(self, pattern, *x):
        # Ensure MLX arrays
        xs = [self._to_mx(xi) for xi in x]
        return mx.einsum(pattern, *xs)
