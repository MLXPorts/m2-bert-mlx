# Adapted to MLX from the Mosaic MLX ops library.

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx_ops.einops import rearrange


class StructuredLinear(nn.Module):
    """Base class for structured linear layers implemented with MLX."""

    def __init__(self, in_features, out_features, bias=True):
        """Subclasses should call reset_parameters after setting weight shapes."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if not hasattr(self, 'in_features_extended'):
            self.in_features_extended = in_features
        if not hasattr(self, 'out_features_extended'):
            self.out_features_extended = out_features
        self.bias = mx.zeros((out_features,), dtype=mx.float32) if bias else None

    def reset_parameters(self):
        """Initialize weights using a dense Kaiming uniform initializer."""
        init_fn = partial(self._kaiming_uniform, a=math.sqrt(5))
        self.set_weights_from_dense_init(init_fn)
        self.reset_parameters_bias()

    def _kaiming_uniform(self, tensor, a=0):
        """Return a tensor filled with Kaiming-uniform values."""
        fan_in = tensor.shape[1] if len(tensor.shape) > 1 else tensor.shape[0]
        gain = math.sqrt(2.0 / (1 + a * a))
        std = gain / math.sqrt(max(fan_in, 1))
        bound = math.sqrt(3.0) * std
        bound_mx = mx.array(bound, dtype=mx.float32)
        rand = mx.random.uniform(shape=tensor.shape, dtype=mx.float32)
        return mx.multiply(mx.subtract(rand, mx.array(0.5, dtype=mx.float32)),
                           mx.multiply(mx.array(2.0, dtype=mx.float32), bound_mx))

    def set_weights_from_dense_init(self, dense_init_fn_):
        """Subclasses should override to write structured weights."""
        raise NotImplementedError

    def reset_parameters_bias(self):
        if self.bias is not None:
            fan_in = max(self.in_features, 1)
            bound = 1 / math.sqrt(fan_in)
            bound_mx = mx.array(bound, dtype=mx.float32)
            rand = mx.random.uniform(shape=self.bias.shape, dtype=mx.float32)
            self.bias = mx.multiply(mx.subtract(rand, mx.array(0.5, dtype=mx.float32)),
                                    mx.multiply(mx.array(2.0, dtype=mx.float32), bound_mx))

    @property
    def saving(self):
        raise NotImplementedError

    def convert_to_dense_weight(self):
        eye = mx.eye(self.in_features, dtype=mx.float32)
        dense_weight = self.forward_matmul(eye)
        return mx.transpose(dense_weight, (1, 0))

    def preprocess(self, x):
        in_features = x.shape[-1]
        if in_features < self.in_features_extended:
            pad = self.in_features_extended - in_features
            pad_shape = list(x.shape)
            pad_shape[-1] = pad
            zeros = mx.zeros(pad_shape, dtype=x.dtype)
            x = mx.concatenate([x, zeros], axis=-1)
        return x

    def postprocess(self, output):
        if output.shape[-1] > self.out_features:
            output = output[..., :self.out_features]
        return output

    def forward_matmul(self, x):
        raise NotImplementedError

    def __call__(self, x):
        out = self.forward_matmul(x)
        if self.bias is not None:
            out = mx.add(out, self.bias.astype(out.dtype))
        return out


class BlockdiagSparsityConfig:
    """Block-diagonal sparsity mask helper."""

    def __init__(self, nblocks, block=32, global_size=0):
        self.nblocks = nblocks
        self.block = block
        self.global_size = global_size

    def make_layout(self, out_features, in_features):
        assert out_features % self.block == 0 and in_features % self.block == 0
        assert out_features % self.nblocks == 0 and in_features % self.nblocks == 0
        block_rows = out_features // self.nblocks
        block_cols = in_features // self.nblocks

        # Build block-diagonal layout
        rows = []
        identity_row = mx.ones((block_rows, block_cols), dtype=mx.int32)
        zero = mx.zeros((block_rows, block_cols), dtype=mx.int32)
        for i in range(self.nblocks):
            pieces = []
            for j in range(self.nblocks):
                pieces.append(identity_row if i == j else zero)
            rows.append(mx.concatenate(pieces, axis=1))
        layout = mx.concatenate(rows, axis=0)

        if self.global_size > 0:
            ones = mx.ones_like(layout)
            row_mask = (mx.arange(out_features, dtype=mx.int32) < self.global_size).reshape(-1, 1)
            col_mask = (mx.arange(in_features, dtype=mx.int32) < self.global_size).reshape(1, -1)
            layout = mx.where(row_mask, ones, layout)
            layout = mx.where(col_mask, ones, layout)

        layout = rearrange(
            layout,
            '(p blksz) (r blksz1) -> p r (blksz blksz1)',
            blksz=self.block,
            blksz1=self.block,
        )
        mask = mx.any(layout.astype(mx.bool_), axis=-1)
        return mask.astype(mx.int32)
