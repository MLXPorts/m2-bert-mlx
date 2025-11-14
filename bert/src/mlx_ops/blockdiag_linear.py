#!/usr/bin/env python
"""
MLX Block-diagonal linear layer to mirror src/mm/blockdiag_linear.py.

Implements a structured linear: weight shaped (nblocks, out_blk, in_blk),
applied over the input split into nblocks along the last dimension.
"""

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from .blockdiag_multiply import blockdiag_multiply
from .einops import rearrange


class StructuredLinear(nn.Module):
    """Base class for structured linear layers in MLX."""

    def __init__(self, in_features, out_features, bias=True):
        """Subclasses should call reset_parameters"""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Subclasses may override {in,out}_features_extended
        if not hasattr(self, 'in_features_extended'):
            self.in_features_extended = in_features
        if not hasattr(self, 'out_features_extended'):
            self.out_features_extended = out_features
        if bias:
            self.bias = mx.zeros([out_features], dtype=mx.float32)
        else:
            self.bias = None

    def reset_parameters(self):
        """Initialize weights from dense initialization."""
        self.set_weights_from_dense_init(dense_init_fn_=partial(self._kaiming_uniform, a=math.sqrt(5)))
        self.reset_parameters_bias()

    def _kaiming_uniform(self, tensor, a=0):
        """MLX implementation of Kaiming uniform initialization."""
        fan_in = tensor.shape[1] if len(tensor.shape) > 1 else tensor.shape[0]
        gain = math.sqrt(2.0 / (1 + a * a))
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std
        bound_mx = mx.array(bound, dtype=mx.float32)
        return mx.multiply(mx.subtract(mx.random.uniform(shape=tensor.shape, dtype=mx.float32),
                                       mx.array(0.5, dtype=mx.float32)),
                          mx.multiply(mx.array(2.0, dtype=mx.float32), bound_mx))

    def set_weights_from_dense_init(self, dense_init_fn_):
        """Set weights from dense initialization function."""
        raise NotImplementedError

    def reset_parameters_bias(self):
        """Reset bias parameters."""
        if self.bias is not None:
            fan_in = self.bias.shape[-1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            bound_mx = mx.array(bound, dtype=mx.float32)
            self.bias = mx.multiply(mx.subtract(mx.random.uniform(shape=self.bias.shape, dtype=mx.float32),
                                                mx.array(0.5, dtype=mx.float32)),
                                   mx.multiply(mx.array(2.0, dtype=mx.float32), bound_mx))

    @property
    def saving(self):
        """Return parameter saving ratio."""
        raise NotImplementedError

    def convert_to_dense_weight(self):
        """Convert structured weight to dense weight."""
        eye = mx.eye(self.in_features, dtype=mx.float32)
        dense_weight = self.forward_matmul(eye)
        transpose_axes = mx.array([1, 0], dtype=mx.int32)
        return mx.transpose(dense_weight, transpose_axes)

    def preprocess(self, x):
        """Preprocess input by padding if necessary."""
        in_features = x.shape[-1]
        # Use Python comparison for shape logic (acceptable for control flow)
        if in_features < self.in_features_extended:
            pad_size = self.in_features_extended - in_features
            # Pad the last dimension
            pad_shape = list(x.shape)
            pad_shape[-1] = pad_size
            padding = mx.zeros(pad_shape, dtype=x.dtype)
            x = mx.concatenate([x, padding], axis=-1)
        return x

    def postprocess(self, output):
        """Postprocess output by trimming if necessary."""
        out_features_extended = output.shape[-1]
        # Use Python comparison for shape logic (acceptable for control flow)
        if out_features_extended > self.out_features:
            # Slice to keep only first out_features
            output = output[..., :self.out_features]
        return output

    def forward_matmul(self, x):
        """Forward pass matrix multiplication."""
        raise NotImplementedError

    def __call__(self, x):
        """Forward pass."""
        output = self.forward_matmul(x)
        if self.bias is not None:
            # Ensure bias matches output dtype
            bias = self.bias.astype(output.dtype)
            output = mx.add(output, bias)
        return output


class BlockdiagLinear(StructuredLinear):
    """Block-diagonal linear layer for MLX."""

    def __init__(self, *args, nblocks=4, shuffle=False, **kwargs):
        """
        Initialize block-diagonal linear layer.

        Args:
            nblocks: Number of diagonal blocks
            shuffle: Apply channel_shuffle operation before matmul as in ShuffleNet
        """
        super().__init__(*args, **kwargs)

        # Calculate block sizes
        in_features_mx = mx.array(self.in_features, dtype=mx.float32)
        out_features_mx = mx.array(self.out_features, dtype=mx.float32)
        nblocks_mx = mx.array(nblocks, dtype=mx.float32)

        in_blksz_float = mx.ceil(mx.divide(in_features_mx, nblocks_mx))
        out_blksz_float = mx.ceil(mx.divide(out_features_mx, nblocks_mx))

        # Convert to integers for shape calculations
        in_blksz = int(in_blksz_float.item())
        out_blksz = int(out_blksz_float.item())

        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks
        self.shuffle = shuffle
        self.nblocks = nblocks

        # Initialize weight as MLX array (not nn.Parameter)
        self.weight = mx.zeros([nblocks, out_blksz, in_blksz], dtype=mx.float32)
        self.reset_parameters()

    def set_weights_from_dense_init(self, dense_init_fn_):
        """Initialize weights from dense initialization function."""
        # Create dense weight tensor
        dense_weight = mx.zeros([self.out_features_extended, self.in_features_extended],
                               dtype=mx.float32)
        dense_weight = dense_init_fn_(dense_weight)

        # Calculate scaling factor
        dense_numel_mx = mx.array(self.out_features_extended * self.in_features_extended,
                                 dtype=mx.float32)
        weight_numel_mx = mx.array(self.nblocks * self.weight.shape[1] * self.weight.shape[2],
                                  dtype=mx.float32)
        scaling = mx.sqrt(mx.divide(dense_numel_mx, weight_numel_mx))
        dense_weight = mx.multiply(dense_weight, scaling)

        # Rearrange dense weight to block diagonal format
        # (out_features_extended, in_features_extended) -> (nblocks, nblocks, out_blk, in_blk)
        rearranged = rearrange(dense_weight, '(b o) (b1 i) -> b b1 o i',
                              b=self.nblocks, b1=self.nblocks)
        # Take only diagonal blocks (b == b1)
        self.weight = rearranged[0]

    @property
    def saving(self):
        """Calculate parameter saving ratio."""
        in_features_mx = mx.array(self.in_features, dtype=mx.float32)
        out_features_mx = mx.array(self.out_features, dtype=mx.float32)
        total_dense = mx.multiply(in_features_mx, out_features_mx)

        weight_numel = mx.array(self.nblocks * self.weight.shape[1] * self.weight.shape[2],
                               dtype=mx.float32)
        return mx.divide(weight_numel, total_dense)

    def forward_matmul(self, x):
        """Forward matrix multiplication."""
        x = self.preprocess(x)
        if self.shuffle:
            # Apply channel shuffle: rearrange groups
            x = rearrange(x, '... (group c_per_group) -> ... (c_per_group group)',
                         group=self.weight.shape[0])
        output = blockdiag_multiply(x, self.weight)
        return self.postprocess(output)


class BlockdiagSparsityConfig:
    """Configuration for block-diagonal sparsity pattern."""

    def __init__(self, nblocks, block=32, global_size=0):
        """
        Initialize sparsity configuration.

        Args:
            nblocks: Number of diagonal blocks
            block: Block size for sparsity pattern
            global_size: Size of global (dense) region
        """
        self.nblocks = nblocks
        self.block = block
        self.global_size = global_size

    def make_layout(self, out_features, in_features):
        """
        Create block-diagonal sparsity layout.

        Args:
            out_features: Output feature dimension
            in_features: Input feature dimension

        Returns:
            Sparsity layout mask
        """
        # Check divisibility
        assert out_features % self.block == 0 and in_features % self.block == 0
        assert out_features % self.nblocks == 0 and in_features % self.nblocks == 0

        # Create block diagonal layout
        block_out = out_features // self.nblocks
        block_in = in_features // self.nblocks

        # Initialize layout as zeros
        layout = mx.zeros([out_features, in_features], dtype=mx.int32)

        # Fill diagonal blocks with ones
        for i in range(self.nblocks):
            i_mx = mx.array(i, dtype=mx.int32)
            one_mx = mx.array(1, dtype=mx.int32)
            out_start = mx.multiply(i_mx, mx.array(block_out, dtype=mx.int32))
            out_end = mx.multiply(mx.add(i_mx, one_mx), mx.array(block_out, dtype=mx.int32))
            in_start = mx.multiply(i_mx, mx.array(block_in, dtype=mx.int32))
            in_end = mx.multiply(mx.add(i_mx, one_mx), mx.array(block_in, dtype=mx.int32))

            # Create ones block
            ones_block = mx.ones([block_out, block_in], dtype=mx.int32)
            layout = mx.slice_update(layout, ones_block, [out_start, in_start], [out_end, in_end])

        # Add global region if specified
        # Use Python comparison for control flow (acceptable)
        if self.global_size > 0:
            # Set first global_size rows to 1
            ones_row = mx.ones([self.global_size, in_features], dtype=mx.int32)
            zero_mx = mx.array(0, dtype=mx.int32)
            global_size_mx = mx.array(self.global_size, dtype=mx.int32)
            in_features_mx = mx.array(in_features, dtype=mx.int32)
            out_features_mx = mx.array(out_features, dtype=mx.int32)
            layout = mx.slice_update(layout, ones_row, [zero_mx, zero_mx],
                                    [global_size_mx, in_features_mx])
            # Set first global_size columns to 1
            ones_col = mx.ones([out_features, self.global_size], dtype=mx.int32)
            layout = mx.slice_update(layout, ones_col, [zero_mx, zero_mx],
                                    [out_features_mx, global_size_mx])

        # Convert from (out_features, in_features) mask to
        # (out_features // block, in_features // block) mask
        layout = rearrange(layout, '(p blksz) (r blksz1) -> p r (blksz blksz1)',
                          blksz=self.block, blksz1=self.block)

        # Check if any element in last dimension is > 0
        zero_compare = mx.array(0, dtype=mx.int32)
        mask = mx.greater(layout, zero_compare)
        # Reduce along last dimension
        result = mx.any(mask, axis=-1)
        return result.astype(mx.int32)