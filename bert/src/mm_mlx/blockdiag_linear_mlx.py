#!/usr/bin/env python
"""
MLX Block-diagonal linear layer to mirror src/mm/blockdiag_linear.py.

Implements a structured linear: weight shaped (nblocks, out_blk, in_blk),
applied over the input split into nblocks along the last dimension.
"""

import mlx.core as mx
import mlx.nn as nn
from .blockdiag_ops_mlx import blockdiag_multiply


class BlockdiagLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, nblocks=4, shuffle=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nblocks = nblocks
        self.shuffle = shuffle

        # Integer ceil division to avoid float->int casts
        self.in_blksz = (in_features + nblocks - 1) // nblocks
        self.out_blksz = (out_features + nblocks - 1) // nblocks
        self.in_features_extended = self.in_blksz * nblocks
        self.out_features_extended = self.out_blksz * nblocks

        # Parameters
        self.weight = mx.random.normal((nblocks, self.out_blksz, self.in_blksz)) * mx.array(0.02, dtype=mx.float32)
        self.use_bias = bias
        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

    def preprocess(self, x):
        # Pad last dim to in_features_extended
        pad = self.in_features_extended - x.shape[-1]
        if pad > 0:
            x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
        return x

    def postprocess(self, y):
        # Truncate to out_features
        if y.shape[-1] > self.out_features:
            y = y[..., :self.out_features]
        return y

    def __call__(self, x):
        # x: (..., in_features)
        x = self.preprocess(x)
        *batch, n = x.shape
        # Flatten batch dims for compute
        B = 1
        for d in batch:
            B *= d
        x2 = x.reshape(B, n)  # (B, in)
        # Reshape into blocks: (B, K, Pin)
        x_blk = x2.reshape(B, self.nblocks, self.in_blksz)
        # Weight: (K, Pout, Pin)
        W = self.weight
        # Vectorized block matmul
        y = blockdiag_multiply(x2, W, nblocks=self.nblocks).reshape(*batch, -1)
        y = self.postprocess(y)
        if self.bias is not None:
            y = y + self.bias.reshape(1, -1)
        return y
