# MLX block-diagonal linear layer mirroring the Torch structure.

import math

import mlx.core as mx

from mlx_ops.blockdiag_multiply import blockdiag_multiply
from mlx_ops.einops import rearrange

from .structured_linear import StructuredLinear


class BlockdiagLinear(StructuredLinear):
    """Block-diagonal linear layer implemented with MLX arrays."""

    def __init__(self, *args, nblocks=4, shuffle=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.nblocks = nblocks
        in_blksz = math.ceil(self.in_features / nblocks)
        out_blksz = math.ceil(self.out_features / nblocks)
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks
        self.shuffle = shuffle
        self.weight = mx.zeros((nblocks, out_blksz, in_blksz), dtype=mx.float32)
        self.reset_parameters()

    def set_weights_from_dense_init(self, dense_init_fn_):
        dense = mx.zeros(
            (self.out_features_extended, self.in_features_extended), dtype=mx.float32
        )
        dense = dense_init_fn_(dense)
        dense_numel = self.out_features_extended * self.in_features_extended
        block_numel = self.nblocks * self.weight.shape[1] * self.weight.shape[2]
        scaling = math.sqrt(dense_numel / block_numel)
        dense = dense * mx.array(scaling, dtype=dense.dtype)

        rearranged = rearrange(
            dense,
            '(b o) (b1 i) -> b b1 o i',
            b=self.nblocks,
            b1=self.nblocks,
        )
        self.weight = rearranged[0]

    @property
    def saving(self):
        total_dense = self.in_features * self.out_features
        block_params = self.nblocks * self.weight.shape[1] * self.weight.shape[2]
        return block_params / total_dense

    def forward_matmul(self, x):
        x = self.preprocess(x)
        if self.shuffle:
            x = rearrange(
                x, '... (group c_per_group) -> ... (c_per_group group)', group=self.weight.shape[0]
            )
        y = blockdiag_multiply(x, self.weight)
        return self.postprocess(y)
