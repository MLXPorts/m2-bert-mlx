# Adapted from https://github.com/HazyResearch/fly/tree/master/src/models/layers

import mlx.core as mx
from bert.src.mlx_ops.einops import rearrange

from .blockdiag_multiply import blockdiag_multiply
from .structured_linear import StructuredLinear


class BlockdiagLinear(StructuredLinear):

    def __init__(self, *args, nblocks=4, shuffle=False, **kwargs):
        """shuffle: apply channel_shuffle operation before the matmul as in ShuffleNet."""
        super().__init__(*args, **kwargs)
        # Use MLX ops for all math
        in_features_mx = mx.array(self.in_features, dtype=mx.float32)
        out_features_mx = mx.array(self.out_features, dtype=mx.float32)
        nblocks_mx = mx.array(nblocks, dtype=mx.float32)

        in_blksz_float = mx.divide(in_features_mx, nblocks_mx)
        out_blksz_float = mx.divide(out_features_mx, nblocks_mx)
        in_blksz_mx = mx.ceil(in_blksz_float).astype(mx.int32)
        out_blksz_mx = mx.ceil(out_blksz_float).astype(mx.int32)

        nblocks_mx_int = mx.array(nblocks, dtype=mx.int32)

        self.in_features_extended = mx.multiply(in_blksz_mx, nblocks_mx_int)
        self.out_features_extended = mx.multiply(out_blksz_mx, nblocks_mx_int)
        self.shuffle = shuffle
        self.nblocks = nblocks
        self.weight = mx.zeros((nblocks, out_blksz_mx, in_blksz_mx), dtype=mx.float32)
        self.reset_parameters()

    def set_weights_from_dense(self, dense_weight: mx.array):
        # Scale by sqrt because the weight is sparse
        # Compute product of shape dimensions using MLX
        total_dense = mx.array(1, dtype=mx.int32)
        for dim in dense_weight.shape:
            total_dense = mx.multiply(total_dense, mx.array(dim, dtype=mx.int32))

        total_sparse = mx.array(1, dtype=mx.int32)
        for dim in self.weight.shape:
            total_sparse = mx.multiply(total_sparse, mx.array(dim, dtype=mx.int32))

        # Convert to float for division and sqrt
        total_dense_f = mx.array(total_dense, dtype=mx.float32)
        total_sparse_f = mx.array(total_sparse, dtype=mx.float32)
        ratio = mx.divide(total_dense_f, total_sparse_f)
        scaling = mx.sqrt(ratio)
        scaling_typed = mx.array(scaling, dtype=dense_weight.dtype)
        dense_weight = mx.multiply(dense_weight, scaling_typed)
        nblocks = self.weight.shape[0]
        reshaped = rearrange(dense_weight, '(b o) (b1 i) -> b b1 o i', b=nblocks, b1=nblocks)
        self.weight = reshaped[0]

    @property
    def saving(self):
        # Compute product of weight shape using MLX
        weight_prod = mx.array(1, dtype=mx.int32)
        for dim in self.weight.shape:
            weight_prod = mx.multiply(weight_prod, mx.array(dim, dtype=mx.int32))

        # Compute product of in_features * out_features using MLX
        in_f = mx.array(self.in_features, dtype=mx.int32)
        out_f = mx.array(self.out_features, dtype=mx.int32)
        total_features = mx.multiply(in_f, out_f)

        # Divide and return as MLX scalar
        weight_prod_f = mx.array(weight_prod, dtype=mx.float32)
        total_features_f = mx.array(total_features, dtype=mx.float32)
        return mx.divide(weight_prod_f, total_features_f)

    def forward_matmul(self, x):
        x = self.preprocess(x)
        if self.shuffle:
            x = rearrange(
                x, '... (group c_per_group) -> ... (c_per_group group)', group=self.weight.shape[0]
            )
        output = blockdiag_multiply(x, self.weight)
        return self.postprocess(output)


class BlockdiagSparsityConfig:

    def __init__(self, nblocks, block=32, global_size=0):
        """Sparsity description used for structured pruning."""
        self.nblocks = nblocks
        self.block = block
        self.global_size = global_size

    def make_layout(self, out_features, in_features):
        assert out_features % self.block == 0 and in_features % self.block == 0
        assert out_features % self.nblocks == 0 and in_features % self.nblocks == 0

        # Use MLX divmod for floor division
        out_features_mx = mx.array(out_features, dtype=mx.int32)
        in_features_mx = mx.array(in_features, dtype=mx.int32)
        nblocks_mx = mx.array(self.nblocks, dtype=mx.int32)

        block_rows, _ = mx.divmod(out_features_mx, nblocks_mx)
        block_cols, _ = mx.divmod(in_features_mx, nblocks_mx)

        one_block = mx.ones((block_rows.item(), block_cols.item()), dtype=mx.int32)
        zero_block = mx.zeros((block_rows.item(), block_cols.item()), dtype=mx.int32)

        rows = []
        for i in range(self.nblocks):
            blocks = []
            for j in range(self.nblocks):
                blocks.append(one_block if i == j else zero_block)
            rows.append(mx.concatenate(blocks, axis=1))
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
        return (layout > 0).any(axis=-1).astype(mx.int32)
