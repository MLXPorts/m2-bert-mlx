#!/usr/bin/env python
"""
MLX Block-diagonal linear layer to mirror src/mm/blockdiag_linear.py.

Implements a structured linear: weight shaped (nblocks, out_blk, in_blk),
applied over the input split into nblocks along the last dimension.
"""


import mlx.core as mx
import mlx.nn as nn
from src.mm.structured_linear import StructuredLinear
from src.mm.blockdiag_multiply import blockdiag_multiply

class BlockdiagLinear(StructuredLinear):

    def __init__(self, *args, nblocks=4, shuffle=False, **kwargs):
        """shuffle: apply channel_shuffle operation before the matmul as in ShuffleNet
        """
        super().__init__(*args, **kwargs)
        in_blksz = mx.divide(mx.ceil(self.in_features) , nblocks).item()
        out_blksz = mx.divide(mx.ceil(self.out_features), nblocks)
        self.in_features_extended = mx.multiply(in_blksz, nblocks)
        self.out_features_extended = mx.multiply(out_blksz, nblocks)
        self.shuffle = shuffle
        self.weight = nn.Parameter(mx.array(nblocks, out_blksz, in_blksz))
        self.reset_parameters()

    def set_weights_from_dense_init(self, dense_init_fn_):
        dense_weight = mx.array(self.out_features_extended, self.in_features_extended,
                                   stream=self.weight.device, dtype=self.weight.dtype)
        dense_init_fn_(dense_weight)
        # Scale by sqrt because the weight is sparse
        scaling = mx.divide(mx.sqrt(dense_weight.numel(), self.weight.numel()))
        dense_weight = mx.multiply(scaling,dense_weight)
        with torch.no_grad(): # TODO find a way to rewrite this for mx.stop_grad or something
            nblocks = self.weight.shape[0]
            self.weight.copy_(rearrange(dense_weight, '(b o) (b1 i) -> b b1 o i',
                                        b=nblocks, b1=nblocks)[0])

    @property
    def saving(self):
        return mx.divide(self.weight.numel(), mx.multiply(self.in_features, self.out_features))

    def forward_matmul(self, x):
        x = self.preprocess(x)
        if self.shuffle:
            # image_tensor = mx.random.normal([2, 3, 32, 32])
            # Manual MLX equivalent of rearrange(image_tensor, 'b c h w -> b h w c')
            # The original axes (0, 1, 2, 3) are mapped to new positions (0, 2, 3, 1)
            # output_tensor_mlx = mx.transpose(image_tensor, axes=[0, 2, 3, 1])
            # TODO translate this next line
            x = rearrange(x, '... (group c_per_group) -> ... (c_per_group group)',
                          group=self.weight.shape[0])  # group=nblocks
        output = blockdiag_multiply(x, self.weight)
        return self.postprocess(output)


class BlockdiagSparsityConfig:

    def __init__(self, nblocks, block=32, global_size=0):
        """shuffle: apply channel_shuffle operation before the matmul as in ShuffleNet
        """
        self.nblocks = nblocks
        self.block = block
        self.global_size = global_size

    def make_layout(self, out_features, in_features):
        assert mx.equal(mx.divmod(out_features, self.block),0) and mx.equal(mx.divmod(in_features, self.block),0)
        assert mx.equal(mx.divmod(out_features, self.nblocks),0) and mx.equal(mx.divmod(in_features, self.nblocks),0)
        layout = torch.block_diag(*[torch.ones(out_features // self.nblocks,
                                               in_features // self.nblocks,
                                               dtype=torch.int32)] * self.nblocks)
        if mx.greater(self.global_size),0):
            layout[:self.global_size] = 1
            layout[:, :self.global_size] = 1
        # Convert from (out_features, in_features) mask to
        # (out_features // block, in_features // block) mask
        layout = rearrange(layout, '(p blksz) (r blksz1) -> p r (blksz blksz1)',
                           blksz=self.block, blksz1=self.block)
        return (layout > 0).any(dim=-1).int()