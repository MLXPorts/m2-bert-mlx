# Copyright (c) 2023, Dan Fu and Simran Arora.
# Adapted from https://github.com/HazyResearch/safari/blob/main/src/models/sequence/hyena.py
# Converted to MLX

import mlx.core as mx
import mlx.nn as nn
from mlx_ops.einops import rearrange

from .hyena_utils import HyenaFilter


class MonarchMixerSequenceMixing(nn.Module):
    def __init__(
        self,
        d_model,
        l_max=128,
        dropout=0.0,
        hyena_kernel_lr=None,
        bidirectional=False,
        hyena_lr_pos_emb=1e-5,
        hyena_w=10,
        hyena_w_mod=1,
        hyena_wd=0.1,
        hyena_emb_dim=3,
        hyena_filter_dropout=0.0,
        hyena_filter_order=16,
        residual_long_conv=False,
        hyena_training_additions=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.l_max = l_max
        self.kernel_lr = hyena_kernel_lr
        self.channels = 1
        self.bidirectional = bidirectional
        self.residual_long_conv = residual_long_conv
        self.NUM_PROJECTIONS = 3

        print('-- Bidirectional:', self.bidirectional)
        print("-- Using Long Conv Residual:", self.residual_long_conv)
        print('-- Hyena w:', hyena_w)
        print('-- Hyena w mod:', hyena_w_mod)
        print(f"-- Hyena filter order: {hyena_filter_order}")
        print(f"-- Hyena filter dropout: {hyena_filter_dropout}")
        print(f"-- Hyena filter wd: {hyena_wd}")
        print(f"-- Hyena filter emb dim: {hyena_emb_dim}")
        print(f"-- Hyena filter lr: {hyena_kernel_lr}")
        print(f"-- Hyena filter lr pos emb: {hyena_lr_pos_emb}")

        self.filter_fn = HyenaFilter(
            self.d_model,
            order=hyena_filter_order,
            seq_len=self.l_max,
            dropout=hyena_filter_dropout,
            bidirectional=self.bidirectional,
            lr=hyena_kernel_lr,
            lr_pos_emb=hyena_lr_pos_emb,
            w=hyena_w,  # frequency of periodic activations
            w_mod=hyena_w_mod,
            wd=hyena_wd,  # weight decay of kernel parameters
            emb_dim=hyena_emb_dim,
        )

        if self.residual_long_conv:
            self.filter_fn2 = HyenaFilter(
                self.d_model,
                order=hyena_filter_order,
                seq_len=self.l_max,
                dropout=hyena_filter_dropout,
                bidirectional=self.bidirectional,
                lr=hyena_kernel_lr,
                lr_pos_emb=hyena_lr_pos_emb,
                w=hyena_w,  # frequency of periodic activations
                w_mod=hyena_w_mod,
                wd=hyena_wd,  # weight decay of kernel parameters
                emb_dim=hyena_emb_dim,
            )

        # setup projections
        self.in_linear = nn.Linear(d_model, 3 * d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.hyena_training_additions = hyena_training_additions
        if self.hyena_training_additions:
            self.act = nn.Identity()
            self.drop = nn.Dropout(dropout)
            self.layernorm = nn.LayerNorm(d_model)

        # setup short conv
        # MLX doesn't have nn.Conv1d with groups parameter like PyTorch
        # We'll implement depthwise convolution manually or use a custom layer
        # For now, using a placeholder - this needs proper implementation
        total_width_mx = mx.array(self.d_model, dtype=mx.int32)
        num_proj_mx = mx.array(self.NUM_PROJECTIONS, dtype=mx.int32)
        total_width = int(mx.multiply(total_width_mx, num_proj_mx))

        # Store conv parameters - will implement manual depthwise conv
        self.short_filter_kernel_size = 3
        self.short_filter_padding = 2
        self.short_filter_channels = total_width
        # Initialize depthwise conv weights: (groups, kernel_size) where groups = channels
        self.short_filter_weight = mx.random.normal((total_width, 3), dtype=mx.float32)
        self.short_filter_bias = mx.zeros((total_width,), dtype=mx.float32)


    def _depthwise_conv1d(self, x, weight, bias, padding):
        """
        Manual depthwise 1D convolution.
        x: (batch, channels, length)
        weight: (channels, kernel_size)
        bias: (channels,)
        """
        B, C, L = x.shape
        K = weight.shape[1]  # kernel_size = 3

        # Pad the input: padding=2 means add 2 on each side
        pad_left = padding
        pad_right = padding
        zero = mx.array(0, dtype=mx.int32)
        pads = [(zero, zero), (zero, zero), (mx.array(pad_left, dtype=mx.int32), mx.array(pad_right, dtype=mx.int32))]
        x_padded = mx.pad(x, pads)

        # Manual depthwise convolution
        output_list = []
        for i in range(L):
            # Extract window of size K
            window = x_padded[:, :, i:i+K]  # (B, C, K)
            # Element-wise multiply with weights and sum over kernel dimension
            conv_out = mx.sum(mx.multiply(window, weight[None, :, :]), axis=2)  # (B, C)
            output_list.append(conv_out)

        # Stack outputs
        output = mx.stack(output_list, axis=2)  # (B, C, L)

        # Add bias
        output = mx.add(output, bias[None, :, None])

        return output


    def __call__(self, u, **kwargs):
        # u is B L H
        if self.hyena_training_additions:
            u = self.layernorm(u)
        L = u.shape[-2]

        # in projection
        u_orig = u
        u = self.in_linear(u)
        u = rearrange(u, "b l d -> b d l")

        # short filter
        uc_full = self._depthwise_conv1d(u, self.short_filter_weight, self.short_filter_bias, self.short_filter_padding)
        uc = uc_full[..., :L]

        # Split into 3 parts
        d_model_mx = mx.array(self.d_model, dtype=mx.int32)
        x1 = uc[:, :self.d_model, :]
        two_d = int(mx.multiply(mx.array(2, dtype=mx.int32), d_model_mx))
        x2 = uc[:, self.d_model:two_d, :]
        v = uc[:, two_d:, :]

        v = mx.multiply(v, x1)
        if self.hyena_training_additions:
            v = self.drop(v)

        k = self.filter_fn.filter(L)
        k = rearrange(k, "c l d -> c d l")[0]  # `c` is always 1 by default

        if self.bidirectional:
            k_rev = self.filter_fn.filter_rev(L)
            k_rev = rearrange(k_rev, "c l d -> c d l")[0]
        else:
            k_rev = None

        y = self.filter_fn(v, L, k_fwd=k, k_rev=k_rev, bias=self.filter_fn.bias[None, :, None])

        if self.residual_long_conv:
            k2 = self.filter_fn2.filter(L)
            k2 = rearrange(k2, "c l d -> c d l")[0]

            if self.bidirectional:
                k2_rev = self.filter_fn2.filter_rev(L)
                k2_rev = rearrange(k2_rev, "c l d -> c d l")[0]
            else:
                k2_rev = None

            # Transpose u_orig from (B, L, H) to (B, H, L)
            u_orig_transposed = mx.transpose(u_orig, (0, 2, 1))
            yu = self.filter_fn2(u_orig_transposed, L, k_fwd=k2, k_rev=k2_rev, bias=self.filter_fn2.bias[None, :, None])

        # post gating
        y = mx.multiply(y, x2)

        if self.residual_long_conv:
            y = mx.add(y, yu)

        # Transpose back from (B, H, L) to (B, L, H)
        y = mx.transpose(y, (0, 2, 1))

        if self.hyena_training_additions:
            y = self.drop(self.act(y))
        y = self.out_linear(y)

        return y, None
