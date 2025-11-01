#!/usr/bin/env python
"""
Complete Monarch Mixer Sequence Mixing in MLX

Full port of MonarchMixerSequenceMixing - the core M2-BERT innovation
that replaces self-attention with sub-quadratic long convolutions.

Architecture:
1. Input projection (d_model → 3*d_model)
2. Short conv (3x1 depthwise convolution)
3. Split into x1, x2, v
4. Input gate: v = v * x1
5. Long conv with learnable Hyena filter
6. Post-gate: y = y * x2
7. Output projection

Optional: Residual long conv (second Hyena path on input)
"""

import mlx.core as mx
import mlx.nn as nn

from .hyena_filter_mlx import HyenaFilter


class MonarchMixerSequenceMixing(nn.Module):
    """
    Complete Monarch Mixer for sequence mixing (replaces attention)

    Uses Hyena-style gated long convolution instead of O(L²) attention.
    Complexity: O(L log L) via FFT convolution.

    Args:
        d_model: Model dimension
        l_max: Maximum sequence length
        dropout: Dropout rate
        hyena_kernel_lr: Learning rate for Hyena filter
        bidirectional: Whether to use bidirectional filtering
        hyena_lr_pos_emb: LR for positional embeddings
        hyena_w: Frequency of periodic activations
        hyena_w_mod: Modulation of frequency
        hyena_wd: Weight decay for Hyena parameters
        hyena_emb_dim: Embedding dimension for Hyena filter
        hyena_filter_dropout: Dropout for Hyena filter
        hyena_filter_order: Order (width) of implicit MLP
        residual_long_conv: Whether to add residual long conv path
        hyena_training_additions: Whether to add extra layernorm/dropout
        fft_chunk_size: Optional chunk size for FFTs
    """

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
        fft_chunk_size=None,
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

        # Main Hyena filter
        self.filter_fn = HyenaFilter(
            d_model,
            order=hyena_filter_order,
            seq_len=l_max,
            dropout=hyena_filter_dropout,
            bidirectional=bidirectional,
            lr=hyena_kernel_lr if hyena_kernel_lr else 1e-3,
            lr_pos_emb=hyena_lr_pos_emb,
            w=hyena_w,
            w_mod=hyena_w_mod,
            wd=hyena_wd,
            emb_dim=hyena_emb_dim,
            fft_chunk_size=fft_chunk_size,
        )

        # Residual long conv filter (optional)
        if self.residual_long_conv:
            self.filter_fn2 = HyenaFilter(
                d_model,
                order=hyena_filter_order,
                seq_len=l_max,
                dropout=hyena_filter_dropout,
                bidirectional=bidirectional,
                lr=hyena_kernel_lr if hyena_kernel_lr else 1e-3,
                lr_pos_emb=hyena_lr_pos_emb,
                w=hyena_w,
                w_mod=hyena_w_mod,
                wd=hyena_wd,
                emb_dim=hyena_emb_dim,
                fft_chunk_size=fft_chunk_size,
            )

        # Input projection: d_model → 3 * d_model
        self.in_linear = nn.Linear(d_model, 3 * d_model)

        # Output projection
        self.out_linear = nn.Linear(d_model, d_model)

        # Optional training additions
        self.hyena_training_additions = hyena_training_additions
        if self.hyena_training_additions:
            self.act = nn.Identity()
            self.drop = nn.Dropout(dropout)
            self.layernorm = nn.LayerNorm(d_model)

        # Short convolution (depthwise)
        total_width = self.d_model * self.NUM_PROJECTIONS

        # MLX doesn't have Conv1d with groups, so we implement depthwise manually
        # For now, use a simple linear approximation
        self.short_filter_weight = mx.random.normal((total_width, 3)) * mx.array(0.02, dtype=mx.float32)

    def depthwise_conv1d(self, x, kernel_size=3, padding=2):
        """
        Depthwise 1D convolution (manual implementation)

        Args:
            x: (batch, channels, length)
            kernel_size: Convolution kernel size
            padding: Padding

        Returns:
            out: (batch, channels, length)
        """
        batch, channels, length = x.shape

        # Pad input
        x_padded = mx.pad(x, [(0, 0), (0, 0), (padding, padding)])

        # Apply depthwise conv manually (channel-wise)
        outputs = []
        for i in range(channels):
            channel_data = x_padded[:, i:i + 1, :]  # (batch, 1, L+padding)
            kernel = self.short_filter_weight[i, :]  # (kernel_size,)

            conv_out = []
            for t in range(length):
                window = channel_data[:, 0, t:t + kernel_size]  # (batch, kernel_size)
                out_val = mx.sum(window * kernel.reshape(1, -1), axis=1)  # (batch,)
                conv_out.append(out_val)

            conv_result = mx.stack(conv_out, axis=1)  # (batch, length)
            outputs.append(conv_result)

        result = mx.stack(outputs, axis=1)  # (batch, channels, length)
        return result

    def __call__(self, u, tracer=None, **kwargs):
        """
        Forward pass

        Args:
            u: (batch, length, d_model) - input

        Returns:
            y: (batch, length, d_model)
            None: (for API compatibility)
        """
        # Optional pre-layernorm
        if self.hyena_training_additions:
            u = self.layernorm(u)

        L = u.shape[1]  # Sequence length

        # Store input for residual conv
        u_orig = u

        # Input projection
        u = self.in_linear(u)  # (batch, L, 3*d_model)
        if tracer is not None:
            try:
                from src.utils.tracer import Tracer  # noqa: F401
                tracer.log('monarch.in_linear', u, framework='mlx')
            except Exception:
                pass

        # Transpose for convolution: (batch, L, d) → (batch, d, L)
        u = u.transpose(0, 2, 1)  # (batch, 3*d_model, L)

        # Short convolution
        uc = self.depthwise_conv1d(u, kernel_size=3, padding=2)
        uc = uc[..., :L]  # Truncate to original length
        if tracer is not None:
            try:
                tracer.log('monarch.depthwise3', uc, framework='mlx')
            except Exception:
                pass

        # Split into x1, x2, v
        x1 = uc[:, :self.d_model, :]  # (batch, d_model, L)
        x2 = uc[:, self.d_model:2 * self.d_model, :]  # (batch, d_model, L)
        v = uc[:, 2 * self.d_model:, :]  # (batch, d_model, L)

        # Input gate: v = v * x1
        v = v * x1
        if tracer is not None:
            try:
                tracer.log('monarch.gated_v', v, framework='mlx')
            except Exception:
                pass

        # Optional dropout
        if self.hyena_training_additions:
            v = self.drop(v)

        # Generate Hyena filter
        k_fwd = self.filter_fn.filter(L)  # (1, L, d_model)
        k_fwd = k_fwd.transpose(0, 2, 1)[0]  # (1, d_model, L) -> (d_model, L)

        # Bidirectional filter
        if self.bidirectional:
            k_rev = self.filter_fn.filter_rev(L)  # (1, L, d_model)
            k_rev = k_rev.transpose(0, 2, 1)[0]  # (1, d_model, L) -> (d_model, L)
        else:
            k_rev = None

        # Reshape bias: (d_model,) -> (1, d_model, 1)
        bias = self.filter_fn.bias.reshape(1, -1, 1)

        # Apply long convolution (Hyena)
        y = self.filter_fn(v, L, k_fwd=k_fwd, k_rev=k_rev, bias=bias, tracer=tracer)
        if tracer is not None:
            try:
                tracer.log('monarch.hyena_out', y, framework='mlx')
            except Exception:
                pass

        # Residual long conv path (optional)
        if self.residual_long_conv:
            # Apply second Hyena filter to original input
            k2_fwd = self.filter_fn2.filter(L)
            k2_fwd = k2_fwd.transpose(0, 2, 1)[0]

            if self.bidirectional:
                k2_rev = self.filter_fn2.filter_rev(L)
                k2_rev = k2_rev.transpose(0, 2, 1)[0]
            else:
                k2_rev = None

            bias2 = self.filter_fn2.bias.reshape(1, -1, 1)

            u_orig_t = u_orig.transpose(0, 2, 1)
            yu = self.filter_fn2(u_orig_t, L, k_fwd=k2_fwd, k_rev=k2_rev, bias=bias2)
        else:
            yu = None

        # Post-gate: y = y * x2
        y = y * x2
        if tracer is not None:
            try:
                tracer.log('monarch.post_gate', y, framework='mlx')
            except Exception:
                pass

        # Add residual path
        if self.residual_long_conv and yu is not None:
            y = y + yu

        # Transpose back: (batch, d_model, L) → (batch, L, d_model)
        y = y.transpose(0, 2, 1)

        # Optional activation + dropout
        if self.hyena_training_additions:
            y = self.drop(self.act(y))

        # Output projection
        y = self.out_linear(y)
        if tracer is not None:
            try:
                tracer.log('monarch.out_linear', y, framework='mlx')
            except Exception:
                pass

        return y, None


def _demo():
    batch_size = 2
    seq_len = 64
    d_model = 768
    mixer = MonarchMixerSequenceMixing(
        d_model=d_model,
        l_max=seq_len,
        bidirectional=True,
        hyena_filter_order=64,
        residual_long_conv=True,
    )
    x = mx.random.normal((batch_size, seq_len, d_model))
    y, _ = mixer(x)
    print('Input:', x.shape, 'Output:', y.shape)


if __name__ == '__main__':
    _demo()
