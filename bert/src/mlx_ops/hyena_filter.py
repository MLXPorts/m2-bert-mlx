#!/usr/bin/env python
"""
Complete Hyena filter implementation in MLX

Full port of hyena_utils.py with all features:
- Complex exponential positional embeddings
- Exponential modulation
- Learnable sinusoidal activations
- Implicit filter MLP
- FFT-based convolution with optional streaming
- Bidirectional support
"""

import mlx.core as mx
import mlx.nn as nn
from .math_ops import PI, sqrt_2_over_pi

from .kernels.metal_fft_conv import MetalFFTConv
from .kernels.metal_fft_conv_streamed import MetalFFTConvStreamed
from .hyperprofiles_mlx import get_profile


class Sin(nn.Module):
    """Learnable sinusoidal activation."""

    def __init__(self, dim, w=10, w_mod=1, train_freq=True):
        super().__init__()
        self.w_mod = mx.array(w_mod, dtype=mx.float32)
        self.train_freq = train_freq
        self.freq = mx.multiply(mx.ones((1, dim), dtype=mx.float32), mx.array(w, dtype=mx.float32))

    def __call__(self, x):
        return mx.sin(mx.multiply(self.w_mod, mx.multiply(self.freq, x)))


class PositionalEmbedding(nn.Module):
    """Complex exponential positional embeddings for Hyena filters."""

    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.lr_pos_emb = lr_pos_emb

        # Build t in [0,1] without Python float literals
        # t = arange(seq_len) / (seq_len-1)
        sl = mx.array(seq_len, dtype=mx.int32)
        ar_t = mx.arange(seq_len, dtype=mx.float32).reshape(1, seq_len, 1)
        denom = mx.maximum(mx.subtract(sl, mx.array(1, dtype=mx.int32)), mx.array(1, dtype=mx.int32))
        denomf = denom.astype(mx.float32)
        t = mx.divide(ar_t, denomf)

        # Use curated constants (import must succeed)
        PI_CONST = PI

        if emb_dim > 1:
            # bands computed with MLX ops, avoid Python arithmetic
            emb_i32 = mx.array(emb_dim, dtype=mx.int32)
            emb_m1_i32 = mx.subtract(emb_i32, mx.array(1, dtype=mx.int32))
            bands_i32 = mx.floor_divide(emb_m1_i32, mx.array(2, dtype=mx.int32))
            bands_f  = bands_i32.astype(mx.float32)

            t_rescaled = mx.arange(seq_len, dtype=mx.float32).reshape(1, seq_len, 1)
            two = mx.array(2.0, dtype=mx.float32)
            # PI is an MLX array from math_ops; use sl (int32) for denominator
            w = mx.multiply(mx.multiply(two, PI_CONST), mx.divide(t_rescaled, sl.astype(mx.float32)))

            # Tail indices 0..(emb_dim-2), map to band index via modulo bands
            idx = mx.arange(emb_dim, dtype=mx.int32)[1:]
            idx_mod = mx.remainder(idx, bands_i32)
            idx_mod_f = idx_mod.astype(mx.float32)

            # Frequency ladder f in [f_lo, bands-1], scale by denom=max(bands-1,1)
            f_lo = mx.array(1e-4, dtype=mx.float32)
            maxf = mx.subtract(bands_f, mx.array(1.0, dtype=mx.float32))
            denomf2 = mx.maximum(maxf, mx.array(1.0, dtype=mx.float32))
            frac = mx.divide(idx_mod_f, denomf2)
            f = mx.add(f_lo, mx.multiply(mx.subtract(maxf, f_lo), frac))

            phase = mx.multiply(mx.negative(f), w)  # (1,L,emb_dim-1)
            zr = mx.cos(phase)
            zi = mx.sin(phase)

            # First half slots are real, second half are imag
            mask_real = mx.less(idx, bands_i32)
            mask_real_f = mask_real.astype(mx.float32)
            z_tail = mx.add(mx.multiply(mask_real_f, zr), mx.multiply(mx.subtract(mx.array(1.0, dtype=mx.float32), mask_real_f), zi))
            z = mx.concatenate([t, z_tail], axis=-1)
        else:
            z = t

        self.z = z
        self.t = t

    def __call__(self, L):
        return self.z[:, :L, :], self.t[:, :L, :]


class ExponentialModulation(nn.Module):
    """Exponential modulation with learnable decay rates."""

    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        shift: float = 0.0,
    ):
        super().__init__()
        self.shift = mx.array(shift, dtype=mx.float32)
        max_decay = mx.divide(mx.log(mx.array(target, dtype=mx.float32)), mx.array(fast_decay_pct, dtype=mx.float32))
        min_decay = mx.divide(mx.log(mx.array(target, dtype=mx.float32)), mx.array(slow_decay_pct, dtype=mx.float32))
        lin = mx.linspace(0.0, 1.0, d_model).astype(mx.float32).reshape(1, 1, d_model)
        deltas = mx.add(min_decay, mx.multiply(mx.subtract(max_decay, min_decay), lin))
        self.deltas = deltas
        self.modulation_lr = modulation_lr

    def __call__(self, t, x):
        decay = mx.exp(mx.multiply(mx.array(-1.0, dtype=mx.float32), mx.multiply(t, mx.abs(self.deltas))))
        return mx.multiply(x, mx.add(decay, self.shift))


def _hyena_fft_conv(u, k_time, D, seqlen, fft_chunk_size=None, gelu=False, dropout_mask=None, tracer=None, prefix: str = "hyena"):
    """FFT-based 1D convolution using JIT Metal kernels.

    Automatically switches between:
    - Unified kernel (fast, all-in-one) for L <= 2048
    - Streamed kernel (4-phase pipeline) for L > 2048 (supports unlimited length)

    Args:
        u: (batch, d_model, seqlen)
        k_time: (d_model, seqlen)
        D: (1, d_model, 1)
        seqlen: sequence length L
    """
    # Initialize both kernels on first use
    if not hasattr(_hyena_fft_conv, "_conv_unified"):
        _hyena_fft_conv._conv_unified = MetalFFTConv()
        _hyena_fft_conv._conv_streamed = MetalFFTConvStreamed()

    # Automatic kernel selection based on sequence length
    # Unified kernel: supports L up to 2048 (N=2*L=4096 fits in threadgroup memory)
    # Streamed kernel: splits into 4 phases, supports arbitrary L
    if seqlen <= 2048:
        y = _hyena_fft_conv._conv_unified(u, k_time, D)
    else:
        y = _hyena_fft_conv._conv_streamed(u, k_time, D)

    if gelu:
        prof = get_profile()
        if prof.gelu_mode == "tanh":
            c = sqrt_2_over_pi()
            y3 = mx.power(y, mx.array(3.0, dtype=mx.float32))
            inner = mx.multiply(c, mx.add(y, mx.multiply(mx.array(0.044715, dtype=mx.float32), y3)))
            y = mx.multiply(mx.multiply(mx.array(0.5, dtype=mx.float32), y), mx.add(mx.array(1.0, dtype=mx.float32), mx.tanh(inner)))
        else:
            y = nn.gelu(y)
    if dropout_mask is not None:
        y = mx.multiply(y, dropout_mask.reshape(-1, 1, 1))
    return y


class HyenaFilter(nn.Module):
    """Complete Hyena filter with implicit MLP parameterization."""

    def __init__(
        self,
        d_model,
        emb_dim=3,
        order=16,
        seq_len=1024,
        lr=1e-3,
        lr_pos_emb=1e-5,
        dropout=0.0,
        w=1,
        w_mod=1,
        wd=0,
        bias=True,
        num_inner_mlps=2,
        linear_mixer=False,
        modulate: bool = True,
        normalized=False,
        bidirectional=False,
        fft_chunk_size=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.modulate = modulate
        self.use_bias = bias
        self.bidirectional = bidirectional
        self.normalized = normalized
        self.dropout_rate = dropout
        self.fft_chunk_size = fft_chunk_size
        self.w = w
        self.w_mod = w_mod

        # Bias parameter
        self.bias = mx.random.normal((d_model,)) * mx.array(0.02, dtype=mx.float32)

        # emb_dim must be odd and >= 3 (validated by higher-level config)
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        if not linear_mixer:
            # Build explicit alternating Linear/Sin stacks, keep references
            Ls = [nn.Linear(emb_dim, order)]
            Ss = [Sin(dim=order, w=w, w_mod=w_mod)]
            for _ in range(num_inner_mlps):
                Ls.append(nn.Linear(order, order))
                Ss.append(Sin(dim=order, w=w, w_mod=w_mod))
            Ls.append(nn.Linear(order, d_model, bias=False))
            self.implicit_linears = Ls
            self.implicit_sins = Ss
            # For parity tooling
            self.implicit_filter_layers = Ls

            if bidirectional:
                Ls_rev = [nn.Linear(emb_dim, order)]
                Ss_rev = [Sin(dim=order, w=w, w_mod=w_mod)]
                for _ in range(num_inner_mlps):
                    Ls_rev.append(nn.Linear(order, order))
                    Ss_rev.append(Sin(dim=order, w=w, w_mod=w_mod))
                Ls_rev.append(nn.Linear(order, d_model, bias=False))
                self.implicit_linears_rev = Ls_rev
                self.implicit_sins_rev = Ss_rev
                self.implicit_filter_layers_rev = Ls_rev
        else:
            self.implicit_filter = nn.Linear(emb_dim, d_model, bias=False)
            if bidirectional:
                self.implicit_filter_rev = nn.Linear(emb_dim, d_model, bias=False)

        self.modulation = ExponentialModulation(
            d_model,
            fast_decay_pct=0.3,
            slow_decay_pct=1.5,
            target=1e-2,
            modulation_lr=0.0,
            shift=0.0,
        )

        self.lr = lr
        self.lr_pos_emb = lr_pos_emb
        self.wd = wd

    def filter(self, L):
        z, t = self.pos_emb(L)
        if hasattr(self, 'implicit_linears'):
            x = self.implicit_linears[0](z)
            for i, _ in enumerate(self.implicit_sins[:-1]):
                x = self.implicit_sins[i](x)
                x = self.implicit_linears[i + 1](x)
            x = self.implicit_sins[-1](x)
            h = self.implicit_linears[-1](x)
        else:
            h = self.implicit_filter(z)
        if self.modulate:
            h = self.modulation(t, h)
        if self.normalized:
            h_norm = mx.sum(mx.abs(h), axis=-1, keepdims=True)
            h = mx.divide(h, mx.add(h_norm, mx.array(1e-8, dtype=mx.float32)))
        return h

    def filter_rev(self, L):
        z, t = self.pos_emb(L)
        if hasattr(self, 'implicit_linears_rev'):
            x = self.implicit_linears_rev[0](z)
            for i, _ in enumerate(self.implicit_sins_rev[:-1]):
                x = self.implicit_sins_rev[i](x)
                x = self.implicit_linears_rev[i + 1](x)
            x = self.implicit_sins_rev[-1](x)
            h = self.implicit_linears_rev[-1](x)
        else:
            h = self.implicit_filter_rev(z)
        if self.modulate:
            h = self.modulation(t, h)
        if self.normalized:
            h_norm = mx.sum(mx.abs(h), axis=-1, keepdims=True)
            h = mx.divide(h, mx.add(h_norm, mx.array(1e-8, dtype=mx.float32)))
        return h

    def __call__(self, x, L, k_fwd=None, k_rev=None, bias=None, tracer=None):
        # Generate filters if not provided
        if k_fwd is None:
            axes = mx.array([0, 2, 1], dtype=mx.int32)
            k_fwd = mx.transpose(self.filter(L), axes)[0]
            if self.bidirectional and k_rev is None:
                k_rev = mx.transpose(self.filter_rev(L), axes)[0]

        if isinstance(k_fwd, tuple):
            k_fwd = k_fwd[0]

        if bias is None:
            D = self.bias.reshape(1, -1, 1)
        else:
            D = bias

        prof = get_profile()

        # Combine kernels according to profile: in time or frequency domain
        if k_rev is not None:
            space = getattr(prof, 'bidir_space', 'time')
            combine = getattr(prof, 'bidir_combine', 'sum')
            # Always combine in time-domain to avoid native FFT usage
            k_fwd_time = k_fwd
            idx = mx.arange(k_rev.shape[1], dtype=mx.int32)[::-1]
            k_rev_time = k_rev[:, idx]
            k_time = mx.add(k_fwd_time, k_rev_time)
            if combine == 'avg':
                k_time = mx.multiply(k_time, mx.array(0.5, dtype=mx.float32))
        else:
            k_time = k_fwd

        # JIT Metal FFT convolution (no native FFT)
        y = _hyena_fft_conv(x, k_time, D, L, fft_chunk_size=self.fft_chunk_size, gelu=False, tracer=tracer, prefix="hyena")
        return y


# Demo moved to tests to keep compute module scalar-clean.
