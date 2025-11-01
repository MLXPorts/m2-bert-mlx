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

import math
import os
import importlib.util
import mlx.core as mx
import mlx.nn as nn

# Robustly resolve HyperProfile loader whether imported as a package or by path
def _resolve_get_profile():
    try:
        from .hyperprofiles_mlx import get_profile  # type: ignore
        return get_profile
    except Exception:
        pass
    try:
        from mm_mlx.hyperprofiles_mlx import get_profile  # type: ignore
        return get_profile
    except Exception:
        pass
    # Load by file path relative to this file
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        hp_path = os.path.join(here, 'hyperprofiles_mlx.py')
        spec = importlib.util.spec_from_file_location('mlx_hp_local', hp_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return getattr(mod, 'get_profile')
    except Exception:
        pass
    # Final fallback: sane defaults
    def _fallback():
        class _P:
            gelu_mode = "erf"
            fft_norm = "forward"
            bidir_combine = "sum"  # default to Torch-like
        return _P()
    return _fallback

get_profile = _resolve_get_profile()


class Sin(nn.Module):
    """Learnable sinusoidal activation."""

    def __init__(self, dim, w=10, w_mod=1, train_freq=True):
        super().__init__()
        self.w_mod = w_mod
        self.train_freq = train_freq
        self.freq = mx.ones((1, dim), dtype=mx.float32) * mx.array(w, dtype=mx.float32)

    def __call__(self, x):
        return mx.sin(self.w_mod * self.freq * x)


class PositionalEmbedding(nn.Module):
    """Complex exponential positional embeddings for Hyena filters."""

    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.lr_pos_emb = lr_pos_emb

        t = mx.linspace(0.0, 1.0, seq_len).astype(mx.float32).reshape(1, seq_len, 1)

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
            t_rescaled = mx.arange(seq_len, dtype=mx.float32).reshape(1, seq_len, 1)
            two = mx.array(2.0, dtype=mx.float32)
            pi = mx.array(3.141592653589793, dtype=mx.float32)
            w = (two * pi) * t_rescaled / mx.array(seq_len, dtype=mx.float32)
            ar = mx.arange(bands, dtype=mx.float32).reshape(1, 1, bands)
            maxf = mx.array(bands - 1, dtype=mx.float32)
            f = mx.array(1e-4, dtype=mx.float32) + (maxf - mx.array(1e-4, dtype=mx.float32)) * (ar / mx.maximum(maxf, mx.array(1.0, dtype=mx.float32)))
            phase = -f * w
            z_real = mx.cos(phase)
            z_imag = mx.sin(phase)
            z = mx.concatenate([t, z_real, z_imag], axis=-1)
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
        self.shift = shift
        max_decay = mx.log(mx.array(target, dtype=mx.float32)) / mx.array(fast_decay_pct, dtype=mx.float32)
        min_decay = mx.log(mx.array(target, dtype=mx.float32)) / mx.array(slow_decay_pct, dtype=mx.float32)
        lin = mx.linspace(0.0, 1.0, d_model).astype(mx.float32).reshape(1, 1, d_model)
        deltas = min_decay + (max_decay - min_decay) * lin
        self.deltas = deltas
        self.modulation_lr = modulation_lr

    def __call__(self, t, x):
        decay = mx.exp(-t * mx.abs(self.deltas))
        return x * (decay + self.shift)


def _hyena_fft_conv(u, k_f, D, seqlen, fft_chunk_size=None, gelu=False, dropout_mask=None, tracer=None, prefix: str = "hyena"):
    """FFT-based 1D convolution over channels using MLX streams for overlap.

    u: (batch, d_model, seqlen)
    k_f: (d_model, fft_bins)
    D: (1, d_model, 1)
    """
    batch, d_model, _ = u.shape
    fft_size = 2 * seqlen

    NUM_STREAMS = 4
    streams = [mx.new_stream(mx.default_device()) for _ in range(NUM_STREAMS)]

    CHANNEL_BATCH_SIZE = 64
    bands = [(s, min(s + CHANNEL_BATCH_SIZE, d_model)) for s in range(0, d_model, CHANNEL_BATCH_SIZE)]

    outputs = [None] * len(bands)
    for idx, (ch_start, ch_end) in enumerate(bands):
        st = streams[idx % NUM_STREAMS]
        with mx.stream(st):
            u_batch = u[:, ch_start:ch_end, :]
            k_f_batch = k_f[ch_start:ch_end, :]
            u_freq = mx.fft.rfft(u_batch, n=fft_size, axis=-1)
            if tracer is not None:
                try:
                    from src.utils.tracer import Tracer  # noqa: F401
                    tracer.log(f"{prefix}.u_freq[{ch_start}:{ch_end}]", u_freq, framework='mlx')
                except Exception:
                    pass
            y_freq = mx.multiply(u_freq, k_f_batch[None, :, :])
            if tracer is not None:
                try:
                    tracer.log(f"{prefix}.y_freq[{ch_start}:{ch_end}]", y_freq, framework='mlx')
                except Exception:
                    pass
            y_batch = mx.fft.irfft(y_freq, n=fft_size, axis=-1)
            y_batch = y_batch[..., :seqlen]
            outputs[idx] = y_batch

    mx.synchronize()
    y = mx.concatenate(outputs, axis=1)

    y = mx.add(y, mx.multiply(u, D))
    if tracer is not None:
        try:
            tracer.log(f"{prefix}.bias_add", y, framework='mlx')
        except Exception:
            pass
    if gelu:
        prof = get_profile()
        if prof.gelu_mode == "tanh":
            c = mx.sqrt(mx.divide(mx.array(2.0, dtype=mx.float32), mx.array(3.141592653589793, dtype=mx.float32)))
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

        assert emb_dim % 2 != 0 and emb_dim >= 3, "emb_dim must be odd and >= 3"
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
            for i in range(len(self.implicit_sins) - 1):
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
            h = h / (h_norm + mx.array(1e-8, dtype=mx.float32))
        return h

    def filter_rev(self, L):
        z, t = self.pos_emb(L)
        if hasattr(self, 'implicit_linears_rev'):
            x = self.implicit_linears_rev[0](z)
            for i in range(len(self.implicit_sins_rev) - 1):
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
            h = h / (h_norm + mx.array(1e-8, dtype=mx.float32))
        return h

    def __call__(self, x, L, k_fwd=None, k_rev=None, bias=None, tracer=None):
        # Generate filters if not provided
        if k_fwd is None:
            k_fwd = self.filter(L).transpose(0, 2, 1)[0]
            if self.bidirectional and k_rev is None:
                k_rev = self.filter_rev(L).transpose(0, 2, 1)[0]

        if isinstance(k_fwd, tuple):
            k_fwd = k_fwd[0]

        if bias is None:
            D = self.bias.reshape(1, -1, 1)
        else:
            D = bias

        # Prepare kernels in frequency domain (match Torch: 2*L)
        fft_size = 2 * L

        prof = None
        try:
            prof = get_profile()
        except Exception:
            prof = None

        # Combine kernels according to profile: in time or frequency domain
        if k_rev is not None:
            space = getattr(prof, 'bidir_space', 'time') if prof is not None else 'time'
            combine = getattr(prof, 'bidir_combine', 'sum') if prof is not None else 'sum'
            if space == 'freq':
                k_f = mx.fft.rfft(k_fwd, n=fft_size, axis=-1)
                k_rev_f = mx.fft.rfft(k_rev, n=fft_size, axis=-1)
                k_f = mx.add(k_f, k_rev_f)
                if combine == 'avg':
                    k_f = mx.multiply(k_f, mx.array(0.5, dtype=mx.float32))
                if tracer is not None:
                    try:
                        tracer.log("hyena.k_f", k_f, framework='mlx')
                    except Exception:
                        pass
            else:
                k_fwd_time = mx.pad(k_fwd, [(0, 0), (0, L)])
                idx = mx.arange(k_rev.shape[1], dtype=mx.int32)[::-1]
                k_rev_time = k_rev[:, idx]
                k_rev_time = mx.pad(k_rev_time, [(0, 0), (L, 0)])
                k_time = mx.add(k_fwd_time, k_rev_time)
                if combine == 'avg':
                    k_time = mx.multiply(k_time, mx.array(0.5, dtype=mx.float32))
                k_f = mx.fft.rfft(k_time, n=fft_size, axis=-1)
                if tracer is not None:
                    try:
                        tracer.log("hyena.k_time", k_time, framework='mlx')
                        tracer.log("hyena.k_f", k_f, framework='mlx')
                    except Exception:
                        pass
        else:
            k_f = mx.fft.rfft(k_fwd, n=fft_size, axis=-1)
            if tracer is not None:
                try:
                    tracer.log("hyena.k_f", k_f, framework='mlx')
                except Exception:
                    pass

        # Do not divide by n here; MLX irfft applies 1/n so net matches Torch's choice.

        y = _hyena_fft_conv(x, k_f, D, L, fft_chunk_size=self.fft_chunk_size, gelu=False, tracer=tracer, prefix="hyena")
        return y


def _demo():
    batch_size = 2
    d_model = 768
    seq_len = 128
    hyena = HyenaFilter(d_model=d_model, emb_dim=5, order=64, seq_len=seq_len, bidirectional=True)
    x = mx.random.normal((batch_size, d_model, seq_len))
    y = hyena(x, seq_len)
    print('Hyena demo — input:', x.shape, 'output:', y.shape)


if __name__ == '__main__':
    _demo()
