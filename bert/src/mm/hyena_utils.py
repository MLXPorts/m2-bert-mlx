# Copyright (c) 2023, Dan Fu and Simran Arora.
# Adapted from https://github.com/HazyResearch/safari/blob/main/src/models/sequence/hyena.py
# Converted to MLX

import mlx.core as mx
import mlx.nn as nn

from mlx_ops.einops import rearrange

from src.utils.train import OptimModule


def fftconv_ref(u_variable, k, D_variable, dropout_mask, gelu=True, k_rev=None, flashfft=None):
    # u.shape:   B H L
    seqlen = u_variable.shape[-1]

    if flashfft is not None:
        y = flashfft(u_variable.astype(mx.bfloat16), k)
    else:
        fft_size_mx = mx.multiply(mx.array(2, dtype=mx.int32), mx.array(seqlen, dtype=mx.int32))
        fft_size_f = mx.array(fft_size_mx, dtype=mx.float32)

        k_f_raw = mx.fft.rfft(k, n=fft_size_mx)
        k_f = mx.divide(k_f_raw, fft_size_f)

        if k_rev is not None:
            k_rev_f_raw = mx.fft.rfft(k_rev, n=fft_size_mx)
            k_rev_f = mx.divide(k_rev_f_raw, fft_size_f)
            k_f = mx.add(k_f, mx.conjugate(k_rev_f))

        u_f = mx.fft.rfft(u_variable.astype(k.dtype), n=fft_size_mx)

        if len(u_variable.shape) > 3:
            k_f = mx.expand_dims(k_f, axis=1)

        y_full = mx.fft.irfft(mx.multiply(u_f, k_f), n=fft_size_mx)
        y = y_full[..., :seqlen]

    out = mx.add(y, mx.multiply(u_variable, D_variable))

    if gelu:
        # MLX doesn't have F.gelu, use approximate gelu
        # gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = mx.array(0.7978845608, dtype=out.dtype)  # sqrt(2/pi)
        coeff = mx.array(0.044715, dtype=out.dtype)
        three = mx.array(3.0, dtype=out.dtype)
        one = mx.array(1.0, dtype=out.dtype)
        half = mx.array(0.5, dtype=out.dtype)

        x_cubed = mx.power(out, three)
        inner = mx.add(out, mx.multiply(coeff, x_cubed))
        tanh_arg = mx.multiply(sqrt_2_over_pi, inner)
        tanh_val = mx.tanh(tanh_arg)
        out = mx.multiply(mx.multiply(half, out), mx.add(one, tanh_val))

    if dropout_mask is not None:
        return mx.multiply(out, rearrange(dropout_mask, "b H -> b H 1")).astype(u_variable.dtype)
    else:
        return out.astype(u_variable.dtype)


def mul_sum(q, y):
    return mx.sum(mx.multiply(q, y), axis=1)


class Sin(nn.Module):
    def __init__(self, dim, w=10, w_mod=1, train_freq=True):
        super().__init__()

        init_tensor = mx.ones((1, dim), dtype=mx.float32)
        w_mx = mx.array(w, dtype=mx.float32)

        if train_freq:
            self.freq = mx.multiply(w_mx, init_tensor)
        else:
            self.freq = mx.multiply(w_mx, mx.ones((1, dim), dtype=mx.float32))

        self.w_mod = mx.array(w_mod, dtype=mx.float32)

    def __call__(self, x):
        return mx.sin(mx.multiply(self.w_mod, mx.multiply(self.freq, x)))


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len

        # The time embedding fed to the filters is normalized so that t_f = 1
        zero = mx.array(0.0, dtype=mx.float32)
        one = mx.array(1.0, dtype=mx.float32)
        seq_len_mx = mx.array(seq_len, dtype=mx.float32)

        t_1d = mx.linspace(zero, one, seq_len, dtype=mx.float32)
        t = mx.reshape(t_1d, (1, seq_len, 1))

        if emb_dim > 1:
            bands_mx = mx.array(emb_dim, dtype=mx.int32)
            one_int = mx.array(1, dtype=mx.int32)
            bands_minus_1 = mx.subtract(bands_mx, one_int)
            two = mx.array(2, dtype=mx.int32)
            bands_result, _ = mx.divmod(bands_minus_1, two)
            bands = bands_result  # Keep as MLX scalar

        # To compute the right embeddings we use the "proper" linspace
        seq_len_minus_1 = mx.subtract(seq_len_mx, one)
        t_rescaled_1d = mx.linspace(zero, seq_len_minus_1, seq_len, dtype=mx.float32)
        t_rescaled = mx.reshape(t_rescaled_1d, (1, seq_len, 1))

        # w = 2 * pi * t_rescaled / seq_len
        two_f = mx.array(2.0, dtype=mx.float32)
        pi = mx.array(3.14159265358979323846, dtype=mx.float32)
        two_pi = mx.multiply(two_f, pi)
        numerator = mx.multiply(two_pi, t_rescaled)
        w = mx.divide(numerator, seq_len_mx)

        # f = linspace(1e-4, bands - 1, bands)
        bands_f = mx.array(bands, dtype=mx.float32)
        bands_minus_1_f = mx.subtract(bands_f, one)
        f_1d = mx.linspace(mx.array(1e-4, dtype=mx.float32), bands_minus_1_f, bands, dtype=mx.float32)
        f = mx.reshape(f_1d, (1, 1, bands))

        # z = exp(-1j * f * w)
        # In MLX, complex numbers: real + 1j * imag
        minus_one = mx.array(-1.0, dtype=mx.float32)
        fw = mx.multiply(f, w)

        # exp(-1j * fw) = cos(-fw) + 1j * sin(-fw) = cos(fw) - 1j * sin(fw)
        # Since we're doing -1j, the imaginary part is -sin(fw)
        # But actually, we need the real and imaginary parts separately
        # z.real = cos(fw), z.imag = -sin(fw)
        neg_fw = mx.multiply(minus_one, fw)
        z_real = mx.cos(fw)
        z_imag = mx.sin(neg_fw)

        # Concatenate: [t, z.real, z.imag]
        z = mx.concatenate([t, z_real, z_imag], axis=-1)

        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def __call__(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    def __init__(
        self,
        d_model,
        fast_decay_pct=0.3,
        slow_decay_pct=1.5,
        target=1e-2,
        modulation_lr=0.0,
        shift: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        shift_mx = mx.array(shift, dtype=mx.float32)
        self.shift = shift_mx

        target_mx = mx.array(target, dtype=mx.float32)
        fast_decay_pct_mx = mx.array(fast_decay_pct, dtype=mx.float32)
        slow_decay_pct_mx = mx.array(slow_decay_pct, dtype=mx.float32)

        log_target = mx.log(target_mx)
        max_decay = mx.divide(log_target, fast_decay_pct_mx)
        min_decay = mx.divide(log_target, slow_decay_pct_mx)

        zero = mx.array(0.0, dtype=mx.float32)
        one = mx.array(1.0, dtype=mx.float32)
        lin = mx.linspace(zero, one, d_model, dtype=mx.float32)

        diff = mx.subtract(max_decay, min_decay)
        product = mx.multiply(diff, lin)
        deltas_1d = mx.add(min_decay, product)
        deltas = mx.reshape(deltas_1d, (1, 1, d_model))

        self.register("deltas", deltas, lr=modulation_lr)

    def __call__(self, t, x):
        minus_one = mx.array(-1.0, dtype=self.deltas.dtype)
        t_times_deltas = mx.multiply(t, mx.abs(self.deltas))
        neg_t_times_deltas = mx.multiply(minus_one, t_times_deltas)
        decay = mx.exp(neg_t_times_deltas)
        decay_plus_shift = mx.add(decay, self.shift)
        x = mx.multiply(x, decay_plus_shift)
        return x


class HyenaFilter(OptimModule):
    def __init__(
        self,
        d_model,
        emb_dim=3,  # dim of input to MLP, augments with positional encoding
        order=16,  # width of the implicit MLP
        seq_len=1024,
        lr=1e-3,
        lr_pos_emb=1e-5,
        dropout=0.0,
        w=1,  # frequency of periodic activations
        w_mod=1, # non-learnable modification of w
        wd=0,  # weight decay of kernel parameters
        bias=True,
        num_inner_mlps=2,
        linear_mixer=False,
        modulate: bool = True,
        normalized=False,
        bidirectional=False,
        **kwargs,
    ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP

        Note:
            filter_dropout is not implemented
        """
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.modulate = modulate
        self.use_bias = bias
        self.bidirectional = bidirectional

        self.bias = mx.random.normal((d_model,), dtype=mx.float32)
        self.dropout = nn.Dropout(p=dropout)

        act = Sin(dim=order, w=w, w_mod=w_mod)
        # Check emb_dim using MLX ops
        emb_dim_mx = mx.array(emb_dim, dtype=mx.int32)
        two_mx = mx.array(2, dtype=mx.int32)
        three_mx = mx.array(3, dtype=mx.int32)
        _, remainder = mx.divmod(emb_dim_mx, two_mx)
        is_odd = mx.not_equal(remainder, mx.array(0, dtype=mx.int32))
        is_ge_3 = mx.greater_equal(emb_dim_mx, three_mx)
        assert bool(is_odd) and bool(is_ge_3), "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        # uses a variable number of inner linear layers
        if linear_mixer is False:
            layers = []
            layers.append(nn.Linear(emb_dim, order))
            layers.append(act)

            for i in range(num_inner_mlps):
                layers.append(nn.Linear(order, order))
                layers.append(act)

            layers.append(nn.Linear(order, d_model, bias=False))
            self.implicit_filter = nn.Sequential(*layers)
        else:
            self.implicit_filter = nn.Sequential(nn.Linear(emb_dim, d_model, bias=False))

        if self.bidirectional:
            layers_rev = []
            layers_rev.append(nn.Linear(emb_dim, order))
            layers_rev.append(act)

            for i in range(num_inner_mlps):
                layers_rev.append(nn.Linear(order, order))
                layers_rev.append(act)

            layers_rev.append(nn.Linear(order, d_model, bias=False))
            self.implicit_filter_rev = nn.Sequential(*layers_rev)

        self.modulation = ExponentialModulation(d_model, **kwargs)

        self.normalized = normalized

        # MLX doesn't have the same optimizer attachment mechanism as PyTorch
        # This would need to be handled differently in MLX training loop

        self.flashfft = None

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        if self.modulate:
            h = self.modulation(t, h)
        if self.normalized:
            # h = h / norm(h, dim=-1, p=1, keepdim=True)
            norm_h = mx.sum(mx.abs(h), axis=-1, keepdims=True)
            h = mx.divide(h, norm_h)
        return h

    def filter_rev(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter_rev(z)
        if self.modulate:
            h = self.modulation(t, h)
        if self.normalized:
            norm_h = mx.sum(mx.abs(h), axis=-1, keepdims=True)
            h = mx.divide(h, norm_h)
        return h

    def __call__(self, x, L, k_fwd=None, k_rev=None, bias=None, *args, **kwargs):
        if k_fwd is None:
            k_fwd = self.filter(L)
            if self.bidirectional and k_rev is None:
                k_rev = self.filter_rev(L)

        # Ensure compatibility with filters that return a tuple
        k_fwd = k_fwd[0] if type(k_fwd) is tuple else k_fwd
        if bias is None:
            bias = self.bias

        # bias = bias if self.use_bias else 0 * bias
        if self.use_bias:
            bias_to_use = bias
        else:
            zero = mx.array(0.0, dtype=bias.dtype)
            bias_to_use = mx.multiply(zero, bias)

        if self.bidirectional:
            k_rev = k_rev[0] if type(k_rev) is tuple else k_rev
            # k = pad(k_fwd, (0, L)) + pad(k_rev.flip(-1), (L, 0))
            zero_mx = mx.array(0, dtype=mx.int32)
            L_mx = mx.array(L, dtype=mx.int32)
            ndim = len(k_fwd.shape)

            # Create padding list: [(0,0), (0,0), ..., (0, L)]
            # Build list by repetition (list ops are allowed for non-tensor data structures)
            pad_fwd = [(zero_mx, zero_mx)] * (ndim - 1) + [(zero_mx, L_mx)]
            k_fwd_padded = mx.pad(k_fwd, pad_fwd)

            k_rev_flipped = mx.flip(k_rev, axis=-1)
            # Create padding list: [(0,0), (0,0), ..., (L, 0)]
            pad_rev = [(zero_mx, zero_mx)] * (ndim - 1) + [(L_mx, zero_mx)]
            k_rev_padded = mx.pad(k_rev_flipped, pad_rev)

            k = mx.add(k_fwd_padded, k_rev_padded)
        else:
            k = k_fwd

        y = fftconv_ref(
            x,
            k,
            bias_to_use,
            dropout_mask=None,
            gelu=False,
            flashfft=self.flashfft,
        )

        # Apply dropout (respects training mode)
        y = self.dropout(y)

        return y.astype(x.dtype)
