#!/usr/bin/env python
"""
FlashMM Sequence Mixing (MLX).

This module keeps the original FlashMM API surface but delegates to the shared
MLX kernels:

* Metal FFT convolution and depthwise 3-tap kernels live in
  :mod:`mlx_ops.kernels`.
* Hyena filter construction uses the same metal-accelerated helpers that power
  the Monarch mixer.
* The public classes (`FastFilter`, `FlashMMSequenceMixing`) remain here so
  higher-level code can import them exactly as before, yet the implementations
  call MLX primitives exclusively.

Design choices:
    - Inputs/weights stay as MLX arrays end-to-end—no NumPy/Torch crossing.
    - FFT buffer sizing follows the Torch reference (N=2*L) to preserve parity.
    - Long docstring here to explain the architecture; the heavy work happens in
      the imported kernels, keeping this file focused on intent rather than
      hardware details.
"""

# Copyright (c) 2023, Dan Fu and Simran Arora.
# Converted to MLX with Metal kernels

import mlx.core as mx
import mlx.nn as nn

# Absolute imports work from bert/src directory
from mlx_ops.kernels.metal_fft_conv import MetalFFTConv
from mlx_ops.kernels.metal_flash_mm import hyena_filter_fwd, exp_mod_in_place_fwd
from mlx_ops.kernels.metal_kernels import depthwise_conv_3tap


def pos_emb_init(seq_len, emb_dim):
    """Initialize positional embeddings."""
    one = mx.array(1.0, dtype=mx.float32)
    zero = mx.array(0.0, dtype=mx.float32)
    seq_len_mx = mx.array(seq_len, dtype=mx.float32)

    t = mx.linspace(zero, one, seq_len)[None, :, None]  # (1, L, 1)

    if emb_dim > 1:
        emb_dim_mx = mx.array(emb_dim, dtype=mx.int32)
        one_mx = mx.array(1, dtype=mx.int32)
        two_mx = mx.array(2, dtype=mx.int32)

        # bands = (emb_dim - 1) // 2
        bands_numerator = mx.subtract(emb_dim_mx, one_mx)
        bands, _ = mx.divmod(bands_numerator, two_mx)
        bands = bands.item()  # Boundary: extract for mx.linspace
    else:
        bands = 0  # Python int is acceptable for loop index / metadata

    # Compute proper linspace for rescaled t
    seq_len_minus_one = mx.subtract(seq_len_mx, one)
    t_rescaled = mx.linspace(zero, seq_len_minus_one, seq_len)[None, :, None]

    # w = 2 * pi * t_rescaled / seq_len
    two_pi = mx.array(6.283185307179586, dtype=mx.float32)
    w = mx.divide(mx.multiply(two_pi, t_rescaled), seq_len_mx)  # (1, L, 1)

    if bands > 0:
        # f = linspace(1e-4, bands - 1, bands)
        min_f = mx.array(1e-4, dtype=mx.float32)
        bands_mx = mx.array(bands, dtype=mx.float32)
        max_f = mx.subtract(bands_mx, one)
        f = mx.linspace(min_f, max_f, bands)[None, None]  # (1, 1, bands)

        # z = exp(-1j * f * w)
        # exp(-1j * fw) = cos(fw) - 1j * sin(fw)
        minus_one_mx = mx.array(-1.0, dtype=mx.float32)
        fw = mx.multiply(f, w)  # (1, L, bands)

        neg_fw = mx.multiply(minus_one_mx, fw)
        z_real = mx.cos(fw)
        z_imag = mx.sin(neg_fw)

        # Concatenate: [t, z.real, z.imag]
        z = mx.concatenate([t, z_real, z_imag], axis=-1)
    else:
        z = t

    return z


class FastFilter(nn.Module):
    """Fast Hyena filter using Metal kernels."""

    def __init__(
        self,
        d_model,
        channels,
        bidirectional=True,
        order=16,
        seq_len=128,
        lr=None,  # Ignored in MLX (no per-parameter LR)
        lr_pos_emb=None,  # Ignored
        w=1,
        wd=0,  # Weight decay
        emb_dim=5,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        if self.bidirectional:
            # Use mx.add instead of *= operator
            channels_mx = mx.array(channels, dtype=mx.int32)
            two_mx = mx.array(2, dtype=mx.int32)
            channels = mx.multiply(channels_mx, two_mx).item()  # Boundary
        self.channels = channels

        # Create positional embeddings
        z = pos_emb_init(seq_len, emb_dim)
        # Repeat for channels: (channels, seqlen, emb_dim)
        z = mx.tile(z, (self.channels, 1, 1))
        self.z = z

        # Sin frequencies
        w_mx = mx.array(w, dtype=mx.float32)
        self.sin_freq = mx.multiply(w_mx, mx.ones((self.channels, order), dtype=mx.float32))

        # Create parameters for eo_mat, eo_bias
        eo_linears = [nn.Linear(emb_dim, order) for _ in range(self.channels)]
        # Stack weights: (channels, order, emb_dim) -> transpose to (channels, emb_dim, order)
        eo_mat = mx.stack([l.weight for l in eo_linears], axis=0)
        eo_mat = mx.transpose(eo_mat, (0, 2, 1))  # (channels, emb_dim, order)
        self.eo_mat = eo_mat
        self.eo_bias = mx.stack([l.bias for l in eo_linears], axis=0)

        # Create parameters for oo1_mat, oo1_bias
        oo1_linears = [nn.Linear(order, order) for _ in range(self.channels)]
        oo1_mat = mx.stack([l.weight for l in oo1_linears], axis=0)
        oo1_mat = mx.transpose(oo1_mat, (0, 2, 1))  # (channels, order, order)
        self.oo1_mat = oo1_mat
        self.oo1_bias = mx.stack([l.bias for l in oo1_linears], axis=0)

        # Create parameters for oo2_mat, oo2_bias
        oo2_linears = [nn.Linear(order, order) for _ in range(self.channels)]
        oo2_mat = mx.stack([l.weight for l in oo2_linears], axis=0)
        oo2_mat = mx.transpose(oo2_mat, (0, 2, 1))  # (channels, order, order)
        self.oo2_mat = oo2_mat
        self.oo2_bias = mx.stack([l.bias for l in oo2_linears], axis=0)

        # Create parameters for oh_mat (order -> d_model projection)
        oh_linears = [nn.Linear(order, d_model, bias=False) for _ in range(self.channels)]
        oh_mat = mx.stack([l.weight for l in oh_linears], axis=0)
        oh_mat = mx.transpose(oh_mat, (0, 2, 1))  # (channels, order, d_model)
        self.oh_mat = oh_mat

        # Create reverse parameter
        if self.bidirectional:
            channels_half = self.channels // 2
            # Build list then convert to tensor
            reverse_list = []
            for _ in range(channels_half):
                reverse_list.extend([0, 1])
            reverse = mx.array(reverse_list, dtype=mx.int32)
        else:
            reverse = mx.zeros((self.channels,), dtype=mx.int32)
        self.reverse = reverse

        # Exponential modulation parameters
        target = mx.array(1e-2, dtype=mx.float32)
        fast_decay_pct = mx.array(0.3, dtype=mx.float32)
        slow_decay_pct = mx.array(1.5, dtype=mx.float32)

        self.min_decay = mx.divide(mx.log(target), slow_decay_pct)
        self.max_decay = mx.divide(mx.log(target), fast_decay_pct)
        self.shift = mx.array(0.0, dtype=mx.float32)

    def __call__(self):
        """Generate filter coefficients using Metal kernel."""
        # Use Metal kernel for hyena filter
        k = hyena_filter_fwd(
            self.z,
            self.sin_freq,
            self.eo_mat,
            self.eo_bias,
            self.oo1_mat,
            self.oo1_bias,
            self.oo2_mat,
            self.oo2_bias,
            self.reverse
        )  # (C, L, ORDER)

        # Project to d_model: k @ oh_mat
        # k: (C, L, ORDER), oh_mat: (C, ORDER, d_model)
        # Want: (C, L, d_model)
        # Use batched matmul
        k = mx.matmul(k, self.oh_mat)  # (C, L, d_model)

        # Apply exponential modulation
        k = exp_mod_in_place_fwd(
            k,
            self.reverse,
            self.min_decay.item(),  # Boundary: scalar parameters
            self.max_decay.item(),
            self.shift.item()
        )

        return k


class FlashMMSequenceMixing(nn.Module):
    """Flash Monarch Mixer Sequence Mixing using Metal kernels."""

    def __init__(
        self,
        d_model,
        l_max=128,
        hyena_kernel_lr=None,
        bidirectional=False,
        hyena_lr_pos_emb=None,
        hyena_w=10,
        hyena_w_mod=1,  # Ignored
        hyena_wd=None,
        hyena_emb_dim=5,
        hyena_filter_order=128,
        residual_long_conv=False,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.l_max = l_max
        self.kernel_lr = hyena_kernel_lr
        self.channels = mx.array(1, dtype=mx.int32)
        self.bidirectional = bidirectional
        self.residual_long_conv = residual_long_conv

        print('Using Flash MM Sequence Mixing (Metal kernels)')
        print('-- Bidirectional:', self.bidirectional)
        print("-- Using Long Conv Residual:", self.residual_long_conv)
        print('-- Hyena w:', hyena_w)
        print('-- Hyena filter order:', hyena_filter_order)

        channels = 1
        if self.residual_long_conv:
            channels_mx = mx.array(channels, dtype=mx.int32)
            two_mx = mx.array(2, dtype=mx.int32)
            channels = mx.multiply(channels_mx, two_mx).item()  # Boundary

        self.fast_filter = FastFilter(
            self.d_model,
            channels=channels,
            bidirectional=self.bidirectional,
            order=hyena_filter_order,
            seq_len=self.l_max,
            lr=hyena_kernel_lr,
            lr_pos_emb=hyena_lr_pos_emb,
            w=hyena_w,
            wd=hyena_wd if hyena_wd is not None else 0.0,
            emb_dim=hyena_emb_dim,
        )

        # Setup projections
        three_d = d_model * 3  # List repetition allowed for metadata
        self.in_linear = nn.Linear(d_model, three_d)
        self.out_linear = nn.Linear(d_model, d_model)

        # Short convolution filters (depthwise 3-tap)
        # Initialize with random normal
        self.x1_s = mx.random.normal((d_model, 3), dtype=mx.float32)
        self.x2_s = mx.random.normal((d_model, 3), dtype=mx.float32)
        self.v_s = mx.random.normal((d_model, 3), dtype=mx.float32)
        self.x1_s_bias = mx.zeros((d_model,), dtype=mx.float32)
        self.x2_s_bias = mx.zeros((d_model,), dtype=mx.float32)
        self.v_s_bias = mx.zeros((d_model,), dtype=mx.float32)

        # Per-channel bias
        self.bias = mx.zeros((self.d_model,), dtype=mx.float32)
        if self.residual_long_conv:
            self.residual_bias = mx.zeros((self.d_model,), dtype=mx.float32)
        else:
            self.residual_bias = None

        # FFT convolution kernel
        self.fft_conv = MetalFFTConv()

    def __call__(self, u, **kwargs):
        """
        Forward pass using FFT convolution.

        Args:
            u: (B, L, H) input tensor

        Returns:
            y: (B, L, H) output tensor
            None: placeholder for compatibility
        """
        B, L, H = u.shape

        # 1. Input projection: u -> [x1, x2, v]
        x1x2v = self.in_linear(u)  # (B, L, 3*H)

        # Split into three parts (H is Python int from shape, acceptable for slicing)
        H2 = H + H  # Avoid * operator
        x1 = x1x2v[:, :, :H]  # (B, L, H)
        x2 = x1x2v[:, :, H:H2]  # (B, L, H)
        v = x1x2v[:, :, H2:]  # (B, L, H)

        # Transpose to (B, H, L) for depthwise conv
        x1 = mx.transpose(x1, (0, 2, 1))  # (B, H, L)
        x2 = mx.transpose(x2, (0, 2, 1))  # (B, H, L)
        v = mx.transpose(v, (0, 2, 1))  # (B, H, L)

        # 2. Short depthwise convolutions (3-tap)
        x1 = depthwise_conv_3tap(x1, self.x1_s, self.x1_s_bias)  # (B, H, L)
        x2 = depthwise_conv_3tap(x2, self.x2_s, self.x2_s_bias)  # (B, H, L)
        v = depthwise_conv_3tap(v, self.v_s, self.v_s_bias)  # (B, H, L)

        # 3. First gating: v = v * x1
        v = mx.multiply(v, x1)  # (B, H, L)

        # 4. Generate filter kernels
        all_kernels = self.fast_filter()  # (C, L_filt, H)
        C, L_filt, _ = all_kernels.shape

        # Split kernels if using residual
        if self.residual_long_conv:
            C_half, _ = mx.divmod(mx.array(C, dtype=mx.int32), mx.array(2, dtype=mx.int32))
            C_half_int = C_half.item()  # Boundary
            k = all_kernels[:C_half_int]  # (C/2, L_filt, H)
            k_resid = all_kernels[C_half_int:]  # (C/2, L_filt, H)

            # Take first filter and reshape for FFT conv
            # MetalFFTConv expects k: (H, L_filt)
            k = k[0]  # (L_filt, H)
            k = mx.transpose(k, (1, 0))  # (H, L_filt)

            k_resid = k_resid[0]  # (L_filt, H)
            k_resid = mx.transpose(k_resid, (1, 0))  # (H, L_filt)
        else:
            # Take first filter
            k = all_kernels[0]  # (L_filt, H)
            k = mx.transpose(k, (1, 0))  # (H, L_filt)
            k_resid = None

        # Pad/truncate k to match L
        if L_filt < L:
            # Pad k
            pad_amount = L - L_filt
            zero = mx.array(0, dtype=mx.int32)
            pad_amount_mx = mx.array(pad_amount, dtype=mx.int32)
            pads = [(zero, zero), (zero, pad_amount_mx)]
            k = mx.pad(k, pads)
            if k_resid is not None:
                k_resid = mx.pad(k_resid, pads)
        elif L_filt > L:
            # Truncate k
            k = k[:, :L]
            if k_resid is not None:
                k_resid = k_resid[:, :L]

        # 5. FFT convolution: y = fft_conv(v, k, bias)
        y = self.fft_conv(v, k, self.bias)  # (B, H, L)

        # 6. Add residual if needed
        if self.residual_long_conv and k_resid is not None:
            u_transposed = mx.transpose(u, (0, 2, 1))  # (B, H, L)
            y_resid = self.fft_conv(u_transposed, k_resid, self.residual_bias)  # (B, H, L)
            y = mx.add(y, y_resid)

        # 7. Second gating: y = y * x2
        y = mx.multiply(y, x2)  # (B, H, L)

        # 8. Transpose back to (B, L, H) and output projection
        y = mx.transpose(y, (0, 2, 1))  # (B, L, H)
        y = self.out_linear(y)  # (B, L, H)

        return y, None
