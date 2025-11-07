#!/usr/bin/env python
import os
import sys
import numpy as np
import mlx.core as mx

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(repo_root, 'src'))

import torch  # type: ignore
from mm_mlx.metal_fft_conv import (
    _get_rfft_rows,
    _get_col_fft_twiddle,
    _get_mul_pointwise,
    _get_col_ifft_twiddle,
    _get_ifft_rows_write,
)


def six_step_numpy(x, k, N, n1, n2):
    # Pad to N
    x_pad = np.zeros(N, dtype=np.complex64)
    x_pad[: x.shape[-1]] = x.astype(np.float32)
    k_pad = np.zeros(N, dtype=np.complex64)
    k_pad[: k.shape[-1]] = k.astype(np.float32)

    X = x_pad.reshape(n2, n1)
    K = k_pad.reshape(n2, n1)

    # Row FFTs length n1
    X = np.fft.fft(X, n=n1, axis=1)
    K = np.fft.fft(K, n=n1, axis=1)

    # Twiddle
    r = np.arange(n2)[:, None]
    k1 = np.arange(n1)[None, :]
    W = np.exp(-2j * np.pi * (r * k1) / float(N)).astype(np.complex64)
    X *= W
    K *= W

    # Column FFTs length n2
    X = np.fft.fft(X, n=n2, axis=0)
    K = np.fft.fft(K, n=n2, axis=0)

    # Multiply
    Y = X * K

    # Keep a copy of spectrum for debug (flatten k = k1*n2 + k2)
    Yspec = Y.copy()
    # Flatten with k = k2*n1 + k1 mapping
    Yspec_flat = Yspec.reshape(-1)

    # IFFT columns + conj twiddle
    Y = np.fft.ifft(Y, n=n2, axis=0)
    Y *= np.conj(W)

    # IFFT rows
    y = np.fft.ifft(Y, n=n1, axis=1)
    y_time = y.reshape(N).real.astype(np.float32)
    return y_time


def main():
    L = 4096 // 2  # N=4096
    N = 4096
    n1 = 1024
    n2 = N // n1

    mx.random.seed(0)
    u = mx.random.normal((1, 1, L)).astype(mx.float32)
    k = mx.random.normal((1, L)).astype(mx.float32)
    D = mx.zeros((1,), dtype=mx.float32)

    # Build params
    params_rows = mx.array([1, 1, L, N, n1, n2], dtype=mx.uint32)
    params_cols = mx.array([1, 1, N, n1, n2], dtype=mx.uint32)

    # Buffers
    size_bins = N
    wr = mx.zeros((size_bins,), dtype=mx.float32)
    wi = mx.zeros((size_bins,), dtype=mx.float32)
    kr = mx.zeros((size_bins,), dtype=mx.float32)
    ki = mx.zeros((size_bins,), dtype=mx.float32)
    wr2 = mx.zeros((size_bins,), dtype=mx.float32)
    wi2 = mx.zeros((size_bins,), dtype=mx.float32)
    kr2 = mx.zeros((size_bins,), dtype=mx.float32)
    ki2 = mx.zeros((size_bins,), dtype=mx.float32)
    wr3 = mx.zeros((size_bins,), dtype=mx.float32)
    wi3 = mx.zeros((size_bins,), dtype=mx.float32)

    one = mx.array(1, dtype=mx.uint32)
    tpg_rows = mx.array(n1, dtype=mx.uint32)
    tpg_cols = mx.array(n2, dtype=mx.uint32)
    grid_rows = (mx.array(n1, dtype=mx.uint32), one, one)  # single gid since batch*ch=1
    grid_cols = (mx.array(n2, dtype=mx.uint32), one, one)

    rfft_rows = _get_rfft_rows()
    col_fft_twiddle = _get_col_fft_twiddle()
    mul_pointwise = _get_mul_pointwise()
    col_ifft_twiddle = _get_col_ifft_twiddle()
    ifft_rows_write = _get_ifft_rows_write()

    # Stage 1
    (wr, wi) = rfft_rows(
        inputs=[params_rows, u.reshape(-1)],
        output_shapes=[(size_bins,), (size_bins,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(mx.array(n1, dtype=mx.uint32), one, one),
        threadgroup=(tpg_rows, one, one),
    )
    (kr, ki) = rfft_rows(
        inputs=[params_rows, k.reshape(-1)],
        output_shapes=[(size_bins,), (size_bins,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(mx.array(n1, dtype=mx.uint32), one, one),
        threadgroup=(tpg_rows, one, one),
    )

    # Stage 2
    (wr2, wi2) = col_fft_twiddle(
        inputs=[params_cols, wr, wi],
        output_shapes=[(size_bins,), (size_bins,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(mx.array(n2, dtype=mx.uint32), one, one),
        threadgroup=(tpg_cols, one, one),
    )
    (kr2, ki2) = col_fft_twiddle(
        inputs=[params_cols, kr, ki],
        output_shapes=[(size_bins,), (size_bins,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(mx.array(n2, dtype=mx.uint32), one, one),
        threadgroup=(tpg_cols, one, one),
    )

    # Stage 3 multiply
    total = mx.array(size_bins, dtype=mx.uint32)
    params_mul = mx.array([size_bins], dtype=mx.uint32)
    tpg_mul = mx.array(256, dtype=mx.uint32)
    total_threads_mul = mx.multiply(
        mx.divide(mx.add(total, mx.subtract(tpg_mul, mx.array(1, dtype=mx.uint32))), tpg_mul), tpg_mul)
    (wr2, wi2) = mul_pointwise(
        inputs=[params_mul, wr2, wi2, kr2, ki2],
        output_shapes=[(size_bins,), (size_bins,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(total_threads_mul, mx.array(1, dtype=mx.uint32), mx.array(1, dtype=mx.uint32)),
        threadgroup=(tpg_mul, mx.array(1, dtype=mx.uint32), mx.array(1, dtype=mx.uint32)),
    )

    # Stage 4
    (wr3, wi3) = col_ifft_twiddle(
        inputs=[params_cols, wr2, wi2],
        output_shapes=[(size_bins,), (size_bins,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(mx.array(n2, dtype=mx.uint32), one, one),
        threadgroup=(tpg_cols, one, one),
    )

    # Stage 5
    (y_flat,) = ifft_rows_write(
        inputs=[params_rows, wr3, wi3, u.reshape(-1), D.reshape(-1)],
        output_shapes=[(L,)],
        output_dtypes=[mx.float32],
        grid=(mx.array(n1, dtype=mx.uint32), one, one),
        threadgroup=(tpg_rows, one, one),
    )
    mx.eval(wr, wi, wr2, wi2, wr3, wi3, y_flat)

    # Reference: six-step numpy and torch direct
    u_np = np.array(u).reshape(L)
    k_np = np.array(k).reshape(L)
    y_np_six = six_step_numpy(u_np, k_np, N, n1, n2)
    y_np_six = y_np_six[:L]

    u_t = torch.from_numpy(u_np)
    k_t = torch.from_numpy(k_np)
    k_f = torch.fft.rfft(k_t, n=N, dim=-1)
    u_f = torch.fft.rfft(u_t, n=N, dim=-1)
    y_f = u_f * k_f
    y_ref = torch.fft.irfft(y_f, n=N, dim=-1)[:L]

    # Compare
    y_mlx = np.array(y_flat)
    # Compare spectra too
    X_full = np.fft.fft(np.pad(u_np.astype(np.float32), (0, N - L))).astype(np.complex64)
    K_full = np.fft.fft(np.pad(k_np.astype(np.float32), (0, N - L))).astype(np.complex64)
    Y_full = X_full * K_full
    # Recreate six-step spectrum for comparison
    x_pad = np.zeros(N, dtype=np.complex64); x_pad[:L] = u_np.astype(np.float32)
    k_pad = np.zeros(N, dtype=np.complex64); k_pad[:L] = k_np.astype(np.float32)
    X = x_pad.reshape(n2, n1)
    K = k_pad.reshape(n2, n1)
    X = np.fft.fft(X, n=n1, axis=1)
    K = np.fft.fft(K, n=n1, axis=1)
    r = np.arange(n2)[:, None]; k1 = np.arange(n1)[None, :]
    W = np.exp(-2j * np.pi * (r * k1) / float(N)).astype(np.complex64)
    X *= W; K *= W
    X = np.fft.fft(X, n=n2, axis=0)
    K = np.fft.fft(K, n=n2, axis=0)
    Yspec_flat = (X * K).transpose(1, 0).reshape(-1)

    print('six-step vs direct torch  max|Δ|:', float(np.max(np.abs(y_np_six - y_ref.numpy()))))
    print('spectra (six vs direct)  max|Δ|:', float(np.max(np.abs(Yspec_flat - Y_full))))
    print('mlx vs six-step          max|Δ|:', float(np.max(np.abs(y_mlx - y_np_six))))
    print('mlx vs torch             max|Δ|:', float(np.max(np.abs(y_mlx - y_ref.numpy()))))


if __name__ == '__main__':
    main()
