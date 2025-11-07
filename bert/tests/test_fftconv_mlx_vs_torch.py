#!/usr/bin/env python
import os
import sys

import mlx.core as mx
import numpy as np

try:
    import torch
except Exception as e:
    torch = None

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(repo_root, 'src'))

from mm_mlx.metal_fft_conv import MetalFFTConv  # noqa: E402


def parity_run(batch=2, channels=8, L=512, seed=0):
    mx.random.seed(seed)
    conv = MetalFFTConv()

    u = mx.random.normal((batch, channels, L)).astype(mx.float32)
    k = mx.random.normal((channels, L)).astype(mx.float32)
    D = mx.zeros((1, channels, 1), dtype=mx.float32)

    y_mlx = conv(u, k, D)
    mx.eval(y_mlx)

    if torch is None:
        return None, None

    N = 2 * L
    mx.eval(u, k, D)
    u_t = torch.from_numpy(np.array(u))
    k_t = torch.from_numpy(np.array(k))
    D_t = torch.from_numpy(np.array(D).reshape(1, channels, 1))

    k_f = torch.fft.rfft(k_t, n=N, dim=-1)
    u_f = torch.fft.rfft(u_t, n=N, dim=-1)
    y_f = u_f * k_f.unsqueeze(0)
    y_ref = torch.fft.irfft(y_f, n=N, dim=-1)[..., :L]
    y_ref = y_ref + u_t * D_t
    diff = (y_ref.numpy() - np.array(y_mlx))
    return float(abs(diff).max()), float(abs(diff).mean())


def main():
    sizes = [64, 128, 256, 512, 1024, 2048, 4096]
    results = []
    for L in sizes:
        m, M = parity_run(L=L)
        if m is None:
            print("PyTorch not available; skipping.")
            return
        results.append((L, m, M))
        print(f"L={L:<5d} | max |Δ| = {m:9.3e} | mean |Δ| = {M:9.3e}")

    # basic assertion for small sizes
    small_ok = all(m < 5e-5 for L, m, M in results if L <= 512)
    print("small_ok:", small_ok)


if __name__ == "__main__":
    main()
