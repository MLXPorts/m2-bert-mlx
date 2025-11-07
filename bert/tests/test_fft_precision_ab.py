#!/usr/bin/env python
import os, sys
import numpy as np
import mlx.core as mx
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(repo_root, 'src'))

from mm_mlx.metal_fft_conv import MetalFFTConv


def run_case(L=1024, dd=False, tables=True, seed=0):
    mx.random.seed(seed)
    B, C = 2, 8
    u = mx.random.normal((B, C, L)).astype(mx.float32)
    k = mx.random.normal((C, L)).astype(mx.float32)
    D = mx.zeros((1, C, 1), dtype=mx.float32)

    N = 2 * L
    u_t = torch.from_numpy(np.array(u))
    k_t = torch.from_numpy(np.array(k))
    y_ref = torch.fft.irfft(torch.fft.rfft(u_t, n=N, dim=-1) * torch.fft.rfft(k_t, n=N, dim=-1).unsqueeze(0), n=N, dim=-1)[..., :L]

    conv = MetalFFTConv(dd_mode=dd, use_twiddle_tables=tables)
    y = conv(u, k, D); mx.eval(y)
    diff = np.abs(y_ref.numpy() - np.array(y))
    return float(diff.max()), float(diff.mean())


def main():
    sizes = [1024, 2048, 4096]
    for L in sizes:
        for dd in [False, True]:
            for tables in [False, True]:
                m, M = run_case(L=L, dd=dd, tables=tables)
                print(f"L={L:<5} dd={int(dd)} tables={int(tables)} | max {m:.3e} mean {M:.3e}")


if __name__ == '__main__':
    main()

