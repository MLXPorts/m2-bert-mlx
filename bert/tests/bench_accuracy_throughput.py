#!/usr/bin/env python
import csv
import os
import sys
import time

import mlx.core as mx
import numpy as np
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(repo_root, 'src'))

from mm_mlx.metal_fft_conv import MetalFFTConv

def float64_baseline(u, k, L):
    N = 2*L
    u64 = np.array(u, dtype=np.float64)
    k64 = np.array(k, dtype=np.float64)
    y64 = np.fft.irfft(np.fft.rfft(u64, n=N, axis=-1) * np.fft.rfft(k64, n=N, axis=-1)[None, :, :], n=N, axis=-1)[..., :L]
    return y64

def stats(a):
    return float(np.max(np.abs(a))), float(np.mean(np.abs(a)))

def run_case(L=1024, B=2, C=8, iters=5):
    mx.random.seed(0)
    N = 2*L
    u = mx.random.normal((B, C, L)).astype(mx.float32)
    k = mx.random.normal((C, L)).astype(mx.float32)
    D = mx.zeros((1, C, 1), dtype=mx.float32)

    # baseline
    y64 = float64_baseline(u, k, L)

    # Removed MLX native FFT path per kernel-only policy

    # Torch accuracy (CPU) — throughput not measured here
    u_t = torch.from_numpy(np.array(u))
    k_t = torch.from_numpy(np.array(k))
    y_t = torch.fft.irfft(torch.fft.rfft(u_t, n=N, dim=-1)*torch.fft.rfft(k_t, n=N, dim=-1).unsqueeze(0), n=N, dim=-1)[..., :L]
    e_tch = stats(y_t.numpy().astype(np.float64) - y64)

    # JIT f32
    conv_f32 = MetalFFTConv(match_torch=True)
    t = 0.0
    for i in range(iters+1):
        t0 = time.perf_counter()
        y_jit = conv_f32(u, k, D); mx.eval(y_jit)
        if i>0:
            t += time.perf_counter() - t0
    jit_f32_ms = 1000.0*t/iters
    e_jit_f32 = stats(np.array(y_jit, dtype=np.float64) - y64)

    # JIT DD
    conv_dd = MetalFFTConv(match_torch=False)
    t = 0.0
    for i in range(iters+1):
        t0 = time.perf_counter()
        y_jit = conv_dd(u, k, D); mx.eval(y_jit)
        if i>0:
            t += time.perf_counter() - t0
    jit_dd_ms = 1000.0*t/iters
    e_jit_dd = stats(np.array(y_jit, dtype=np.float64) - y64)

    return {
        'L': L, 'B': B, 'C': C,
        'torch_max': e_tch[0], 'torch_mean': e_tch[1],
        'jit_f32_max': e_jit_f32[0], 'jit_f32_mean': e_jit_f32[1], 'jit_f32_ms': jit_f32_ms,
        'jit_dd_max': e_jit_dd[0], 'jit_dd_mean': e_jit_dd[1], 'jit_dd_ms': jit_dd_ms,
    }

def main():
    sizes = [256, 512, 1024, 2048, 4096]
    writer = csv.DictWriter(sys.stdout, fieldnames=[
        'L','B','C',
        'torch_max','torch_mean',
        'jit_f32_max','jit_f32_mean','jit_f32_ms',
        'jit_dd_max','jit_dd_mean','jit_dd_ms'
    ])
    writer.writeheader()
    for L in sizes:
        row = run_case(L=L, B=2, C=8, iters=5)
        writer.writerow(row)

if __name__ == '__main__':
    main()
