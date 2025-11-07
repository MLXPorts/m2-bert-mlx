#!/usr/bin/env python
import os, sys
import numpy as np
import mlx.core as mx
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(repo_root, 'src'))

from mm_mlx.metal_fft_conv import MetalFFTConv

def compare_one(L=1024, B=2, C=8, match_torch=False):
    N = 2*L
    mx.random.seed(0)
    u = mx.random.normal((B, C, L)).astype(mx.float32)
    k = mx.random.normal((C, L)).astype(mx.float32)
    D = mx.zeros((1, C, 1), dtype=mx.float32)

    # Float64 baseline
    u64 = np.array(u, dtype=np.float64)
    k64 = np.array(k, dtype=np.float64)
    y64 = np.fft.irfft(np.fft.rfft(u64, n=N, axis=-1) * np.fft.rfft(k64, n=N, axis=-1)[None, :, :], n=N, axis=-1)[..., :L]

    # MLX
    k_f = mx.fft.rfft(k, n=N, axis=-1)
    u_f = mx.fft.rfft(u, n=N, axis=-1)
    y_mlx = mx.fft.irfft(u_f * k_f.reshape(1, C, -1), n=N, axis=-1)[..., :L]
    mx.eval(y_mlx)

    # Torch
    u_t = torch.from_numpy(np.array(u))
    k_t = torch.from_numpy(np.array(k))
    y_t = torch.fft.irfft(torch.fft.rfft(u_t, n=N, dim=-1) * torch.fft.rfft(k_t, n=N, dim=-1).unsqueeze(0), n=N, dim=-1)[..., :L]

    # JIT
    conv = MetalFFTConv(match_torch=match_torch)
    y_jit = conv(u, k, D); mx.eval(y_jit)

    def stats(a):
        return float(np.max(np.abs(a))), float(np.mean(np.abs(a)))

    e_mlx = stats(np.array(y_mlx, dtype=np.float64) - y64)
    e_tch = stats(y_t.numpy().astype(np.float64) - y64)
    e_jit = stats(np.array(y_jit, dtype=np.float64) - y64)

    return e_mlx, e_tch, e_jit


def main():
    sizes = [256, 512, 1024, 2048, 4096]
    print("match_torch=False (extended DD multiply)")
    for L in sizes:
        e_mlx, e_tch, e_jit = compare_one(L=L, match_torch=False)
        print(f"L={L:<5} | MLX max {e_mlx[0]:.3e} mean {e_mlx[1]:.3e} | Torch max {e_tch[0]:.3e} mean {e_tch[1]:.3e} | JIT-DD max {e_jit[0]:.3e} mean {e_jit[1]:.3e}")
    print("\nmatch_torch=True (pure f32 multiply)")
    for L in sizes:
        e_mlx, e_tch, e_jit = compare_one(L=L, match_torch=True)
        print(f"L={L:<5} | MLX max {e_mlx[0]:.3e} mean {e_mlx[1]:.3e} | Torch max {e_tch[0]:.3e} mean {e_tch[1]:.3e} | JIT-f32 max {e_jit[0]:.3e} mean {e_jit[1]:.3e}")

if __name__ == '__main__':
    main()
