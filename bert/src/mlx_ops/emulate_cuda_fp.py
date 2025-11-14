#!/usr/bin/env python
"""
Emulate CUDA-style float32 arithmetic to hand-calculate expected rounding.

We simulate two multiply/add strategies for complex multiply:
  - separate: (ar*br) and (ai*bi) are rounded to f32 individually, then summed/subtracted in f32
  - fma_sum:  real = fma(ar, br, -round(ai*bi))  (one rounding for the sum, product of ai*bi rounded once)
              imag = fma(ar, bi,  round(ai*br))

This is not a perfect model of every GPU/compiler, but it brackets the
behavior commonly observed in CUDA codegen.

Note: This module is for diagnostics only. It should not be used in hot paths.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def f32(x: np.ndarray | float) -> np.float32:
    return np.float32(x)


def f32_mul(a: np.float32, b: np.float32) -> np.float32:
    return np.float32(np.float64(a) * np.float64(b))


def f32_add(a: np.float32, b: np.float32) -> np.float32:
    return np.float32(np.float64(a) + np.float64(b))


def f32_sub(a: np.float32, b: np.float32) -> np.float32:
    return np.float32(np.float64(a) - np.float64(b))


def f32_fma(a: np.float32, b: np.float32, c: np.float32) -> np.float32:
    """Emulate fused multiply-add: round(a*b + c) once to float32.

    We compute in float64 then round once to float32.
    """
    return np.float32(np.float64(a) * np.float64(b) + np.float64(c))


def complex_mul_f32(ar: np.float32, ai: np.float32, br: np.float32, bi: np.float32, mode: str = "separate") -> Tuple[np.float32, np.float32]:
    """Complex multiply with explicit rounding model.

    (ar + i ai) * (br + i bi)
    modes:
      - 'separate': round products, then sum/sub in f32
      - 'fma_sum' : real = fma(ar, br, -round(ai*bi)) ; imag = fma(ar, bi,  round(ai*br))
    """
    if mode == "separate":
        t0 = f32_mul(ar, br)
        t1 = f32_mul(ai, bi)
        real = f32_sub(t0, t1)
        t2 = f32_mul(ar, bi)
        t3 = f32_mul(ai, br)
        imag = f32_add(t2, t3)
        return real, imag
    elif mode == "fma_sum":
        m1 = f32_mul(ai, bi)
        real = f32_fma(ar, br, f32(-m1))
        m2 = f32_mul(ai, br)
        imag = f32_fma(ar, bi, m2)
        return real, imag
    else:
        raise ValueError(f"unknown mode: {mode}")


def fftconv_emulated(u: np.ndarray, k: np.ndarray, D: np.ndarray, mode: str = "fma_sum") -> np.ndarray:
    """Emulate FFT convolution with explicit float32 rounding in complex multiply.

    Args:
      u: (B, C, L) float32
      k: (C, L) float32
      D: (1, C, 1) float32
    Returns:
      y: (B, C, L) float32
    """
    B, C, L = u.shape
    N = 2 * L
    uf = np.fft.rfft(u.astype(np.float32), n=N, axis=-1).astype(np.complex64)
    kf = np.fft.rfft(k.astype(np.float32), n=N, axis=-1).astype(np.complex64)

    # Elementwise complex multiply with rounding model
    out = np.empty_like(uf)
    for b in range(B):
        for c in range(C):
            ar = uf[b, c].real.astype(np.float32)
            ai = uf[b, c].imag.astype(np.float32)
            br = kf[c].real.astype(np.float32)
            bi = kf[c].imag.astype(np.float32)
            rr = np.empty_like(ar)
            ri = np.empty_like(ai)
            for i in range(ar.shape[0]):
                r, im = complex_mul_f32(ar[i], ai[i], br[i], bi[i], mode=mode)
                rr[i] = r
                ri[i] = im
            out[b, c] = rr.astype(np.float32) + 1j * ri.astype(np.float32)

    y = np.fft.irfft(out, n=N, axis=-1)[..., :L].astype(np.float32)
    y = y + (u * D).astype(np.float32)
    return y


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))

