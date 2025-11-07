#!/usr/bin/env python
"""
MetalFFTConv: MLX JIT-compiled Metal kernel for 1D FFT-convolution

- No fallbacks. Always uses our JIT Metal kernel.
- Global kernel cache to avoid recompilation.
- Pure MLX scalars/tensors for all math (no Python numerics in compute).

Current implementation:
- Single all-in-one kernel per (batch, channel) that computes:
  1) FFT(k_time) in threadgroup memory, stores Kf to global work buffers
  2) FFT(u) in threadgroup memory
  3) Pointwise complex multiply U * K
  4) IFFT → write y (time) and add per-channel bias D

Notes:
- Uses fixed MAX_N=4096 threadgroup buffer; supports L up to 2048 (N=2*L up to 4096).
- Twiddles computed on-the-fly via sin/cos to keep host simple for now.
- Grid semantics per mx.fast.metal_kernel: total threads = (B*C * tpg), one TG per (b,c).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .typed import f32, u32
# math_ops is alongside mm_mlx at bert/src; import absolute
from math_ops import PI


# -----------------------------------------------------------------------------
# Metal kernel (header + body-only source)
# -----------------------------------------------------------------------------

_FFTCONV_HEADER = """#include <metal_stdlib>
using namespace metal;

// Constants
constant float PI_F = 3.14159265358979323846f;
struct Complex { float real; float imag; };

inline Complex cadd(Complex a, Complex b) { return Complex{a.real + b.real, a.imag + b.imag}; }
inline Complex csub(Complex a, Complex b) { return Complex{a.real - b.real, a.imag - b.imag}; }
inline Complex cmul(Complex a, Complex b) { return Complex{fma(-a.imag, b.imag, a.real * b.real), fma(a.real, b.imag, a.imag * b.real)}; }
// Double-double helpers for optional high-precision complex multiply
struct dd_t { float hi; float lo; };
inline dd_t quick_two_sum(float a, float b) { float s = a + b; float e = b - (s - a); return dd_t{s, e}; }
inline dd_t two_sum(float a, float b) { float s = a + b; float v = s - a; float e = (a - (s - v)) + (b - v); return dd_t{s, e}; }
inline dd_t two_prod(float a, float b) { float p = a * b; float e = fma(a, b, -p); return dd_t{p, e}; }
inline dd_t dd_add(dd_t a, dd_t b) { dd_t s = two_sum(a.hi, b.hi); dd_t t = two_sum(a.lo, b.lo); s.lo += t.hi; s = quick_two_sum(s.hi, s.lo); s.lo += t.lo; s = quick_two_sum(s.hi, s.lo); return s; }
inline dd_t dd_sub(dd_t a, dd_t b) { return dd_add(a, dd_t{-b.hi, -b.lo}); }
inline dd_t dd_mul(dd_t a, dd_t b) { dd_t p = two_prod(a.hi, b.hi); p.lo += a.hi * b.lo + a.lo * b.hi; p = quick_two_sum(p.hi, p.lo); return p; }
inline float dd_to_float(dd_t a) { return a.hi + a.lo; }
struct cdd_t { dd_t re; dd_t im; };
inline cdd_t cdd_mul(cdd_t a, cdd_t b) { dd_t ac = dd_mul(a.re, b.re); dd_t bd = dd_mul(a.im, b.im); dd_t re = dd_sub(ac, bd); dd_t ad = dd_mul(a.re, b.im); dd_t bc = dd_mul(a.im, b.re); dd_t im = dd_add(ad, bc); return cdd_t{re, im}; }

inline uint bit_reverse(uint x, uint logn) {
    uint r = 0u;
    for (uint i = 0u; i < logn; ++i) {
        r = (r << 1u) | (x & 1u);
        x >>= 1u;
    }
    return r;
}

// In-place iterative FFT/IFFT in threadgroup memory
inline void fft_inplace_global_table(
    device float* re, device float* im,
    device const float* Twr, device const float* Twi,
    uint N, uint tid, uint tpg, bool inverse
) {
    // Bit reversal permutation in global memory
    uint logn = 0u; { uint n = N; while (n > 1u) { n >>= 1u; ++logn; } }
    for (uint i = tid; i < N; i += tpg) {
        uint j = bit_reverse(i, logn);
        if (j > i) {
            float tr = re[i]; float ti = im[i];
            re[i] = re[j]; im[i] = im[j];
            re[j] = tr;    im[j] = ti;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stages with twiddle table: W_N^(j*stride)
    for (uint s = 1u; s <= logn; ++s) {
        uint m = (1u << s);
        uint halfm = (m >> 1u);
        uint stride = N / m;

        for (uint k = tid; k < (N >> 1u); k += tpg) {
            uint j = k & (halfm - 1u);
            uint g = k >> (s - 1u);
            uint idx1 = g * m + j;
            uint idx2 = idx1 + halfm;
            // Use sincos for now (shared range reduction); table kept for future A/B
            float ang = (inverse ? 1.0f : -1.0f) * 2.0f * PI_F * (float(j) / float(m));
            float cw; float sw = sincos(ang, cw);
            float r2 = re[idx2], i2 = im[idx2];
            // t = w * data[idx2]
            float tr = cw * r2 - sw * i2;
            float ti = cw * i2 + sw * r2;
            float ur = re[idx1], ui = im[idx1];
            // idx1 = u + t; idx2 = u - t
            re[idx1] = ur + tr; im[idx1] = ui + ti;
            re[idx2] = ur - tr; im[idx2] = ui - ti;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (inverse) {
        float invN = 1.0f / float(N);
        for (uint i = tid; i < N; i += tpg) {
            re[i] *= invN; im[i] *= invN;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"""


_FFTCONV_SOURCE = r"""
    // params: [B, C, L, N]
    uint B = params[0];
    uint C = params[1];
    uint L = params[2];
    uint N = params[3]; // must be power of two, N = 2*L

    uint gtid = thread_position_in_grid.x;
    uint tpg  = threads_per_threadgroup.x;
    uint grp  = gtid / tpg;            // which (b,c)
    uint tid  = gtid - grp * tpg;      // local thread id
    if (grp >= B * C) return;

    uint b = grp / C;
    uint c = grp - b * C;

    // Base pointers (flattened indices)
    uint u_base  = (b * C + c) * L;
    uint y_base  = u_base;
    uint k_base  = c * L;
    uint f_base  = (b * C + c) * N; // workspace spectra per (b,c)

    // flags: bit0=dd_mode, bit1=use_tables
    uint flags_v = flags[0];
    bool dd_mode = (flags_v & 1u) != 0u;
    bool use_tables = (flags_v & 2u) != 0u;

    // Workspaces are global arrays passed as outputs: Ur/Ui, Kr/Ki (each length B*C*N)
    // 1) K: time → freq
    for (uint i = tid; i < N; i += tpg) {
        float re = (i < L) ? k_time[k_base + i] : 0.0f;
        Kr[f_base + i] = re;
        Ki[f_base + i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (use_tables) {
        fft_inplace_global_table(&Kr[f_base], &Ki[f_base], Twr, Twi, N, tid, tpg, /*inverse=*/ false);
    } else {
        // Fallback should not be used in production, but keep for A/B
        fft_inplace_global_table(&Kr[f_base], &Ki[f_base], Twr, Twi, N, tid, tpg, /*inverse=*/ false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) U: time → freq
    for (uint i = tid; i < N; i += tpg) {
        float re = (i < L) ? u[u_base + i] : 0.0f;
        Ur[f_base + i] = re;
        Ui[f_base + i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (use_tables) {
        fft_inplace_global_table(&Ur[f_base], &Ui[f_base], Twr, Twi, N, tid, tpg, /*inverse=*/ false);
    } else {
        fft_inplace_global_table(&Ur[f_base], &Ui[f_base], Twr, Twi, N, tid, tpg, /*inverse=*/ false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) Pointwise multiply in freq domain: U *= K → in-place on U
    for (uint i = tid; i < N; i += tpg) {
        float ur = Ur[f_base + i], ui = Ui[f_base + i];
        float kr = Kr[f_base + i], ki = Ki[f_base + i];
        if (!dd_mode) {
            float rr = fma(-ui, ki, ur * kr);
            float ri = fma(ur,  ki, ui * kr);
            Ur[f_base + i] = rr; Ui[f_base + i] = ri;
        } else {
            cdd_t ua = cdd_t{ dd_t{ur, 0.0f}, dd_t{ui, 0.0f} };
            cdd_t kb = cdd_t{ dd_t{kr, 0.0f}, dd_t{ki, 0.0f} };
            cdd_t rc = cdd_mul(ua, kb);
            Ur[f_base + i] = dd_to_float(rc.re);
            Ui[f_base + i] = dd_to_float(rc.im);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4) IFFT(U) and write y (take first L) with per-channel bias
    if (use_tables) {
        fft_inplace_global_table(&Ur[f_base], &Ui[f_base], Twr, Twi, N, tid, tpg, /*inverse=*/ true);
    } else {
        fft_inplace_global_table(&Ur[f_base], &Ui[f_base], Twr, Twi, N, tid, tpg, /*inverse=*/ true);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bias = D[c];
    for (uint i = tid; i < L; i += tpg) {
        float yv = Ur[f_base + i] + u[u_base + i] * bias;
        y[y_base + i] = yv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""


_FFTCONV_KERNEL = None


def _get_fftconv_kernel():
    global _FFTCONV_KERNEL
    if _FFTCONV_KERNEL is None:
        _FFTCONV_KERNEL = mx.fast.metal_kernel(
            name="fftconv1d_unified",
            input_names=["params", "u", "k_time", "D", "Twr", "Twi", "flags"],
            output_names=["y", "Ur", "Ui", "Kr", "Ki"],  # global workspaces
            header=_FFTCONV_HEADER,
            source=_FFTCONV_SOURCE,
        )
    return _FFTCONV_KERNEL


class MetalFFTConv(nn.Module):
    """
    JIT Metal FFT-convolution layer.

    Always uses our compiled kernel; no fallbacks. Kernels are compiled once globally.
    """

    def __init__(self, match_torch: bool = False, use_twiddle_tables: bool = True, dd_mode: bool = False):
        super().__init__()
        # If True, we may switch multiply path to pure f32 in future; kernel is already f32.
        self.match_torch = match_torch
        self.use_twiddle_tables = use_twiddle_tables
        self.dd_mode = dd_mode
        self.kernel = _get_fftconv_kernel()
        self._twiddle_cache: dict[int, tuple[mx.array, mx.array]] = {}

    def _twiddles(self, N: int) -> tuple[mx.array, mx.array]:
        if N in self._twiddle_cache:
            return self._twiddle_cache[N]
        # Build forward twiddle table W_N^k = cos(-2πk/N) + i sin(-2πk/N) for k=0..N/2-1
        half = N // 2
        k = mx.arange(half, dtype=mx.float32)
        two = mx.array(2.0, dtype=mx.float32)
        Nf = mx.array(N, dtype=mx.float32)
        angle = mx.multiply(mx.multiply(mx.negative(two), PI), mx.divide(k, Nf))  # -2πk/N
        twr = mx.cos(angle)
        twi = mx.sin(angle)
        # Exact corners: k=0 → (1,0)
        idx0 = mx.array(0, dtype=mx.int32)
        idx_all = mx.arange(half, dtype=mx.int32)
        mask0 = mx.equal(idx_all, idx0)
        twr = mx.where(mask0, mx.array(1.0, dtype=mx.float32), twr)
        twi = mx.where(mask0, mx.array(0.0, dtype=mx.float32), twi)
        if (N % 4) == 0:
            kq = mx.array(N // 4, dtype=mx.int32)
            maskq = mx.equal(idx_all, kq)
            twr = mx.where(maskq, mx.array(0.0, dtype=mx.float32), twr)
            twi = mx.where(maskq, mx.array(-1.0, dtype=mx.float32), twi)
        self._twiddle_cache[N] = (twr.astype(mx.float32), twi.astype(mx.float32))
        return self._twiddle_cache[N]

    def __call__(self, u: mx.array, k: mx.array, D: mx.array) -> mx.array:  # type: ignore[name-defined]
        # Shapes: u (B, C, L), k (C, L), D (1, C, 1) or (C,)
        assert len(u.shape) == 3, "u must be (B,C,L)"
        assert len(k.shape) == 2, "k must be (C,L)"
        B, C, L = u.shape

        # Enforce dtypes
        u = u.astype(mx.float32)
        k = k.astype(mx.float32)
        if D.ndim == 3:
            D_plane = D.reshape(-1).astype(mx.float32)  # (C,)
        else:
            D_plane = D.astype(mx.float32)

        # Flattened buffers
        u_flat = u.reshape(-1)
        k_flat = k.reshape(-1)

        # Params buffer
        params = mx.stack([u32(B), u32(C), u32(L), mx.add(u32(L), u32(L))])

        # Grid configuration (MLX scalars)
        groups = mx.multiply(u32(B), u32(C))          # B*C threadgroups
        N_u32 = mx.add(u32(L), u32(L))                # 2*L
        tpg = mx.where(mx.greater_equal(N_u32, u32(256)), u32(256), N_u32)
        one = u32(1)
        total_threads = mx.multiply(groups, tpg)
        grid = (total_threads, one, one)
        threadgroup = (tpg, one, one)

        # Output buffers: shapes use MLX u32 scalars (no Python arithmetic)
        y_elems = mx.multiply(mx.multiply(u32(B), u32(C)), u32(L))
        n_elems = mx.multiply(mx.multiply(u32(B), u32(C)), mx.add(u32(L), u32(L)))
        y_flat_shape = (y_elems,)
        work_shape = (n_elems,)

        # Twiddle tables
        Twr, Twi = self._twiddles(int(2 * L)) if self.use_twiddle_tables else (mx.zeros((1,), dtype=mx.float32), mx.zeros((1,), dtype=mx.float32))
        # Flags: bit0=dd_mode, bit1=use_tables
        flags = mx.array(int(self.dd_mode) | (int(self.use_twiddle_tables) << 1), dtype=mx.uint32)

        (y_flat, _ur, _ui, _kr, _ki) = self.kernel(
            inputs=[params, u_flat, k_flat, D_plane.reshape(-1), Twr, Twi, flags.reshape(-1)],
            output_shapes=[y_flat_shape, work_shape, work_shape, work_shape, work_shape],
            output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32, mx.float32],
            grid=grid,
            threadgroup=threadgroup,
        )

        y = y_flat.reshape((B, C, L))
        return y
