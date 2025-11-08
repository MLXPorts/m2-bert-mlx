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
# No host math needed


# -----------------------------------------------------------------------------
# Metal kernel (header + body-only source)
# -----------------------------------------------------------------------------

_FFTCONV_HEADER = """#include <metal_stdlib>
using namespace metal;

// Constants
constant float PI_F = 3.14159265358979323846f;
// Twiddle capacity for N up to 4096 (half = 2048)
#define MAX_TW 4096
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

// Vectorized double-double using float2 (hi, lo)
inline float2 dd2_quick_two_sum(float a, float b) { float s = a + b; float e = b - (s - a); return float2(s, e); }
inline float2 dd2_two_sum(float a, float b) { float s = a + b; float v = s - a; float e = (a - (s - v)) + (b - v); return float2(s, e); }
inline float2 dd2_two_prod(float a, float b) { float p = a * b; float e = fma(a, b, -p); return float2(p, e); }
inline float2 dd2_add(float2 a, float2 b) { float2 s = dd2_two_sum(a.x, b.x); float2 t = dd2_two_sum(a.y, b.y); s.y += t.x; s = dd2_quick_two_sum(s.x, s.y); s.y += t.y; s = dd2_quick_two_sum(s.x, s.y); return s; }
inline float2 dd2_sub(float2 a, float2 b) { return dd2_add(a, float2(-b.x, -b.y)); }
inline float2 dd2_mul(float2 a, float2 b) { float2 p = dd2_two_prod(a.x, b.x); p.y += a.x * b.y + a.y * b.x; p = dd2_quick_two_sum(p.x, p.y); return p; }
inline float dd2_to_float(float2 a) { return a.x + a.y; }
inline void cdd2_mul(float2 a_re, float2 a_im, float2 b_re, float2 b_im, thread float2 &out_re, thread float2 &out_im) {
    float2 ac = dd2_mul(a_re, b_re);
    float2 bd = dd2_mul(a_im, b_im);
    out_re = dd2_sub(ac, bd);
    float2 ad = dd2_mul(a_re, b_im);
    float2 bc = dd2_mul(a_im, b_re);
    out_im = dd2_add(ad, bc);
}

inline uint bit_reverse(uint x, uint logn) {
    uint r = 0u;
    for (uint i = 0u; i < logn; ++i) {
        r = (r << 1u) | (x & 1u);
        x >>= 1u;
    }
    return r;
}

// In-place iterative FFT/IFFT in threadgroup memory
inline float2 dd2_mul_f(float2 a, float f) {
    float2 p = dd2_two_prod(a.x, f);
    p.y += a.y * f;
    return dd2_quick_two_sum(p.x, p.y);
}

inline void fft_inplace_global_table(
    device float* re, device float* im,
    threadgroup const float* Twr, threadgroup const float* Twi,
    uint N, uint tid, uint tpg, bool inverse, bool dd_bfly
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

    // Optional Torch-like early stages (radix-2 then radix-4)
    uint s_begin = 1u;
    if (logn >= 2u) {
        // Stage s=1 (m=2): pairs (0,1),(2,3),...
        for (uint p = tid; p < (N >> 1); p += tpg) {
            uint idx1 = (p << 1);
            uint idx2 = idx1 + 1u;
            float ur = re[idx1], ui = im[idx1];
            float vr = re[idx2], vi = im[idx2];
            re[idx1] = ur + vr; im[idx1] = ui + vi;
            re[idx2] = ur - vr; im[idx2] = ui - vi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stage s=2 (m=4): j=0 (w=1), j=1 (w = (0,-1) for forward, (0,1) for inverse)
        float sw1 = inverse ? 1.0f : -1.0f; // sin for j=1 twiddle; cos = 0
        for (uint p = tid; p < (N >> 2); p += tpg) {
            uint base = (p << 2);
            // j=0: (base, base+2)
            {
                uint i1 = base; uint i2 = base + 2u;
                float ur = re[i1], ui = im[i1];
                float vr = re[i2], vi = im[i2];
                re[i1] = ur + vr; im[i1] = ui + vi;
                re[i2] = ur - vr; im[i2] = ui - vi;
            }
            // j=1: (base+1, base+3) with w=(0,sw1)
            {
                uint i1 = base + 1u; uint i2 = base + 3u;
                float ur = re[i1], ui = im[i1];
                float vr = re[i2], vi = im[i2];
                float tr = -sw1 * vi;
                float ti =  sw1 * vr;
                re[i1] = ur + tr; im[i1] = ui + ti;
                re[i2] = ur - tr; im[i2] = ui - ti;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s_begin = 3u;
    }

    // Stages with twiddle table: W_N^(j*stride)
    for (uint s = s_begin; s <= logn; ++s) {
        uint m = (1u << s);
        uint halfm = (m >> 1u);
        uint stride = N / m;
        uint blocks = N / m;

        // Each thread iterates its own j across all blocks; fetch W_j per j (stable)
        for (uint j = tid; j < halfm; j += tpg) {
            uint tw_idx = j * stride; // W^(j*stride)
            float cw = Twr[tw_idx];
            float sw = Twi[tw_idx];
            if (inverse) sw = -sw;
            for (uint g = 0u; g < blocks; ++g) {
                uint idx1 = g * m + j;
                uint idx2 = idx1 + halfm;
                if (!dd_bfly) {
                    float r2 = re[idx2], i2 = im[idx2];
                    float tr = cw * r2 - sw * i2;
                    float ti = cw * i2 + sw * r2;
                    float ur = re[idx1], ui = im[idx1];
                    re[idx1] = ur + tr; im[idx1] = ui + ti;
                    re[idx2] = ur - tr; im[idx2] = ui - ti;
                } else {
                    // Double-double butterfly using float2
                    float2 ur = float2(re[idx1], 0.0f);
                    float2 ui = float2(im[idx1], 0.0f);
                    float2 r2 = float2(re[idx2], 0.0f);
                    float2 i2 = float2(im[idx2], 0.0f);
                    // t = w * (r2 + i*i2)
                    float2 tr = dd2_sub(dd2_mul_f(r2, cw), dd2_mul_f(i2, sw));
                    float2 ti = dd2_add(dd2_mul_f(r2, sw), dd2_mul_f(i2, cw));
                    // u +/- t
                    float2 r1 = dd2_add(ur, tr);
                    float2 i1 = dd2_add(ui, ti);
                    float2 r0 = dd2_sub(ur, tr);
                    float2 i0 = dd2_sub(ui, ti);
                    re[idx1] = dd2_to_float(r1); im[idx1] = dd2_to_float(i1);
                    re[idx2] = dd2_to_float(r0); im[idx2] = dd2_to_float(i0);
                }
            }
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

    // Precision-first path (no runtime flags)
    const bool hermitian_exact = true;
    const bool comp_bfly = true;

    // Precompute twiddle tables in threadgroup memory (size N/2)
    threadgroup float Twr[MAX_TW];
    threadgroup float Twi[MAX_TW];
    uint halfN = N >> 1;
    for (uint i = tid; i < halfN && i < MAX_TW; i += tpg) {
        float ang = -2.0f * PI_F * (float(i) / float(N));
        float cw; float sw = precise::sincos(ang, cw);
        Twr[i] = cw; Twi[i] = sw;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        Twr[0] = 1.0f; Twi[0] = 0.0f;
        if ( (N & 3u) == 0u ) { uint q = (N >> 2); if (q < MAX_TW) { Twr[q] = 0.0f; Twi[q] = -1.0f; } }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Workspaces are global arrays passed as outputs: Ur/Ui, Kr/Ki (each length B*C*N)
    // 1) K: time → freq
    for (uint i = tid; i < N; i += tpg) {
        float re = (i < L) ? k_time[k_base + i] : 0.0f;
        Kr[f_base + i] = re;
        Ki[f_base + i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fft_inplace_global_table(&Kr[f_base], &Ki[f_base], Twr, Twi, N, tid, tpg, /*inverse=*/ false, comp_bfly);
    if (hermitian_exact) {
        if (tid == 0u) { Ki[f_base + 0u] = 0.0f; if ((N & 1u) == 0u) { Ki[f_base + (N >> 1)] = 0.0f; } }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) U: time → freq
    for (uint i = tid; i < N; i += tpg) {
        float re = (i < L) ? u[u_base + i] : 0.0f;
        Ur[f_base + i] = re;
        Ui[f_base + i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fft_inplace_global_table(&Ur[f_base], &Ui[f_base], Twr, Twi, N, tid, tpg, /*inverse=*/ false, comp_bfly);
    if (hermitian_exact) {
        if (tid == 0u) { Ui[f_base + 0u] = 0.0f; if ((N & 1u) == 0u) { Ui[f_base + (N >> 1)] = 0.0f; } }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) Pointwise multiply in freq domain: U *= K → in-place on U
    for (uint i = tid; i < N; i += tpg) {
        float ur = Ur[f_base + i], ui = Ui[f_base + i];
        float kr = Kr[f_base + i], ki = Ki[f_base + i];
        // Always use float2 double-double multiply for U*K
        float2 ar = float2(ur, 0.0f), ai = float2(ui, 0.0f);
        float2 br = float2(kr, 0.0f), bi = float2(ki, 0.0f);
        thread float2 rr; thread float2 ri;
        cdd2_mul(ar, ai, br, bi, rr, ri);
        Ur[f_base + i] = dd2_to_float(rr);
        Ui[f_base + i] = dd2_to_float(ri);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4) IFFT(U) and write y (take first L) with per-channel bias
    fft_inplace_global_table(&Ur[f_base], &Ui[f_base], Twr, Twi, N, tid, tpg, /*inverse=*/ true, comp_bfly);
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
            input_names=["params", "u", "k_time", "D"],
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

    def __init__(self, match_torch: bool = False):
        super().__init__()
        # If True, we may switch multiply path to pure f32 in future; kernel is already f32.
        self.match_torch = match_torch
        # Precision-first path is always enabled in-kernel (no flags)
        self.kernel = _get_fftconv_kernel()
        # No host twiddle cache needed; computed per-launch in kernel

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

        (y_flat, _ur, _ui, _kr, _ki) = self.kernel(
            inputs=[params, u_flat, k_flat, D_plane.reshape(-1)],
            output_shapes=[y_flat_shape, work_shape, work_shape, work_shape, work_shape],
            output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32, mx.float32],
            grid=grid,
            threadgroup=threadgroup,
        )

        y = y_flat.reshape((B, C, L))
        return y
