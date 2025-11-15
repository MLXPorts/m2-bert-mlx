#!/usr/bin/env python
"""
Metal kernels for Flash Monarch Mixer operations.

Fused kernels for:
1. hyena_filter_fwd - MLP on positional embeddings to generate filter coefficients
2. exp_mod - exponential modulation of filter
3. mm_block_fwd - fused short conv + FFT conv + gating (uses existing FFT kernel)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# Type helpers
f32 = lambda x: mx.array(x, dtype=mx.float32)
u32 = lambda x: mx.array(x, dtype=mx.uint32)
i32 = lambda x: mx.array(x, dtype=mx.int32)

# Global kernel cache
_HYENA_FILTER_KERNEL = None
_EXP_MOD_KERNEL = None

# -----------------------------------------------------------------------------
# Hyena Filter Kernel
# -----------------------------------------------------------------------------

_HYENA_FILTER_HEADER = """#include <metal_stdlib>
using namespace metal;

constant float PI = 3.14159265358979323846f;
"""

_HYENA_FILTER_SOURCE = """
// Hyena filter forward: MLP on positional embeddings
// Each thread computes one (C, L, ORDER) output element
// Grid: total threads = C * L * ORDER
uint tid = thread_position_in_grid.x;

uint C = params[0];
uint L = params[1];
uint EMB_DIM = params[2];
uint ORDER = params[3];

uint total = C * L * ORDER;
if (tid >= total) return;

// Decompose thread ID into (C_id, L_id, order_id)
uint C_id = tid / (L * ORDER);
uint rem = tid % (L * ORDER);
uint L_id = rem / ORDER;
uint order_id = rem % ORDER;

// Reverse check
bool reverse = (reverse_flags[C_id] == 1);
uint L_id_out = reverse ? (L - 1 - L_id) : L_id;

// Load positional embeddings z[C_id, L_id, :]
float z_vals[16];  // Max EMB_DIM=16
for (uint i = 0; i < EMB_DIM; i++) {
    uint z_idx = (C_id * L + L_id) * EMB_DIM + i;
    z_vals[i] = z[z_idx];
}

float my_freq = sin_freq[C_id * ORDER + order_id];

// Stage 1: eo_mat * z + eo_bias -> sin(val * freq)
float val1 = 0.0f;
for (uint i = 0; i < EMB_DIM; i++) {
    uint eo_idx = (C_id * EMB_DIM + i) * ORDER + order_id;
    val1 += eo_mat[eo_idx] * z_vals[i];
}
val1 += eo_bias[C_id * ORDER + order_id];
val1 = sin(val1 * my_freq);

// For stages 2 and 3, we need results from all ORDER threads
// Since we're running fully parallel, we need to recompute stage 1 for all ORDER values
// This is inefficient but avoids synchronization

// Stage 1 for all ORDER positions (needed for stage 2)
float stage1_vals[128];  // Max ORDER=128
for (uint o = 0; o < ORDER; o++) {
    float s1 = 0.0f;
    for (uint i = 0; i < EMB_DIM; i++) {
        uint eo_idx = (C_id * EMB_DIM + i) * ORDER + o;
        s1 += eo_mat[eo_idx] * z_vals[i];
    }
    s1 += eo_bias[C_id * ORDER + o];
    stage1_vals[o] = sin(s1 * sin_freq[C_id * ORDER + o]);
}

// Stage 2: oo1_mat * stage1 + oo1_bias -> sin(val * freq)
float val2 = 0.0f;
for (uint i = 0; i < ORDER; i++) {
    uint oo1_idx = (C_id * ORDER + i) * ORDER + order_id;
    val2 += oo1_mat[oo1_idx] * stage1_vals[i];
}
val2 += oo1_bias[C_id * ORDER + order_id];
val2 = sin(val2 * my_freq);

// Stage 2 for all ORDER positions (needed for stage 3)
float stage2_vals[128];  // Max ORDER=128
for (uint o = 0; o < ORDER; o++) {
    float s2 = 0.0f;
    for (uint i = 0; i < ORDER; i++) {
        uint oo1_idx = (C_id * ORDER + i) * ORDER + o;
        s2 += oo1_mat[oo1_idx] * stage1_vals[i];
    }
    s2 += oo1_bias[C_id * ORDER + o];
    stage2_vals[o] = sin(s2 * sin_freq[C_id * ORDER + o]);
}

// Stage 3: oo2_mat * stage2 + oo2_bias -> sin(val * freq)
float val3 = 0.0f;
for (uint i = 0; i < ORDER; i++) {
    uint oo2_idx = (C_id * ORDER + i) * ORDER + order_id;
    val3 += oo2_mat[oo2_idx] * stage2_vals[i];
}
val3 += oo2_bias[C_id * ORDER + order_id];
val3 = sin(val3 * my_freq);

// Write output
uint out_idx = (C_id * L + L_id_out) * ORDER + order_id;
output[out_idx] = val3;
"""

def _get_hyena_filter_kernel():
    global _HYENA_FILTER_KERNEL
    if _HYENA_FILTER_KERNEL is None:
        _HYENA_FILTER_KERNEL = mx.fast.metal_kernel(
            name="hyena_filter_fwd",
            input_names=["params", "z", "sin_freq", "eo_mat", "eo_bias", "oo1_mat", "oo1_bias", "oo2_mat", "oo2_bias", "reverse_flags"],
            output_names=["output"],
            header=_HYENA_FILTER_HEADER,
            source=_HYENA_FILTER_SOURCE,
        )
    return _HYENA_FILTER_KERNEL


def hyena_filter_fwd(z, sin_freq, eo_mat, eo_bias, oo1_mat, oo1_bias, oo2_mat, oo2_bias, reverse):
    """
    Hyena filter forward pass using Metal kernel.

    Args:
        z: (C, L, EMB_DIM) positional embeddings
        sin_freq: (C, ORDER) sin frequencies
        eo_mat: (C, EMB_DIM, ORDER) embedding-to-order matrix
        eo_bias: (C, ORDER) bias
        oo1_mat: (C, ORDER, ORDER) order-to-order matrix 1
        oo1_bias: (C, ORDER) bias 1
        oo2_mat: (C, ORDER, ORDER) order-to-order matrix 2
        oo2_bias: (C, ORDER) bias 2
        reverse: (C,) int32 reverse flags (0 or 1)

    Returns:
        output: (C, L, ORDER) filter coefficients
    """
    C, L, EMB_DIM = z.shape
    ORDER = eo_mat.shape[-1]

    # Ensure float32
    z = z.astype(mx.float32)
    sin_freq = sin_freq.astype(mx.float32)
    eo_mat = eo_mat.astype(mx.float32)
    eo_bias = eo_bias.astype(mx.float32)
    oo1_mat = oo1_mat.astype(mx.float32)
    oo1_bias = oo1_bias.astype(mx.float32)
    oo2_mat = oo2_mat.astype(mx.float32)
    oo2_bias = oo2_bias.astype(mx.float32)
    reverse = reverse.astype(mx.int32)

    # Parameters buffer
    params = mx.array([C, L, EMB_DIM, ORDER], dtype=mx.uint32)

    # Get kernel
    kernel = _get_hyena_filter_kernel()

    # Grid: total threads = C * L * ORDER
    # In MLX, grid is total threads, not number of threadgroups
    total = mx.multiply(mx.multiply(u32(C), u32(L)), u32(ORDER))
    one = u32(1)
    tpg = u32(256)
    grid = (total, one, one)
    threadgroup = (tpg, one, one)

    (output,) = kernel(
        inputs=[params, z, sin_freq, eo_mat, eo_bias, oo1_mat, oo1_bias, oo2_mat, oo2_bias, reverse],
        output_shapes=[(C, L, ORDER)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=threadgroup,
    )

    return output


# -----------------------------------------------------------------------------
# Exponential Modulation Kernel
# -----------------------------------------------------------------------------

_EXP_MOD_HEADER = """#include <metal_stdlib>
using namespace metal;
"""

_EXP_MOD_SOURCE = """
// Exponential modulation in-place
// Each thread handles one element
uint tid = thread_position_in_grid.x;

uint C = params[0];
uint L = params[1];
uint H = params[2];

uint total = C * L * H;
if (tid >= total) return;

// Decompose thread ID
uint C_id = tid / (L * H);
uint rem = tid % (L * H);
uint L_id = rem / H;
uint H_id = rem % H;

bool reverse = (reverse_flags[C_id] == 1);

// Compute fractions
float L_frac;
if (L > 1) {
    if (reverse) {
        L_frac = -1.0f + float(L_id) / float(L - 1);
    } else {
        L_frac = -1.0f * float(L_id) / float(L - 1);
    }
} else {
    L_frac = 0.0f;
}

float H_frac;
if (H > 1) {
    H_frac = (float(H_id) / float(H - 1)) * (max_decay - min_decay) + min_decay;
} else {
    H_frac = min_decay;
}

// Apply exponential modulation
float val = k[tid];
val = val * exp(abs(H_frac) * L_frac + shift);
k_out[tid] = val;
"""

def _get_exp_mod_kernel():
    global _EXP_MOD_KERNEL
    if _EXP_MOD_KERNEL is None:
        _EXP_MOD_KERNEL = mx.fast.metal_kernel(
            name="exp_mod_in_place",
            input_names=["params", "k", "reverse_flags", "min_decay", "max_decay", "shift"],
            output_names=["k_out"],
            header=_EXP_MOD_HEADER,
            source=_EXP_MOD_SOURCE,
        )
    return _EXP_MOD_KERNEL


def exp_mod_in_place_fwd(k, reverse, min_decay, max_decay, shift):
    """
    Apply exponential modulation to filter in-place.

    Args:
        k: (C, L, H) filter tensor
        reverse: (C,) int32 reverse flags
        min_decay: float min decay rate
        max_decay: float max decay rate
        shift: float shift value

    Returns:
        k: (C, L, H) modulated filter (in-place)
    """
    C, L, H = k.shape

    k = k.astype(mx.float32)
    reverse = reverse.astype(mx.int32)

    # Parameters
    params = mx.array([C, L, H], dtype=mx.uint32)
    min_decay_mx = f32(min_decay)
    max_decay_mx = f32(max_decay)
    shift_mx = f32(shift)

    # Get kernel
    kernel = _get_exp_mod_kernel()

    # Grid: total threads = C * L * H
    # In MLX, grid is total threads, not number of threadgroups
    C_mx = u32(C)
    L_mx = u32(L)
    H_mx = u32(H)
    total = mx.multiply(mx.multiply(C_mx, L_mx), H_mx)
    one = u32(1)
    tpg = u32(256)
    grid = (total, one, one)
    threadgroup = (tpg, one, one)

    (k_out,) = kernel(
        inputs=[params, k, reverse, min_decay_mx, max_decay_mx, shift_mx],
        output_shapes=[(C, L, H)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=threadgroup,
    )

    return k_out
