# Numerical Precision in GPU Computing: A Comprehensive Guide

## Executive Summary

This document explains why numerical precision matters in GPU computing, documents our investigation into MLX vs PyTorch numerical differences, and provides actionable guidance for maintaining stability over long-running workloads.

**Key Finding**: Running 300 million operations per second for 24 hours = 25.9 trillion operations. Each float32 operation introduces approximately 6×10⁻⁸ relative error (half-ULP). Even "tiny" 10⁻⁶ differences compound when repeated trillions of times.

**Root Cause**: Not mathematical errors, but:
1. Multiple rounding points in standard pipelines
2. Backend differences in accumulation order and FMA usage
3. Hidden dtype/device conversions adding extra rounding
4. Inconsistent FFT normalization conventions

## Table of Contents

1. [The Investigation: What We Found](#the-investigation)
2. [Mathematical Background](#mathematical-background)
3. [Common Precision-Breaking Patterns](#precision-breaking-patterns)
4. [The Solution: Extended Precision](#extended-precision)
5. [Implementation Tiers](#implementation-tiers)
6. [Results and Verification](#results)
7. [Practical Usage Guide](#usage-guide)

---

## The Investigation

### Starting Point

MLX Hyena vs PyTorch Hyena showed approximately 10⁻⁶ relative differences in outputs. Both implementations were mathematically correct, but numerically different. Over billions of iterations, this drift becomes significant.

### Source Code Analysis

We examined the actual C++ implementations of both frameworks:

**PyTorch FFT** (`pytorch/aten/src/ATen/native/SpectralOps.cpp`):
- Line 117: `norm_from_string` defines normalization semantics
- `norm='forward'`: forward transform divides by N, inverse does not
- `norm='backward'` (default): forward does not scale, inverse divides by N
- `norm='ortho'`: both divide by √N

**PyTorch MPS Backend** (`pytorch/aten/src/ATen/native/mps/operations/FastFourierTransform.mm`):
- Line 62: TODO comment explicitly acknowledging "numerical discrepancies"
- Uses MPSGraph FFT primitives (Apple's implementation)

**MLX FFT** (`mlx/fft.cpp`, `backend/cpu/fft.cpp`, `backend/metal/fft.cpp`):
- CPU: uses pocketfft directly (identical to PyTorch CPU)
- GPU: custom Metal implementation with multi-pass 1D FFTs
- Default: unnormalized forward, 1/N scaling on inverse

### Critical Discovery: Multiple Rounding Points

Current float32 FFT-based convolution has **four rounding points**:

```
Input (float32)
  → FFT (internally float32, output complex64)     [ROUND 1]
  → Multiply (complex64 × complex64)               [ROUND 2]
  → IFFT (internally float32)                      [ROUND 3]
  → Scale & bias (float32)                         [ROUND 4]
  → Output (float32)
```

Each rounding compounds error. With four rounding points, drift accumulates significantly faster than a single-round path.

### Comparison Summary

| Area | PyTorch Behavior | MLX Behavior | Impact |
|------|------------------|--------------|--------|
| **FFT normalization** | `irfft(..., norm='forward')` applies no implicit scale; manually divide kernel spectrum by N | `irfft(...)` applies 1/N by default; must NOT pre-divide spectrum | Double scaling or missing scaling gives order-one amplitude errors |
| **Kernel combine** | Time-domain pad + reverse + sum (default in reference code) | Selectable: time-domain sum or frequency-domain average | Different combine domains re-order additions; affects numerical stability |
| **Linear/GEMM ops** | Backend-specific accumulation order and FMA handling | Same issue (different Metal kernels) | Approximately 10⁻⁷ relative drift from order-of-operations alone |
| **float()/int() usage** | Casting tensors breaks graph, forces CPU copy, rounds twice | Identical behavior | Strict error in our linting |
| **numpy.fft** | Promotes to float64, runs on CPU | N/A | Any parity test touching NumPy is invalid |
| **Device fallbacks** | MPSGraph FFT has known accuracy limitations | MLX has custom Metal FFT | Tiny differences remain even with matching formulas |

---

## Mathematical Background

### Why Float32 Isn't Associative

In IEEE-754 floating-point arithmetic:

- **(a + b) + c ≠ a + (b + c)** — addition is not associative
- **a × b then + c ≠ fma(a, b, c)** — FMA rounds once, separate operations round twice
- Different backends use different reduction trees, yielding different answers

All are IEEE-compliant, but they produce different results at the ULP level.

### Error Accumulation Models

For a sum of N float32 terms with unit magnitude:

- **Worst case (pathological ordering)**: error ≤ κ·N·u where u ≈ 5.96×10⁻⁸
- **Random signs (typical)**: error ≈ κ·√N·u
- **With Kahan compensation**: error ≈ κ·u (constant, independent of N)

For N = 2.6×10¹³ (24-hour scenario at 300M ops/sec):

- **Worst case**: 5.1×10⁶ · 5.96×10⁻⁸ ≈ **0.3 (30% drift)**
- **Random**: √(2.6×10¹³) · 5.96×10⁻⁸ ≈ **3×10⁻⁴ (0.03%)**
- **Kahan**: ≈10⁻⁷ (negligible)

### The Convolution Theorem and Normalization

The discrete Fourier transform convolution theorem states:

```
y = IFFT(FFT(x) ⊙ FFT(h))
```

This requires **exactly one 1/N factor** across the forward/inverse pair. Placement is conventional:

**Convention A (PyTorch reference code)**: Apply 1/N on the spectrum
```python
k_f = torch.fft.rfft(k, n=2*L) / (2*L)  # scale here
u_f = torch.fft.rfft(u, n=2*L)
y = torch.fft.irfft(u_f * k_f, n=2*L, norm='forward')  # no inverse scaling
```

**Convention B (MLX default)**: Apply 1/N on the inverse
```python
k_f = mx.fft.rfft(k, n=2*L)  # no scaling
u_f = mx.fft.rfft(u, n=2*L)
y = mx.fft.irfft(u_f * k_f, n=2*L)  # inverse applies 1/N
```

**Both are mathematically correct.** Mixing them is wrong:

**INCORRECT (double scaling)**:
```python
k_f = mx.fft.rfft(k, n=2*L) / (2*L)  # scaled once
y = mx.fft.irfft(u_f * k_f, n=2*L)   # MLX inverse scales by 1/N again
# Result is 1/N² too small
```

**INCORRECT (no scaling)**:
```python
k_f = torch.fft.rfft(k, n=2*L)  # not scaled
y = torch.fft.irfft(u_f * k_f, n=2*L, norm='forward')  # inverse doesn't scale
# Result is N times too large
```

---

## Precision-Breaking Patterns

### Pattern 1: Python Scalars in Tensor Math

**Problem:**
```python
y = x * 0.5  # BAD
k_f = torch.fft.rfft(k) / (2 * L)  # BAD
```

**What happens:**
1. Python `0.5` is float64
2. Framework either:
   - Promotes x to float64 (expensive, wrong precision)
   - Demotes 0.5 to float32 (adds a rounding step)
   - Breaks lazy graph and computes immediately
3. Result: **two roundings instead of one**

Over 25 trillion operations, these extra roundings accumulate to visible drift.

**Fix (PyTorch):**
```python
half = torch.tensor(0.5, dtype=torch.float32, device=x.device)
y = torch.mul(x, half)

n = torch.tensor(2 * L, dtype=torch.float32, device=x.device)
k_f = torch.divide(torch.fft.rfft(k), n)
```

**Fix (MLX):**
```python
half = mx.array(0.5, dtype=mx.float32)
y = mx.multiply(x, half)

n = mx.array(2 * L, dtype=mx.float32)
k_f = mx.divide(mx.fft.rfft(k), n)
```

### Pattern 2: NumPy in Compute Paths

**Problem:**
```python
import numpy as np
x_np = np.random.randn(B, H, L).astype(np.float32)  # BAD
x_mx = mx.array(x_np)
```

**What happens:**
1. `np.random.randn` creates float64
2. `.astype(np.float32)` rounds to float32
3. `mx.array` copies to GPU
4. **First rounding wasted; should generate on device**

**Fix (MLX):**
```python
key = mx.random.key(0)
x_mx = mx.random.normal(shape=(B, H, L), dtype=mx.float32, key=key)
```

**Fix (PyTorch):**
```python
x = torch.randn(B, H, L, dtype=torch.float32, device='mps')
```

### Pattern 3: Graph-Breaking Conversions

**Problem:**
```python
similarity = compute_similarity(text1, text2)
if similarity.item() > threshold:  # BAD
    ...
```

**What happens:**
1. `.item()` forces GPU→CPU copy
2. Breaks lazy computation graph
3. Destroys Metal buffer link (no gradient tracking in MLX)
4. Adds extra rounding (tensor → Python scalar)
5. Cannot fuse with subsequent operations

**Fix (PyTorch):**
```python
threshold_t = torch.tensor(threshold, dtype=torch.float32, device=similarity.device)
if torch.greater(similarity, threshold_t):
    ...
```

**Fix (MLX):**
```python
threshold_a = mx.array(threshold, dtype=mx.float32)
if mx.greater(similarity, threshold_a):
    ...
```

### Pattern 4: FFT Normalization Errors

See "The Convolution Theorem and Normalization" section above. Key rule:

**Apply exactly one 1/N factor across the forward/inverse pair.** Document which convention you use and enforce it consistently.

### Pattern 5: Device Hops and Hidden Copies

**Problem:**
```python
u = mx.array(data, dtype=mx.float32)  # on GPU
bias = mx.array(0.1, dtype=mx.float32)  # might be on CPU
y = u + bias  # BAD if devices differ
```

**What happens:**
1. Framework detects device mismatch
2. Copies bias GPU↔CPU
3. Re-rounds during copy (layout change)
4. Loses kernel fusion opportunities

**Fix:**
```python
# PyTorch: create on target device explicitly
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
u = torch.tensor(data, dtype=torch.float32, device=device)
bias = torch.tensor(0.1, dtype=torch.float32, device=device)

# MLX: defaults to GPU; just ensure dtype
u = mx.array(data, dtype=mx.float32)
bias = mx.array(0.1, dtype=mx.float32)
```

---

## The Solution: Extended Precision

### Why Double-Double?

Apple M-series GPUs do not have native float64 ALUs. Performance would be orders of magnitude slower. Instead, we emulate higher precision using **two float32 values**:

```c
struct dd {
    float hi;  // High-order 23 bits mantissa
    float lo;  // Low-order error term (another ~23 bits)
};
// Together: ~46 bits mantissa ≈ 30-32 decimal digits precision
// Compare: float32 = 23 bits ≈ 7 digits
//          float64 = 52 bits ≈ 16 digits
```

### Error-Free Transforms

**Two-Sum (Kahan algorithm):**
```c
// Compute a + b exactly as (sum, error)
dd two_sum(float a, float b) {
    float s = a + b;
    float a_prime = s - b;
    float b_prime = s - a_prime;
    float delta_a = a - a_prime;
    float delta_b = b - b_prime;
    float e = delta_a + delta_b;
    return dd{s, e};
}
```

**Two-Product (using FMA):**
```c
// Compute a × b exactly as (product, error)
dd two_prod(float a, float b) {
    float p = a * b;
    float e = fma(a, b, -p);  // error = (a*b) - round(a*b)
    return dd{p, e};
}
```

### Extended Precision Pipeline

**Standard (4 rounding points):**
```
Input (float32)
  → FFT (round) → complex64
  → Multiply (round) → complex64
  → IFFT (round) → float32
  → Bias (round) → float32
```

**Extended Precision (1 rounding point):**
```
Input (float32)
  → upcast to dd
  → FFT (all operations in dd, twiddles in dd)
  → Complex multiply (dd × dd in frequency domain)
  → IFFT (all operations in dd)
  → Scale & bias (dd)
  → Round ONCE to float32
  → Output
```

**Result**: Orders of magnitude lower drift for the same mathematical computation.

---

## Implementation Tiers

### Tier 0: Hygiene (Zero Cost, Maximum Benefit)

**Always implement these:**

1. **Single device, single dtype throughout**
   - Keep all tensors on one device (GPU)
   - No device hops, no dtype mixing

2. **Backend tensor scalars only**
   - Every constant: `mx.array(v, dtype=mx.float32)` or `torch.tensor(v, dtype=torch.float32, device=device)`
   - Use backend ops: `mx.multiply`, `torch.add` (not Python `*`, `+`)

3. **One normalization contract**
   - Pick exactly one FFT normalization convention
   - Document it in code
   - Enforce via linting (EmberCoach)

4. **No graph-breaking conversions**
   - No `.item()`, `.numpy()`, `float()`, `int()` in hot paths
   - Keep values as tensors
   - Use tensor comparisons

### Tier 1: Deterministic Numeric (DN) Mode

**For reproducible results:**

1. **Fixed reduction order**
   - Serial K-loop for small reductions
   - Deterministic pairwise tree for larger ones
   - No split-K or nondeterministic parallelism

2. **Explicit FMA control**
   - Use FMA only for error-free transforms (two_prod)
   - Keep mul/add separate elsewhere for reproducibility

3. **Fixed combine semantics**
   - Document TIME vs FREQ domain choice
   - Document SUM vs AVG choice
   - Enforce via HyperProfile configuration

### Tier 2: Extended Precision (EP) Mode

**For extreme stability:**

1. **Critical operations in double-double**
   - Complex multiply in frequency domain (biggest win, low cost)
   - Depthwise convolutions (cheap, high leverage)
   - Optional: FFT butterflies (heavier, for extreme requirements)

2. **Compensated summation (Kahan)**
   - In long reductions and dot products
   - Approximately 2× cost, but O(u) error instead of O(N·u)

3. **Profile-driven selection**
   - `ep_freqmul=true`: dd complex multiply
   - `ep_depthwise=true`: dd depthwise convolution
   - `ep_fft=true`: dd butterfly (heavier)

### Tier 3: Reference/Validation

**For debugging and verification:**

1. **Periodic rebase**
   - Every N steps, recompute with CPU float64 reference
   - Snap state to reference
   - Bounds drift over long runs

2. **Strict DFT for small cases**
   - O(L²) exact DFT with per-operation rounding
   - Debug mode only
   - Validates FFT implementations

---

## Results and Verification

### Before Hygiene (Mixed Conventions, Python Scalars)

- Relative error: ~0.99 (nearly 100% difference)
- Maximum absolute error: ~2.7
- Root cause: Mixed normalization conventions, Python scalar leakage

### After Tier 0 (Hygiene + Single Normalization)

- Relative error: ~4.8×10⁻⁷
- Mean squared error: ~10⁻¹³
- **Improvement: 2 million times better**
- Achieved through:
  - Unified FFT normalization
  - Backend tensor scalars only
  - No NumPy in compute paths
  - Single device throughout

### Expected with Tier 2 (Extended Precision for Critical Ops)

- Relative error: <10⁻¹⁰
- Mean squared error: <10⁻²⁰
- Near bit-exact for practical purposes
- Deterministic across runs

### Test Configurations

From `bert/tests/test_mlx_hyena_parity.py`:

| Config | Batch | d_model | seq_len | order | y_mse | y_rel |
|--------|-------|---------|---------|-------|-------|-------|
| 1 | 2 | 64 | 128 | 16 | 1.10×10⁻¹⁴ | 4.8×10⁻⁷ |
| 2 | 2 | 128 | 256 | 32 | 1.06×10⁻¹³ | 3.6×10⁻⁷ |
| 3 | 1 | 192 | 512 | 32 | 1.97×10⁻¹³ | 5.5×10⁻⁷ |

Kernel generation MSE: ~4×10⁻¹⁷ (effectively identical).

---

## Practical Usage Guide

### For Data Scientists and ML Engineers

1. **Use EmberCoach on your codebase:**
   ```bash
   python tools/embercoach.py your_module/
   python tools/embercoach.py --verbose your_module/  # detailed explanations
   ```

2. **Check FFT normalization:**
   - Read your framework's documentation
   - Apply exactly one 1/N across rfft/irfft
   - Document your choice in code comments

3. **Profile-driven precision:**
   - Use HyperProfiles to switch precision modes
   - Start with Tier 0 (always)
   - Add Tier 1 for long-running jobs
   - Add Tier 2 only if drift monitoring shows need

### For Kernel Developers

1. **Implement error-free primitives:**
   - two_sum, two_prod
   - dd arithmetic (add, mul, div)
   - Complex dd operations

2. **Make precision opt-in:**
   - Keep float32 fast path as default
   - Expose flags for DN and EP modes via HyperProfile
   - Allow per-operation precision control

3. **Test determinism:**
   - Same inputs yield same outputs (bit-exact)
   - Round-trip tests: ifft(fft(x)) ≈ x within tolerance
   - Energy conservation (Parseval's theorem)
   - Long-run drift monitoring

### HyperProfile Configuration

Profiles live in `bert/profiles/`:

**torch_like.json** (matches PyTorch reference exactly):
```json
{
  "bidir_space": "time",
  "bidir_combine": "sum",
  "fft_norm": "forward",
  "ep_freqmul": false,
  "ep_depthwise": false,
  "strict_kernels": false
}
```

**mlx_stable.json** (MLX default with stability enhancements):
```json
{
  "bidir_space": "time",
  "bidir_combine": "avg",
  "fft_norm": "backward",
  "ep_freqmul": false,
  "ep_depthwise": false,
  "strict_kernels": false
}
```

**Extended precision variant** (for long-running workloads):
```json
{
  "bidir_space": "time",
  "bidir_combine": "sum",
  "fft_norm": "forward",
  "ep_freqmul": true,
  "ep_depthwise": true,
  "strict_kernels": true
}
```

Select via environment variable:
```bash
export MLX_M2_PROFILE=torch_like
python your_script.py
```

---

## The Three Laws of GPU Precision

### Law 1: No Python Numerics in Tensor Math

Every constant is a backend tensor. Every operation is a backend operation. No exceptions except simple integer indexing.

**Rationale**: Python scalars are float64. Using them in float32 tensor math adds dtype conversions and extra rounding. Over billions of operations, this compounds to significant drift.

### Law 2: Exactly One Rounding Per Pipeline

Round once, at the final output boundary. Carry extended precision through critical segments when stability requirements demand it.

**Rationale**: Each rounding introduces approximately half-ULP error. Four rounding points compound errors faster than one. Extended precision (double-double) eliminates intermediate roundings.

### Law 3: One Device, One Dtype, One Normalization

No device hops, no dtype mixing, no convention surprises. Document your choices and enforce them through linting and testing.

**Rationale**: Device copies and dtype conversions introduce additional rounding and break kernel fusion. Inconsistent FFT normalization gives order-one amplitude errors.

---

## Resources and References

### Framework Documentation

- [PyTorch FFT Normalization](https://pytorch.org/docs/stable/generated/torch.fft.fft.html)
- [NumPy FFT (caution: promotes to float64)](https://numpy.org/doc/stable/reference/routines.fft.html)
- [Apple vDSP FFT Scaling](https://developer.apple.com/documentation/accelerate/vdsp)

### Academic Background

- [Convolution Theorem](https://en.wikipedia.org/wiki/Convolution_theorem)
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)
- [FFTW Accuracy](http://www.fftw.org/accuracy/)
- [IEEE-754 and FMA](https://docs.nvidia.com/cuda/floating-point/index.html)

### Our Implementation

- PyTorch source (external): `pytorch/aten/src/ATen/native/SpectralOps.cpp`
- MLX source (external): `mlx/fft.cpp`, `mlx/backend/cpu/fft.cpp`, `mlx/backend/metal/fft.cpp`
- Metal kernels: `experimental/metal_bitexact/`
- Parity tests: `bert/tests/test_mlx_hyena_parity.py`, `bert/tests/test_mlx_monarch_parity.py`
- Investigation tools: `experiments/lab/trace_*.py`

---

## Summary

**The Problem**: 25.9 trillion operations over 24 hours. Even 10⁻⁷ differences per operation compound to visible drift.

**The Cause**: Multiple rounding points, inconsistent normalization, Python scalar leakage, device hops, backend accumulation order differences.

**The Solution**:
- **Tier 0 (always)**: Backend tensor scalars, single device/dtype, one normalization contract
- **Tier 1 (reproducibility)**: Deterministic reduction order, fixed combine semantics
- **Tier 2 (extreme stability)**: Extended precision (double-double) for critical operations

**The Results**: 2 million times improvement in relative error through Tier 0 alone. Near bit-exact results achievable with Tier 2 for long-running workloads.

**Follow these principles, and your 300M ops/sec for 24 hours will be numerically stable and reproducible.**
