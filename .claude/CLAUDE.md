# MLX Metal Kernel Development Guide

**Hard-fought lessons from integrating Metal kernels into m2-bert-mlx**

## Core Principle: 100% Backend Tensors

MLX is **NOT** NumPy. Do not assume NumPy patterns work.

### Golden Rule
**Once you extract to Python scalars, the original GPU buffer is destroyed.**

```python
# WRONG - buffer destroyed
x = mx.array([1, 2, 3])
val = x[0].item()  # GPU→CPU copy, graph break
y = mx.array(val)  # New buffer, original x connection lost

# CORRECT - stay in MLX
x = mx.array([1, 2, 3])
y = x[0]  # Still an mx.array, lazy, same buffer family
```

## MLX Metal Kernel API (`mx.fast.metal_kernel`)

### 1. Kernel Source is Body-Only

The JIT generates function signatures automatically from `input_names` and `output_names`.

```python
# CORRECT - body only
_KERNEL_SOURCE = r"""
    uint tid = thread_position_in_grid.x;
    if (tid >= n) return;
    out[tid] = inp[tid] * 2.0f;
"""

# WRONG - don't include function signature
_KERNEL_SOURCE = r"""
kernel void my_kernel(device float* out [[buffer(0)]]) {  // BAD
    ...
}
"""
```

### 2. Compilation Pattern

```python
_KERNEL = None

_HEADER = """#include <metal_stdlib>
using namespace metal;
"""

_SOURCE = r"""
    // kernel body here
"""

def my_operation(x):
    global _KERNEL

    # Compile once on first use
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="my_kernel",
            input_names=["params", "x"],
            output_names=["y"],
            header=_HEADER,
            source=_SOURCE
        )

    # Dispatch
    (result,) = _KERNEL(
        inputs=[params, x],
        output_shapes=[(shape,)],
        output_dtypes=[mx.float32],
        grid=(num_groups, mx.array(1, dtype=mx.uint32), mx.array(1, dtype=mx.uint32)),
        threadgroup=(threads_per_group, mx.array(1, dtype=mx.uint32), mx.array(1, dtype=mx.uint32))
    )
    return result
```

### 3. Grid/Threadgroup Must Use MLX Scalars in Tuples

```python
# WRONG - Python arithmetic
tpg = 256
grid = ((total + tpg - 1) // tpg, 1, 1)  # Python scalars break buffer connections

# CORRECT - MLX ops, MLX scalars in tuples
tpg = mx.array(256, dtype=mx.uint32)
one = mx.array(1, dtype=mx.uint32)
total_mx = mx.array(total, dtype=mx.uint32)
num_groups = mx.divide(mx.add(total_mx, mx.subtract(tpg, one)), tpg)
grid = (num_groups, one, one)  # MLX scalars in tuple - GOOD
threadgroup = (tpg, one, one)
```

## No Python Arithmetic Operators

MLX does **not** overload `+, -, *, /, %` like NumPy does.

```python
# WRONG - Python operators
y = x * 2 + 1  # Triggers float64 promotion, breaks lazy graph
z = x / n      # Python scalar division

# CORRECT - Backend operations
two = mx.array(2.0, dtype=mx.float32)
one = mx.array(1.0, dtype=mx.float32)
y = mx.add(mx.multiply(x, two), one)

n_mx = mx.array(n, dtype=mx.float32)
z = mx.divide(x, n_mx)
```

## No Scalar Extraction

```python
# WRONG - Graph breaks
if x.max() > threshold:  # .max() returns mx.array, comparison converts to Python
    ...

# WRONG - Explicit extraction
val = float(x[0])  # Destroys buffer connection
val = x[0].item()  # Forces GPU→CPU copy

# CORRECT - Stay in MLX
threshold_mx = mx.array(threshold, dtype=x.dtype)
mask = mx.greater(x, threshold_mx)
result = mx.where(mask, x, mx.zeros_like(x))
```

## Complex64 Limitations

**Complex64 support on Metal GPU is incomplete as of 2025.**

Many operations are CPU-only or not yet implemented:
- `mx.sqrt()` on complex arrays (kernel not available)
- GPU scatter with complex64
- Some FFT operations fall back to CPU

### Workaround
Use `mx.multiply()` for complex operations instead of custom kernels:

```python
# Skip custom complex multiply kernel
# Just use MLX's built-in (works on GPU for basic ops)
result = mx.multiply(a_complex, b_complex)
```

## EmberCoach is the Truth

**Never assume code is clean until EmberCoach validates it.**

```bash
python tools/embercoach.py path/to/file.py
```

Common violations:
- `PYTHON-SCALAR-002`: Python numeric in tensor expression
- `ASSIGN-SCALAR-001`: Bare Python scalar in assignment
- `GRAPH-BREAK-001`: `.item()` breaks lazy graph
- `FFT-NORM-001`: FFT normalization needs checking

## Example: Depthwise Convolution Kernel

Working example from `metal_kernels.py`:

```python
_DEPTHWISE_SOURCE = r"""
    uint B = params[0];
    uint C = params[1];
    uint L = params[2];

    uint tid = thread_position_in_grid.x;
    uint total = B * C * L;
    if (tid >= total) return;

    uint b = tid / (C * L);
    uint rem = tid % (C * L);
    uint c = rem / L;
    uint t = rem % L;

    // Zero-pad access
    float v0 = (t >= 1) ? x[b * C * L + c * L + (t - 1)] : 0.0f;
    float v1 = x[b * C * L + c * L + t];
    float v2 = (t + 1 < L) ? x[b * C * L + c * L + (t + 1)] : 0.0f;

    float k0 = k[c * 3 + 0];
    float k1 = k[c * 3 + 1];
    float k2 = k[c * 3 + 2];

    // Fixed order for determinism
    float acc = v0 * k0;
    acc = fma(v1, k1, acc);
    acc = fma(v2, k2, acc);

    y[tid] = acc;
"""

def depthwise_conv_3tap(x, kernel):
    global _DEPTHWISE_KERNEL

    batch, channels, length = x.shape

    # Ensure float32
    x = x.astype(mx.float32)
    kernel = kernel.astype(mx.float32)

    if _DEPTHWISE_KERNEL is None:
        _DEPTHWISE_KERNEL = mx.fast.metal_kernel(
            name="depthwise3",
            input_names=["params", "x", "k"],
            output_names=["y"],
            header=_HEADER,
            source=_DEPTHWISE_SOURCE
        )

    # Params buffer
    params = mx.array([batch, channels, length], dtype=mx.uint32)

    # Grid calculation with MLX scalars
    batch_mx = mx.array(batch, dtype=mx.uint32)
    channels_mx = mx.array(channels, dtype=mx.uint32)
    length_mx = mx.array(length, dtype=mx.uint32)
    total_threads = mx.multiply(mx.multiply(batch_mx, channels_mx), length_mx)
    tpg = mx.array(256, dtype=mx.uint32)
    one = mx.array(1, dtype=mx.uint32)
    num_groups = mx.divide(mx.add(total_threads, mx.subtract(tpg, one)), tpg)
    grid = (num_groups, one, one)
    threadgroup = (tpg, one, one)

    (y,) = _DEPTHWISE_KERNEL(
        inputs=[params, x, kernel],
        output_shapes=[(batch, channels, length)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=threadgroup
    )

    return y
```

## Integration Pattern

When replacing Python loops with Metal kernels:

1. **Read existing code** - understand what it does
2. **Write kernel body** - no function signature
3. **Test kernel standalone** - small inputs, verify correctness
4. **Run EmberCoach** - fix all violations
5. **Integrate** - replace Python loop
6. **Test integration** - verify output matches original
7. **Run EmberCoach again** - ensure no new violations

## Common Mistakes (and Fixes)

### Mistake: Using .metal Files Directly
```python
# WRONG - can't load .metal files with full signatures
source = open("MyKernel.metal").read()
kernel = mx.fast.metal_kernel(source=source)  # Will fail
```

Fix: Extract body only, provide as inline Python string.

### Mistake: Python Integer Loop Indices
```python
# ACCEPTABLE - loop indices can be Python ints
for i in range(channels):  # OK - loop index
    ...

# WRONG - tensor operations with Python scalars
y = x * 0.5  # BAD - Python float
```

Fix: Use MLX operations: `y = mx.multiply(x, mx.array(0.5, dtype=mx.float32))`

### Mistake: Assuming NumPy Broadcasting
```python
# WRONG - shape extraction loses buffer connection
shape = x.shape
scale = mx.array(shape[0])  # New buffer, x connection lost
```

Fix: Keep operations in MLX graph.

## Testing Workflow

```bash
# 1. Test kernel standalone
python -c "
import mlx.core as mx
from mm_mlx.metal_kernels import depthwise_conv_3tap
x = mx.random.normal((2, 64, 128))
kernel = mx.random.normal((64, 3))
y = depthwise_conv_3tap(x, kernel)
mx.eval(y)
print('✓ Kernel works')
"

# 2. Run EmberCoach
python tools/embercoach.py bert/src/mm_mlx/metal_kernels.py

# 3. Test integration
python -c "
from mm_mlx.monarch_mixer_mlx import MonarchMixerSequenceMixing
import mlx.core as mx
mixer = MonarchMixerSequenceMixing(d_model=768, l_max=64)
x = mx.random.normal((2, 64, 768))
y, _ = mixer(x)
mx.eval(y)
print('✓ Integration works')
"
```

## Resources

- **EmberCoach**: `tools/embercoach.py` - Numerical precision validator
- **MLX Docs**: https://ml-explore.github.io/mlx/
- **xLSTM-metal reference**: `/Volumes/emberstuff/xLSTM-metal/` - Working kernel examples
- **Precision guide**: `docs/NUMERICAL_PRECISION_GUIDE.md`

---

**Remember**: When in doubt, run it. You can't learn Metal kernels by reading - you must compile, test, and iterate. EmberCoach will tell you the truth.
