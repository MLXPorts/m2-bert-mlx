#!/usr/bin/env python
"""
Block-diagonal matrix multiplication in MLX using xLSTM Metal GEMM kernels

Port of blockdiag_multiply.py to MLX with Metal acceleration
"""

import mlx.core as mx
import mlx.nn as nn
import sys
import os

# Add xLSTM kernels to path
xlstm_kernels_path = '/Volumes/emberstuff/xLSTM/experimental_kernels/mlx_fast_kernels'
if xlstm_kernels_path not in sys.path:
    sys.path.insert(0, xlstm_kernels_path)

from gemm_kernels import gemm_av, gemm_at_b


def blockdiag_multiply_reference(x, weight):
    """
    Reference implementation (slow but correct)

    Arguments:
        x: (..., n)
        weight: (nblocks, q, n / nblocks)
    Outputs:
        out: (..., nblocks * q)
    """
    n = x.shape[-1]
    nblocks, q, p = weight.shape
    assert nblocks * p == n, f"Shape mismatch: {nblocks} * {p} != {n}"

    # Reshape x to separate blocks
    batch_shape = x.shape[:-1]
    x_reshaped = x.reshape(*batch_shape, nblocks, p)

    # Apply each block separately
    outputs = []
    for i in range(nblocks):
        # x_block: (..., p)
        # weight[i]: (q, p)
        x_block = x_reshaped[..., i, :]

        # Matrix multiply: (..., p) @ (p, q) -> (..., q)
        out_block = x_block @ weight[i].T
        outputs.append(out_block)

    # Stack and reshape: (..., nblocks, q) -> (..., nblocks * q)
    result = mx.stack(outputs, axis=-2)
    result = result.reshape(*batch_shape, nblocks * q)

    return result


def blockdiag_multiply_gemm(x, weight):
    """
    Fast implementation using xLSTM GEMM Metal kernels

    Arguments:
        x: (..., n)
        weight: (nblocks, q, n / nblocks)
    Outputs:
        out: (..., nblocks * q)

    Uses Metal-accelerated GEMM for each block
    """
    n = x.shape[-1]
    nblocks, q, p = weight.shape
    assert nblocks * p == n, f"Shape mismatch: {nblocks} * {p} != {n}"

    batch_shape = x.shape[:-1]
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim

    # Reshape to (batch, nblocks, p)
    x_reshaped = x.reshape(batch_size, nblocks, p)

    # Process each block with GEMM kernel
    outputs = []
    for i in range(nblocks):
        # x_block: (batch, p)
        # weight[i]: (q, p)
        x_block = x_reshaped[:, i, :]  # (batch, p)

        # Transpose for GEMM: weight[i].T: (p, q)
        # GEMM: (p, q) @ (p, batch)^T = (p, q) @ (batch, p)
        # We want: (batch, p) @ (q, p)^T = (batch, q)

        # Use gemm_av: (A: q×p, v: p×batch) -> (q×batch)
        W_block = weight[i]  # (q, p)
        x_block_T = x_block.T  # (p, batch)

        result = gemm_av(W_block, x_block_T)  # (q, batch)
        result = result.T  # (batch, q)

        outputs.append(result)

    # Stack: (batch, nblocks, q)
    result = mx.stack(outputs, axis=1)

    # Reshape back: (..., nblocks * q)
    result = result.reshape(*batch_shape, nblocks * q)

    return result


def blockdiag_multiply(x, weight, use_metal=False):
    """
    Block-diagonal matrix multiplication

    Arguments:
        x: (..., n) - Input tensor
        weight: (nblocks, q, n / nblocks) - Block-diagonal weight matrix
        use_metal: Whether to use Metal GEMM kernels (forward only, no gradients)

    Returns:
        out: (..., nblocks * q) - Output tensor

    Note:
        Metal GEMM kernels are faster but don't support gradients yet.
        Use use_metal=False for training (default).

    Example:
        >>> x = mx.random.normal((4, 768))
        >>> weight = mx.random.normal((4, 192, 192))  # 4 blocks of 192x192
        >>> out = blockdiag_multiply(x, weight)
        >>> out.shape
        (4, 768)  # 4 blocks * 192 = 768
    """
    # Use reference implementation for now (supports gradients)
    # TODO: Implement custom vjp for Metal GEMM
    return blockdiag_multiply_reference(x, weight)


class BlockDiagLinear(nn.Module):
    """
    Linear layer with block-diagonal weight matrix

    More efficient than dense layer: O(n * q) instead of O(n^2)

    Args:
        in_features: Input dimension (n)
        out_features: Output dimension (nblocks * q)
        nblocks: Number of diagonal blocks
        bias: Whether to include bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        nblocks: int,
        bias: bool = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.nblocks = nblocks

        assert in_features % nblocks == 0, \
            f"in_features ({in_features}) must be divisible by nblocks ({nblocks})"
        assert out_features % nblocks == 0, \
            f"out_features ({out_features}) must be divisible by nblocks ({nblocks})"

        self.block_in = in_features // nblocks  # p
        self.block_out = out_features // nblocks  # q

        # Weight: (nblocks, q, p)
        self.weight = mx.random.normal(
            (nblocks, self.block_out, self.block_in)
        ) * 0.02

        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

    def __call__(self, x):
        """
        Forward pass

        Args:
            x: (..., in_features)
        Returns:
            out: (..., out_features)
        """
        # Use reference for training (supports gradients)
        out = blockdiag_multiply(x, self.weight, use_metal=False)

        if self.bias is not None:
            out = out + self.bias

        return out


def test_blockdiag_multiply():
    """Test block-diagonal multiplication"""
    print("Testing block-diagonal multiply (MLX + Metal GEMM)...")
    print()

    # Test 1: Simple case
    batch_size = 4
    n = 768
    nblocks = 4
    q = 192
    p = 192  # n // nblocks

    x = mx.random.normal((batch_size, n))
    weight = mx.random.normal((nblocks, q, p))

    print(f"Input: {x.shape}")
    print(f"Weight: {weight.shape} ({nblocks} blocks of {q}x{p})")

    # Reference implementation
    out_ref = blockdiag_multiply_reference(x, weight)
    print(f"Output (reference): {out_ref.shape}")

    # Metal GEMM implementation
    out_gemm = blockdiag_multiply_gemm(x, weight)
    print(f"Output (Metal GEMM): {out_gemm.shape}")

    # Check correctness
    diff = mx.abs(out_ref - out_gemm).max()
    print(f"Max difference: {diff.item():.6e}")

    if diff < 1e-4:  # Relaxed tolerance for float32
        print("✅ Results match (within tolerance)!")
    else:
        print("❌ Results differ significantly!")

    print()

    # Test 2: BlockDiagLinear layer
    print("Testing BlockDiagLinear layer...")

    layer = BlockDiagLinear(
        in_features=768,
        out_features=768,
        nblocks=4,
        bias=True
    )

    x_test = mx.random.normal((2, 768))
    out = layer(x_test)

    print(f"Layer input: {x_test.shape}")
    print(f"Layer output: {out.shape}")
    print(f"Parameters: {layer.weight.shape} + bias {layer.bias.shape if layer.bias is not None else None}")

    # Check gradient computation
    def loss_fn(layer):
        out = layer(x_test)
        return mx.mean(out ** 2)

    loss_and_grad = nn.value_and_grad(layer, loss_fn)
    loss, grads = loss_and_grad(layer)

    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient keys: {list(grads.keys())}")

    if 'weight' in grads:
        print(f"Weight gradient shape: {grads['weight'].shape}")
        print("✅ Gradient computation works!")
    else:
        print("❌ No weight gradient!")

    print()

    # Test 3: Benchmark
    print("Benchmarking Metal GEMM vs reference...")
    import time

    x_bench = mx.random.normal((128, 768))
    weight_bench = mx.random.normal((4, 192, 192))

    # Warmup
    for _ in range(10):
        _ = blockdiag_multiply_reference(x_bench, weight_bench)
        _ = blockdiag_multiply_gemm(x_bench, weight_bench)

    # Benchmark reference
    mx.eval(x_bench)  # Ensure evaluated
    start = time.time()
    for _ in range(100):
        out_ref = blockdiag_multiply_reference(x_bench, weight_bench)
        mx.eval(out_ref)
    ref_time = (time.time() - start) / 100

    # Benchmark Metal GEMM
    mx.eval(x_bench)
    start = time.time()
    for _ in range(100):
        out_gemm = blockdiag_multiply_gemm(x_bench, weight_bench)
        mx.eval(out_gemm)
    gemm_time = (time.time() - start) / 100

    print(f"Reference time: {ref_time*1000:.3f} ms")
    print(f"Metal GEMM time: {gemm_time*1000:.3f} ms")
    print(f"Speedup: {ref_time/gemm_time:.2f}x")
    print()

    print("✅ All tests complete!")


if __name__ == '__main__':
    test_blockdiag_multiply()
