#!/usr/bin/env python
# Adapted from https://github.com/HazyResearch/fly/tree/master/src/models/layers
# Converted to MLX

import mlx.core as mx
from mlx_ops.einops import rearrange


def blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=2):
    """
    This implementation is slow but more likely to be correct.
    There are 3 implementations, which should all yield the same answer.

    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)

    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """
    if version not in [1, 2, 3]:
        raise NotImplementedError('version must be either 1, 2, or 3')

    batch, n = x.shape
    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape

    # Assertions - shape values are already Python ints
    assert k * p == n
    assert l * r == k * q

    # Use einops rearrange
    x_reshaped = rearrange(x, 'b (k p) -> b k p', k=k)

    if version == 1:  # Implementation 1 (only works for when k = q = p = l = s = r = sqrt(n))
        # Use MLX sqrt for the actual computation
        n_tensor = mx.array(n, dtype=mx.float32)
        sqrtn_tensor = mx.sqrt(n_tensor)
        sqrtn_rounded = mx.round(sqrtn_tensor)

        # For assertion only, we can use the fact that k should equal sqrtn
        # Check that all dimensions are equal and that n is a perfect square
        assert k == q == p == l == s == r, "All dimensions must be equal for version 1"
        assert k * k == n, "n must be a perfect square for version 1"

        result = mx.einsum('bkp,kqp,qlk->blq', x_reshaped, w1_bfly, w2_bfly)
        return mx.reshape(result, (batch, n))

    elif version == 2:  # Implementation 2
        out1 = mx.einsum('kqp,bkp->bkq', w1_bfly, x_reshaped)

        # Use einops for complex reshapes
        out1 = rearrange(rearrange(out1, 'b k q -> b (k q)'), 'b (r l) -> b l r', l=l)

        result = mx.einsum('lsr,blr->bsl', w2_bfly, out1)
        return mx.reshape(result, (batch, s * l))

    # Implementation 3: most likely to be correct, but it's the slowest
    elif version == 3:
        # Create block diagonal manually for w1
        # Each block is (q, p), total size (k*q, k*p) = (k*q, n)
        w1_dense = mx.zeros((k * q, k * p), dtype=w1_bfly.dtype)

        for i in range(k):
            row_start = i * q
            col_start = i * p
            w1_dense = mx.slice_update(
                w1_dense,
                w1_bfly[i],
                mx.array([row_start, col_start]),
                [0, 1]
            )

        # Linear is just matmul with transpose: F.linear(x, w) = x @ w.T
        out1 = mx.matmul(x, mx.transpose(w1_dense, (1, 0)))

        # Use einops rearrange
        out1 = rearrange(out1, 'b (r l) -> b (l r)', l=l)

        # Create block diagonal manually for w2
        # Each block is (s, r), total size (l*s, l*r)
        w2_dense = mx.zeros((l * s, l * r), dtype=w2_bfly.dtype)

        for i in range(l):
            row_start = i * s
            col_start = i * r
            w2_dense = mx.slice_update(
                w2_dense,
                w2_bfly[i],
                mx.array([row_start, col_start]),
                [0, 1]
            )

        out2 = mx.matmul(out1, mx.transpose(w2_dense, (1, 0)))

        # Use einops rearrange
        out2 = rearrange(out2, 'b (l s) -> b (s l)', l=l)

        return out2


def blockdiag_butterfly_multiply(x, w1_bfly, w2_bfly):
    """
    Optimized implementation using batched matrix multiplication.

    In MLX, gradients are automatic - no need for custom autograd.
    This implementation uses careful memory layout for fast performance.

    Arguments:
        x: (batch, n) or (*batch_shape, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)

    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """
    batch_shape = x.shape[:-1]
    n = x.shape[-1]

    # Compute batch_dim as product of batch dimensions using MLX ops
    batch_dim = mx.array(1, dtype=mx.int32)
    for dim in batch_shape:
        batch_dim = mx.multiply(batch_dim, mx.array(dim, dtype=mx.int32))

    k, q, p = w1_bfly.shape
    l, s, r = w2_bfly.shape

    assert k * p == n
    assert l * r == k * q

    # Reshape and transpose for batched matmul
    # x: (batch_dim, k, p) with batch along k dimension for matmul
    x_reshaped = mx.reshape(x, (batch_dim, k, p))
    x_reshaped = mx.transpose(x_reshaped, (1, 0, 2))  # (k, batch_dim, p)

    # First butterfly multiply: (k, batch_dim, p) @ (k, p, q) -> (k, batch_dim, q)
    # w1_bfly is (k, q, p), need to transpose last two dims
    w1_transposed = mx.transpose(w1_bfly, (0, 2, 1))  # (k, p, q)

    # Batched matmul
    out1 = mx.matmul(x_reshaped, w1_transposed)  # (k, batch_dim, q)

    # Reshape for second butterfly
    # out1: (batch_dim, k, q) -> (batch_dim, k*q) -> (batch_dim, r, l) -> (l, batch_dim, r)
    out1 = mx.transpose(out1, (1, 0, 2))  # (batch_dim, k, q)
    out1 = mx.reshape(out1, (batch_dim, k * q))
    out1 = mx.reshape(out1, (batch_dim, r, l))
    out1 = mx.transpose(out1, (2, 0, 1))  # (l, batch_dim, r)

    # Make contiguous for better performance
    out1 = mx.contiguous(out1)

    # Second butterfly multiply: (l, batch_dim, r) @ (l, r, s) -> (l, batch_dim, s)
    w2_transposed = mx.transpose(w2_bfly, (0, 2, 1))  # (l, r, s)

    out2 = mx.matmul(out1, w2_transposed)  # (l, batch_dim, s)

    # Reshape to final output
    # (l, batch_dim, s) -> (batch_dim, l, s) -> (batch_dim, s, l) -> (*batch_shape, s*l)
    out2 = mx.transpose(out2, (1, 0, 2))  # (batch_dim, l, s)
    out2 = mx.transpose(out2, (0, 2, 1))  # (batch_dim, s, l)

    out_shape = list(batch_shape) + [s * l]
    out2 = mx.reshape(out2, out_shape)

    return out2


# Alias for compatibility
blockdiag_butterfly_multiply.apply = blockdiag_butterfly_multiply
