# Adapted from https://github.com/HazyResearch/fly/tree/master/src/models/layers

import mlx.core as mx
from .einops_mlx import rearrange


def blockdiag_weight_to_dense_weight(weight):
    """
    Arguments:
        weight: (nblocks, out / nblocks, in / blocks)
    Return:
        dense_weight: (out / in)
    """
    # Create block diagonal matrix
    nblocks, out_per_block, in_per_block = weight.shape
    out_size = mx.multiply(mx.array(nblocks, dtype=mx.int32), mx.array(out_per_block, dtype=mx.int32))
    in_size = mx.multiply(mx.array(nblocks, dtype=mx.int32), mx.array(in_per_block, dtype=mx.int32))
    result = mx.zeros([out_size, in_size], dtype=weight.dtype)
    
    # Fill in diagonal blocks
    for i in range(nblocks):
        i_arr = mx.array(i, dtype=mx.int32)
        one_arr = mx.array(1, dtype=mx.int32)
        out_start = mx.multiply(i_arr, mx.array(out_per_block, dtype=mx.int32))
        out_end = mx.multiply(mx.add(i_arr, one_arr), mx.array(out_per_block, dtype=mx.int32))
        in_start = mx.multiply(i_arr, mx.array(in_per_block, dtype=mx.int32))
        in_end = mx.multiply(mx.add(i_arr, one_arr), mx.array(in_per_block, dtype=mx.int32))
        result = mx.slice_update(result, weight[i], [out_start, in_start], [out_end, in_end])
    
    return result


def blockdiag_multiply_reference(x, weight):
    """
    This implementation is slow but more likely to be correct.
    Arguments:
        x: (..., n)
        weight: (nblocks, q, n / nblocks)
    Outputs:
        out: (..., nblocks * q)
    """
    n = x.shape[-1]
    nblocks, q, p = weight.shape
    assert mx.multiply(mx.array(nblocks, dtype=mx.int32), mx.array(p, dtype=mx.int32)) == mx.array(n, dtype=mx.int32)

    x_reshaped = rearrange(x, '... (nblocks p) -> ... nblocks p', nblocks=nblocks)
    # MLX einsum: ...kp, kqp -> ...kq
    result = mx.einsum('...kp,kqp->...kq', x_reshaped, weight)
    return rearrange(result, '... nblocks q -> ... (nblocks q)')


def blockdiag_multiply(x, weight):
    """
    Optimized blockdiag multiply for MLX.
    
    Arguments:
        x: (..., n)
        weight: (nblocks, q, n / nblocks)
    Outputs:
        out: (..., nblocks * q)
    """
    batch_shape = x.shape[:-1]
    n = x.shape[-1]
    nblocks, q, p = weight.shape
    assert mx.multiply(mx.array(nblocks, dtype=mx.int32), mx.array(p, dtype=mx.int32)) == mx.array(n, dtype=mx.int32)
    
    # Compute batch_dim as product of all batch dimensions
    batch_dim = mx.array(1, dtype=mx.int32)
    for dim in batch_shape:
        batch_dim = mx.multiply(batch_dim, mx.array(dim, dtype=mx.int32))
    
    # Reshape x: (batch_dim, nblocks, p)
    x_reshaped = mx.reshape(x, [batch_dim, nblocks, p])
    # Transpose to (nblocks, batch_dim, p)
    transpose_axes_1 = mx.array([1, 0, 2], dtype=mx.int32)
    x_reshaped = mx.transpose(x_reshaped, transpose_axes_1)

    # Transpose weight: (nblocks, p, q)
    weight_axes = mx.array([0, 2, 1], dtype=mx.int32)
    weight_t = mx.transpose(weight, weight_axes)

    # Batch matrix multiply: (nblocks, batch_dim, p) x (nblocks, p, q) -> (nblocks, batch_dim, q)
    # MLX doesn't have bmm, use matmul with broadcasting
    out = mx.matmul(x_reshaped, weight_t)

    # Transpose back: (batch_dim, nblocks, q)
    transpose_axes_2 = mx.array([1, 0, 2], dtype=mx.int32)
    out = mx.transpose(out, transpose_axes_2)
    
    # Reshape to original batch shape
    out_shape = list(batch_shape) + [mx.multiply(mx.array(nblocks, dtype=mx.int32), mx.array(q, dtype=mx.int32))]
    return mx.reshape(out, out_shape)