#!/usr/bin/env python
"""
Shared block‑diagonal ops for MLX.

Implements a vectorized block‑diag multiply that mirrors src/mm/blockdiag_multiply.py.
"""

import mlx.core as mx


def blockdiag_multiply(x: mx.array, weight: mx.array, *, nblocks: int) -> mx.array:
    """
    x: (..., n)
    weight: (nblocks, q, n/nblocks)
    returns: (..., nblocks*q)
    """
    *batch, n = x.shape
    _, q, p = weight.shape
    assert n % nblocks == 0 and (n // nblocks) == p, "incompatible block sizes"
    B = 1
    for d in batch:
        B *= d
    x2 = x.reshape(B, n)
    x_blk = x2.reshape(B, nblocks, p)  # (B,K,P)
    y_blk = mx.einsum('bkp,kqp->bkq', x_blk, weight)  # (B,K,Q)
    y2 = y_blk.reshape(B, nblocks * q)
    return y2.reshape(*batch, nblocks * q)

