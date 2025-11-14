#!/usr/bin/env python
"""
Block-Diagonal Multiply (legacy import shim).

The real implementation lives in :func:`mlx_ops.blockdiag_multiply`.  This file
re-exports that symbol under the original ``src.mm`` namespace so the canonical
model definitions do not need to change.  Keeping the forwarding layer also
gives us one place to document the design intent: all specialised kernels
reside in :mod:`mlx_ops`, while the rest of the codebase keeps its familiar
imports.
"""

from mlx_ops.blockdiag_multiply import blockdiag_multiply  # noqa: F401

__all__ = ['blockdiag_multiply']
