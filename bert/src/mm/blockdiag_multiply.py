#!/usr/bin/env python
"""
MLX wrapper for block-diagonal multiply.

Inference-only path that re-exports the MLX implementation.
"""

from mlx_ops.blockdiag_multiply import blockdiag_multiply  # noqa: F401

__all__ = ['blockdiag_multiply']
