"""
Structured Linear Utilities (Torch import compatibility layer).

The historical MosaicML BERT code imported `StructuredLinear` and
`BlockdiagSparsityConfig` from ``src.mm.structured_linear``.  The real MLX
implementations now live in :mod:`mlx_ops.blockdiag_linear`, where they share
code with the Monarch mixer and other MLX backends.  This module exists purely
to keep the canonical model topology untouched: it forwards the old import path
to the new MLX-native helpers and documents the architectural trade-offs.

Design choices
==============
* Maintain backwards-compatible module paths (`src.mm.*`) so higher level code
  and downstream notebooks do not need to change their imports.
* Re-export the MLX implementations unmodified; no proxy wrappers or runtime
  checks are introduced.  This keeps the execution surface identical to the
  shared MLX library and avoids divergence.
* Provide a single place to describe how block-diagonal sparsity layouts work,
  while keeping all math kernels inside :mod:`mlx_ops`.
"""

from mlx_ops.blockdiag_linear import (
    BlockdiagSparsityConfig as _BlockdiagSparsityConfig,
    StructuredLinear as _StructuredLinear,
)

# Re-exported symbols for legacy imports.
StructuredLinear = _StructuredLinear
BlockdiagSparsityConfig = _BlockdiagSparsityConfig

__all__ = ['StructuredLinear', 'BlockdiagSparsityConfig']
