"""
Block-Diagonal Linear Layer (Torch import compatibility layer).

The shared MLX implementation of block-diagonal linear layers lives in
:mod:`mlx_ops.blockdiag_linear`.  This module simply re-exports those classes
under the historical ``src.mm.blockdiag_linear`` namespace so that the rest of
the BERT codebase can remain close to its PyTorch roots.  All compute, Metal
kernels, and parameter initialisation logic live in the shared MLX library.

Design choices
==============
* No proxy classes: we re-export the MLX versions directly to avoid subtle
  behavioural differences or duplicated code paths.
* Rich documentation stays in this module to explain *why* the abstractions
  exist, while the implementation details (padding, Kaiming init, Metal
  kernels) stay in :mod:`mlx_ops`.
* Future architectural experiments (e.g., custom sparsity) should land in the
  shared MLX library and will automatically be available here.
"""

from mlx_ops.blockdiag_linear import BlockdiagLinear as _BlockdiagLinear

BlockdiagLinear = _BlockdiagLinear

__all__ = ['BlockdiagLinear']
