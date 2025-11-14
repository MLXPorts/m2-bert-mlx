"""
Legacy ``src.mm`` namespace (MLX-backed import shims).

The original MosaicML code imported low-level layers from ``src.mm.*``.
All heavy implementations now live in :mod:`mlx_ops`, so this package’s job is
to keep the import paths stable while pointing to the MLX-native code.  Each
submodule contains a long design doc explaining how the abstraction maps onto
MLX kernels.  Keeping the shims in one place lets the canonical BERT model
mirror the PyTorch reference with minimal diffs while still benefiting from the
shared MLX library.
"""
