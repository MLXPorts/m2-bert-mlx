"""
Monarch Mixer Sequence Mixing (Torch import compatibility layer).

The full MLX implementation lives in :mod:`mlx_ops.monarch_mixer`.  This module
simply re-exports :class:`MonarchMixerSequenceMixing` under the legacy
``src.mm`` namespace so BERT’s model definitions keep their minimal diff from
the PyTorch originals.  All convolution kernels, Hyena filters, and Metal
optimisations are handled by the shared MLX library.
"""

from mlx_ops.monarch_mixer import MonarchMixerSequenceMixing as _MonarchMixerSequenceMixing

MonarchMixerSequenceMixing = _MonarchMixerSequenceMixing

__all__ = ['MonarchMixerSequenceMixing']
