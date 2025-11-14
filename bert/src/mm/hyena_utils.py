"""
Hyena Filter (Torch import compatibility + thin wrappers).

Historically the MosaicML code imported :class:`HyenaFilter` and the reference
FFT convolution helper (:func:`fftconv_ref`) from ``src.mm.hyena_utils``.  The
true MLX implementation now lives in :mod:`mlx_ops.hyena_filter`.  This module
keeps the old import surface stable while documenting how the MLX backend is
structured:

* All heavy lifting (parameter initialisation, FFT kernels, Metal dispatch) is
  implemented once inside :mod:`mlx_ops.hyena_filter`.
* This shim simply re-exports :class:`HyenaFilter` and exposes a convenience
  wrapper :func:`fftconv_ref` for tooling/tests that still expect the legacy
  helper.
* No try/except fallbacks are used—the code is MLX-only and assumes the shared
  MLX library is available.
"""

import mlx.core as mx

from mlx_ops.hyena_filter import HyenaFilter as _HyenaFilter
from mlx_ops.hyena_filter import _hyena_fft_conv as _mlx_fftconv


def fftconv_ref(u, k_time, D, dropout_mask=None, gelu=False, k_rev=None, flashfft=None):
    """
    Mirror the legacy Torch helper by calling the shared MLX FFT convolution.

    Args:
        u: `(batch, d_model, seqlen)` input activations
        k_time: `(d_model, seqlen)` forward-time kernel
        D: `(1, d_model, 1)` diagonal term
        dropout_mask: optional `(batch,)` mask
        gelu: whether to apply GELU inside the FFT helper (defaults False)
        k_rev: optional reverse-time kernel; if supplied we sum it with `k_time`
        flashfft: kept for API compatibility; ignored in MLX
    """
    seqlen = u.shape[-1]
    kernel = k_time
    if k_rev is not None:
        kernel = kernel + mx.flip(k_rev, axis=-1)
    return _mlx_fftconv(u, kernel, D, seqlen, gelu=gelu, dropout_mask=dropout_mask)


HyenaFilter = _HyenaFilter

__all__ = ['HyenaFilter', 'fftconv_ref']
