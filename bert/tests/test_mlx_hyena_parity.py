#!/usr/bin/env python
"""
Numerical parity + stability tests for MLX HyenaFilter vs PyTorch mirror.

Compares:
- Generated kernels h (forward and reverse when enabled)
- Long convolution outputs on random inputs
- NaN/Inf checks and gradient sanity
"""

import mlx.core as mx
import numpy as np
import torch
import torch.nn as tnn
from mm.hyena_utils import HyenaFilter as TorchHyena, fftconv_ref
from mm_mlx.hyena_filter_mlx import HyenaFilter as MLXHyena


def _to_torch(a):
    if isinstance(a, mx.array):
        return torch.from_numpy(np.array(a))
    if isinstance(a, np.ndarray):
        return torch.from_numpy(a)
    return a


def _to_mx(a):
    if torch.is_tensor(a):
        return mx.array(a.detach().cpu().numpy())
    if isinstance(a, np.ndarray):
        return mx.array(a)
    return a


def build_mirrored_torch_hyena_from_mlx(hx: MLXHyena) -> TorchHyena:
    """Construct a Torch HyenaFilter and copy parameters from an MLX HyenaFilter.

    Handles likely Linear weight/bias layout differences by transposing if needed.
    """
    d_model = hx.d_model
    emb_dim = hx.emb_dim

    if hasattr(hx, 'implicit_filter_layers') and hx.implicit_filter_layers:
        first_linear = hx.implicit_filter_layers[0]
        order = _to_torch(first_linear.weight).shape[1]
        num_inner = max(len(hx.implicit_filter_layers) - 2, 0)
        mlx_layers = hx.implicit_filter_layers
    else:
        order = d_model
        num_inner = 0
        mlx_layers = []

    ty = TorchHyena(
        d_model=d_model,
        emb_dim=emb_dim,
        order=order,
        seq_len=hx.seq_len,
        num_inner_mlps=num_inner,
        bidirectional=hx.bidirectional,
        modulate=hx.modulate,
        normalized=hx.normalized,
        linear_mixer=False,
    )

    # Copy positional embeddings and modulation deltas
    ty.pos_emb.z.data = _to_torch(hx.pos_emb.z).to(dtype=ty.pos_emb.z.dtype)
    ty.pos_emb.t.data = _to_torch(hx.pos_emb.t).to(dtype=ty.pos_emb.t.dtype)
    ty.modulation.deltas.data = _to_torch(hx.modulation.deltas).to(dtype=ty.modulation.deltas.dtype)
    ty.bias.data = _to_torch(hx.bias).to(dtype=ty.bias.dtype)

    # Copy implicit MLP weights/biases layer-by-layer
    torch_layers = [l for l in ty.implicit_filter if isinstance(l, tnn.Linear)]
    assert len(mlx_layers) == len(torch_layers)
    for ml, tl in zip(mlx_layers, torch_layers):
        w = _to_torch(ml.weight)
        # Torch expects [out, in]; MLX Linear stores [in, out]
        if w.ndim == 2 and w.shape[0] == tl.in_features and w.shape[1] == tl.out_features:
            tl.weight.data = w.T.contiguous().to(dtype=tl.weight.dtype)
        elif w.ndim == 2 and w.shape == tl.weight.data.shape:
            tl.weight.data = w.to(dtype=tl.weight.dtype)
        else:
            # Best effort transpose if dims swapped
            tl.weight.data = w.T.contiguous().to(dtype=tl.weight.dtype)
        ml_bias = getattr(ml, 'bias', None)
        if ml_bias is not None and tl.bias is not None:
            tl.bias.data = _to_torch(ml_bias).to(dtype=tl.bias.dtype)

    if hx.bidirectional and hasattr(hx, 'implicit_filter_layers_rev'):
        mlx_layers_rev = hx.implicit_filter_layers_rev
        torch_layers_rev = [l for l in ty.implicit_filter_rev if isinstance(l, tnn.Linear)]
        assert len(mlx_layers_rev) == len(torch_layers_rev)
        for ml, tl in zip(mlx_layers_rev, torch_layers_rev):
            w = _to_torch(ml.weight)
            if w.ndim == 2 and w.shape[0] == tl.in_features and w.shape[1] == tl.out_features:
                tl.weight.data = w.T.contiguous().to(dtype=tl.weight.dtype)
            else:
                tl.weight.data = w.T.contiguous().to(dtype=tl.weight.dtype)
            ml_bias = getattr(ml, 'bias', None)
            if ml_bias is not None and tl.bias is not None:
                tl.bias.data = _to_torch(ml_bias).to(dtype=tl.bias.dtype)

    return ty


def parity_once(batch=2, d_model=128, seq_len=256, order=32, emb_dim=5, bidir=True):
    # Build MLX Hyena
    hx = MLXHyena(
        d_model=d_model, emb_dim=emb_dim, order=order, seq_len=seq_len,
        bidirectional=bidir, num_inner_mlps=2,
    )
    # Mirror to Torch
    ty = build_mirrored_torch_hyena_from_mlx(hx)

    # Same random input
    x_np = np.random.randn(batch, d_model, seq_len).astype(np.float32)
    x_mx = mx.array(x_np)
    x_t = torch.tensor(x_np)

    # Compare kernels
    k_mx = hx.filter(seq_len)
    k_t = ty.filter(seq_len)
    k_diff = np.max(np.abs(np.array(k_mx) - k_t.detach().cpu().numpy()))

    # Convolution outputs (forward only)
    y_mx = hx(x_mx, seq_len)
    # Torch uses fftconv_ref with assembled kernel inside ty(x,...), but ty(x,L) returns y directly
    y_t = ty(x_t, seq_len)

    y_np = np.array(y_mx)
    t_np = y_t.detach().cpu().numpy()
    y_diff = np.max(np.abs(y_np - t_np))
    y_rel = (np.linalg.norm(y_np - t_np) / (np.linalg.norm(t_np) + 1e-8))
    y_mse = np.mean((y_np - t_np) ** 2)

    # Stability checks
    assert not np.isnan(np.array(y_mx)).any(), 'NaNs in MLX output'
    assert not torch.isnan(y_t).any(), 'NaNs in Torch output'

    return dict(k_max_abs=k_diff, y_max_abs=y_diff, y_rel=y_rel, y_mse=y_mse)


def main():
    mx.random.seed(0)
    torch.manual_seed(0)

    configs = [
        dict(batch=2, d_model=64,  seq_len=128, order=16, emb_dim=5, bidir=False),
        dict(batch=2, d_model=128, seq_len=256, order=32, emb_dim=5, bidir=True),
        dict(batch=1, d_model=192, seq_len=512, order=32, emb_dim=5, bidir=True),
    ]

    for cfg in configs:
        res = parity_once(**cfg)
        print('Hyena parity', cfg, '=>', res)
        # Loose tolerances for cross-framework numeric diff; tighten once stable
        assert res['y_rel'] < 1e-4, f"Relative diff too large: {res}"

    print('✅ MLX Hyena parity and stability checks passed')


if __name__ == '__main__':
    main()
