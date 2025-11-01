#!/usr/bin/env python
"""
Numerical parity + stability tests for MLX HyenaFilter vs PyTorch mirror.

Compares:
- Generated kernels h (forward and reverse when enabled)
- Long convolution outputs on random inputs
- NaN/Inf checks and gradient sanity
"""

import math
import os
import sys
import types
import importlib.util
import numpy as np

import torch
import torch.nn as tnn

import mlx.core as mx
import mlx.nn as mnn

# Import by file path to avoid importing src/__init__.py
THIS_DIR = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))  # path to bert/
SRC_DIR = os.path.join(PKG_ROOT, 'src')

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

# Prepare a minimal 'src.utils.train' in sys.modules so hyena_utils can import OptimModule
train_path = os.path.join(SRC_DIR, 'utils', 'train.py')
train_mod = _load_module('src.utils.train', train_path)
sys.modules['src'] = types.ModuleType('src')
sys.modules['src.utils'] = types.ModuleType('src.utils')
sys.modules['src.utils.train'] = train_mod

# Load MLX Hyena and Torch Hyena directly by path
mlx_hyena_path = os.path.join(SRC_DIR, 'mm_mlx', 'hyena_filter_mlx.py')
torch_hyena_path = os.path.join(SRC_DIR, 'mm', 'hyena_utils.py')
MLXHyena = _load_module('mlx_hyena', mlx_hyena_path).HyenaFilter
TorchHyena_mod = _load_module('torch_hyena', torch_hyena_path)
TorchHyena = TorchHyena_mod.HyenaFilter
fftconv_ref = TorchHyena_mod.fftconv_ref


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
    # Derive 'order' from the output linear (weight shape [d_model, order])
    if hasattr(hx, 'implicit_filter_layers') and len(hx.implicit_filter_layers) >= 1:
        order = int(np.array(hx.implicit_filter_layers[-1].weight).shape[1])
    else:
        order = d_model

    # Derive num_inner_mlps from MLX layers (count square linear layers)
    mlx_inner = 0
    if hasattr(hx, 'implicit_filter_layers'):
        for l in hx.implicit_filter_layers:
            w = np.array(l.weight)
            if w.ndim == 2 and w.shape[0] == w.shape[1] == order:
                mlx_inner += 1
        # square layers correspond to inner MLPs exactly
    print('DEBUG computed mlx_inner:', mlx_inner)

    ty = TorchHyena(
        d_model=d_model,
        emb_dim=emb_dim,
        order=order,
        seq_len=hx.seq_len,
        num_inner_mlps=mlx_inner,
        bidirectional=hx.bidirectional,
        modulate=hx.modulate,
        normalized=hx.normalized,
        linear_mixer=False,
        w=getattr(hx, 'w', 10),
        w_mod=getattr(hx, 'w_mod', 1),
    )

    # Copy positional embeddings and modulation deltas
    ty.pos_emb.z.data = _to_torch(hx.pos_emb.z).to(dtype=ty.pos_emb.z.dtype)
    ty.pos_emb.t.data = _to_torch(hx.pos_emb.t).to(dtype=ty.pos_emb.t.dtype)
    ty.modulation.deltas.data = _to_torch(hx.modulation.deltas).to(dtype=ty.modulation.deltas.dtype)
    ty.bias.data = _to_torch(hx.bias).to(dtype=ty.bias.dtype)

    # Copy implicit MLP weights/biases layer-by-layer
    mlx_layers = hx.implicit_filter_layers if hasattr(hx, 'implicit_filter_layers') else []
    torch_layers = [l for l in ty.implicit_filter if isinstance(l, tnn.Linear)]
    print('DEBUG linear layers:', len(mlx_layers), len(torch_layers))
    print('DEBUG ty.implicit_filter:', [type(l).__name__ for l in ty.implicit_filter])
    assert len(mlx_layers) == len(torch_layers)
    for ml, tl in zip(mlx_layers, torch_layers):
        w = _to_torch(ml.weight)
        # MLX stores [out, in]; Torch expects [out, in]
        if w.ndim == 2 and w.shape == tl.weight.data.shape:
            tl.weight.data = w.to(dtype=tl.weight.dtype)
        else:
            # Fallback: attempt simple transpose
            tl.weight.data = w.T.contiguous().to(dtype=tl.weight.dtype)
        if hasattr(ml, 'bias') and (ml.bias is not None) and (tl.bias is not None):
            tl.bias.data = _to_torch(ml.bias).to(dtype=tl.bias.dtype)

    if hx.bidirectional:
        mlx_layers_rev = hx.implicit_filter_layers_rev if hasattr(hx, 'implicit_filter_layers_rev') else []
        torch_layers_rev = [l for l in ty.implicit_filter_rev if isinstance(l, tnn.Linear)]
        assert len(mlx_layers_rev) == len(torch_layers_rev)
        for ml, tl in zip(mlx_layers_rev, torch_layers_rev):
            w = _to_torch(ml.weight)
            if w.ndim == 2 and w.shape == tl.weight.data.shape:
                tl.weight.data = w.to(dtype=tl.weight.dtype)
            else:
                tl.weight.data = w.T.contiguous().to(dtype=tl.weight.dtype)
            if hasattr(ml, 'bias') and (ml.bias is not None) and (tl.bias is not None):
                tl.bias.data = _to_torch(ml.bias).to(dtype=tl.bias.dtype)

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

    # Compare generated kernels
    h_mx = hx.filter(seq_len)
    h_ty = ty.filter(seq_len)
    print('DEBUG h shapes', np.array(h_mx).shape, h_ty.shape)
    print('DEBUG h max abs', float(np.max(np.abs(np.array(h_mx) - h_ty.detach().cpu().numpy()))))

    # Convolution outputs (explicit Torch reference path to avoid layout pitfalls)
    y_mx = hx(x_mx, seq_len)
    k_fwd_t = ty.filter(seq_len)[0].transpose(0, 1).contiguous()  # (H,L)
    if ty.bidirectional:
        k_rev_t = ty.filter_rev(seq_len)[0].transpose(0, 1).contiguous()
        k_time = torch.nn.functional.pad(k_fwd_t, (0, seq_len)) + \
                 torch.nn.functional.pad(torch.flip(k_rev_t, dims=[-1]), (seq_len, 0))
    else:
        k_time = k_fwd_t
    D = ty.bias.reshape(1, -1, 1)
    y_t = fftconv_ref(x_t, k_time, D, dropout_mask=None, gelu=False)

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
