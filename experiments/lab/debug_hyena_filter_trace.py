#!/usr/bin/env python
import os, sys, importlib.util, types
import numpy as np
import torch
import torch.nn as tnn
import mlx.core as mx
import mlx.nn as mnn

THIS_DIR = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
SRC_DIR = os.path.join(PKG_ROOT, 'src')

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

# Prepare torch-side OptimModule import shim
train_path = os.path.join(SRC_DIR, 'utils', 'train.py')
sys.modules['src'] = types.ModuleType('src')
sys.modules['src.utils'] = types.ModuleType('src.utils')
sys.modules['src.utils.train'] = load('src.utils.train', train_path)

mlx_hyena = load('mlx_hyena', os.path.join(SRC_DIR, 'mm_mlx', 'hyena_filter_mlx.py'))
torch_hyena_mod = load('torch_hyena', os.path.join(SRC_DIR, 'mm', 'hyena_utils.py'))

HyenaMLX = mlx_hyena.HyenaFilter
HyenaTorch = torch_hyena_mod.HyenaFilter

def to_t(a):
    if isinstance(a, mx.array):
        return torch.from_numpy(np.array(a))
    if isinstance(a, np.ndarray):
        return torch.from_numpy(a)
    return a

def to_mx(a):
    if torch.is_tensor(a):
        return mx.array(a.detach().cpu().numpy())
    if isinstance(a, np.ndarray):
        return mx.array(a)
    return a

def _np(a):
    if isinstance(a, mx.array):
        return np.array(a)
    if torch.is_tensor(a):
        return a.detach().cpu().numpy()
    return np.array(a)

def maxabs(a, b):
    return float(np.max(np.abs(_np(a) - _np(b))))

def mse(a, b):
    aa = _np(a); bb = _np(b)
    return float(np.mean((aa - bb) ** 2))

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    d_model, emb_dim, order, L = 128, 5, 32, 256
    hx = HyenaMLX(d_model=d_model, emb_dim=emb_dim, order=order, seq_len=L, bidirectional=False, num_inner_mlps=2)
    ty = HyenaTorch(d_model=d_model, emb_dim=emb_dim, order=order, seq_len=L, num_inner_mlps=2, bidirectional=False,
                    modulate=hx.modulate, normalized=hx.normalized, linear_mixer=False,
                    w=getattr(hx, 'w', 10), w_mod=getattr(hx, 'w_mod', 1))

    # Mirror weights layer-by-layer
    mlx_layers = getattr(hx, 'implicit_filter_layers', getattr(hx, 'implicit_linears', []))
    torch_layers = [l for l in ty.implicit_filter if isinstance(l, tnn.Linear)]
    assert len(mlx_layers) == len(torch_layers)
    for ml, tl in zip(mlx_layers, torch_layers):
        w = to_t(ml.weight)
        if w.ndim == 2 and w.shape == tl.weight.data.shape:
            tl.weight.data = w.to(dtype=tl.weight.dtype)
        else:
            tl.weight.data = w.T.contiguous().to(dtype=tl.weight.dtype)
        if hasattr(ml, 'bias') and (ml.bias is not None) and (tl.bias is not None):
            tl.bias.data = to_t(ml.bias).to(dtype=tl.bias.dtype)

    # 1) Positional embeddings
    z_mx, t_mx = hx.pos_emb(L)
    z_t, t_t = ty.pos_emb(L)
    print('z maxabs:', maxabs(z_mx, z_t), 'mse:', mse(z_mx, z_t))
    print('t maxabs:', maxabs(t_mx, t_t), 'mse:', mse(t_mx, t_t))

    # 2) Implicit filter end-to-end
    if hasattr(hx, 'implicit_linears'):
        x = hx.implicit_linears[0](z_mx)
        for i in range(len(hx.implicit_sins) - 1):
            x = hx.implicit_sins[i](x)
            x = hx.implicit_linears[i + 1](x)
        x = hx.implicit_sins[-1](x)
        hpre_mx = hx.implicit_linears[-1](x)
    else:
        hpre_mx = hx.implicit_filter(z_mx)
    hpre_t = ty.implicit_filter(to_t(z_mx)).detach()
    print('implicit_filter out maxabs:', maxabs(hpre_mx, hpre_t), 'mse:', mse(hpre_mx, hpre_t))

    # 2b) Linear-only pass (skip Sin) to detect missing nonlinearity on MLX side
    x_mx = z_mx
    for lin in mlx_layers:
        x_mx = lin(x_mx)
    x_t = to_t(z_mx)
    for lin in torch_layers:
        x_t = lin(x_t)
    print('linear-only MLX vs Torch maxabs:', maxabs(x_mx, x_t), 'mse:', mse(x_mx, x_t))
    print('linear-only vs implicit_filter (MLX) maxabs:', maxabs(x_mx, hpre_mx), 'mse:', mse(x_mx, hpre_mx))
    print('linear-only vs implicit_filter (Torch) maxabs:', maxabs(x_t, hpre_t), 'mse:', mse(x_t, hpre_t))

    # 3) Modulation compare
    h_mx = hx.modulation(t_mx, hpre_mx)
    h_t = ty.modulation(t_t, hpre_t)
    print('Modulation maxabs:', maxabs(h_mx, h_t), 'mse:', mse(h_mx, h_t))

if __name__ == '__main__':
    main()
