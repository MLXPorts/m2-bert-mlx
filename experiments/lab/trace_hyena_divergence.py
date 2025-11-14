#!/usr/bin/env python
"""
Trace MLX vs Torch Hyena step-by-step and report where outputs diverge.

Prints per-stage max_abs, MSE, and max ULP difference to pinpoint the smallest
operation responsible for any discrepancy.
"""
import importlib.util
import os
import sys
import types

import mlx.core as mx
import numpy as np
import torch
import torch.nn as tnn

THIS_DIR = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
SRC_DIR = os.path.join(PKG_ROOT, 'src')

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

# Prepare shim for src.utils.train
train_path = os.path.join(SRC_DIR, 'utils', 'train.py')
sys.modules['src'] = types.ModuleType('src')
sys.modules['src.utils'] = types.ModuleType('src.utils')
sys.modules['src.utils.train'] = load('src.utils.train', train_path)

mlx_hyena = load('mlx_hyena', os.path.join(SRC_DIR, 'mm_mlx', 'hyena_filter.py'))
torch_hyena_mod = load('torch_hyena', os.path.join(SRC_DIR, 'mm', 'hyena_utils.py'))

HyenaMLX = mlx_hyena.HyenaFilter
HyenaTorch = torch_hyena_mod.HyenaFilter

def np32(a):
    if isinstance(a, mx.array):
        return np.array(a, dtype=np.float32)
    if torch.is_tensor(a):
        return a.detach().cpu().to(dtype=torch.float32).numpy()
    return np.array(a, dtype=np.float32)

def maxabs(a,b):
    aa, bb = np32(a), np32(b)
    return float(np.max(np.abs(aa - bb)))

def mse(a,b):
    aa, bb = np32(a), np32(b)
    return float(np.mean((aa - bb) ** 2))

def ulp_max(a,b):
    aa, bb = np32(a).view(np.uint32), np32(b).view(np.uint32)
    return int(np.max(np.abs(aa.astype(np.int64) - bb.astype(np.int64))))

def trace_once(batch=2, d_model=128, seq_len=256, order=32, emb_dim=5, bidir=True):
    print(f"CFG: b={batch} d={d_model} L={seq_len} order={order} emb={emb_dim} bidir={bidir}")
    hx = HyenaMLX(d_model=d_model, emb_dim=emb_dim, order=order, seq_len=seq_len, bidirectional=bidir, num_inner_mlps=2)
    ty = HyenaTorch(d_model=d_model, emb_dim=emb_dim, order=order, seq_len=seq_len, num_inner_mlps=2, bidirectional=bidir,
                    modulate=hx.modulate, normalized=hx.normalized, linear_mixer=False, w=getattr(hx,'w',10), w_mod=getattr(hx,'w_mod',1))

    # Mirror linears MLX->Torch
    mlx_layers = getattr(hx, 'implicit_filter_layers', getattr(hx, 'implicit_linears', []))
    torch_layers = [l for l in ty.implicit_filter if isinstance(l, tnn.Linear)]
    assert len(mlx_layers) == len(torch_layers)
    for ml, tl in zip(mlx_layers, torch_layers):
        w = torch.from_numpy(np.array(ml.weight)).to(dtype=tl.weight.dtype)
        tl.weight.data = w if w.shape == tl.weight.data.shape else w.t().contiguous()
        if hasattr(ml,'bias') and ml.bias is not None and tl.bias is not None:
            tl.bias.data = torch.from_numpy(np.array(ml.bias)).to(dtype=tl.bias.dtype)

    # 1) Positional embeddings
    z_mx, t_mx = hx.pos_emb(seq_len)
    z_t, t_t = ty.pos_emb(seq_len)
    print('pos.z  max=', maxabs(z_mx, z_t), ' mse=', mse(z_mx, z_t), ' ulp=', ulp_max(z_mx, z_t))
    print('pos.t  max=', maxabs(t_mx, t_t), ' mse=', mse(t_mx, t_t), ' ulp=', ulp_max(t_mx, t_t))

    # 2) Implicit filter output
    # MLX explicit evaluation when available
    if hasattr(hx, 'implicit_linears'):
        x = hx.implicit_linears[0](z_mx)
        for i in range(len(hx.implicit_sins) - 1):
            x = hx.implicit_sins[i](x)
            x = hx.implicit_linears[i + 1](x)
        x = hx.implicit_sins[-1](x)
        hpre_mx = hx.implicit_linears[-1](x)
    else:
        hpre_mx = hx.implicit_filter(z_mx)
    hpre_t = ty.implicit_filter(z_t)
    print('impl out max=', maxabs(hpre_mx, hpre_t), ' mse=', mse(hpre_mx, hpre_t), ' ulp=', ulp_max(hpre_mx, hpre_t))

    # 3) Modulation
    h_mx = hx.modulation(t_mx, hpre_mx)
    h_t = ty.modulation(t_t, hpre_t)
    print('mod  out max=', maxabs(h_mx, h_t), ' mse=', mse(h_mx, h_t), ' ulp=', ulp_max(h_mx, h_t))

    # 4) Convolution pieces
    x_np = np.random.randn(batch, d_model, seq_len).astype(np.float32)
    x_mx = mx.array(x_np)
    x_t = torch.tensor(x_np)

    # Kernels time-domain
    # MLX filter() returns (1, L, H); Torch returns (1, L, H)
    k_fwd_mx = np.swapaxes(np32(hx.filter(seq_len)), 1, 2)[0]  # (H,L)
    k_fwd_t  = ty.filter(seq_len)[0].transpose(0,1).contiguous()
    print('k_fwd max=', maxabs(k_fwd_mx, k_fwd_t), ' mse=', mse(k_fwd_mx, k_fwd_t), ' ulp=', ulp_max(k_fwd_mx, k_fwd_t))

    if bidir:
        k_rev_mx = np.swapaxes(np32(hx.filter_rev(seq_len)), 1, 2)[0]
        k_rev_t  = ty.filter_rev(seq_len)[0].transpose(0,1).contiguous()
        print('k_rev max=', maxabs(k_rev_mx, k_rev_t), ' mse=', mse(k_rev_mx, k_rev_t), ' ulp=', ulp_max(k_rev_mx, k_rev_t))

    # rFFT on inputs
    fft_size = 2*seq_len
    u_mxf = mx.fft.rfft(x_mx, n=fft_size, axis=-1)
    u_tf = torch.fft.rfft(x_t, n=fft_size)
    print('u_f  max=', maxabs(u_mxf, u_tf), ' mse=', mse(u_mxf, u_tf), ' ulp=', ulp_max(u_mxf, u_tf))

    # Assemble time-kernel like Torch
    if bidir:
        kt_mx = mx.add(mx.pad(k_fwd_mx, [(0,0),(0,seq_len)]), mx.pad(mx.slice(k_rev_mx, (0,0), (k_rev_mx.shape[0], k_rev_mx.shape[1]), (-1,-1)), [(0,0),(seq_len,0)]))
        kt_t  = torch.nn.functional.pad(k_fwd_t, (0, seq_len)) + torch.nn.functional.pad(torch.flip(k_rev_t, dims=[-1]), (seq_len, 0))
    else:
        kt_mx = k_fwd_mx
        kt_t  = k_fwd_t
    print('k_time max=', maxabs(kt_mx, kt_t), ' mse=', mse(kt_mx, kt_t), ' ulp=', ulp_max(kt_mx, kt_t))

    k_mxf = mx.fft.rfft(mx.array(kt_mx), n=fft_size, axis=-1)
    k_tf  = torch.fft.rfft(kt_t, n=fft_size)
    print('k_f  max=', maxabs(k_mxf, k_tf), ' mse=', mse(k_mxf, k_tf), ' ulp=', ulp_max(k_mxf, k_tf))

    y_mxf = mx.fft.irfft(mx.multiply(u_mxf, k_mxf), n=fft_size, axis=-1)[..., :seq_len]
    y_tf  = torch.fft.irfft(u_tf * k_tf, n=fft_size)[..., :seq_len]
    print('irfft max=', maxabs(y_mxf, y_tf), ' mse=', mse(y_mxf, y_tf), ' ulp=', ulp_max(y_mxf, y_tf))

    # Bias add
    D_mx = hx.bias.reshape(1, -1, 1)
    D_t  = ty.bias.reshape(1, -1, 1)
    out_mx = mx.add(y_mxf, mx.multiply(x_mx, D_mx))
    out_t  = y_tf + x_t * D_t
    print('final max=', maxabs(out_mx, out_t), ' mse=', mse(out_mx, out_t), ' ulp=', ulp_max(out_mx, out_t))

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    for cfg in [
        (2,64,128,16,5,False),
        (2,128,256,32,5,True),
    ]:
        trace_once(*cfg)
