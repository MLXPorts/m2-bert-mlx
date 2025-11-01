#!/usr/bin/env python
"""
Torch <-> MLX Hyena parity without NumPy.
Uses Python lists + torch tensors + MLX arrays only.
"""
import os, sys, types, importlib.util, random
import torch
import torch.nn as tnn
import mlx.core as mx

THIS_DIR = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
SRC_DIR = os.path.join(PKG_ROOT, 'src')

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

# Shim for src.utils.train
train_path = os.path.join(SRC_DIR, 'utils', 'train.py')
sys.modules['src'] = types.ModuleType('src')
sys.modules['src.utils'] = types.ModuleType('src.utils')
sys.modules['src.utils.train'] = load('src.utils.train', train_path)

HyenaMLX = load('mlx_hyena', os.path.join(SRC_DIR, 'mm_mlx', 'hyena_filter_mlx.py')).HyenaFilter
HyenaTorch = load('torch_hyena', os.path.join(SRC_DIR, 'mm', 'hyena_utils.py')).HyenaFilter

def to_torch_from_mx(a):
    return torch.tensor(mx.array(a).tolist(), dtype=torch.float32)

def to_mx_from_list(lst):
    return mx.array(lst, dtype=mx.float32)

def mse_torch(a, b):
    d = a - b
    return float(torch.mean(d * d).item())

def rand_input(batch, d_model, L):
    # Deterministic list-based randoms
    rnd = random.Random(0)
    data = [[[rnd.uniform(-1.0, 1.0) for _ in range(L)] for _ in range(d_model)] for _ in range(batch)]
    return data

def mirror_weights_mlx_to_torch(hx, ty):
    import mlx.nn as mnn
    torch_layers = [l for l in ty.implicit_filter if isinstance(l, tnn.Linear)]
    mlx_layers = getattr(hx, 'implicit_filter_layers', getattr(hx, 'implicit_linears', []))
    assert len(mlx_layers) == len(torch_layers)
    for ml, tl in zip(mlx_layers, torch_layers):
        w = torch.tensor(ml.weight.tolist(), dtype=tl.weight.dtype)
        tl.weight.data = w if w.shape == tl.weight.data.shape else w.t().contiguous()
        if hasattr(ml, 'bias') and ml.bias is not None and tl.bias is not None:
            tl.bias.data = torch.tensor(ml.bias.tolist(), dtype=tl.bias.dtype)

def run_case(batch=2, d_model=128, L=256, order=32, emb_dim=5, bidir=True):
    hx = HyenaMLX(d_model=d_model, emb_dim=emb_dim, order=order, seq_len=L, bidirectional=bidir, num_inner_mlps=2)
    ty = HyenaTorch(d_model=d_model, emb_dim=emb_dim, order=order, seq_len=L, num_inner_mlps=2, bidirectional=bidir,
                    modulate=hx.modulate, normalized=hx.normalized, linear_mixer=False,
                    w=getattr(hx,'w',10), w_mod=getattr(hx,'w_mod',1))
    mirror_weights_mlx_to_torch(hx, ty)

    x_list = rand_input(batch, d_model, L)
    x_mx = to_mx_from_list(x_list)
    x_t = torch.tensor(x_list, dtype=torch.float32)

    y_mx = hx(x_mx, L)
    # Manual Torch FFT long-conv identical to hyena_utils but standalone
    k_fwd_t = ty.filter(L)[0].transpose(0,1).contiguous()
    if ty.bidirectional:
        k_rev_t = ty.filter_rev(L)[0].transpose(0,1).contiguous()
        k_time = torch.nn.functional.pad(k_fwd_t, (0, L)) + torch.nn.functional.pad(torch.flip(k_rev_t, dims=[-1]), (L, 0))
    else:
        k_time = k_fwd_t
    fft_size = 2 * L
    u_f = torch.fft.rfft(x_t, n=fft_size)
    k_f = torch.fft.rfft(k_time, n=fft_size)
    y_t = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :L]
    y_t = y_t + x_t * ty.bias.reshape(1, -1, 1)
    y_mx_t = to_torch_from_mx(y_mx)
    err = mse_torch(y_mx_t, y_t.detach().to(dtype=torch.float32))
    print({'cfg':(batch,d_model,L,order,emb_dim,bidir),'y_mse':err})

def main():
    for cfg in [
        (2,64,128,16,5,False),
        (2,128,256,32,5,True),
        (1,192,512,32,5,True),
    ]:
        run_case(*cfg)

if __name__ == '__main__':
    main()
