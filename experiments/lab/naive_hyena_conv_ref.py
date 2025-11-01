#!/usr/bin/env python
"""
Naive float32 time-domain Hyena long-conv reference vs MLX/Torch.

Computes y[b,c,t] = sum_i f32( f32(v[b,c,t-i]) * f32(k_time[c,i]) ) with zero-padding,
adding bias D channel-wise, rounding to float32 at each multiply and add via struct.
Used to decide whether MLX or Torch deviates more from a fixed-order FP32 reference.
"""

import os
import sys
import struct
import importlib.util
import numpy as np
import torch
import mlx.core as mx

THIS_DIR = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
SRC_DIR = os.path.join(PKG_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

def f32(x: float) -> float:
    return struct.unpack('!f', struct.pack('!f', x))[0]

def f32_mul(a: float, b: float) -> float:
    return f32(f32(a) * f32(b))

def f32_add(a: float, b: float) -> float:
    return f32(f32(a) + f32(b))

def naive_conv_f32(v: np.ndarray, k_time: np.ndarray, D: np.ndarray) -> np.ndarray:
    # v: (B,C,L), k_time: (C, 2L), D: (1,C,1)
    B, C, L = v.shape
    KT = k_time.shape[1]
    assert KT == 2*L
    out = np.zeros((B,C,L), dtype=np.float32)
    for b in range(B):
        for c in range(C):
            for t in range(L):
                acc = f32(0.0)
                for i in range(KT):
                    ti = t - i
                    vv = f32(0.0) if ti < 0 or ti >= L else float(v[b,c,ti])
                    ww = float(k_time[c,i])
                    acc = f32_add(acc, f32_mul(vv, ww))
                # bias add: y + v*D (match Hyena)
                acc = f32_add(acc, f32_mul(float(v[b,c,t]), float(D[0,c,0])))
                out[b,c,t] = acc
    return out

def build_time_kernel_from_mlx(hx, L: int, combine: str = 'sum') -> np.ndarray:
    # match hyena_filter_mlx time combine
    k_fwd = np.array(hx.filter(L)).transpose(0,2,1)[0]  # (C,L)
    if hx.bidirectional:
        k_rev = np.array(hx.filter_rev(L)).transpose(0,2,1)[0]
        k_fwd_time = np.pad(k_fwd, ((0,0),(0,L)))
        k_rev_time = np.pad(k_rev[:, ::-1], ((0,0),(L,0)))
        k_time = k_fwd_time + k_rev_time
        if combine == 'avg':
            k_time = k_time * np.float32(0.5)
    else:
        k_time = k_fwd
    return k_time.astype(np.float32)

def run_once(B=2, C=64, L=128, combine='sum'):
    from mm_mlx.hyena_filter_mlx import HyenaFilter as MLXHyena
    hx = MLXHyena(d_model=C, emb_dim=5, order=32, seq_len=L, bidirectional=True)
    v_np = np.random.randn(B,C,L).astype(np.float32)
    v_mx = mx.array(v_np)
    v_t  = torch.tensor(v_np)
    D = np.array(hx.bias).reshape(1,-1,1).astype(np.float32)
    # MLX output
    y_mx = np.array(hx(v_mx, L))
    # Torch output (use reference path from hyena_utils)
    hyena_utils_path = os.path.join(SRC_DIR, 'mm', 'hyena_utils.py')
    # Load torch hyena utils with stub for src.utils.train
    train_path = os.path.join(SRC_DIR, 'utils', 'train.py')
    spec_train = importlib.util.spec_from_file_location('src.utils.train', train_path)
    mod_train = importlib.util.module_from_spec(spec_train); spec_train.loader.exec_module(mod_train)
    sys.modules['src'] = importlib.util.module_from_spec(importlib.util.spec_from_loader('src', loader=None))
    sys.modules['src.utils'] = importlib.util.module_from_spec(importlib.util.spec_from_loader('src.utils', loader=None))
    sys.modules['src.utils.train'] = mod_train
    spec = importlib.util.spec_from_file_location('hyena_utils', hyena_utils_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    k_time = build_time_kernel_from_mlx(hx, L, combine=combine)
    k_time_t = torch.tensor(k_time)
    D_t = torch.tensor(D)
    y_t = mod.fftconv_ref(v_t, k_time_t, D_t, dropout_mask=None, gelu=False)
    # Naive reference
    y_ref = naive_conv_f32(v_np, k_time, D)
    def stats(name, a, b):
        d = np.max(np.abs(a-b)); r = np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-8)
        print(f'{name}: max_abs={d:.6g} rel={r:.6g}')
    print(f'-- combine={combine}')
    stats('MLX vs REF', y_mx, y_ref)
    stats('Torch vs REF', y_t.detach().cpu().numpy(), y_ref)
    stats('MLX vs Torch', y_mx, y_t.detach().cpu().numpy())

def main():
    np.random.seed(0); torch.manual_seed(0); mx.random.seed(0)
    run_once(combine='sum')
    run_once(combine='avg')

if __name__ == '__main__':
    main()
