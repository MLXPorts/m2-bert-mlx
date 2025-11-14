#!/usr/bin/env python
"""
Compare MLX builtin activations (mlx.nn.activations) vs our registry (bert/src/mm_mlx/activations.py).
Checks elementwise parity across a range of values for:
- tanh, sigmoid, relu, silu
- gelu exact (erf) vs gelu_tanh (approx)
- lecun_tanh (with float32 constants)
"""
import importlib.util
import os

import mlx.core as mx
import numpy as np

THIS_DIR = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
SRC_DIR = os.path.join(PKG_ROOT, 'src')

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

act_mod = load('activations_mlx', os.path.join(SRC_DIR, 'mm_mlx', 'activations.py'))

import mlx.nn as nn

def np_allclose(a,b,rtol=1e-6, atol=1e-6):
    return np.allclose(a,b, rtol=rtol, atol=atol)

def check(name_builtin, our_name, xvals):
    # MLX builtin function/class dispatch
    if name_builtin == 'gelu':
        builtin = nn.gelu
    elif name_builtin == 'gelu_tanh':
        builtin = nn.gelu_approx
    elif name_builtin == 'tanh':
        builtin = mx.tanh
    elif name_builtin == 'sigmoid':
        builtin = mx.sigmoid
    elif name_builtin == 'relu':
        builtin = nn.relu
    elif name_builtin == 'silu':
        builtin = nn.silu
    else:
        builtin = None

    ours = act_mod.get_activation(our_name)
    x = mx.array(xvals, dtype=mx.float32)
    y_b = builtin(x) if builtin is not None else None
    y_o = ours(x)
    b = None if y_b is None else np.array(y_b)
    o = np.array(y_o)
    return b, o

def main():
    xs = np.linspace(-8, 8, 257).astype(np.float32)
    cases = [
        ('tanh','tanh'),
        ('sigmoid','sigmoid'),
        ('relu','relu'),
        ('silu','silu'),
        ('gelu','gelu'),          # exact/erf
        ('gelu_tanh','gelu_tanh') # tanh approx
    ]
    for bname, oname in cases:
        yb, yo = check(bname, oname, xs)
        if yb is None:
            print(f"[WARN] No builtin for {bname}, skipped")
            continue
        ok = np_allclose(yb, yo, rtol=1e-6, atol=1e-6)
        diff = float(np.nanmax(np.abs(yb - yo)))
        print(f"{bname:10s} vs {oname:10s} | close={ok} max_abs={diff:.3e}")

    # LeCun tanh: compare our implementation to composing MLX ops
    x = mx.array(xs)
    ours = act_mod.get_activation('lecun_tanh')(x)
    # Equivalent composed: 1.7159 * tanh(0.666 * x)
    eq = mx.multiply(mx.array(1.7159, dtype=mx.float32), mx.tanh(mx.multiply(mx.array(0.666, dtype=mx.float32), x)))
    ok = np_allclose(np.array(ours), np.array(eq))
    diff = float(np.nanmax(np.abs(np.array(ours) - np.array(eq))))
    print(f"lecun_tanh   parity | close={ok} max_abs={diff:.3e}")

if __name__ == '__main__':
    main()

