#!/usr/bin/env python
import os
import sys
import importlib.util
import numpy as np
import torch

import mlx.core as mx

THIS_DIR = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
SRC_DIR = os.path.join(PKG_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

from mm_mlx.blockdiag_ops_mlx import blockdiag_multiply as mx_blockdiag_multiply

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

torch_block_path = os.path.join(SRC_DIR, 'mm', 'blockdiag_multiply.py')
tb = _load_module('torch_blockdiag', torch_block_path)


def run_once(n=256, q=64, nblocks=4, batch=(2,3)):
    p = n // nblocks
    x_np = np.random.randn(*batch, n).astype(np.float32)
    w_np = np.random.randn(nblocks, q, p).astype(np.float32)
    xm = mx.array(x_np)
    wm = mx.array(w_np)
    xt = torch.tensor(x_np)
    wt = torch.tensor(w_np)
    ym = mx_blockdiag_multiply(xm, wm, nblocks=nblocks)
    yt = tb.blockdiag_multiply(xt, wt)
    diff = np.max(np.abs(np.array(ym) - yt.detach().cpu().numpy()))
    rel  = np.linalg.norm(np.array(ym) - yt.detach().cpu().numpy()) / (np.linalg.norm(yt.detach().cpu().numpy()) + 1e-8)
    return dict(y_max_abs=diff, y_rel=rel)


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    for cfg in [dict(n=256,q=64,nblocks=4,batch=(2,3)), dict(n=384,q=96,nblocks=6,batch=(1,4))]:
        res = run_once(**cfg)
        print('BlockDiag parity', cfg, '=>', res)
        assert res['y_rel'] < 1e-6

if __name__ == '__main__':
    main()

