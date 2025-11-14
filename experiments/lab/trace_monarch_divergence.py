#!/usr/bin/env python
"""
Step-by-step numeric trace for Monarch (MLX vs Torch mirror).

Prints stats at each fine-grained stage and reports the first stage where
abs/rel/ULP deltas exceed thresholds.
"""

import importlib.util
import os
import sys

import mlx.core as mx
import numpy as np
import torch

# Ensure MLX Hyena uses frequency-domain average combine to match Torch path
os.environ.setdefault('MLX_M2_PROFILE', 'torch_like')

THIS_DIR = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
SRC_DIR = os.path.join(PKG_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

from utils.tracer import Tracer, ulp_distance


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def build_mlx_and_mirror(d_model=64, seq_len=128):
    # MLX Monarch
    from mm_mlx.monarch_mixer_mlx import MonarchMixerSequenceMixing as MLXMixer
    mlx_m = MLXMixer(d_model=d_model, l_max=seq_len, bidirectional=True, residual_long_conv=False, hyena_filter_order=32)

    # Torch mirror class from parity test
    test_path = os.path.join(THIS_DIR, 'test_mlx_monarch_parity.py')
    tmod = _load_module('mirror', test_path)
    torch_m = tmod.TorchMixerMirror(mlx_m)
    return mlx_m, torch_m, tmod


def stats_diff(name, a, b, tr: Tracer):
    an = np.array(a, dtype=np.float32)
    bn = np.array(b, dtype=np.float32)
    diff = np.max(np.abs(an - bn))
    rel = np.linalg.norm(an - bn) / (np.linalg.norm(bn) + 1e-8)
    ulp = ulp_distance(an, bn)
    print(f"DIFF {name}: max_abs={diff:.8g} rel={rel:.8g} ulp_avg={ulp:.4f}")
    if diff > 1e-5 or rel > 1e-4:
        # Show a small slice where it’s worst
        idx = np.unravel_index(np.argmax(np.abs(an - bn)), an.shape)
        print(f"  worst idx {idx}: mlx={an[idx]:.8g} torch={bn[idx]:.8g} delta={(an-bn)[idx]:.8g}")
        return True
    return False


def main():
    mx.random.seed(0); torch.manual_seed(0)
    tr = Tracer(enabled=True)
    d, L, B = 64, 128, 2
    mlx_m, torch_m, tmod = build_mlx_and_mirror(d, L)

    x_np = np.random.randn(B, L, d).astype(np.float32)
    x_mx = mx.array(x_np)
    x_t = torch.tensor(x_np)

    # In-proj
    u_mx = mlx_m.in_linear(x_mx)
    u_t = torch_m.in_linear(x_t)
    tr.log('in_linear.mlx', u_mx, 'mlx'); tr.log('in_linear.torch', u_t, 'torch')
    if stats_diff('in_linear', u_mx, u_t.detach().cpu().numpy(), tr):
        return

    # Depthwise 3-tap
    uc_mx = mlx_m.depthwise_conv1d(u_mx.transpose(0,2,1), kernel_size=3, padding=2)[..., :L]
    dw = tmod.TorchDepthwise3Tap(3*d)
    dw.weight.data = torch.from_numpy(np.array(mlx_m.short_filter_weight))
    uc_t = dw(u_t.transpose(1,2))[..., :L]
    tr.log('depthwise3.mlx', uc_mx, 'mlx'); tr.log('depthwise3.torch', uc_t, 'torch')
    if stats_diff('depthwise3', uc_mx, uc_t.detach().cpu().numpy(), tr):
        return

    # Split and gate
    x1_mx = uc_mx[:, :d, :]; x2_mx = uc_mx[:, d:2*d, :]; v_mx = uc_mx[:, 2*d:, :]
    v_mx = v_mx * x1_mx
    x1_t = uc_t[:, :d, :]; x2_t = uc_t[:, d:2*d, :]; v_t = uc_t[:, 2*d:, :]
    v_t = v_t * x1_t
    tr.log('gated_v.mlx', v_mx, 'mlx'); tr.log('gated_v.torch', v_t, 'torch')
    if stats_diff('gated_v', v_mx, v_t.detach().cpu().numpy(), tr):
        return

    # Build mirrored Torch Hyena from MLX filter (matches test harness exactly)
    hyena_test = _load_module('hyena_parity', os.path.join(THIS_DIR, 'test_mlx_hyena_parity.py'))
    ty = hyena_test.build_mirrored_torch_hyena_from_mlx(mlx_m.filter_fn)

    # MLX hyena (with internal tracing)
    tr_mlx = Tracer(enabled=True, prefix='mlx:')
    y_mx = mlx_m.filter_fn(v_mx, L, tracer=tr_mlx)
    # Torch hyena via reference conv (matches parity harness exactly)
    hyena_utils = _load_module('torch_hyena', os.path.join(SRC_DIR, 'mm', 'hyena_utils.py'))
    k_fwd_t = ty.filter(L)[0].transpose(0, 1).contiguous()  # (H,L)
    k_rev_t = ty.filter_rev(L)[0].transpose(0, 1).contiguous()  # (H,L)
    k_time = torch.nn.functional.pad(k_fwd_t, (0, L)) + \
             torch.nn.functional.pad(torch.flip(k_rev_t, dims=[-1]), (L, 0))
    D = ty.bias.reshape(1, -1, 1)
    y_t = hyena_utils.fftconv_ref(v_t, k_time, D, dropout_mask=None, gelu=False)
    tr.log('hyena_out.mlx', y_mx, 'mlx'); tr.log('hyena_out.torch', y_t, 'torch')
    if stats_diff('hyena_out', y_mx, y_t.detach().cpu().numpy(), tr):
        return

    # Post gate
    y_mx2 = y_mx * x2_mx
    y_t2 = y_t * x2_t
    tr.log('post_gate.mlx', y_mx2, 'mlx'); tr.log('post_gate.torch', y_t2, 'torch')
    if stats_diff('post_gate', y_mx2, y_t2.detach().cpu().numpy(), tr):
        return

    # Out-proj
    out_mx = mlx_m.out_linear(y_mx2.transpose(0,2,1)).transpose(0,2,1)
    out_t = torch_m.out_linear(y_t2.transpose(1,2)).transpose(1,2)
    tr.log('out_linear.mlx', out_mx, 'mlx'); tr.log('out_linear.torch', out_t, 'torch')
    stats_diff('out_linear', out_mx, out_t.detach().cpu().numpy(), tr)


if __name__ == '__main__':
    main()
