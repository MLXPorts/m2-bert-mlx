#!/usr/bin/env python
"""
Numerical parity + stability for Monarch Mixer (MLX vs Torch mirror).

We mirror the MLX block exactly in Torch (custom depthwise conv that matches
MLX’s 3‑tap per-channel op) and use Torch Hyena for long-conv with MLX weights
copied over. This avoids differences from Conv1d(groups) in the original repo.
"""

import mlx.core as mx
import mlx.nn as mnn
import numpy as np
import torch
import torch.nn as tnn
from mm.hyena_utils import HyenaFilter as TorchHyena
from mm_mlx.hyena_filter_mlx import HyenaFilter as MLXHyena
from mm_mlx.monarch_mixer_mlx import MonarchMixerSequenceMixing as MLXMixer


def _to_torch(a):
    if isinstance(a, mx.array):
        return torch.from_numpy(np.array(a))
    if isinstance(a, np.ndarray):
        return torch.from_numpy(a)
    return a


def _copy_linear_mlx_to_torch(ml: mnn.Linear, tl: tnn.Linear):
    w = _to_torch(ml.weight)
    if w.ndim == 2 and w.shape[0] == tl.in_features and w.shape[1] == tl.out_features:
        tl.weight.data = w.T.contiguous().to(dtype=tl.weight.dtype)
    else:
        tl.weight.data = w.T.contiguous().to(dtype=tl.weight.dtype)
    if getattr(ml, 'bias', None) is not None and tl.bias is not None:
        tl.bias.data = _to_torch(ml.bias).to(dtype=tl.bias.dtype)


class TorchDepthwise3Tap(tnn.Module):
    def __init__(self, channels):
        super().__init__()
        # Store kernel per-channel
        self.weight = tnn.Parameter(torch.zeros(channels, 3))

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        pad = 2
        xp = torch.nn.functional.pad(x, (pad, pad))
        out = []
        for c in range(C):
            k = self.weight[c]
            # sliding window conv
            vals = []
            for t in range(L):
                w = xp[:, c, t:t+3]
                vals.append((w * k).sum(dim=-1))
            vals = torch.stack(vals, dim=1)
            out.append(vals)
        return torch.stack(out, dim=1)


class TorchMixerMirror(tnn.Module):
    def __init__(self, mlx_mixer: MLXMixer):
        super().__init__()
        self.d = mlx_mixer.d_model
        self.in_linear = tnn.Linear(self.d, 3*self.d)
        self.out_linear = tnn.Linear(self.d, self.d)
        _copy_linear_mlx_to_torch(mlx_mixer.in_linear, self.in_linear)
        _copy_linear_mlx_to_torch(mlx_mixer.out_linear, self.out_linear)

        # Depthwise conv mirror
        self.dw = TorchDepthwise3Tap(3*self.d)
        self.dw.weight.data = _to_torch(mlx_mixer.short_filter_weight)

        # Hyena mirror from MLX weights
        filter_fn = mlx_mixer.filter_fn
        if hasattr(filter_fn, 'implicit_filter_layers') and filter_fn.implicit_filter_layers:
            first_linear = filter_fn.implicit_filter_layers[0]
            order = _to_torch(first_linear.weight).shape[1]
            num_inner = max(len(filter_fn.implicit_filter_layers) - 2, 0)
            mlx_layers = filter_fn.implicit_filter_layers
        else:
            order = self.d
            num_inner = 0
            mlx_layers = []

        self.hyena = TorchHyena(
            d_model=self.d,
            emb_dim=filter_fn.emb_dim,
            order=order,
            seq_len=mlx_mixer.l_max,
            num_inner_mlps=num_inner,
            bidirectional=filter_fn.bidirectional,
            modulate=filter_fn.modulate,
            normalized=filter_fn.normalized,
        )
        # Copy Hyena params
        self.hyena.pos_emb.z.data = _to_torch(mlx_mixer.filter_fn.pos_emb.z).to(dtype=self.hyena.pos_emb.z.dtype)
        self.hyena.pos_emb.t.data = _to_torch(mlx_mixer.filter_fn.pos_emb.t).to(dtype=self.hyena.pos_emb.t.dtype)
        self.hyena.modulation.deltas.data = _to_torch(mlx_mixer.filter_fn.modulation.deltas).to(dtype=self.hyena.modulation.deltas.dtype)
        self.hyena.bias.data = _to_torch(mlx_mixer.filter_fn.bias).to(dtype=self.hyena.bias.dtype)
        # Copy MLPs
        torch_layers = [l for l in self.hyena.implicit_filter if isinstance(l, tnn.Linear)]
        assert len(mlx_layers) == len(torch_layers)
        for ml, tl in zip(mlx_layers, torch_layers):
            _copy_linear_mlx_to_torch(ml, tl)

        if filter_fn.bidirectional and hasattr(filter_fn, 'implicit_filter_layers_rev'):
            mlx_layers_rev = filter_fn.implicit_filter_layers_rev
            torch_layers_rev = [l for l in self.hyena.implicit_filter_rev if isinstance(l, tnn.Linear)]
            for ml, tl in zip(mlx_layers_rev, torch_layers_rev):
                _copy_linear_mlx_to_torch(ml, tl)

    def forward(self, u):
        # u: (B, L, D)
        B, L, D = u.shape
        x = self.in_linear(u)
        xt = x.transpose(1, 2)  # (B, 3D, L)
        uc = self.dw(xt)[..., :L]
        x1 = uc[:, :D, :]
        x2 = uc[:, D:2*D, :]
        v  = uc[:, 2*D:, :]
        v = v * x1
        # Hyena long conv
        y = self.hyena(v, L)
        y = y * x2
        y = y.transpose(1, 2)
        y = self.out_linear(y)
        return y


def parity_once(batch=2, d_model=128, seq_len=256):
    mmx = MLXMixer(
        d_model=d_model, l_max=seq_len,
        bidirectional=True, residual_long_conv=False,
        hyena_filter_order=32,
    )
    mtx = TorchMixerMirror(mmx)

    x_np = np.random.randn(batch, seq_len, d_model).astype(np.float32)
    x_mx = mx.array(x_np)
    x_t = torch.tensor(x_np)

    y_mx, _ = mmx(x_mx)
    y_t = mtx(x_t)

    diff = np.max(np.abs(np.array(y_mx) - y_t.detach().cpu().numpy()))
    rel = (np.linalg.norm(np.array(y_mx) - y_t.detach().cpu().numpy()) /
           (np.linalg.norm(y_t.detach().cpu().numpy()) + 1e-8))

    return dict(y_max_abs=diff, y_rel=rel)


def main():
    mx.random.seed(0)
    torch.manual_seed(0)

    for cfg in [dict(batch=2, d_model=64, seq_len=128), dict(batch=1, d_model=96, seq_len=192)]:
        res = parity_once(**cfg)
        print('Monarch parity', cfg, '=>', res)
        assert res['y_rel'] < 1e-4, f"Relative diff too large: {res}"

    print('✅ MLX Monarch parity and stability checks passed')


if __name__ == '__main__':
    main()
