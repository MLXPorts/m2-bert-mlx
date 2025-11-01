#!/usr/bin/env python
"""
Simple tracer for step-by-step numeric debugging.

Usage:
    from src.utils.tracer import Tracer
    tr = Tracer(enabled=True)
    tr.log('stage-name', array_like, framework='mlx')

Emits concise stats (shape, dtype, min/max/mean, l2, max-abs) and can be
paired with a diff routine to locate first divergence.
"""

from __future__ import annotations

import sys
import json
from typing import Any, Optional
import numpy as np


class Tracer:
    def __init__(self, enabled: bool = True, stream = sys.stdout, jlines: bool = False, prefix: str = "") -> None:
        self.enabled = enabled
        self.stream = stream
        self.jlines = jlines
        self.prefix = prefix

    def log(self, name: str, arr: Any, framework: str = "") -> None:
        if not self.enabled:
            return
        x = self._to_np(arr)
        if x is None:
            return
        stats = self._stats(x)
        stats.update(dict(name=name, fw=framework))
        if self.jlines:
            self.stream.write(json.dumps(stats) + "\n")
        else:
            self.stream.write(f"TRACE {self.prefix}{name} [{framework}] shape={stats['shape']} dtype={stats['dtype']} "
                              f"min={stats['min']:.8g} max={stats['max']:.8g} mean={stats['mean']:.8g} "
                              f"l2={stats['l2']:.8g}\n")
        self.stream.flush()

    @staticmethod
    def _to_np(arr: Any) -> Optional[np.ndarray]:
        try:
            import mlx.core as mx  # type: ignore
        except Exception:
            mx = None  # type: ignore
        if mx is not None:
            from mlx.core import array as mx_array  # type: ignore
        else:
            mx_array = None
        try:
            import torch  # type: ignore
        except Exception:
            torch = None  # type: ignore

        if mx_array is not None and isinstance(arr, mx_array):  # type: ignore
            return np.array(arr)
        if torch is not None and torch.is_tensor(arr):
            return arr.detach().cpu().numpy()
        if isinstance(arr, np.ndarray):
            return arr
        return None

    @staticmethod
    def _stats(x: np.ndarray) -> dict:
        x = x.astype(np.float32, copy=False)
        return dict(
            shape=tuple(x.shape),
            dtype=str(x.dtype),
            min=float(x.min(initial=np.float32(0.0))),
            max=float(x.max(initial=np.float32(0.0))),
            mean=float(x.mean()),
            l2=float(np.linalg.norm(x.ravel())),
        )


def ulp_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return average ULP distance between two float32 arrays."""
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    ai = a.view(np.int32).copy()
    bi = b.view(np.int32).copy()
    # Make lexicographically ordered by flipping sign bit ordering
    ai ^= (ai >> 31) & 0x7FFFFFFF
    bi ^= (bi >> 31) & 0x7FFFFFFF
    return float(np.mean(np.abs(ai - bi)))

