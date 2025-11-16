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
import mlx.core as mx


class Tracer:
    def __init__(self, enabled: bool = True, stream = sys.stdout, jlines: bool = False, prefix: str = "") -> None:
        self.enabled = enabled
        self.stream = stream
        self.jlines = jlines
        self.prefix = prefix

    def log(self, name: str, arr: Any, framework: str = "") -> None:
        if not self.enabled:
            return
        x = self._to_mxarray(arr)
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
    def _to_mxarray(arr: Any) -> Optional[mx.array]:
        import mlx.core as mx
        return mx.array(arr)

    @staticmethod
    def _stats(x: mx.array) -> dict:
        x = x.astype(mx.float32)
        return dict(
            shape=tuple(x.shape),
            dtype=str(x.dtype),
            min=float(x.min()),
            max=float(x.max()),
            mean=float(x.mean()),
            l2=float(mx.linalg.norm(x.ravel())),
        )


def ulp_distance(a: mx.array, b: mx.array) -> float:
    """Return average ULP distance between two float32 arrays."""
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    ai = a.view(mx.int32)
    bi = b.view(mx.int32)
    # Make lexicographically ordered by flipping sign bit ordering
    ai ^= (ai >> 31) & 0x7FFFFFFF
    bi ^= (bi >> 31) & 0x7FFFFFFF
    return float(mx.mean(mx.abs(ai - bi)))

