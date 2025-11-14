#!/usr/bin/env python
"""
HyperProfile loader for MLX parity/stability knobs.

Profiles are small JSON files that tune numerics to mimic Torch
or prefer MLX-stable defaults. Example keys:

- gelu_mode: "erf" | "tanh"
- fft_norm: "forward" | "backward"   # scaling convention for FFT long‑conv
- bidir_combine: "avg" | "sum"        # how to combine fwd/rev kernels
- layer_norm_strict: bool             # use Torch-like LayerNorm variant
- strict_kernels: bool                # use bit‑exact kernels when available

Use set_profile(name_or_path) at program start, or rely on
MLX_M2_PROFILE env var. Defaults to 'mlx_stable'.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import Any, Dict

_DEFAULTS = {
    "gelu_mode": "erf",
    "fft_norm": "forward",
    "bidir_combine": "avg",
    "bidir_space": "time",
    "layer_norm_strict": False,
    "strict_kernels": False,
}


@dataclass(frozen=True)
class HyperProfile:
    gelu_mode: str = "erf"
    fft_norm: str = "forward"
    bidir_combine: str = "avg"
    bidir_space: str = "time"
    layer_norm_strict: bool = False
    strict_kernels: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HyperProfile":
        merged = {**_DEFAULTS, **(d or {})}
        # Filter unknown keys (e.g., 'name', 'description')
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in merged.items() if k in valid}
        return cls(**filtered)  # type: ignore[arg-type]


_current: HyperProfile | None = None


def _profiles_root() -> str:
    # Resolve to package-relative profiles folder: bert/profiles
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.abspath(os.path.join(here, ".."))  # bert/src -> bert
    return os.path.join(root, "profiles")


def _resolve_profile_path(name_or_path: str) -> str:
    if os.path.isfile(name_or_path):
        return name_or_path
    # Try bundled profiles
    cand = os.path.join(_profiles_root(), f"{name_or_path}.json")
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError(f"HyperProfile not found: {name_or_path}")


def load_profile(name_or_path: str) -> HyperProfile:
    path = _resolve_profile_path(name_or_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return HyperProfile.from_dict(data)


def set_profile(name_or_path: str) -> HyperProfile:
    global _current
    _current = load_profile(name_or_path)
    return _current


def get_profile() -> HyperProfile:
    global _current
    if _current is not None:
        return _current
    # Lazy init from env or default (must exist; no fallback)
    env_name = os.environ.get("MLX_M2_PROFILE", "mlx_stable")
    _current = load_profile(env_name)
    return _current
