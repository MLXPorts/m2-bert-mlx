#!/usr/bin/env python
"""
Checkpoint-agnostic config inference dispatcher.

Tries to infer config from checkpoint (safetensors first),
falls back to config.json with warning.
GGUF direct parsing removed (mlx.load handles GGUF and
config should accompany the checkpoint).
"""

import warnings
from pathlib import Path
from typing import Dict, Any

from .config_loader import load_config
from .infer_config_from_safetensors import infer_config_from_safetensors


def infer_config_from_checkpoint(model_path: str) -> Dict[str, Any]:
    """Infer model configuration from checkpoint.

    Priority order:
    1. Safetensors (if model.safetensors.index.json exists)
    2. config.json (with warning if no checkpoint index present)

    Args:
        model_path: Path to model directory

    Returns:
        Dict with model configuration
    """
    p = Path(model_path)

    # Safetensors index present → derive from weights
    if (p / "model.safetensors.index.json").exists():
        print("Inferring config from safetensors checkpoint...")
        return infer_config_from_safetensors(model_path)

    # Fallback to config.json
    config_path = p / "config.json"
    if config_path.exists():
        warnings.warn(
            f"No safetensors index found in {model_path}. "
            "Falling back to config.json (may be less model-agnostic).",
            UserWarning,
            stacklevel=2
        )
        print("Loading config from config.json...")
        return load_config(model_path)

    raise FileNotFoundError(
        f"No checkpoint or config found in {model_path}. "
        "Expected model.safetensors.index.json or config.json"
    )
