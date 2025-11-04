"""Test helpers for MLX BERT.

Ensures src/ is on sys.path so tests can import MLX/Torch modules without
requiring composer or other optional deps.
"""

import pathlib
import sys

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
