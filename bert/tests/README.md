MLX Numeric Parity & Stability Tests
===================================

This folder contains tests that compare the MLX implementations of
Hyena and Monarch Mixer against PyTorch reference implementations
(built to mirror the same math) for numerical parity and stability.

Tests
-----
- `test_mlx_hyena_parity.py` – Compares MLX HyenaFilter kernels and
  long-convolution outputs with a mirrored PyTorch Hyena implementation
  using the same weights/parameters.
- `test_mlx_monarch_parity.py` – End-to-end parity for the Monarch Mixer
  block (input projection, short depthwise conv, gated long-conv, output
  projection) between MLX and a mirrored PyTorch block.

Quick Start
-----------
1. Install deps:
   - `pip install -r bert/requirements.txt`
   - `pip install -r bert/requirements-embeddings.txt` (optional)
   - `pip install -r bert/requirements-mlx.txt`
2. Run a single test:
   - `python -m bert.tests.test_mlx_hyena_parity`
   - `python -m bert.tests.test_mlx_monarch_parity`

Notes
-----
- These tests do not modify the original CUDA fused paths. They operate
  on standalone modules to validate math and stability on Apple Silicon.
- Monarch short conv in MLX is currently a simple depthwise 3‑tap; the
  PyTorch mirror replicates the same math rather than using Conv1d(groups).

