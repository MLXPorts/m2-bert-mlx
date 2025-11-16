MLX Port Notes (Work in Progress)
=================================

Goal
----
- Provide an MLX backend for core M2-BERT components so the model runs efficiently on Apple Silicon without CUDA.
- Start with the Monarch Mixer sequence mixer (Hyena long-conv replacement for attention) and Hyena filter generator.

What’s Included
----------------
- `bert/src/mm_mlx/hyena_filter_mlx.py` — MLX implementation of the Hyena filter generator.
- `bert/src/mm_mlx/monarch_mixer_mlx.py` — MLX implementation of Monarch Mixer sequence mixing (attention replacement).
- `bert/src/mm_mlx/hyperprofiles_mlx.py` — Parity/stability knobs (Torch‑like vs MLX‑stable).
- `bert/src/mm_mlx/blockdiag_linear_mlx.py` — Block‑diagonal Linear for Monarch MLP.
- `bert/profiles/{mlx_stable,torch_like}.json` — Example HyperProfiles.
- `bert/requirements-mlx.txt` — minimal deps for running MLX modules.
- `bert/tests/` — MLX↔Torch numerical parity and stability tests.

How To Try
----------
- Install deps: `pip install -r bert/requirements-mlx.txt`
- Quick demo (standalone layer):
  - `python bert/src/mm_mlx/hyena_filter_mlx.py`
  - `python bert/src/mm_mlx/monarch_mixer_mlx.py`
- Parity tests:
  - `bash bert/tests/run_all_parity.sh`
  - or run individually via `python -m bert.tests.test_mlx_hyena_parity`

Parity Knobs (HyperProfiles)
----------------------------
- Choose a profile via env var: `export MLX_M2_PROFILE=torch_like` (default: `mlx_stable`).
- Profiles live in `bert/profiles/*.json` and are loaded by `mm_mlx/hyperprofiles_mlx.py`.
- Affects: GELU choice, FFT long‑conv scaling, bidirectional kernel combine, LayerNorm variant.

Design Mapping
--------------
- Mirrors the PyTorch API in `bert/src/mm/monarch_mixer_sequence_mixer.py` and `bert/src/mm/hyena_utils.py` but implemented with MLX (`mlx.core` and `mlx.nn`).
- Long convolution is performed via FFT (`mx.fft.rfft/irfft`) with simple multi-stream overlap to utilize GPU.
- Short depthwise conv is emulated (MLX lacks grouped Conv1d). This is sufficient for functional parity and can be optimized later.

Next Steps
----------
- Hook an `MLX` model variant mirroring `BertLayer`/`BertEncoder` to enable end-to-end inference/training in MLX.
- Replace ad-hoc depthwise conv with a faster kernel once MLX exposes grouped 1D conv or we add a fused path.
- Add unit tests that compare outputs between PyTorch Hyena/Monarch layers and MLX counterparts on random inputs.

Notes
-----
- CUDA-specific fused kernels under `csrc/flashmm` are kept intact for PyTorch. The MLX backend avoids these by using pure-MLX FFTs.
