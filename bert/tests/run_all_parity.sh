#!/usr/bin/env bash
set -euo pipefail

# Hyena parity expects Torch-like bidirectional combine (sum)
MLX_M2_PROFILE=torch_like PYTHONPATH=. python bert/tests/test_mlx_hyena_parity.py

# Monarch parity mirror uses average combine in frequency domain
MLX_M2_PROFILE=torch_avg PYTHONPATH=. python bert/tests/test_mlx_monarch_parity.py

# Block‑diag vectorized multiply parity
PYTHONPATH=. python bert/tests/test_mlx_blockdiag_parity.py

echo "All MLX parity tests completed."
