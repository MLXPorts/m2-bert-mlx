#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export MLX_M2_PROFILE=${MLX_M2_PROFILE:-torch_like}

python -m tests.test_mlx_hyena_parity
python -m tests.test_mlx_monarch_parity

echo "All MLX parity tests completed."
