#!/usr/bin/env bash
set -euo pipefail

python -m bert.tests.test_mlx_hyena_parity
python -m bert.tests.test_mlx_monarch_parity

echo "All MLX parity tests completed."

