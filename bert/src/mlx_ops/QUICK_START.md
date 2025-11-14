# MLX Ops Quick Start Guide

## Installation
Already installed! Library is at `bert/src/mlx_ops/`

## Import
```python
# Add to path if needed
import sys
sys.path.insert(0, 'bert/src')

# Import operations
from mlx_ops import rearrange, repeat, reduce
from mlx_ops import conv1d, conv1d_fft, depthwise_conv1d
from mlx_ops import load_checkpoint, match_and_load_weights
```

## Common Operations

### Tensor Rearrangement
```python
import mlx.core as mx
from mlx_ops import rearrange

x = mx.ones((2, 3, 4))  # (batch, seq, dim)

# Transpose
y = rearrange(x, 'b n d -> b d n')  # (2, 4, 3)

# Flatten last two dims
y = rearrange(x, 'b n d -> b (n d)')  # (2, 12)

# Split last dim
x = mx.ones((2, 12))
y = rearrange(x, 'b (n d) -> b n d', n=3, d=4)  # (2, 3, 4)
```

### Convolution
```python
from mlx_ops import conv1d

x = mx.ones((2, 100, 64))  # (batch, length, channels)
weight = mx.ones((128, 64, 3))  # (out_ch, in_ch, kernel)
bias = mx.ones((128,))

# Conv1d with bias
y = conv1d(x, weight, bias, padding=1)
```

### Load Model Weights
```python
from mlx_ops import load_checkpoint

# Load checkpoint
state_dict = load_checkpoint('.model/model.pt')

# state_dict is now Dict[str, mx.array]
# Use it to initialize your model
```

## Testing
```bash
python3 bert/tests/test_mlx_operations.py
```

## Documentation
See `bert/src/mlx_ops/README.md` for full details.
