#!/usr/bin/env python
"""
Complete Monarch MLP with Gated Linear Unit (GLU) in MLX

Full port of BertGatedLinearUnitMLP with block-diagonal Monarch matrices
"""

import math
import mlx.core as mx
import mlx.nn as nn

from .blockdiag_multiply_mlx import BlockDiagLinear


class MonarchGLUMLP(nn.Module):
    """
    Complete Gated Linear Unit MLP with Monarch (block-diagonal) matrices

    Architecture:
        1. Input projection with GLU (2x intermediate_size)
        2. Split into gate and value
        3. gate = GELU(gate_proj)
        4. hidden = gate * value
        5. Output projection
        6. LayerNorm + residual connection

    Args:
        hidden_size: Model dimension
        intermediate_size: MLP hidden dimension
        nblocks: Number of diagonal blocks for Monarch matrices
        dropout: Dropout rate
        layer_norm_eps: LayerNorm epsilon
        use_monarch: Whether to use block-diagonal (True) or dense (False)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        nblocks: int = 4,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        use_monarch: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.nblocks = nblocks
        self.use_monarch = use_monarch

        # Create linear layers (either block-diagonal or dense)
        if use_monarch:
            # Block-diagonal Monarch matrices
            self.gated_layers = BlockDiagLinear(
                in_features=hidden_size,
                out_features=intermediate_size * 2,  # 2x for GLU
                nblocks=nblocks,
                bias=False
            )
            self.wo = BlockDiagLinear(
                in_features=intermediate_size,
                out_features=hidden_size,
                nblocks=nblocks,
                bias=True
            )
        else:
            # Dense matrices
            self.gated_layers = nn.Linear(
                hidden_size,
                intermediate_size * 2,
                bias=False
            )
            self.wo = nn.Linear(
                intermediate_size,
                hidden_size,
                bias=True
            )

        # Layer normalization
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def __call__(self, hidden_states):
        """
        Forward pass with Gated Linear Unit

        Args:
            hidden_states: (..., hidden_size)

        Returns:
            output: (..., hidden_size)
        """
        # Store for residual connection
        residual = hidden_states

        # Project to 2x intermediate size
        hidden_states = self.gated_layers(hidden_states)  # (..., 2 * intermediate_size)

        # Split into gate and value
        # gate: will be passed through GELU
        # value: will be multiplied with gated output
        gated = hidden_states[..., :self.intermediate_size]
        non_gated = hidden_states[..., self.intermediate_size:]

        # GLU operation: GELU(gate) * value
        # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x = gated
        gelu_out = 0.5 * x * (1 + mx.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

        hidden_states = gelu_out * non_gated  # (..., intermediate_size)

        # Dropout
        hidden_states = self.dropout(hidden_states)

        # Output projection
        hidden_states = self.wo(hidden_states)  # (..., hidden_size)

        # Residual connection + LayerNorm
        output = self.layernorm(hidden_states + residual)

        return output


class MonarchMLP(nn.Module):
    """
    Standard MLP with Monarch matrices (without GLU)

    Architecture:
        1. Input projection
        2. GELU activation
        3. Dropout
        4. Output projection
        5. LayerNorm + residual connection

    Args:
        hidden_size: Model dimension
        intermediate_size: MLP hidden dimension
        nblocks: Number of diagonal blocks for Monarch matrices
        dropout: Dropout rate
        layer_norm_eps: LayerNorm epsilon
        use_monarch: Whether to use block-diagonal (True) or dense (False)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        nblocks: int = 4,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        use_monarch: bool = True
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.nblocks = nblocks
        self.use_monarch = use_monarch

        # Create linear layers
        if use_monarch:
            self.gated_layers = BlockDiagLinear(
                in_features=hidden_size,
                out_features=intermediate_size,
                nblocks=nblocks,
                bias=False
            )
            self.wo = BlockDiagLinear(
                in_features=intermediate_size,
                out_features=hidden_size,
                nblocks=nblocks,
                bias=True
            )
        else:
            self.gated_layers = nn.Linear(
                hidden_size,
                intermediate_size,
                bias=False
            )
            self.wo = nn.Linear(
                intermediate_size,
                hidden_size,
                bias=True
            )

        # Layer normalization
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def __call__(self, hidden_states):
        """
        Forward pass

        Args:
            hidden_states: (..., hidden_size)

        Returns:
            output: (..., hidden_size)
        """
        residual = hidden_states

        # Up-projection
        hidden_states = self.gated_layers(hidden_states)

        # GELU activation
        x = hidden_states
        hidden_states = 0.5 * x * (1 + mx.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

        # Dropout
        hidden_states = self.dropout(hidden_states)

        # Down-projection
        hidden_states = self.wo(hidden_states)

        # Residual + LayerNorm
        output = self.layernorm(hidden_states + residual)

        return output


def test_monarch_mlp():
    """Test Monarch MLP implementations"""
    print("="*70)
    print("Testing Complete Monarch MLP with GLU (MLX)")
    print("="*70)
    print()

    batch_size = 4
    seq_len = 128
    hidden_size = 768
    intermediate_size = 3072
    nblocks = 4

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Intermediate size: {intermediate_size}")
    print(f"  Number of blocks: {nblocks}")
    print()

    # Test 1: Monarch GLU MLP
    print("Test 1: Monarch GLU MLP")
    print("-" * 70)

    mlp_glu = MonarchGLUMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        nblocks=nblocks,
        dropout=0.1,
        use_monarch=True
    )

    x_input = mx.random.normal((batch_size, seq_len, hidden_size))
    y_output = mlp_glu(x_input)

    print(f"  Input: {x_input.shape}")
    print(f"  Output: {y_output.shape}")
    print(f"  Output range: [{y_output.min().item():.3f}, {y_output.max().item():.3f}]")
    print("  ✅ Monarch GLU MLP works!")
    print()

    # Test 2: Standard Monarch MLP (no GLU)
    print("Test 2: Standard Monarch MLP (no GLU)")
    print("-" * 70)

    mlp_standard = MonarchMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        nblocks=nblocks,
        dropout=0.1,
        use_monarch=True
    )

    y_standard = mlp_standard(x_input)

    print(f"  Input: {x_input.shape}")
    print(f"  Output: {y_standard.shape}")
    print(f"  Output range: [{y_standard.min().item():.3f}, {y_standard.max().item():.3f}]")
    print("  ✅ Standard Monarch MLP works!")
    print()

    # Test 3: Dense GLU MLP (no Monarch)
    print("Test 3: Dense GLU MLP (no Monarch)")
    print("-" * 70)

    mlp_dense = MonarchGLUMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dropout=0.1,
        use_monarch=False  # Dense matrices
    )

    y_dense = mlp_dense(x_input)

    print(f"  Input: {x_input.shape}")
    print(f"  Output: {y_dense.shape}")
    print(f"  Output range: [{y_dense.min().item():.3f}, {y_dense.max().item():.3f}]")
    print("  ✅ Dense GLU MLP works!")
    print()

    # Test 4: Parameter count comparison
    print("Test 4: Parameter Count Comparison")
    print("-" * 70)

    def count_parameters(model):
        total = 0
        params = model.parameters()

        def count_nested(d):
            t = 0
            for k, v in d.items():
                if isinstance(v, dict):
                    t += count_nested(v)
                else:
                    t += v.size
            return t

        return count_nested(params)

    params_monarch = count_parameters(mlp_glu)
    params_dense = count_parameters(mlp_dense)

    print(f"  Monarch GLU parameters: {params_monarch:,}")
    print(f"  Dense GLU parameters: {params_dense:,}")
    print(f"  Reduction: {(1 - params_monarch/params_dense)*100:.1f}%")
    print(f"  Compression ratio: {params_dense/params_monarch:.2f}x")
    print()

    # Test 5: Gradient computation
    print("Test 5: Gradient Computation")
    print("-" * 70)

    def loss_fn(mlp):
        x_test = mx.random.normal((2, 64, hidden_size))
        y = mlp(x_test)
        return mx.mean(y ** 2)

    loss_and_grad = nn.value_and_grad(mlp_glu, loss_fn)
    loss, grads = loss_and_grad(mlp_glu)

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradient keys: {list(grads.keys())}")

    # Check key gradients
    if 'gated_layers' in grads:
        gated_grads = grads['gated_layers']
        print(f"  Gated layers gradients: {list(gated_grads.keys())}")

    if 'wo' in grads:
        wo_grads = grads['wo']
        print(f"  Output projection gradients: {list(wo_grads.keys())}")

    print("  ✅ Gradient computation works!")
    print()

    # Test 6: Numerical stability
    print("Test 6: Numerical Stability")
    print("-" * 70)

    # Test with very small and very large inputs
    x_small = mx.random.normal((2, 32, hidden_size)) * 0.001
    x_large = mx.random.normal((2, 32, hidden_size)) * 10.0

    y_small = mlp_glu(x_small)
    y_large = mlp_glu(x_large)

    print(f"  Small input range: [{x_small.min().item():.6f}, {x_small.max().item():.6f}]")
    print(f"  Small output range: [{y_small.min().item():.6f}, {y_small.max().item():.6f}]")
    print(f"  Large input range: [{x_large.min().item():.3f}, {x_large.max().item():.3f}]")
    print(f"  Large output range: [{y_large.min().item():.3f}, {y_large.max().item():.3f}]")

    has_nan = mx.any(mx.isnan(y_small)) or mx.any(mx.isnan(y_large))
    has_inf = mx.any(mx.isinf(y_small)) or mx.any(mx.isinf(y_large))

    if not has_nan and not has_inf:
        print("  ✅ No NaN or Inf values!")
    else:
        print("  ❌ Numerical instability detected!")

    print()
    print("="*70)
    print("✅ All Monarch MLP tests complete!")
    print("="*70)


if __name__ == '__main__':
    test_monarch_mlp()
