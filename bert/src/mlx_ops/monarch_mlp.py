#!/usr/bin/env python
"""
Complete Monarch MLP with Gated Linear Unit (GLU) in MLX

Full port of BertGatedLinearUnitMLP with block-diagonal Monarch matrices
"""

import mlx.core as mx
import mlx.nn as nn

from bert.src.mlx_ops.blockdiag_multiply import BlockDiagLinear
# numpy removed from compute module; tests live in bert/tests
from math_ops import sqrt_2_over_pi


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
                out_features=intermediate_size + intermediate_size,  # 2x for GLU
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
                intermediate_size + intermediate_size,
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
        half = mx.array(0.5, dtype=mx.float32)
        one  = mx.array(1.0, dtype=mx.float32)
        two  = mx.array(2.0, dtype=mx.float32)
        # Curated sqrt(2/pi)
        c = sqrt_2_over_pi()
        x3   = mx.power(x, mx.array(3.0, dtype=mx.float32))
        inner= mx.multiply(c, mx.add(x, mx.multiply(mx.array(0.044715, dtype=mx.float32), x3)))
        gelu = mx.multiply(mx.multiply(half, x), mx.add(one, mx.tanh(inner)))

        hidden_states = mx.multiply(gelu, non_gated)  # (..., intermediate_size)

        # Dropout
        hidden_states = self.dropout(hidden_states)

        # Output projection
        hidden_states = self.wo(hidden_states)  # (..., hidden_size)

        # Residual connection + LayerNorm
        output = self.layernorm(mx.add(hidden_states, residual))

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

        # GELU activation (backend ops only)
        x = hidden_states
        half = mx.array(0.5, dtype=mx.float32)
        one  = mx.array(1.0, dtype=mx.float32)
        two  = mx.array(2.0, dtype=mx.float32)
        c = sqrt_2_over_pi()
        x3   = mx.power(x, mx.array(3.0, dtype=mx.float32))
        inner= mx.multiply(c, mx.add(x, mx.multiply(mx.array(0.044715, dtype=mx.float32), x3)))
        hidden_states = mx.multiply(mx.multiply(half, x), mx.add(one, mx.tanh(inner)))

        # Dropout
        hidden_states = self.dropout(hidden_states)

        # Down-projection
        hidden_states = self.wo(hidden_states)

        # Residual + LayerNorm
        output = self.layernorm(mx.add(hidden_states, residual))

        return output


# Demo/tests moved to bert/tests to keep compute module scalar-clean.
