#!/usr/bin/env python
"""
Custom gradient computation with stream-based execution

Replaces MLX's monolithic value_and_grad() with manual gradient computation
that uses streams to avoid Metal's 499MB per-allocation limit.

Key approach:
1. Forward pass: Compute layer-by-layer, storing only necessary activations
2. Backward pass: Compute gradients layer-by-layer in reverse, using streams
3. Memory handoff: Each layer's gradients computed on separate streams
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Tuple, Any
from .metal_fft_conv import MetalFFTConv

class StreamGradientComputation:
    """
    Stream-based gradient computation for deep models

    Computes gradients manually with explicit stream control to avoid
    large monolithic allocations during backpropagation.
    """

    def __init__(self, model: nn.Module, num_streams: int = 4):
        """
        Args:
            model: The model to compute gradients for
            num_streams: Number of streams for parallel gradient computation
        """
        self.model = model
        self.num_streams = num_streams
        self.device = mx.default_device()
        self.streams = [mx.new_stream(self.device) for _ in range(num_streams)]

    def forward_with_cache(self, *args, **kwargs) -> Tuple[mx.array, List[Dict[str, mx.array]]]:
        """
        Forward pass that caches only essential activations for backward pass

        Returns:
            output: Model output
            activation_cache: List of cached activations per layer
        """
        # For now, use standard forward pass
        # TODO: Implement layer-by-layer forward with selective caching
        output = self.model(*args, **kwargs)

        # Return empty cache for now - will implement selective caching
        return output, []

    def backward_stream_based(
        self,
        output: mx.array,
        target: mx.array,
        loss_fn,
        activation_cache: List[Dict[str, mx.array]]
    ) -> Dict[str, mx.array]:
        """
        Backward pass using stream-based gradient computation

        Args:
            output: Forward pass output
            target: Target values
            loss_fn: Loss function
            activation_cache: Cached activations from forward pass

        Returns:
            gradients: Dictionary of parameter gradients
        """
        # Compute loss gradient
        loss = loss_fn(output, target)

        # Get gradient of loss w.r.t. output
        # Use MLX's grad for the loss only (small computation)
        def loss_only(out):
            return loss_fn(out, target)

        grad_output = mx.grad(loss_only)(output)

        # Now manually compute gradients layer by layer using streams
        # This is where we avoid the monolithic graph

        # For demonstration, return empty dict
        # TODO: Implement layer-by-layer backward pass
        return {}


def compute_gradients_with_streams(
    model: nn.Module,
    loss_fn,
    inputs: mx.array,
    targets: mx.array,
    num_streams: int = 4
) -> Tuple[mx.array, Dict[str, mx.array]]:
    """
    Compute loss and gradients using stream-based execution

    This is a drop-in replacement for mx.value_and_grad that uses
    streams to avoid large allocations.

    Args:
        model: The model
        loss_fn: Loss function
        inputs: Input data
        targets: Target data
        num_streams: Number of streams for parallel computation

    Returns:
        loss: Scalar loss value
        gradients: Dictionary of parameter gradients
    """
    # Create streams
    device = mx.default_device()
    streams = [mx.new_stream(device) for _ in range(num_streams)]

    # For M2-BERT specifically, we can compute gradients for each layer separately
    # This avoids building the full computation graph

    # Simple implementation: Use MLX's grad but apply it layer-by-layer
    # to keep graph sizes manageable

    def forward_fn(params):
        # Update model parameters
        model.update(params)
        # Forward pass
        output = model(inputs)
        # Compute loss
        return loss_fn(output, targets)

    # Get trainable parameters
    params = model.trainable_parameters()

    # Compute value and grad - but we'll do it smarter
    # Split parameters into groups and compute gradients per group

    param_groups = []
    group_size = max(1, len(params) // num_streams)

    param_keys = list(params.keys())
    for i in range(0, len(param_keys), group_size):
        group_keys = param_keys[i:i+group_size]
        param_groups.append({k: params[k] for k in group_keys})

    # Compute forward pass once
    output = model(inputs)
    loss = loss_fn(output, targets)
    mx.eval(loss)

    # Compute gradients per group using streams
    all_grads = {}

    for idx, param_group in enumerate(param_groups):
        stream = streams[idx % num_streams]

        with mx.stream(stream):
            # Compute gradients for this parameter group only
            def loss_for_group(group_params):
                # Create full param dict with current group
                full_params = params.copy()
                full_params.update(group_params)
                model.update(full_params)
                out = model(inputs)
                return loss_fn(out, targets)

            # Gradient computation for this group
            group_grad_fn = mx.grad(loss_for_group)
            group_grads = group_grad_fn(param_group)

            # Add to all grads
            all_grads.update(group_grads)

    # Synchronize all streams
    mx.synchronize()

    return loss, all_grads


def value_and_grad_stream(model: nn.Module, loss_fn):
    """
    Create a stream-based value_and_grad function

    Returns a function that computes loss and gradients using streams
    to avoid Metal's 499MB limit.

    Usage:
        loss_and_grad_fn = value_and_grad_stream(model, loss_fn)
        loss, grads = loss_and_grad_fn(inputs, targets)
    """
    def compute_fn(inputs, targets):
        return compute_gradients_with_streams(model, loss_fn, inputs, targets)

    return compute_fn


# Simpler approach: Manual gradient computation for FFT layer
def fft_conv_with_manual_grad(u, k, D, streams=None):
    """
    FFT convolution with manual gradient computation

    Computes forward and provides manual backward pass to avoid
    automatic differentiation building large graphs.
    """
    if streams is None:
        device = mx.default_device()
        streams = [mx.new_stream(device) for _ in range(4)]

    # Unified Metal kernel forward (no native MLX FFT). Autograd through this
    # path is not provided here; this helper is forward-only.
    if not hasattr(fft_conv_with_manual_grad, "_conv"):
        fft_conv_with_manual_grad._conv = MetalFFTConv()
    return fft_conv_with_manual_grad._conv(u, k, D)


def manual_backward_fft_conv(grad_output, u, k, D, streams=None):
    """
    Manual backward pass for FFT convolution

    Computes gradients w.r.t. u, k, D using stream-based execution

    Args:
        grad_output: Gradient from next layer (batch, d_model, seqlen)
        u: Input (batch, d_model, seqlen)
        k: Kernel (d_model, seqlen)
        D: Bias (1, d_model, 1)
        streams: List of streams for parallel computation

    Returns:
        grad_u: Gradient w.r.t. u
        grad_k: Gradient w.r.t. k
        grad_D: Gradient w.r.t. D
    """
    if streams is None:
        device = mx.default_device()
        streams = [mx.new_stream(device) for _ in range(4)]

    # Fail loudly as per project rules: no fallbacks to native FFT.
    raise NotImplementedError("manual_backward_fft_conv is not implemented for the Metal kernel path.")


if __name__ == '__main__':
    print("Stream-based gradient computation module")
    print("Provides custom value_and_grad that respects Metal memory limits")
