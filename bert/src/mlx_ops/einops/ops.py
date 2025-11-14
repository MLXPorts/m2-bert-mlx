"""
MLX-adapted einops operations
Core rearrange, reduce, repeat functions for MLX arrays
"""

import mlx.core as mx
from typing import Dict, Tuple, List, Optional, Union
import re

def _parse_shape_spec(pattern: str, known_axes: Dict[str, int] = None) -> Tuple[List, List]:
    """Parse einops pattern like 'b h w c' or '(b h) w c'"""
    if known_axes is None:
        known_axes = {}
    
    # Remove parentheses and split
    axes = []
    groups = []
    current_group = []
    depth = mx.array(0, dtype=mx.int32)
    
    for char in pattern:
        if char == '(':
            depth = mx.add(depth, mx.array(1, dtype=mx.int32))
            if int(depth) == 1:
                current_group = []
        elif char == ')':
            depth = mx.subtract(depth, mx.array(1, dtype=mx.int32))
            if int(depth) == 0 and current_group:
                groups.append(current_group)
                current_group = []
        elif char == ' ' and int(depth) == 0:
            continue
        elif char != ' ' and char.isalnum():
            if int(depth) > 0:
                current_group.append(char)
            else:
                axes.append(char)
    
    return axes, groups

def rearrange(tensor: mx.array, pattern: str, **axes_lengths) -> mx.array:
    """
    Rearrange tensor elements according to pattern.
    
    Examples:
        rearrange(x, 'b h w c -> b (h w) c')
        rearrange(x, 'b c h w -> b h w c')
    """
    pattern_parts = [p.strip() for p in pattern.split('->')]
    if len(pattern_parts) != 2:
        raise ValueError(f"Pattern must have exactly one '->' separator, got: {pattern}")
    
    input_pattern, output_pattern = pattern_parts
    
    # Parse input pattern
    input_axes, input_groups = _parse_shape_spec(input_pattern)
    
    # For simple transpose operations
    if '(' not in input_pattern and '(' not in output_pattern:
        # Build axis mapping
        output_axes = [a.strip() for a in output_pattern.split()]
        axis_map = {ax: i for i, ax in enumerate(input_axes)}
        new_order = [axis_map[ax] for ax in output_axes if ax in axis_map]
        return mx.transpose(tensor, axes=new_order)
    
    # For reshape operations
    input_shape = tensor.shape
    
    # Handle grouped dimensions in input
    if input_groups:
        # Flatten groups first
        new_shape = []
        axis_idx = mx.array(0, dtype=mx.int32)
        for i, ax in enumerate(input_axes):
            if any(ax in group for group in input_groups):
                # This axis is part of a group, will be handled separately
                continue
            else:
                new_shape.append(input_shape[int(axis_idx)])
                axis_idx = mx.add(axis_idx, mx.array(1, dtype=mx.int32))
        
        # Add grouped dimensions
        for group in input_groups:
            group_size = mx.array(1, dtype=mx.int32)
            for ax in group:
                if ax in axes_lengths:
                    group_size = mx.multiply(group_size, mx.array(axes_lengths[ax], dtype=mx.int32))
            new_shape.append(int(group_size))
        
        tensor = mx.reshape(tensor, new_shape)
    
    # Handle output pattern
    output_axes, output_groups = _parse_shape_spec(output_pattern)
    
    # Build output shape
    output_shape = []
    for ax in output_axes:
        if ax in axes_lengths:
            output_shape.append(axes_lengths[ax])
        else:
            # Infer from input
            ax_idx = input_axes.index(ax) if ax in input_axes else None
            if ax_idx is not None:
                output_shape.append(input_shape[ax_idx])
    
    # Handle output groups
    if output_groups:
        for group in output_groups:
            group_size = mx.array(1, dtype=mx.int32)
            for ax in group:
                if ax in axes_lengths:
                    group_size = mx.multiply(group_size, mx.array(axes_lengths[ax], dtype=mx.int32))
            output_shape.append(int(group_size))
    
    if output_shape:
        tensor = mx.reshape(tensor, output_shape)
    
    return tensor

def reduce(tensor: mx.array, pattern: str, reduction: str, **axes_lengths) -> mx.array:
    """
    Reduce tensor along specified axes.
    
    Examples:
        reduce(x, 'b h w c -> b c', 'mean')
        reduce(x, 'b h w c -> b', 'max')
    """
    pattern_parts = [p.strip() for p in pattern.split('->')]
    if len(pattern_parts) != 2:
        raise ValueError(f"Pattern must have exactly one '->' separator")
    
    input_pattern, output_pattern = pattern_parts
    input_axes = input_pattern.split()
    output_axes = output_pattern.split()
    
    # Find axes to reduce
    reduced_axes = [i for i, ax in enumerate(input_axes) if ax not in output_axes]
    
    if not reduced_axes:
        return tensor
    
    # Perform reduction
    if reduction == 'mean':
        result = mx.mean(tensor, axis=reduced_axes, keepdims=False)
    elif reduction == 'sum':
        result = mx.sum(tensor, axis=reduced_axes, keepdims=False)
    elif reduction == 'max':
        result = mx.max(tensor, axis=reduced_axes, keepdims=False)
    elif reduction == 'min':
        result = mx.min(tensor, axis=reduced_axes, keepdims=False)
    elif reduction == 'prod':
        result = mx.prod(tensor, axis=reduced_axes, keepdims=False)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    
    return result

def repeat(tensor: mx.array, pattern: str, **axes_lengths) -> mx.array:
    """
    Repeat tensor elements according to pattern.
    
    Examples:
        repeat(x, 'h w -> h w c', c=3)
        repeat(x, 'b c -> b c h w', h=2, w=2)
    """
    pattern_parts = [p.strip() for p in pattern.split('->')]
    if len(pattern_parts) != 2:
        raise ValueError(f"Pattern must have exactly one '->' separator")
    
    input_pattern, output_pattern = pattern_parts
    input_axes = input_pattern.split()
    output_axes = output_pattern.split()
    
    # Find new axes
    new_axes = [ax for ax in output_axes if ax not in input_axes]
    
    # Add new dimensions
    for ax in new_axes:
        if ax not in axes_lengths:
            raise ValueError(f"Length for new axis '{ax}' must be specified")
        
        # Find position to insert
        pos = output_axes.index(ax)
        tensor = mx.expand_dims(tensor, axis=pos)
        
        # Repeat along new axis
        reps = [mx.array(1, dtype=mx.int32)] * len(tensor.shape)
        reps[pos] = mx.array(axes_lengths[ax], dtype=mx.int32)
        tensor = mx.tile(tensor, tuple(int(r) for r in reps))
    
    return tensor

# Alias for compatibility
einsum = mx.einsum

__all__ = ['rearrange', 'reduce', 'repeat', 'einsum']
