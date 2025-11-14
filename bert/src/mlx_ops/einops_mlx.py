"""
Pure MLX implementation of common einops patterns used in M2-BERT.
NO numpy, strict MLX operations only.
"""

from typing import List

import mlx.core as mx


def rearrange(tensor: mx.array, pattern: str, **axes_lengths) -> mx.array:
    """
    Pure MLX rearrange operation for common M2-BERT patterns.
    
    Supported patterns:
    - 'b n d -> b d n' (transpose)
    - '... (n m) -> ... n m' (split last dim)
    - '... n m -> ... (n m)' (merge last two dims)
    - '(b n) d -> b n d' (split first dim with known n)
    
    Args:
        tensor: Input MLX array
        pattern: Rearrange pattern string
        **axes_lengths: Known axis lengths for splits
        
    Returns:
        Rearranged MLX array
    """
    left, right = [s.strip() for s in pattern.split('->')]
    
    # Remove spaces for parsing
    left_clean = left.replace(' ', '')
    right_clean = right.replace(' ', '')
    
    # Handle simple transpose: 'bnd->bdn'
    if '(' not in left and '(' not in right and '...' not in left:
        left_axes = list(left_clean)
        right_axes = list(right_clean)
        if sorted(left_axes) == sorted(right_axes):
            perm = [left_axes.index(ax) for ax in right_axes]
            return mx.transpose(tensor, perm)
    
    # Handle merge pattern: '...nm->...(nm)' or 'bnd->b(nd)'
    if '(' not in left and '(' in right:
        # Find which dims to merge
        # Pattern: 'b n d -> b (n d)' means merge last 2 dims
        shape = list(tensor.shape)
        
        # Count non-grouped axes on right
        right_parts = _parse_simple_pattern(right_clean)
        left_parts = _parse_simple_pattern(left_clean)
        
        # Find merge group
        for i, part in enumerate(right_parts):
            if isinstance(part, list):  # Found grouped axes
                # These axes should be merged
                merge_count = len(part)
                # Position in original tensor
                merge_start_idx = i
                
                # Build new shape
                new_shape = shape[:merge_start_idx]
                merge_size = mx.array(1, dtype=mx.int32)
                for j in range(merge_count):
                    idx = merge_start_idx + j
                    if idx < len(shape):
                        merge_size = mx.multiply(merge_size, mx.array(shape[idx], dtype=mx.int32))
                new_shape.append(int(merge_size))
                new_shape.extend(shape[merge_start_idx + merge_count:])
                
                return mx.reshape(tensor, new_shape)
    
    # Handle split pattern: '...(nm)->...nm' or 'b(nd)->bnd'
    if '(' in left and '(' not in right:
        shape = list(tensor.shape)
        left_parts = _parse_simple_pattern(left_clean)
        right_parts = _parse_simple_pattern(right_clean)
        
        # Find grouped axis in left
        for i, part in enumerate(left_parts):
            if isinstance(part, list):  # Found group to split
                split_dims = part
                # Need to know the dimensions
                split_sizes = []
                for ax_name in split_dims:
                    if ax_name in axes_lengths:
                        split_sizes.append(axes_lengths[ax_name])
                    else:
                        raise ValueError(f"Must specify length for axis '{ax_name}' when splitting")
                
                # Build new shape
                new_shape = shape[:i]
                new_shape.extend(split_sizes)
                new_shape.extend(shape[i+1:])
                
                return mx.reshape(tensor, new_shape)
    
    # Fallback for complex patterns
    raise NotImplementedError(f"Pattern '{pattern}' not yet implemented for pure MLX")


def repeat(tensor: mx.array, pattern: str, **axes_lengths) -> mx.array:
    """
    Pure MLX repeat operation.
    
    Example: 'h w -> h w c' with c=3 repeats along new axis
    
    Args:
        tensor: Input MLX array
        pattern: Repeat pattern
        **axes_lengths: Lengths for new axes
        
    Returns:
        Repeated MLX array
    """
    left, right = [s.strip() for s in pattern.split('->')]
    left_axes = left.replace(' ', '')
    right_axes = right.replace(' ', '')
    
    # Find new axes
    left_set = set(left_axes.replace('(', '').replace(')', ''))
    right_set = set(right_axes.replace('(', '').replace(')', ''))
    new_axes = right_set - left_set
    
    result = tensor
    for ax_name in new_axes:
        if ax_name not in axes_lengths:
            raise ValueError(f"Must specify length for new axis '{ax_name}'")
        # Expand and tile
        result = mx.expand_dims(result, axis=mx.array(-1, dtype=mx.int32))
        reps = [mx.array(1, dtype=mx.int32)] * (len(result.shape) - 1)
        reps.append(mx.array(axes_lengths[ax_name], dtype=mx.int32))
        result = mx.tile(result, reps)
    
    return result


def reduce(tensor: mx.array, pattern: str, reduction: str = 'mean') -> mx.array:
    """
    Pure MLX reduce operation.
    
    Example: 'b h w c -> b c' reduces over h and w
    
    Args:
        tensor: Input MLX array
        pattern: Reduction pattern
        reduction: Type ('mean', 'sum', 'max', 'min')
        
    Returns:
        Reduced MLX array
    """
    left, right = [s.strip() for s in pattern.split('->')]
    left_axes = list(left.replace(' ', '').replace('(', '').replace(')', ''))
    right_axes = set(right.replace(' ', '').replace('(', '').replace(')', ''))
    
    # Find axes to reduce
    reduce_axes_list = []
    for idx, ax in enumerate(left_axes):
        if ax not in right_axes:
            reduce_axes_list.append(idx)
    
    # Apply reduction
    result = tensor
    for axis_idx in sorted(reduce_axes_list, reverse=True):
        if reduction == 'mean':
            result = mx.mean(result, axis=axis_idx, keepdims=False)
        elif reduction == 'sum':
            result = mx.sum(result, axis=axis_idx, keepdims=False)
        elif reduction == 'max':
            result = mx.max(result, axis=axis_idx, keepdims=False)
        elif reduction == 'min':
            result = mx.min(result, axis=axis_idx, keepdims=False)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    return result


def _parse_simple_pattern(pattern_str: str) -> List:
    """Parse pattern into list of axes or grouped axes."""
    result = []
    in_group = False
    current_group = []
    
    idx = 0
    while idx < len(pattern_str):
        char = pattern_str[idx]
        if char == '(':
            in_group = True
            current_group = []
        elif char == ')':
            if in_group:
                result.append(current_group)
                current_group = []
                in_group = False
        elif char == '.' and idx + 2 < len(pattern_str):
            # Handle '...'
            if pattern_str[idx:idx+3] == '...':
                result.append('...')
                idx += 2  # Will be incremented again
        elif char.isalpha() or char.isdigit():
            if in_group:
                current_group.append(char)
            else:
                result.append(char)
        idx += 1
    
    return result
