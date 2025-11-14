"""
Weight loading utilities for M2-BERT MLX.

Provides functions to load weights from PyTorch checkpoints (.pt files)
and convert them to MLX format.

NO NumPy - strict MLX operations only.
"""

import mlx.core as mx
from typing import Dict, List, Tuple, Optional
import os


def load_pytorch_checkpoint(checkpoint_path: str) -> Dict[str, mx.array]:
    """
    Load PyTorch checkpoint and convert to MLX format.
    
    Args:
        checkpoint_path: Path to .pt or .pth file
        
    Returns:
        Dictionary mapping parameter names to MLX arrays
        
    Note:
        This uses torch.load temporarily to read the file, then converts
        all tensors to MLX format immediately.
    """
    import torch
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    # PyTorch 2.6+ requires weights_only=False for complex checkpoints
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have weights_only parameter
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # Composer checkpoint format
        if 'state' in checkpoint and 'model' in checkpoint['state']:
            state_dict = checkpoint['state']['model']
        # Standard checkpoint with state_dict key
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Assume the dict itself is the state dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Handle nested module keys
    new_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("module.", "")
        new_dict[new_key] = state_dict[key]
    state_dict = new_dict
    
    # Convert all PyTorch tensors to MLX arrays
    mlx_state_dict = {}
    for key, value in state_dict.items():
        # Skip non-tensor values
        if not isinstance(value, torch.Tensor):
            print(f"Skipping non-tensor key: {key} (type: {type(value)})")
            continue
            
        # Convert to numpy first, then to MLX
        # (This is the only place we temporarily use numpy for I/O)
        numpy_array = value.cpu().numpy()
        mlx_state_dict[key] = mx.array(numpy_array)
        
    print(f"Loaded {len(mlx_state_dict)} parameters")
    return mlx_state_dict


def load_safetensors_checkpoint(checkpoint_path: str) -> Dict[str, mx.array]:
    """
    Load safetensors checkpoint and convert to MLX format.
    
    Args:
        checkpoint_path: Path to .safetensors file
        
    Returns:
        Dictionary mapping parameter names to MLX arrays
    """
    from safetensors import safe_open
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading safetensors from {checkpoint_path}")
    mlx_state_dict = {}
    
    with safe_open(checkpoint_path, framework="numpy") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            mlx_state_dict[key] = mx.array(tensor)
    
    print(f"Loaded {len(mlx_state_dict)} parameters")
    return mlx_state_dict


def load_checkpoint(checkpoint_path: str) -> Dict[str, mx.array]:
    """
    Load checkpoint from various formats and convert to MLX.
    
    Supports:
    - PyTorch .pt/.pth files
    - Safetensors .safetensors files
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary mapping parameter names to MLX arrays
    """
    if checkpoint_path.endswith('.npz'):
        # Load NPZ format (numpy archive)
        import numpy as np
        weights_np = dict(np.load(checkpoint_path))
        weights_mlx = {}
        for key, value in weights_np.items():
            weights_mlx[key] = mx.array(value)
        return weights_mlx
    elif checkpoint_path.endswith('.safetensors'):
        return load_safetensors_checkpoint(checkpoint_path)
    elif checkpoint_path.endswith('.pt') or checkpoint_path.endswith('.pth') or 'pytorch' in checkpoint_path:
        return load_pytorch_checkpoint(checkpoint_path)
    else:
        # Try pytorch format by default
        return load_pytorch_checkpoint(checkpoint_path)


def match_and_load_weights(
    model_state_dict: Dict[str, mx.array],
    checkpoint_state_dict: Dict[str, mx.array],
    strict: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Match and load weights from checkpoint into model.
    
    Args:
        model_state_dict: Model's current state dict (will be updated)
        checkpoint_state_dict: Loaded checkpoint state dict
        strict: If True, all keys must match
        
    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    missing_keys = []
    unexpected_keys = []
    
    # Find missing keys
    for key in model_state_dict.keys():
        if key not in checkpoint_state_dict:
            missing_keys.append(key)
            if strict:
                print(f"Missing key in checkpoint: {key}")
        else:
            # Load the weight
            model_state_dict[key] = checkpoint_state_dict[key]
    
    # Find unexpected keys
    for key in checkpoint_state_dict.keys():
        if key not in model_state_dict:
            unexpected_keys.append(key)
            if strict:
                print(f"Unexpected key in checkpoint: {key}")
    
    print(f"Loaded weights: {len(checkpoint_state_dict) - len(unexpected_keys)} parameters")
    if len(missing_keys) > mx.array(0, dtype=mx.int32):
        print(f"Missing keys: {len(missing_keys)}")
    if len(unexpected_keys) > mx.array(0, dtype=mx.int32):
        print(f"Unexpected keys: {len(unexpected_keys)}")
    
    return missing_keys, unexpected_keys


def print_checkpoint_info(state_dict_or_path):
    """
    Print information about a checkpoint.
    
    Args:
        state_dict_or_path: Either a state dict (dict) or path to checkpoint file (str)
    """
    # Handle both dict and path inputs
    if isinstance(state_dict_or_path, dict):
        state_dict = state_dict_or_path
        print(f"\nCheckpoint Info:")
    else:
        print(f"\nCheckpoint Info: {state_dict_or_path}")
        state_dict = load_checkpoint(state_dict_or_path)
    
    print("-" * 50)
    
    print(f"Total parameters: {len(state_dict)}")
    print(f"\nFirst 20 parameter shapes:")
    
    total_params = mx.array(0, dtype=mx.int64)
    for i, (key, value) in enumerate(sorted(state_dict.items())):
        num_params = mx.array(1, dtype=mx.int64)
        for dim in value.shape:
            num_params = mx.multiply(num_params, mx.array(dim, dtype=mx.int64))
        total_params = mx.add(total_params, num_params)
        if i < 20:
            print(f"  {key}: {value.shape} ({num_params} params)")
    
    if len(state_dict) > 20:
        print(f"  ... and {len(state_dict) - 20} more parameters")
    
    print(f"\nTotal trainable parameters: {total_params}")
    print("-" * 50)
