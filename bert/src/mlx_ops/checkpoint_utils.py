#!/usr/bin/env python3
"""
Convert M2-BERT PyTorch checkpoint to MLX format and load into model.

This script:
1. Loads a PyTorch Composer checkpoint
2. Extracts the model state_dict
3. Converts to numpy arrays
4. Saves in .npz format for MLX
5. Provides utilities to load weights into MLX M2-BERT model

Usage:
    # Convert checkpoint
    python -m src.mm_mlx.checkpoint_utils convert --checkpoint ../model/model.pt --output ../model/model_mlx.npz
    
    # Load into model
    from src.mm_mlx.checkpoint_utils import load_pretrained_m2bert
    model = load_pretrained_m2bert('../model/model_mlx.npz')
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def extract_torch_checkpoint(checkpoint_path: str) -> Dict[str, np.ndarray]:
    """
    Extract model state_dict from PyTorch Composer checkpoint and convert to numpy.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        
    Returns:
        Dictionary mapping parameter names to numpy arrays
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to convert checkpoints. Install with: pip install torch")
    
    print(f"Loading PyTorch checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract state dict from Composer format
    if isinstance(checkpoint, dict) and 'state' in checkpoint:
        if 'model' in checkpoint['state']:
            state_dict = checkpoint['state']['model']
            print(f"✓ Extracted {len(state_dict)} parameters from Composer checkpoint")
        else:
            raise ValueError("Could not find 'model' in checkpoint['state']")
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        print(f"✓ Loaded state dict with {len(state_dict)} parameters")
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")
    
    # Convert tensors to numpy
    numpy_dict = {}
    skipped = []
    
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            numpy_dict[key] = value.detach().cpu().numpy()
        else:
            skipped.append((key, type(value).__name__))
    
    if skipped:
        print(f"⚠ Skipped {len(skipped)} non-tensor parameters:")
        for key, dtype in skipped[:5]:
            print(f"    {key}: {dtype}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")
    
    print(f"✓ Converted {len(numpy_dict)} tensors to numpy")
    
    return numpy_dict


def analyze_checkpoint(weights: Dict[str, np.ndarray]) -> Dict[str, any]:
    """
    Analyze checkpoint to extract model configuration.
    
    Args:
        weights: Dictionary of parameter names to numpy arrays
        
    Returns:
        Dictionary of inferred configuration parameters
    """
    config = {}
    
    # Vocabulary and hidden size
    emb_key = 'model.bert.embeddings.word_embeddings.weight'
    if emb_key in weights:
        vocab_size, hidden_size = weights[emb_key].shape
        config['vocab_size'] = int(vocab_size)
        config['hidden_size'] = int(hidden_size)
    
    # Number of layers
    num_layers = 0
    while f'model.bert.encoder.layer.{num_layers}.attention.filter_fn.bias' in weights:
        num_layers += 1
    config['num_hidden_layers'] = num_layers
    
    # Max position embeddings
    pos_key = 'model.bert.embeddings.position_embeddings.weight'
    if pos_key in weights:
        max_pos, _ = weights[pos_key].shape
        config['max_position_embeddings'] = int(max_pos)
    
    # Hyena filter configuration
    sample_layer = 'model.bert.encoder.layer.0.attention.filter_fn.'
    
    # Get pos_emb.z shape to determine hyena_emb_dim
    pos_emb_key = sample_layer + 'pos_emb.z'
    if pos_emb_key in weights:
        _, _, emb_dim = weights[pos_emb_key].shape
        config['hyena_emb_dim'] = int(emb_dim)
    
    # Get hyena_filter_order from implicit_filter.0.weight output dimension
    filter_key = sample_layer + 'implicit_filter.0.weight'
    if filter_key in weights:
        order, _ = weights[filter_key].shape
        config['hyena_filter_order'] = int(order)
    
    # Check bidirectional
    rev_key = sample_layer + 'implicit_filter_rev.0.weight'
    config['bidirectional'] = rev_key in weights
    
    # Check for monarch MLP blocks
    mlp_key = f'model.bert.encoder.layer.0.mlp.dense1_dense2.weight'
    if mlp_key in weights:
        weight_shape = weights[mlp_key].shape
        if len(weight_shape) == 3:  # (nblocks, q, n/nblocks)
            config['nblocks'] = int(weight_shape[0])
            # Intermediate size = nblocks * q
            config['intermediate_size'] = int(weight_shape[0] * weight_shape[1])
        else:
            # Standard MLP
            config['intermediate_size'] = int(weight_shape[0])
            config['nblocks'] = 4  # default
    
    # Alternative: check gated_layers for actual intermediate size
    gated_key = f'model.bert.encoder.layer.0.mlp.gated_layers.weight'
    if gated_key in weights:
        weight_shape = weights[gated_key].shape
        if len(weight_shape) == 3:  # (nblocks, intermediate*2/nblocks, hidden/nblocks)
            config['nblocks'] = int(weight_shape[0])
            # For gated MLP: gated_layers has 2x intermediate (gate + value)
            # shape[1] = (intermediate_size / nblocks) * 2
            # So: intermediate_size = nblocks * shape[1] / 2
            config['intermediate_size'] = int(weight_shape[0] * weight_shape[1] // 2)
    
    # Count total parameters
    total_params = sum(w.size for w in weights.values())
    config['total_parameters'] = int(total_params)
    
    return config


def save_checkpoint_npz(weights: Dict[str, np.ndarray], output_path: str):
    """
    Save weights as .npz file that MLX can load.
    
    Args:
        weights: Dictionary of parameter names to numpy arrays
        output_path: Path to save .npz file
    """
    print(f"\nSaving to {output_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save as compressed npz
    np.savez_compressed(output_path, **weights)
    
    # Verify
    loaded = np.load(output_path)
    assert len(loaded.files) == len(weights), "Verification failed: file count mismatch"
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Saved {len(weights)} arrays ({size_mb:.1f} MB)")


def convert_checkpoint(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert PyTorch checkpoint to MLX format.
    
    Args:
        input_path: Path to PyTorch .pt file
        output_path: Path to save .npz file (auto-generated if None)
        
    Returns:
        Path to converted checkpoint
    """
    # Auto-generate output path if not provided
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.with_suffix('.npz'))
    
    # Extract and convert
    weights = extract_torch_checkpoint(input_path)
    
    # Analyze configuration
    config = analyze_checkpoint(weights)
    
    print("\n" + "="*70)
    print("Checkpoint Configuration:")
    print("="*70)
    for key, value in sorted(config.items()):
        if key == 'total_parameters':
            print(f"  {key:30s}: {value:,} ({value/1e6:.1f}M)")
        else:
            print(f"  {key:30s}: {value}")
    
    # Save
    save_checkpoint_npz(weights, output_path)
    
    # Verify MLX can load it
    print("\nVerifying MLX compatibility...")
    mlx_weights = mx.load(output_path)
    print(f"✓ MLX successfully loaded {len(mlx_weights)} arrays")
    
    print("\n" + "="*70)
    print("✓✓✓ CONVERSION SUCCESSFUL ✓✓✓")
    print("="*70)
    print(f"\nConverted checkpoint saved to: {output_path}")
    print(f"Load in MLX with: mx.load('{output_path}')")
    
    return output_path


def map_pytorch_to_mlx_key(torch_key: str) -> str:
    """
    Map PyTorch parameter name to MLX parameter name.
    
    PyTorch structure:
        model.bert.encoder.layer.N.attention.filter_fn.implicit_filter.M.weight
        
    MLX structure:
        layers.N.sequence_mixing.filter_fn.implicit_linears.I.weight
        
    Where M alternates: 0,2,4,6 are Linear layers (I=0,1,2,3)
                       1,3,5 are Sin layers with freq
    """
    key = torch_key
    
    # Skip non-model parameters (MLM head, etc.)
    if 'model.cls' in key or 'model.lm_head' in key:
        return None
    
    # Remove 'model.bert.' prefix
    if key.startswith('model.bert.'):
        key = key[len('model.bert.'):]
    
    # Map encoder.layer -> layers
    key = key.replace('encoder.layer.', 'layers.')
    
    # Map attention -> sequence_mixing
    key = key.replace('.attention.', '.sequence_mixing.')
    
    # Handle short_filter - map the whole key structure
    if '.short_filter.' in key:
        # short_filter.weight → short_filter_weight
        # short_filter.bias → short_filter_bias
        key = key.replace('.short_filter.weight', '.short_filter_weight')
        key = key.replace('.short_filter.bias', '.short_filter_bias')
    
    # Handle implicit_filter -> implicit_linears mapping
    # PyTorch has: implicit_filter.0, .1, .2, .3, .4, .5, .6
    # Where even indices (0,2,4,6) are Linear layers
    # And odd indices (1,3,5) are Sin layers with .freq
    # MLX has: implicit_linears[0,1,2,3] and implicit_sins[0,1,2]
    
    import re
    
    # Match implicit_filter.N or implicit_filter_rev.N
    pattern = r'(implicit_filter(?:_rev)?)\.(\d+)\.(\w+)'
    match = re.search(pattern, key)
    
    if match:
        filter_name = match.group(1)  # 'implicit_filter' or 'implicit_filter_rev'
        index = int(match.group(2))
        param = match.group(3)  # 'weight', 'bias', or 'freq'
        
        if param == 'freq':
            # Sin layer: map to implicit_sins
            sin_index = index // 2
            if filter_name == 'implicit_filter_rev':
                new_filter = 'implicit_sins_rev'
            else:
                new_filter = 'implicit_sins'
            key = re.sub(pattern, f'{new_filter}.{sin_index}.freq', key)
        else:
            # Linear layer: map to implicit_linears  
            linear_index = index // 2
            if filter_name == 'implicit_filter_rev':
                new_filter = 'implicit_linears_rev'
            else:
                new_filter = 'implicit_linears'
            key = re.sub(pattern, f'{new_filter}.{linear_index}.{param}', key)
    
    return key


def load_pretrained_m2bert(
    checkpoint_path: str,
    config_override: Optional[Dict] = None
) -> 'M2BERTModel':
    """
    Load pretrained M2-BERT model from converted checkpoint.
    
    Args:
        checkpoint_path: Path to .npz checkpoint file
        config_override: Optional dictionary to override inferred config
        
    Returns:
        M2BERTModel with loaded weights
    """
    from .m2bert_model import M2BERTModel
    
    print(f"Loading pretrained M2-BERT from: {checkpoint_path}")
    
    # Load weights
    weights = mx.load(checkpoint_path)
    print(f"✓ Loaded {len(weights)} weight tensors")
    
    # Convert to numpy for analysis
    weights_np = {k: np.array(v) for k, v in weights.items()}
    
    # Infer configuration
    config = analyze_checkpoint(weights_np)
    
    # Apply overrides
    if config_override:
        config.update(config_override)
    
    print("\n" + "="*70)
    print("Model Configuration:")
    print("="*70)
    for key, value in sorted(config.items()):
        if key != 'total_parameters':
            print(f"  {key:30s}: {value}")
    
    # Create model
    print("\nInitializing M2-BERT model...")
    model = M2BERTModel(
        vocab_size=config['vocab_size'],
        hidden_size=config['hidden_size'],
        num_hidden_layers=config['num_hidden_layers'],
        intermediate_size=config.get('intermediate_size', 3072),
        max_position_embeddings=config['max_position_embeddings'],
        nblocks=config.get('nblocks', 4),
        bidirectional=config['bidirectional'],
        hyena_filter_order=config['hyena_filter_order'],
        hyena_emb_dim=config.get('hyena_emb_dim', 3),
    )
    
    # Load weights into model
    print("\nLoading weights into model...")
    model_dict = model.parameters()
    
    # Map and load parameters
    loaded_count = 0
    missing_in_checkpoint = []
    missing_in_model = []
    shape_mismatches = []
    
    def set_nested_param(params_dict, path, value):
        """Set parameter in nested dictionary/list structure."""
        parts = path.split('.')
        current = params_dict
        
        for i, part in enumerate(parts[:-1]):
            # Handle list indices
            if part.isdigit():
                idx = int(part)
                if isinstance(current, list) and idx < len(current):
                    current = current[idx]
                else:
                    return False, "list index out of range"
            elif part in current:
                current = current[part]
            else:
                return False, f"key '{part}' not found"
        
        final_key = parts[-1]
        
        # Handle final list index
        if final_key.isdigit():
            idx = int(final_key)
            if isinstance(current, list) and idx < len(current):
                # Check shape
                if hasattr(current[idx], 'shape') and hasattr(value, 'shape'):
                    if current[idx].shape != value.shape:
                        return False, f"shape mismatch: {current[idx].shape} vs {value.shape}"
                current[idx] = value
                return True, None
            return False, "list index out of range"
        else:
            if final_key in current:
                # Check shape
                if hasattr(current[final_key], 'shape') and hasattr(value, 'shape'):
                    if current[final_key].shape != value.shape:
                        return False, f"shape mismatch: {current[final_key].shape} vs {value.shape}"
                current[final_key] = value
                return True, None
            return False, f"key '{final_key}' not found"
    
    # Load each parameter
    for torch_key, torch_value in weights.items():
        mlx_key = map_pytorch_to_mlx_key(torch_key)
        
        # Skip unmapped keys (like cls.predictions, short_filter.bias)
        if mlx_key is None:
            continue
        
        # Handle shape transformations
        value = torch_value
        
        # short_filter: Conv1d (C, 1, K) → MLX (C, K)
        if 'short_filter_weight' in mlx_key and len(value.shape) == 3:
            value = value.squeeze(1)  # Remove middle dimension
        
        success, error = set_nested_param(model_dict, mlx_key, value)
        
        if success:
            loaded_count += 1
        else:
            if "shape mismatch" in str(error):
                shape_mismatches.append((mlx_key, error))
            else:
                missing_in_model.append((mlx_key, torch_key))
    
    # Update model with loaded weights
    model.update(model_dict)
    
    print(f"✓ Loaded {loaded_count}/{len(weights)} parameters")
    
    if shape_mismatches:
        print(f"\n⚠ {len(shape_mismatches)} shape mismatches:")
        for mlx_key, error in shape_mismatches[:5]:
            print(f"    {mlx_key}: {error}")
        if len(shape_mismatches) > 5:
            print(f"    ... and {len(shape_mismatches) - 5} more")
    
    if missing_in_model:
        print(f"\n⚠ {len(missing_in_model)} parameters not found in model:")
        for mlx_key, torch_key in missing_in_model[:5]:
            print(f"    {mlx_key}")
            print(f"      (from {torch_key})")
        if len(missing_in_model) > 5:
            print(f"    ... and {len(missing_in_model) - 5} more")
    
    success_rate = loaded_count / len(weights) * 100
    print(f"\nLoad success rate: {success_rate:.1f}%")
    
    if success_rate > 80:
        print("\n" + "="*70)
        print("✓✓✓ MODEL LOADED SUCCESSFULLY ✓✓✓")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("⚠ MODEL PARTIALLY LOADED ⚠")
        print("="*70)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="M2-BERT checkpoint utilities for MLX")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert PyTorch checkpoint to MLX format')
    convert_parser.add_argument('--checkpoint', type=str, required=True,
                                help='Path to PyTorch checkpoint (.pt)')
    convert_parser.add_argument('--output', type=str, default=None,
                                help='Path to output .npz file (default: same name as input)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze checkpoint configuration')
    analyze_parser.add_argument('--checkpoint', type=str, required=True,
                                help='Path to checkpoint (.pt or .npz)')
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        convert_checkpoint(args.checkpoint, args.output)
    
    elif args.command == 'analyze':
        if args.checkpoint.endswith('.npz'):
            weights = mx.load(args.checkpoint)
            weights_np = {k: np.array(v) for k, v in weights.items()}
        else:
            weights_np = extract_torch_checkpoint(args.checkpoint)
        
        config = analyze_checkpoint(weights_np)
        
        print("\n" + "="*70)
        print("Checkpoint Configuration:")
        print("="*70)
        for key, value in sorted(config.items()):
            if key == 'total_parameters':
                print(f"  {key:30s}: {value:,} ({value/1e6:.1f}M)")
            else:
                print(f"  {key:30s}: {value}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
