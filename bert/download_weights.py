"""
Download and convert M2-BERT pretrained weights to MLX format.

This script downloads a pretrained M2-BERT model from HuggingFace and converts
the weights to a format that can be loaded by our MLX implementation.
"""

import argparse
from pathlib import Path
from typing import Dict, Any

import mlx.core as mx


def download_model(model_name: str, cache_dir: str = ".model") -> Path:
    """
    Download model from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "danfu09/m2-bert-80m")
        cache_dir: Directory to cache the downloaded model
        
    Returns:
        Path to the downloaded model directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.check_call(["pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {model_name}...")
    model_path = snapshot_download(
        repo_id=model_name,
        cache_dir=str(cache_path),
        allow_patterns=["*.bin", "*.safetensors", "config.json", "*.json"]
    )
    
    return Path(model_path)


def load_pytorch_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load PyTorch checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing the model weights
    """
    import torch
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract state dict (handle different checkpoint formats)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    return state_dict


def convert_pytorch_to_mlx(state_dict: Dict[str, Any]) -> Dict[str, mx.array]:
    """
    Convert PyTorch state dict to MLX arrays.
    
    Args:
        state_dict: PyTorch state dictionary
        
    Returns:
        Dictionary with MLX arrays
    """
    import torch

    mlx_weights = {}
    
    for key, value in state_dict.items():
        # Convert torch tensor to numpy then to MLX
        if isinstance(value, torch.Tensor):
            np_array = value.detach().cpu().numpy()
            mlx_weights[key] = mx.array(np_array)
        else:
            # Keep non-tensor values as is
            mlx_weights[key] = value
    
    return mlx_weights


def save_mlx_weights(weights: Dict[str, mx.array], output_path: Path):
    """
    Save MLX weights to file.
    
    Args:
        weights: Dictionary of MLX arrays
        output_path: Path to save the weights
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert MLX arrays to numpy for saving
    import numpy as np
    np_weights = {}
    for key, value in weights.items():
        if isinstance(value, mx.array):
            np_weights[key] = np.array(value)
        else:
            np_weights[key] = value
    
    # Save as npz file
    np.savez(str(output_path), **np_weights)
    print(f"Saved weights to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and convert M2-BERT weights")
    parser.add_argument(
        "--model-name",
        type=str,
        default="danfu09/m2-bert-80m",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".model",
        help="Output directory for converted weights"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["npz", "safetensors"],
        default="npz",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    # Download the model
    model_path = download_model(args.model_name, args.output_dir)
    print(f"Model downloaded to: {model_path}")
    
    # Find the checkpoint file
    checkpoint_files = list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
    if not checkpoint_files:
        checkpoint_files = list(Path(model_path).rglob("*.bin")) + list(Path(model_path).rglob("*.safetensors"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_path}")
    
    checkpoint_file = checkpoint_files[0]
    print(f"Loading checkpoint from: {checkpoint_file}")
    
    # Load and convert
    if checkpoint_file.suffix == ".bin":
        state_dict = load_pytorch_checkpoint(checkpoint_file)
    elif checkpoint_file.suffix == ".safetensors":
        # Handle safetensors format
        try:
            from safetensors import safe_open
            state_dict = {}
            with safe_open(checkpoint_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        except ImportError:
            print("Installing safetensors...")
            import subprocess
            subprocess.check_call(["pip", "install", "safetensors"])
            from safetensors import safe_open
            state_dict = {}
            with safe_open(checkpoint_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
    
    # Print weight keys for debugging
    print("\nCheckpoint keys:")
    for key in sorted(state_dict.keys())[:20]:
        shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else None
        print(f"  {key}: {shape}")
    if len(state_dict.keys()) > 20:
        print(f"  ... and {len(state_dict.keys()) - 20} more keys")
    
    # Convert to MLX
    mlx_weights = convert_pytorch_to_mlx(state_dict)
    
    # Save converted weights
    output_path = Path(args.output_dir) / f"{args.model_name.split('/')[-1]}_mlx.npz"
    save_mlx_weights(mlx_weights, output_path)
    
    # Also save config if it exists
    config_file = model_path / "config.json"
    if config_file.exists():
        import shutil
        output_config = Path(args.output_dir) / f"{args.model_name.split('/')[-1]}_config.json"
        shutil.copy(config_file, output_config)
        print(f"Saved config to {output_config}")


if __name__ == "__main__":
    main()
