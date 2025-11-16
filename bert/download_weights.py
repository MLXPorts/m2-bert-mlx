"""
Download and convert M2-BERT pretrained weights to MLX format.

This script downloads a pretrained M2-BERT model from HuggingFace and converts
the weights to a format that can be loaded by our MLX implementation.
"""

import argparse
from pathlib import Path
from typing import Dict, Any
import mlx.core as mx
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def load_checkpoint_any(checkpoint_path: Path) -> Dict[str, mx.array]:
    """Load checkpoint (.bin via custom loader, .safetensors via mx.load)."""
    if checkpoint_path.suffix == '.safetensors':
        return mx.load(str(checkpoint_path))
    # Fallback to .bin loader (no numpy involved)
    from utils.pytorch_loader import load_pytorch_bin
    return load_pytorch_bin(checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Download and convert M2-BERT weights")
    parser.add_argument("--model-name", type=str, default="danfu09/m2-bert-80m")
    parser.add_argument("--output-dir", type=str, default=".model")
    # Only safetensors output (numpy forbidden)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output safetensors")
    args = parser.parse_args()

    model_path = download_model(args.model_name, args.output_dir)
    print(f"Model downloaded to: {model_path}")

    # Collect candidate checkpoints
    checkpoint_files = list(model_path.glob("*.safetensors"))
    if not checkpoint_files:
        checkpoint_files = list(model_path.glob("*.bin"))
    if not checkpoint_files:
        # Deep search fallback
        checkpoint_files = list(Path(model_path).rglob("*.safetensors")) or list(Path(model_path).rglob("*.bin"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No .safetensors or .bin files found under {model_path}")

    target_ckpt = checkpoint_files[0]
    print(f"Using checkpoint: {target_ckpt}")

    weights = load_checkpoint_any(target_ckpt)

    out_name = target_ckpt.stem
    out_path = Path(args.output_dir) / f"{out_name}.safetensors"
    if out_path.exists() and not args.force:
        print(f"Output exists, skipping (use --force to overwrite): {out_path}")
    else:
        print(f"Saving safetensors: {out_path}")
        mx.save_safetensors(str(out_path), weights)
        print("✓ Saved safetensors")
    print("Done.")

if __name__ == "__main__":
    main()
