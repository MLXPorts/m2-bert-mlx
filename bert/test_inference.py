"""
Simple inference example for M2-BERT with MLX.

This script demonstrates how to load the pretrained M2-BERT-80M model
and use it for masked language modeling.
"""

import mlx.core as mx
from pathlib import Path
from typing import List, Dict
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mlx_ops import load_checkpoint, print_checkpoint_info


def load_tokenizer():
    """Load BERT tokenizer."""
    try:
        from transformers import BertTokenizer
    except ImportError:
        print("Installing transformers...")
        import subprocess
        subprocess.check_call(["pip", "install", "transformers"])
        from transformers import BertTokenizer
    
    return BertTokenizer.from_pretrained('bert-base-uncased')


def main():
    # Paths
    checkpoint_path = ".model/m2-bert-80m_mlx.npz"
    
    print("="*60)
    print("M2-BERT-80M Inference Example")
    print("="*60)
    
    # Load weights
    print("\n1. Loading weights...")
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please run download_weights.py first:")
        print("  python download_weights.py --model-name danfu09/m2-bert-80m")
        return
    
    weights = load_checkpoint(checkpoint_path)
    print(f"✓ Loaded {len(weights)} parameters")
    
    # Print weight info
    print("\n2. Checkpoint Information:")
    print_checkpoint_info(weights)
    
    # Load tokenizer
    print("\n3. Loading tokenizer...")
    tokenizer = load_tokenizer()
    print(f"✓ Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
    
    # Example text
    text = "The capital of France is [MASK]."
    print(f"\n4. Example text: '{text}'")
    
    # Tokenize
    print("\n5. Tokenizing...")
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Input IDs: {input_ids}")
    
    # Convert to MLX array
    input_tensor = mx.array([input_ids])
    print(f"Input tensor shape: {input_tensor.shape}")
    
    print("\n" + "="*60)
    print("Weight loading successful!")
    print("Next steps:")
    print("  1. Implement the M2-BERT model architecture in MLX")
    print("  2. Load these weights into the model")
    print("  3. Run forward pass for inference")
    print("="*60)


if __name__ == "__main__":
    main()
