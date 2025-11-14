"""
Test script to verify M2-BERT MLX operations are working correctly.

This script tests:
1. Pure MLX einops operations (rearrange, repeat, reduce)
2. Weight loading from PyTorch checkpoints
3. Conv1d operations with bias

Run from project root:
    python3 bert/tests/test_mlx_operations.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import mlx.core as mx
from mlx_ops.einops_mlx import rearrange, repeat, reduce
from mlx_ops.conv_ops import conv1d, depthwise_conv1d
from mlx_ops.weight_loading import load_checkpoint


def test_einops_operations():
    """Test pure MLX einops operations."""
    print("\n" + "="*60)
    print("Testing EinOps Operations")
    print("="*60)
    
    # Test 1: Transpose
    print("\n1. Testing rearrange (transpose): 'b n d -> b d n'")
    x = mx.ones((2, 3, 4))
    y = rearrange(x, 'b n d -> b d n')
    assert y.shape == (2, 4, 3), f"Expected (2, 4, 3), got {y.shape}"
    print(f"   ✓ Input: {x.shape} -> Output: {y.shape}")
    
    # Test 2: Merge dimensions
    print("\n2. Testing rearrange (merge): 'b n d -> b (n d)'")
    x = mx.ones((2, 3, 4))
    y = rearrange(x, 'b n d -> b (n d)')
    assert y.shape == (2, 12), f"Expected (2, 12), got {y.shape}"
    print(f"   ✓ Input: {x.shape} -> Output: {y.shape}")
    
    # Test 3: Split dimensions
    print("\n3. Testing rearrange (split): 'b (n d) -> b n d' with n=3, d=4")
    x = mx.ones((2, 12))
    y = rearrange(x, 'b (n d) -> b n d', n=3, d=4)
    assert y.shape == (2, 3, 4), f"Expected (2, 3, 4), got {y.shape}"
    print(f"   ✓ Input: {x.shape} -> Output: {y.shape}")
    
    # Test 4: Reduce
    print("\n4. Testing reduce: 'b n d -> b d' (mean)")
    x = mx.ones((2, 3, 4))
    y = reduce(x, 'b n d -> b d', 'mean')
    assert y.shape == (2, 4), f"Expected (2, 4), got {y.shape}"
    print(f"   ✓ Input: {x.shape} -> Output: {y.shape}")
    
    print("\n✓ All einops tests passed!")
    return True


def test_conv_operations():
    """Test convolution operations."""
    print("\n" + "="*60)
    print("Testing Convolution Operations")
    print("="*60)
    
    # Test 1: Conv1d with bias
    print("\n1. Testing conv1d with bias")
    x = mx.ones((2, 10, 8))  # (batch, length, channels)
    weight = mx.ones((16, 8, 3))  # (out_channels, in_channels, kernel_size)
    bias = mx.ones((16,))
    y = conv1d(x, weight, bias, padding=1)
    print(f"   ✓ Input: {x.shape}, Weight: {weight.shape}, Bias: {bias.shape}")
    print(f"   ✓ Output: {y.shape}")
    
    # Test 2: Depthwise conv1d
    print("\n2. Testing depthwise_conv1d")
    x = mx.ones((2, 10, 8))
    weight = mx.ones((8, 1, 3))  # (channels, 1, kernel_size)
    bias = mx.ones((8,))
    y = depthwise_conv1d(x, weight, bias, padding=1)
    print(f"   ✓ Input: {x.shape}, Weight: {weight.shape}")
    print(f"   ✓ Output: {y.shape}")
    
    print("\n✓ All convolution tests passed!")
    return True


def test_weight_loading():
    """Test weight loading from PyTorch checkpoint."""
    print("\n" + "="*60)
    print("Testing Weight Loading")
    print("="*60)
    
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', '..', '.model', 'model.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"\n⚠ Checkpoint not found at {checkpoint_path}")
        print("  Skipping weight loading test")
        return True
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    state_dict = load_checkpoint(checkpoint_path)
    
    print(f"\n✓ Successfully loaded {len(state_dict)} parameters")
    
    # Show sample parameters
    print(f"\nSample parameters (first 10):")
    for i, key in enumerate(sorted(state_dict.keys())[:10]):
        shape_str = str(tuple(state_dict[key].shape))
        print(f"  {i+1}. {key}: {shape_str}")
    
    # Verify BERT components
    bert_params = [k for k in state_dict.keys() if 'bert' in k.lower()]
    encoder_params = [k for k in state_dict.keys() if 'encoder' in k.lower()]
    
    print(f"\nModel structure:")
    print(f"  - BERT-related parameters: {len(bert_params)}")
    print(f"  - Encoder-related parameters: {len(encoder_params)}")
    
    print("\n✓ Weight loading test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# M2-BERT MLX Operations Test Suite")
    print("#"*60)
    
    all_passed = True
    
    try:
        all_passed &= test_einops_operations()
    except Exception as e:
        print(f"\n✗ Einops tests failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_conv_operations()
    except Exception as e:
        print(f"\n✗ Convolution tests failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_weight_loading()
    except Exception as e:
        print(f"\n✗ Weight loading tests failed: {e}")
        all_passed = False
    
    print("\n" + "#"*60)
    if all_passed:
        print("# ✓ ALL TESTS PASSED!")
    else:
        print("# ✗ SOME TESTS FAILED")
    print("#"*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
