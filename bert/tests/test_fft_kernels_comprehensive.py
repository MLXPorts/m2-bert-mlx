#!/usr/bin/env python
"""
Comprehensive test suite for FFT convolution kernels:
- MetalFFTConv (unified kernel)
- MetalFFTConvStreamed (stream-chained kernel)
- Integration with HyenaFilter
- Accuracy vs PyTorch
"""
import os
import sys

import mlx.core as mx
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available, skipping accuracy tests")

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(repo_root, 'src'))

from mm_mlx.metal_fft_conv import MetalFFTConv
from mm_mlx.metal_fft_conv_streamed import MetalFFTConvStreamed
from mm_mlx.hyena_filter_mlx import HyenaFilter


def test_unified_kernel(verbose=True):
    """Test unified MetalFFTConv kernel"""
    if verbose:
        print("\n" + "="*70)
        print("TEST 1: Unified MetalFFTConv Kernel")
        print("="*70)
    
    conv = MetalFFTConv()
    sizes = [64, 128, 256, 512, 1024, 2048]
    passed = []
    
    for L in sizes:
        mx.random.seed(42)
        u = mx.random.normal((2, 4, L)).astype(mx.float32)
        k = mx.random.normal((4, L)).astype(mx.float32)
        D = mx.zeros((4,), dtype=mx.float32)
        
        y = conv(u, k, D)
        mx.eval(y)
        
        assert y.shape == (2, 4, L), f"Shape mismatch: {y.shape} != {(2, 4, L)}"
        
        # Check for NaN/Inf
        y_np = np.array(y)
        has_nan = np.isnan(y_np).any()
        has_inf = np.isinf(y_np).any()
        
        if verbose:
            print(f"  L={L:<5} | shape={y.shape} | mean={float(mx.mean(y)):.6f} | std={float(mx.std(y)):.6f}")
        
        assert not has_nan, f"NaN detected at L={L}"
        assert not has_inf, f"Inf detected at L={L}"
        passed.append(L)
    
    if verbose:
        print(f"✓ Unified kernel passed for all sizes: {passed}")
    return True


def test_streamed_kernel(verbose=True):
    """Test stream-chained MetalFFTConvStreamed kernel"""
    if verbose:
        print("\n" + "="*70)
        print("TEST 2: Stream-Chained MetalFFTConvStreamed Kernel")
        print("="*70)
    
    conv = MetalFFTConvStreamed(num_streams=4)
    sizes = [64, 128, 256, 512]  # Smaller sizes for streamed (more overhead)
    passed = []
    
    for L in sizes:
        mx.random.seed(42)
        u = mx.random.normal((2, 4, L)).astype(mx.float32)
        k = mx.random.normal((4, L)).astype(mx.float32)
        D = mx.zeros((4,), dtype=mx.float32)
        
        y = conv(u, k, D)
        mx.eval(y)
        
        assert y.shape == (2, 4, L), f"Shape mismatch: {y.shape} != {(2, 4, L)}"
        
        # Check for NaN/Inf
        y_np = np.array(y)
        has_nan = np.isnan(y_np).any()
        has_inf = np.isinf(y_np).any()
        
        if verbose:
            print(f"  L={L:<5} | shape={y.shape} | mean={float(mx.mean(y)):.6f} | std={float(mx.std(y)):.6f}")
        
        assert not has_nan, f"NaN detected at L={L}"
        assert not has_inf, f"Inf detected at L={L}"
        passed.append(L)
    
    if verbose:
        print(f"✓ Streamed kernel passed for all sizes: {passed}")
    return True


def test_accuracy_vs_torch(verbose=True):
    """Test accuracy against PyTorch reference"""
    if not TORCH_AVAILABLE:
        if verbose:
            print("\n" + "="*70)
            print("TEST 3: Accuracy vs PyTorch - SKIPPED (PyTorch not available)")
            print("="*70)
        return True
    
    if verbose:
        print("\n" + "="*70)
        print("TEST 3: Accuracy vs PyTorch Reference")
        print("="*70)
    
    conv = MetalFFTConv()
    sizes = [128, 256, 512, 1024, 2048]
    max_errors = []
    mean_errors = []
    
    for L in sizes:
        mx.random.seed(42)
        u = mx.random.normal((2, 8, L)).astype(mx.float32)
        k = mx.random.normal((8, L)).astype(mx.float32)
        D = mx.zeros((1, 8, 1), dtype=mx.float32)
        
        # MLX kernel
        y_mlx = conv(u, k, D)
        mx.eval(y_mlx)
        
        # PyTorch reference
        N = 2 * L
        u_t = torch.from_numpy(np.array(u))
        k_t = torch.from_numpy(np.array(k))
        D_t = torch.from_numpy(np.array(D).reshape(1, 8, 1))
        
        k_f = torch.fft.rfft(k_t, n=N, dim=-1)
        u_f = torch.fft.rfft(u_t, n=N, dim=-1)
        y_f = u_f * k_f.unsqueeze(0)
        y_ref = torch.fft.irfft(y_f, n=N, dim=-1)[..., :L]
        y_ref = y_ref + u_t * D_t
        
        diff = np.abs(y_ref.numpy() - np.array(y_mlx))
        max_err = float(diff.max())
        mean_err = float(diff.mean())
        
        max_errors.append(max_err)
        mean_errors.append(mean_err)
        
        if verbose:
            print(f"  L={L:<5} | max error: {max_err:.3e} | mean error: {mean_err:.3e}")
        
        # Reasonable error bounds (float32 precision)
        assert max_err < 1e-4, f"Max error too high at L={L}: {max_err}"
        assert mean_err < 2e-5, f"Mean error too high at L={L}: {mean_err}"
    
    if verbose:
        print(f"✓ All accuracy tests passed")
        print(f"  Max errors range: [{min(max_errors):.3e}, {max(max_errors):.3e}]")
        print(f"  Mean errors range: [{min(mean_errors):.3e}, {max(mean_errors):.3e}]")
    return True


def test_hyena_filter_integration(verbose=True):
    """Test integration with HyenaFilter"""
    if verbose:
        print("\n" + "="*70)
        print("TEST 4: HyenaFilter Integration")
        print("="*70)
    
    sizes = [128, 256, 512]
    
    for L in sizes:
        hf = HyenaFilter(d_model=32, seq_len=L, emb_dim=3, order=16)
        
        mx.random.seed(42)
        x = mx.random.normal((2, 32, L)).astype(mx.float32)
        
        y = hf(x, L=L)
        mx.eval(y)
        
        assert y.shape == (2, 32, L), f"Shape mismatch: {y.shape} != {(2, 32, L)}"
        
        # Check for NaN/Inf
        y_np = np.array(y)
        has_nan = np.isnan(y_np).any()
        has_inf = np.isinf(y_np).any()
        
        if verbose:
            print(f"  L={L:<5} | shape={y.shape} | mean={float(mx.mean(y)):.6f} | std={float(mx.std(y)):.6f}")
        
        assert not has_nan, f"NaN detected at L={L}"
        assert not has_inf, f"Inf detected at L={L}"
    
    if verbose:
        print(f"✓ HyenaFilter integration test passed")
    return True


def test_edge_cases(verbose=True):
    """Test edge cases and boundary conditions"""
    if verbose:
        print("\n" + "="*70)
        print("TEST 5: Edge Cases")
        print("="*70)
    
    conv = MetalFFTConv()
    
    # Test 1: Single batch
    u = mx.random.normal((1, 4, 128)).astype(mx.float32)
    k = mx.random.normal((4, 128)).astype(mx.float32)
    D = mx.zeros((4,), dtype=mx.float32)
    y = conv(u, k, D)
    mx.eval(y)
    assert y.shape == (1, 4, 128)
    if verbose:
        print("  ✓ Single batch test passed")
    
    # Test 2: Single channel
    u = mx.random.normal((2, 1, 128)).astype(mx.float32)
    k = mx.random.normal((1, 128)).astype(mx.float32)
    D = mx.zeros((1,), dtype=mx.float32)
    y = conv(u, k, D)
    mx.eval(y)
    assert y.shape == (2, 1, 128)
    if verbose:
        print("  ✓ Single channel test passed")
    
    # Test 3: Large batch
    u = mx.random.normal((16, 4, 64)).astype(mx.float32)
    k = mx.random.normal((4, 64)).astype(mx.float32)
    D = mx.zeros((4,), dtype=mx.float32)
    y = conv(u, k, D)
    mx.eval(y)
    assert y.shape == (16, 4, 64)
    if verbose:
        print("  ✓ Large batch test passed")
    
    # Test 4: Many channels
    u = mx.random.normal((2, 32, 64)).astype(mx.float32)
    k = mx.random.normal((32, 64)).astype(mx.float32)
    D = mx.zeros((32,), dtype=mx.float32)
    y = conv(u, k, D)
    mx.eval(y)
    assert y.shape == (2, 32, 64)
    if verbose:
        print("  ✓ Many channels test passed")
    
    if verbose:
        print("✓ All edge case tests passed")
    return True


def main():
    print("\n" + "#"*70)
    print("# COMPREHENSIVE FFT CONVOLUTION KERNEL TEST SUITE")
    print("#"*70)
    
    all_passed = True
    
    try:
        test_unified_kernel(verbose=True)
    except Exception as e:
        print(f"✗ Unified kernel test FAILED: {e}")
        all_passed = False
    
    try:
        test_streamed_kernel(verbose=True)
    except Exception as e:
        print(f"✗ Streamed kernel test FAILED: {e}")
        all_passed = False
    
    try:
        test_accuracy_vs_torch(verbose=True)
    except Exception as e:
        print(f"✗ Accuracy test FAILED: {e}")
        all_passed = False
    
    try:
        test_hyena_filter_integration(verbose=True)
    except Exception as e:
        print(f"✗ HyenaFilter integration test FAILED: {e}")
        all_passed = False
    
    try:
        test_edge_cases(verbose=True)
    except Exception as e:
        print(f"✗ Edge cases test FAILED: {e}")
        all_passed = False
    
    print("\n" + "#"*70)
    if all_passed:
        print("# ✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("# ✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("#"*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
