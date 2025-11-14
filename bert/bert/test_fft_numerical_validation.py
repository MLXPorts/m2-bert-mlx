#!/usr/bin/env python
"""
Numerical validation of FFT convolution kernels.

Compare kernel implementations against MLX native FFT reference:
1. Unified kernel vs MLX FFT (L <= 2048)
2. Streamed kernel vs Unified kernel (2048 < L <= 4096)
3. Streamed kernel vs MLX FFT (all lengths)
"""

import sys
sys.path.insert(0, 'src')

import mlx.core as mx
import numpy as np

def mlx_native_fft_conv(u, k_time, D):
    """
    Reference implementation using MLX's native FFT operations.

    Computes CIRCULAR convolution via FFT (same as the Metal kernel).

    Args:
        u: (B, C, L) input
        k_time: (C, L) kernel in time domain
        D: (1, C, 1) or (C,) bias

    Returns:
        y: (B, C, L) output
    """
    B, C, L = u.shape

    # Ensure bias is in correct shape
    if D.ndim == 3:
        D_flat = D.reshape(-1)  # (C,)
    else:
        D_flat = D

    # Circular convolution: no padding needed, FFT size = L
    # FFT
    U_freq = mx.fft.rfft(u, axis=-1)         # (B, C, L//2+1) complex
    K_freq = mx.fft.rfft(k_time, axis=-1)    # (C, L//2+1) complex

    # Broadcast and multiply in frequency domain
    K_freq_expanded = mx.expand_dims(K_freq, axis=0)  # (1, C, L//2+1)
    Y_freq = mx.multiply(U_freq, K_freq_expanded)     # (B, C, L//2+1)

    # IFFT back to time domain
    y = mx.fft.irfft(Y_freq, n=L, axis=-1)  # (B, C, L)

    # Add bias
    D_expanded = mx.expand_dims(mx.expand_dims(D_flat, axis=0), axis=-1)  # (1, C, 1)
    y = mx.add(y, D_expanded)

    return y


def compare_outputs(name1, output1, name2, output2, rtol=1e-4, atol=1e-5):
    """
    Compare two outputs and report differences.

    Returns:
        (passed, max_abs_diff, max_rel_diff, mean_abs_diff)
    """
    # Compute differences
    abs_diff = mx.abs(mx.subtract(output1, output2))

    # Avoid division by zero in relative error
    abs_output2 = mx.abs(output2)
    safe_denominator = mx.maximum(abs_output2, mx.array(1e-10, dtype=mx.float32))
    rel_diff = mx.divide(abs_diff, safe_denominator)

    max_abs_diff = float(mx.max(abs_diff))
    max_rel_diff = float(mx.max(rel_diff))
    mean_abs_diff = float(mx.mean(abs_diff))

    # Check tolerance
    passed = (max_abs_diff < atol) or (max_rel_diff < rtol)

    print(f"  Comparing {name1} vs {name2}:")
    print(f"    Max absolute error: {max_abs_diff:.2e}")
    print(f"    Max relative error: {max_rel_diff:.2e}")
    print(f"    Mean absolute error: {mean_abs_diff:.2e}")
    print(f"    Tolerance: atol={atol:.2e}, rtol={rtol:.2e}")
    print(f"    Result: {'✓ PASS' if passed else '✗ FAIL'}")

    return passed, max_abs_diff, max_rel_diff, mean_abs_diff


def test_numerical_accuracy(seq_len, batch_size=2, d_model=64):
    """Test numerical accuracy at a specific sequence length."""
    print(f"\n{'='*80}")
    print(f"Numerical Validation: L={seq_len}, B={batch_size}, C={d_model}")

    # Create deterministic test data (use fixed seed)
    mx.random.seed(42)
    u = mx.random.normal((batch_size, d_model, seq_len)).astype(mx.float32)
    k_time = mx.random.normal((d_model, seq_len)).astype(mx.float32)
    D = mx.random.normal((1, d_model, 1)).astype(mx.float32)

    # Get reference output using MLX native FFT
    print("  Computing reference (MLX native FFT)...")
    y_reference = mlx_native_fft_conv(u, k_time, D)
    mx.eval(y_reference)

    # Get kernel output
    from mlx_ops.hyena_filter import _hyena_fft_conv
    kernel_type = "Unified" if seq_len <= 2048 else "Streamed"
    print(f"  Computing kernel output ({kernel_type})...")
    y_kernel = _hyena_fft_conv(u, k_time, D, seqlen=seq_len)
    mx.eval(y_kernel)

    # Compare
    passed, max_abs, max_rel, mean_abs = compare_outputs(
        "Kernel", y_kernel,
        "Reference", y_reference,
        rtol=1e-3,  # 0.1% relative error
        atol=1e-4   # Absolute error for small values
    )

    return passed, max_abs, max_rel, mean_abs


def test_unified_vs_streamed():
    """
    Test that unified and streamed kernels produce identical results
    at L=2048 (where both can run).
    """
    print(f"\n{'='*80}")
    print("Testing Unified vs Streamed kernel consistency at L=2048")

    seq_len = 2048
    batch_size = 2
    d_model = 64

    # Create test data
    mx.random.seed(42)
    u = mx.random.normal((batch_size, d_model, seq_len)).astype(mx.float32)
    k_time = mx.random.normal((d_model, seq_len)).astype(mx.float32)
    D = mx.random.normal((1, d_model, 1)).astype(mx.float32)

    # Force unified kernel
    from mlx_ops.kernels.metal_fft_conv import MetalFFTConv
    unified_conv = MetalFFTConv()
    print("  Running unified kernel...")
    y_unified = unified_conv(u, k_time, D)
    mx.eval(y_unified)

    # Force streamed kernel
    from mlx_ops.kernels.metal_fft_conv_streamed import MetalFFTConvStreamed
    streamed_conv = MetalFFTConvStreamed()
    print("  Running streamed kernel...")
    y_streamed = streamed_conv(u, k_time, D)
    mx.eval(y_streamed)

    # Compare - should be nearly bit-identical
    passed, max_abs, max_rel, mean_abs = compare_outputs(
        "Unified", y_unified,
        "Streamed", y_streamed,
        rtol=1e-5,  # Very tight tolerance - should be nearly identical
        atol=1e-6
    )

    return passed


def main():
    print("FFT Convolution Numerical Validation")
    print("="*80)

    # Test at various sequence lengths
    test_configs = [
        (512, "Unified kernel - short"),
        (1024, "Unified kernel - medium"),
        (2048, "Unified kernel - max length"),
        (4096, "Streamed kernel - 2x unified limit"),
        (8192, "Streamed kernel - 4x unified limit"),
        (16384, "Streamed kernel - 8x unified limit"),
        (32768, "Streamed kernel - 16x unified limit"),
    ]

    results = []

    for seq_len, description in test_configs:
        print(f"\n{description}")
        try:
            passed, max_abs, max_rel, mean_abs = test_numerical_accuracy(seq_len)
            results.append((seq_len, description, passed, max_abs, max_rel, mean_abs))
        except Exception as e:
            print(f"  ❌ FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((seq_len, description, False, float('inf'), float('inf'), float('inf')))

    # Test unified vs streamed consistency
    try:
        unified_vs_streamed_passed = test_unified_vs_streamed()
    except Exception as e:
        print(f"  ❌ FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        unified_vs_streamed_passed = False

    # Summary
    print(f"\n{'='*80}")
    print("NUMERICAL VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Length':<10} {'Type':<30} {'Status':<10} {'Max Abs Err':<15} {'Max Rel Err':<15}")
    print(f"{'-'*80}")

    for seq_len, desc, passed, max_abs, max_rel, _ in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{seq_len:<10} {desc:<30} {status:<10} {max_abs:<15.2e} {max_rel:<15.2e}")

    print(f"\nUnified vs Streamed consistency: {'✓ PASS' if unified_vs_streamed_passed else '✗ FAIL'}")

    # Final verdict
    all_passed = all(passed for _, _, passed, _, _, _ in results) and unified_vs_streamed_passed

    if all_passed:
        print(f"\n✓ ALL NUMERICAL VALIDATION TESTS PASSED")
        print("  Kernels produce correct results matching MLX native FFT reference!")
    else:
        print(f"\n✗ SOME TESTS FAILED - numerical accuracy issues detected")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
