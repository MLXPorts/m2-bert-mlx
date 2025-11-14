#!/usr/bin/env python
"""
Test FFT convolution kernel scaling to very long sequences.

Tests automatic switching between:
- Unified kernel (L <= 2048)
- Streamed kernel (L > 2048, supports unlimited length)
"""

import sys
sys.path.insert(0, 'src')

import mlx.core as mx
import time
from mlx_ops.hyena_filter import _hyena_fft_conv

def test_fft_conv_length(batch_size, d_model, seq_len):
    """Test FFT convolution at a specific sequence length."""
    print(f"\n{'='*80}")
    print(f"Testing: B={batch_size}, C={d_model}, L={seq_len}")
    print(f"  Expected kernel: {'UNIFIED (all-in-one)' if seq_len <= 2048 else 'STREAMED (4-phase)'}")

    # Create test data
    u = mx.random.normal((batch_size, d_model, seq_len)).astype(mx.float32)
    k_time = mx.random.normal((d_model, seq_len)).astype(mx.float32)
    D = mx.random.normal((1, d_model, 1)).astype(mx.float32)

    # Warm-up (compile kernel)
    print("  Compiling kernel...")
    _ = _hyena_fft_conv(u, k_time, D, seqlen=seq_len)
    mx.eval(_)

    # Benchmark
    print("  Benchmarking...")
    num_runs = 10
    times = []

    for i in range(num_runs):
        start = time.perf_counter()
        y = _hyena_fft_conv(u, k_time, D, seqlen=seq_len)
        mx.eval(y)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)

    # Verify output shape
    assert y.shape == (batch_size, d_model, seq_len), f"Wrong output shape: {y.shape}"

    # Check for NaNs/Infs
    has_nan = mx.isnan(y).any()
    has_inf = mx.isinf(y).any()
    mx.eval(has_nan)
    mx.eval(has_inf)

    print(f"  ✓ Output shape: {y.shape}")
    print(f"  ✓ Time: avg={avg_time*1000:.2f}ms, min={min_time*1000:.2f}ms")
    print(f"  ✓ NaN check: {bool(has_nan)} | Inf check: {bool(has_inf)}")

    if has_nan or has_inf:
        print(f"  ❌ NUMERICAL ISSUE DETECTED!")
        return False

    # Calculate throughput
    elements = batch_size * d_model * seq_len
    throughput = elements / (avg_time * 1e6)  # M elements/sec
    print(f"  ✓ Throughput: {throughput:.1f} M elements/sec")

    return True

def main():
    print("FFT Convolution Kernel Scaling Test")
    print("="*80)

    # Test configuration
    batch_size = 2
    d_model = 64

    # Test sequence lengths spanning the kernel transition
    test_lengths = [
        512,      # Short - unified kernel
        1024,     # Medium - unified kernel
        2048,     # Max unified kernel
        4096,     # Streamed kernel required
        8192,     # 2x streamed kernel
        16384,    # 4x streamed kernel
        32768,    # 8x streamed kernel
        65536,    # 16x streamed kernel (very long)
        131072,   # 32x streamed kernel (extremely long)
    ]

    results = []

    for seq_len in test_lengths:
        try:
            success = test_fft_conv_length(batch_size, d_model, seq_len)
            results.append((seq_len, success))
        except Exception as e:
            print(f"  ❌ FAILED with error: {e}")
            results.append((seq_len, False))
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Length':<10} {'Kernel Type':<20} {'Status':<10}")
    print(f"{'-'*80}")

    for seq_len, success in results:
        kernel_type = "Unified" if seq_len <= 2048 else "Streamed"
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{seq_len:<10} {kernel_type:<20} {status:<10}")

    # Final verdict
    all_passed = all(success for _, success in results)
    if all_passed:
        print(f"\n✓ ALL TESTS PASSED - FFT convolution scales to {max(test_lengths):,} sequence length!")
    else:
        print(f"\n✗ SOME TESTS FAILED - check errors above")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
