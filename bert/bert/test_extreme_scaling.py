#!/usr/bin/env python
"""
Test FFT convolution at extreme sequence lengths.
Push the limits to find the maximum supported length.
"""

import sys
sys.path.insert(0, 'src')

import mlx.core as mx
import time
from mlx_ops.hyena_filter import _hyena_fft_conv

def test_extreme_length(batch_size, d_model, seq_len):
    """Test FFT convolution at extreme sequence length."""
    print(f"\n{'='*80}")
    print(f"Testing L={seq_len:,} (B={batch_size}, C={d_model})")

    # Memory estimate
    elements = batch_size * d_model * seq_len
    memory_mb = (elements * 4) / (1024 * 1024)  # float32 = 4 bytes
    print(f"  Input size: {memory_mb:.1f} MB ({elements:,} elements)")

    try:
        # Create test data
        print("  Allocating tensors...")
        u = mx.random.normal((batch_size, d_model, seq_len)).astype(mx.float32)
        k_time = mx.random.normal((d_model, seq_len)).astype(mx.float32)
        D = mx.random.normal((1, d_model, 1)).astype(mx.float32)
        mx.eval(u)
        mx.eval(k_time)
        mx.eval(D)

        # Compile kernel
        print("  Compiling kernel...")
        start_compile = time.perf_counter()
        _ = _hyena_fft_conv(u, k_time, D, seqlen=seq_len)
        mx.eval(_)
        compile_time = time.perf_counter() - start_compile
        print(f"  ✓ Kernel compiled in {compile_time:.2f}s")

        # Run inference
        print("  Running inference...")
        start = time.perf_counter()
        y = _hyena_fft_conv(u, k_time, D, seqlen=seq_len)
        mx.eval(y)
        end = time.perf_counter()
        inference_time = end - start

        # Verify
        assert y.shape == (batch_size, d_model, seq_len), f"Wrong shape: {y.shape}"
        has_nan = bool(mx.isnan(y).any())
        has_inf = bool(mx.isinf(y).any())
        mx.eval(has_nan)
        mx.eval(has_inf)

        print(f"  ✓ Inference time: {inference_time*1000:.1f}ms")
        print(f"  ✓ Throughput: {elements/(inference_time*1e6):.1f} M elements/sec")
        print(f"  ✓ NaN/Inf check: {has_nan}/{has_inf}")

        if has_nan or has_inf:
            print(f"  ⚠️  NUMERICAL ISSUE")
            return False

        return True

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

def main():
    print("EXTREME FFT Convolution Scaling Test")
    print("="*80)
    print("Pushing to the limits...")

    # Use smaller batch/channels to maximize sequence length
    batch_size = 1
    d_model = 16

    # Extreme sequence lengths - powers of 2
    test_lengths = [
        2**18,    # 262,144
        2**19,    # 524,288
        2**20,    # 1,048,576 (1M tokens!)
        2**21,    # 2,097,152 (2M tokens!)
        2**22,    # 4,194,304 (4M tokens!)
    ]

    results = []

    for seq_len in test_lengths:
        success = test_extreme_length(batch_size, d_model, seq_len)
        results.append((seq_len, success))

        if not success:
            print(f"\n⛔ Stopped at L={seq_len:,} - reached limit")
            break

    # Summary
    print(f"\n{'='*80}")
    print("EXTREME SCALING SUMMARY")
    print(f"{'='*80}")

    max_working = 0
    for seq_len, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"L={seq_len:>10,}  {status}")
        if success:
            max_working = seq_len

    if max_working > 0:
        print(f"\n🎯 MAXIMUM VALIDATED LENGTH: {max_working:,} tokens")
        print(f"   That's {max_working/1024:.1f}K tokens or {max_working/(1024*1024):.2f}M tokens!")

    return 0

if __name__ == "__main__":
    sys.exit(main())
