#!/usr/bin/env python
import mlx.core as mx


def build_twiddle_tg_kernel():
    # Body-only MSL: touch threadgroup buffers twCos/twSin to ensure allocation
    body = r"""
    uint i = thread_position_in_grid.x;
    if (i >= halfN[0]) return;
    // write some values into TG buffers
    twCos[i] = 1.0f;
    twSin[i] = 0.0f;
    // Optionally read back and write to out to prevent DCE
    out[i] = twCos[i] + twSin[i];
    """
    return mx.fast.metal_kernel(
        name="tg_twiddle_probe",
        input_names=["halfN"],
        output_names=["out"],
        threadgroup_names=["twCos", "twSin"],
        source=body,
    )


def probe_sizes(sizes):
    ker = build_twiddle_tg_kernel()
    results = []
    for N in sizes:
        half = N // 2
        # Allocate output
        out_shape = (half,)
        tg_sizes = [half * 4, half * 4]  # bytes for float32 planes
        try:
            (out,) = ker(
                inputs=[mx.array([half], dtype=mx.uint32)],
                grid=(mx.array(half, dtype=mx.uint32), mx.array(1, dtype=mx.uint32), mx.array(1, dtype=mx.uint32)),
                threadgroup=(mx.array(128, dtype=mx.uint32), mx.array(1, dtype=mx.uint32), mx.array(1, dtype=mx.uint32)),
                output_shapes=[out_shape],
                output_dtypes=[mx.float32],
                threadgroup_sizes=tg_sizes,
            )
            mx.eval(out)
            ok = True
        except Exception as e:
            print(f"N={N} failed: {e}")
            ok = False
        results.append((N, ok))
    return results


def main():
    sizes = [512, 1024, 2048, 4096, 8192, 16384]
    res = probe_sizes(sizes)
    for N, ok in res:
        print(f"N={N:<6} TG twiddle alloc: {'ok' if ok else 'fail'}")


if __name__ == "__main__":
    main()

