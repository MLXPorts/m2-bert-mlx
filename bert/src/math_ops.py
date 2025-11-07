#!/usr/bin/env python
"""
Pure MLX mathematical operations for UKM experiments.

Zero NumPy. All operations use MLX tensors, MLX ops, MLX scalars.
PI is computed using Chudnovsky algorithm with pure MLX.
"""

import mlx.core as mx


def _calculate_pi_mlx(precision_digits=15):
    """
    Calculate pi using the Chudnovsky algorithm with pure MLX.

    Formula:
    1/π = (12/426880√10005) * Σ (6k)!(13591409 + 545140134k) / ((3k)!(k!)^3 * (-640320)^(3k))

    Returns:
        MLX array containing pi value
    """
    # Constants
    C = mx.array(640320, dtype=mx.int32)
    C3_OVER_24 = mx.divide(mx.power(C, mx.array(3, dtype=mx.int32)),
                           mx.array(24, dtype=mx.int32))
    DIGITS_PER_TERM = mx.array(14.1816474627254776555, dtype=mx.float32)

    def binary_split(a, b):
        """Recursive binary split for Chudnovsky."""
        a_val = mx.array(a, dtype=mx.int32)
        b_val = mx.array(b, dtype=mx.int32)
        diff = mx.subtract(b_val, a_val)

        # Base case
        if mx.all(mx.equal(diff, mx.array(1, dtype=mx.int32))):
            if mx.all(mx.equal(a_val, mx.array(0, dtype=mx.int32))):
                Pab = mx.array(1, dtype=mx.int64)
                Qab = mx.array(1, dtype=mx.int64)
            else:
                term1 = mx.subtract(mx.multiply(mx.array(6, dtype=mx.int64), a_val),
                                   mx.array(5, dtype=mx.int64))
                term2 = mx.subtract(mx.multiply(mx.array(2, dtype=mx.int64), a_val),
                                   mx.array(1, dtype=mx.int64))
                term3 = mx.subtract(mx.multiply(mx.array(6, dtype=mx.int64), a_val),
                                   mx.array(1, dtype=mx.int64))
                Pab = mx.multiply(mx.multiply(term1, term2), term3)
                Qab = mx.multiply(mx.power(a_val, mx.array(3, dtype=mx.int32)), C3_OVER_24)

            base_term = mx.array(13591409, dtype=mx.int64)
            multiplier = mx.array(545140134, dtype=mx.int64)
            term = mx.add(base_term, mx.multiply(multiplier, a_val))
            Tab = mx.multiply(Pab, term)

            # Negate if odd
            remainder = mx.remainder(a_val, mx.array(2, dtype=mx.int32))
            is_odd = mx.equal(remainder, mx.array(1, dtype=mx.int32))
            Tab = mx.where(is_odd, mx.negative(Tab), Tab)

            return Pab, Qab, Tab

        # Recursive case
        m_val = mx.floor(mx.divide(mx.add(a_val, b_val), mx.array(2, dtype=mx.int32)))
        m_int = int(m_val.item())

        Pam, Qam, Tam = binary_split(a, m_int)
        Pmb, Qmb, Tmb = binary_split(m_int, b)

        Pab = mx.multiply(Pam, Pmb)
        Qab = mx.multiply(Qam, Qmb)
        Tab = mx.add(mx.multiply(Qmb, Tam), mx.multiply(Pam, Tmb))

        return Pab, Qab, Tab

    # Calculate number of terms needed
    terms_float = mx.divide(mx.array(precision_digits, dtype=mx.float32), DIGITS_PER_TERM)
    terms_float = mx.add(terms_float, mx.array(1.0, dtype=mx.float32))
    terms = int(mx.floor(terms_float).item())

    # Compute binary split
    P, Q, T = binary_split(0, terms)

    # Calculate pi
    sqrt_10005 = mx.sqrt(mx.array(10005.0, dtype=mx.float32))
    numerator = mx.multiply(Q, mx.array(426880.0, dtype=mx.float32))
    numerator = mx.multiply(numerator, sqrt_10005)
    pi_val = mx.divide(numerator, T)

    return pi_val


# Global constant computed at module load
PI = _calculate_pi_mlx(15)


def pi():
    """Return mathematical constant pi as MLX array."""
    return PI


# Additional constants
def e():
    """Return mathematical constant e (Euler's number) as MLX array."""
    # e = sum(1/k!) from k=0 to infinity
    # For float32, 20 terms is sufficient
    result = mx.array(1.0, dtype=mx.float32)
    factorial = mx.array(1.0, dtype=mx.float32)

    for k in range(1, 20):
        k_val = mx.array(k, dtype=mx.float32)
        factorial = mx.multiply(factorial, k_val)
        term = mx.divide(mx.array(1.0, dtype=mx.float32), factorial)
        result = mx.add(result, term)

    return result


E = e()


def sqrt_2():
    """Return sqrt(2) as MLX array."""
    return mx.sqrt(mx.array(2.0, dtype=mx.float32))


def sqrt_2_over_pi():
    """Return sqrt(2/π) as MLX array."""
    two = mx.array(2.0, dtype=mx.float32)
    return mx.sqrt(mx.divide(two, PI))


# Comparison operations that return MLX arrays
def allclose(a, b, rtol=None, atol=None):
    """
    Check if two MLX arrays are element-wise equal within tolerance.

    Args:
        a: First MLX array
        b: Second MLX array
        rtol: Relative tolerance (default: 1e-5 for float32)
        atol: Absolute tolerance (default: 1e-8 for float32)

    Returns:
        MLX scalar (0 or 1) indicating if arrays are close
    """
    if rtol is None:
        rtol = mx.array(1e-5, dtype=mx.float32)
    else:
        rtol = mx.array(rtol, dtype=mx.float32)

    if atol is None:
        atol = mx.array(1e-8, dtype=mx.float32)
    else:
        atol = mx.array(atol, dtype=mx.float32)

    # Convert to MLX arrays
    a_arr = mx.array(a) if not isinstance(a, mx.array) else a
    b_arr = mx.array(b) if not isinstance(b, mx.array) else b

    # |a - b| <= atol + rtol * |b|
    diff = mx.abs(mx.subtract(a_arr, b_arr))
    threshold = mx.add(atol, mx.multiply(rtol, mx.abs(b_arr)))

    return mx.all(mx.less_equal(diff, threshold))


def array_equal(a, b):
    """Check if two MLX arrays are exactly equal."""
    a_arr = mx.array(a) if not isinstance(a, mx.array) else a
    b_arr = mx.array(b) if not isinstance(b, mx.array) else b
    return mx.all(mx.equal(a_arr, b_arr))


__all__ = [
    'PI',
    'E',
    'pi',
    'e',
    'sqrt_2',
    'sqrt_2_over_pi',
    'allclose',
    'array_equal',
]
