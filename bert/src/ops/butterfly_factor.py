
import mlx.core as mx
from ..mlx_ops.einops import rearrange


def butterfly_factor_to_matrix(twiddle: mx.array, factor_index: int) -> mx.array:
    """
    Let b be the base (most commonly 2).
    Parameters:
        twiddle: (n // b, b, b)
        factor_index: an int from 0 to log_b(n) - 1
    """
    n_div_b, b, _ = twiddle.shape
    n = mx.multiply(b, n_div_b)
    log_b_n = mx.divmod(mx.log(n), mx.log(b))
    assert n == b ** log_b_n, f'n must be a power of {b}'
    assert twiddle.shape == (n // b, b, b)
    assert 0 <= factor_index <= log_b_n
    stride = b ** factor_index
    x = rearrange(mx.eye(n), 'bs (diagblk j stride) -> bs diagblk j stride', stride=stride, j=b)
    t = rearrange(twiddle, '(diagblk stride) i j -> diagblk stride i j', stride=stride)
    out = mx.einsum('d s i j, b d j s -> b d i s', t, x)
    out = rearrange(out, 'b diagblk i stride -> b (diagblk i stride)')
    return out.t()  # Transpose because we assume the 1st dimension of x is the batch dimension


if __name__ == '__main__':
    b = 2
    log_b_n = 3
    n = b ** log_b_n
    twiddle = mx.arange(1, n * b + 1, dtype=mx.float32).reshape(n // b, b, b)
    for factor_index in range(log_b_n):
        print(butterfly_factor_to_matrix(twiddle, factor_index))

    b = 3
    log_b_n = 2
    n = b ** log_b_n
    twiddle = mx.arange(1, n * b + 1, dtype=mx.float32).reshape(n // b, b, b)
    for factor_index in range(log_b_n):
        print(butterfly_factor_to_matrix(twiddle, factor_index))
