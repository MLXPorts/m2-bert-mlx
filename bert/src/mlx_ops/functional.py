"""
MLX functional operations that don't have direct equivalents.
"""
import mlx.core as mx


def normalize(x: mx.array, p: float = 2, axis: int = -1, eps: float = 1e-12) -> mx.array:
    """
    Normalize array along given axis using Lp norm.
    
    Args:
        x: Input array
        p: Order of norm (default: 2 for L2 norm)
        axis: Axis along which to normalize
        eps: Small value to avoid division by zero
        
    Returns:
        Normalized array
    """
    if p == mx.array(2):
        # L2 normalization
        norm = mx.sqrt(mx.sum(mx.square(x), axis=axis, keepdims=True))
        norm = mx.maximum(norm, mx.array(eps))
        return mx.divide(x, norm)
    else:
        # General Lp normalization
        norm = mx.power(mx.sum(mx.power(mx.abs(x), mx.array(p)), axis=axis, keepdims=True), mx.divide(mx.array(1.0), mx.array(p)))
        norm = mx.maximum(norm, mx.array(eps))
        return mx.divide(x, norm)


def gelu(x: mx.array) -> mx.array:
    """
    GELU activation function.
    
    Args:
        x: Input array
        
    Returns:
        GELU(x)
    """
    # GEL(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the standard normal distribution
    # Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    return mx.multiply(mx.array(0.5), mx.multiply(x, mx.add(mx.array(1.0), mx.tanh(mx.multiply(mx.sqrt(mx.divide(mx.array(2.0), mx.array(3.14159265359))), mx.add(x, mx.multiply(mx.array(0.044715), mx.power(x, mx.array(3)))))))))


def expand_dims(x: mx.array, axis: int) -> mx.array:
    """
    Expand dimensions of array at given axis.
    
    Args:
        x: Input array
        axis: Axis at which to expand
        
    Returns:
        Array with expanded dimensions
    """
    return mx.expand_dims(x, axis=axis)


def layer_norm(x: mx.array, normalized_shape: int, weight: mx.array = None, bias: mx.array = None, eps: float = 1e-5) -> mx.array:
    """
    Layer normalization.
    
    Args:
        x: Input array
        normalized_shape: Size of the dimension to normalize over
        weight: Optional weight parameter
        bias: Optional bias parameter
        eps: Small value for numerical stability
        
    Returns:
        Layer normalized array
    """
    # Compute mean and variance over the last axis
    mean = mx.mean(x, axis=mx.array(-1), keepdims=True)
    var = mx.var(x, axis=mx.array(-1), keepdims=True)
    
    # Normalize
    x_norm = mx.divide(mx.subtract(x, mean), mx.sqrt(mx.add(var, mx.array(eps))))
    
    # Apply weight and bias if provided
    if weight is not None:
        x_norm = mx.multiply(x_norm, weight)
    if bias is not None:
        x_norm = mx.add(x_norm, bias)
        
    return x_norm
