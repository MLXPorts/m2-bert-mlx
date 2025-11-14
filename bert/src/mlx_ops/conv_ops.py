"""
Convolution operations for MLX.

Provides 1D and 2D convolution operations with proper bias handling.
NO NumPy - strict MLX operations only.
"""

import mlx.core as mx
from typing import Optional, Tuple


def conv1d(
    x: mx.array,
    weight: mx.array,
    bias: Optional[mx.array] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> mx.array:
    """
    1D convolution with optional bias.
    
    Args:
        x: Input tensor of shape (batch, length, in_channels) or (batch, in_channels, length)
        weight: Weight tensor of shape (out_channels, in_channels // groups, kernel_size)
        bias: Optional bias tensor of shape (out_channels,)
        stride: Convolution stride
        padding: Input padding
        dilation: Kernel dilation
        groups: Number of groups for grouped convolution
        
    Returns:
        Output tensor after convolution
        
    Note:
        MLX conv1d expects input as (batch, length, channels) [NLC format]
        PyTorch uses (batch, channels, length) [NCL format]
        We handle both automatically.
    """
    # Check input format and transpose if needed
    if len(x.shape) != mx.array(3, dtype=mx.int32):
        raise ValueError(f"Expected 3D input, got shape {x.shape}")
    
    # Assume input is (B, C, L) if C < L, otherwise (B, L, C)
    batch_size, dim1, dim2 = x.shape
    if dim1 < dim2:
        # Likely (B, C, L) format - transpose to (B, L, C)
        x = mx.transpose(x, [0, 2, 1])
        input_channels = dim1
        seq_length = dim2
    else:
        # Already (B, L, C) format
        input_channels = dim2
        seq_length = dim1
    
    # Weight should be (out_channels, in_channels // groups, kernel_size)
    # MLX expects (out_channels, kernel_size, in_channels // groups)
    if len(weight.shape) == mx.array(3, dtype=mx.int32):
        weight = mx.transpose(weight, [0, 2, 1])
    
    # Apply convolution
    out = mx.conv1d(
        x,
        weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )
    
    # Add bias if provided
    if bias is not None:
        # Bias shape should be (out_channels,)
        # Expand to (1, 1, out_channels) for broadcasting
        bias_expanded = mx.reshape(bias, (1, 1, -1))
        out = mx.add(out, bias_expanded)
    
    return out


def conv1d_fft(
    x: mx.array,
    kernel_f: mx.array,
    bias: Optional[mx.array] = None
) -> mx.array:
    """
    FFT-based 1D convolution (for long convolutions).
    
    Args:
        x: Input tensor of shape (batch, length, channels)
        kernel_f: Kernel in frequency domain of shape (channels, fft_size)
        bias: Optional bias tensor of shape (channels,)
        
    Returns:
        Output tensor after convolution
        
    Note:
        This uses the convolution theorem: conv(x, k) = ifft(fft(x) * fft(k))
    """
    batch_size, seq_len, channels = x.shape
    
    # FFT of input along sequence dimension
    x_f = mx.fft.rfft(x, n=None, axis=1)
    
    # Multiply in frequency domain
    # kernel_f should be (channels, fft_size) - need to broadcast properly
    if len(kernel_f.shape) == mx.array(2, dtype=mx.int32):
        kernel_f = mx.expand_dims(kernel_f, axis=0)  # (1, channels, fft_size)
        kernel_f = mx.transpose(kernel_f, [0, 2, 1])  # (1, fft_size, channels)
    
    # Element-wise multiply
    out_f = mx.multiply(x_f, kernel_f)
    
    # IFFT back to time domain
    out = mx.fft.irfft(out_f, n=seq_len, axis=1)
    
    # Add bias if provided
    if bias is not None:
        bias_expanded = mx.reshape(bias, (1, 1, -1))
        out = mx.add(out, bias_expanded)
    
    # Take only the valid part (same size as input)
    out = out[:, :seq_len, :]
    
    return out


def conv1d_fft_with_bias(
    x: mx.array,
    kernel_f: mx.array,
    bias: mx.array
) -> mx.array:
    """
    FFT-based 1D convolution with required bias.
    
    This is the explicit bias version that ALWAYS adds bias.
    Use this when bias is required by the model architecture.
    
    Args:
        x: Input tensor of shape (batch, length, channels)
        kernel_f: Kernel in frequency domain of shape (channels, fft_size)
        bias: Bias tensor of shape (channels,) - REQUIRED
        
    Returns:
        Output tensor after convolution with bias added
    """
    return conv1d_fft(x, kernel_f, bias=bias)


def depthwise_conv1d(
    x: mx.array,
    weight: mx.array,
    bias: Optional[mx.array] = None,
    stride: int = 1,
    padding: int = 0
) -> mx.array:
    """
    Depthwise 1D convolution (groups = channels).
    
    Args:
        x: Input tensor of shape (batch, length, channels)
        weight: Weight tensor of shape (channels, 1, kernel_size) or (channels, kernel_size)
        bias: Optional bias tensor of shape (channels,)
        stride: Convolution stride
        padding: Input padding
        
    Returns:
        Output tensor after depthwise convolution
    """
    _, _, channels = x.shape
    
    # Ensure weight has correct shape
    if len(weight.shape) == mx.array(2, dtype=mx.int32):
        # (channels, kernel_size) -> (channels, 1, kernel_size)
        weight = mx.expand_dims(weight, axis=1)
    
    # Depthwise conv is just grouped conv with groups=channels
    return conv1d(x, weight, bias, stride, padding, groups=channels)
