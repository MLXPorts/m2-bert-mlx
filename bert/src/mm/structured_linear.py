# Adapted from https://github.com/HazyResearch/fly/tree/master/src/models/layers

import mlx.core as mx
import mlx.nn as nn


class StructuredLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        """Subclasses should call reset_parameters."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Subclasses may override {in,out}_features_extended
        if not hasattr(self, 'in_features_extended'):
            self.in_features_extended = in_features
        if not hasattr(self, 'out_features_extended'):
            self.out_features_extended = out_features
        if bias:
            self.bias = mx.zeros((out_features,), dtype=mx.float32)
        else:
            self.bias = None

    def reset_parameters(self) -> None:
        dense_weight = self._kaiming_uniform(self.out_features_extended, self.in_features_extended)
        self.set_weights_from_dense(dense_weight)
        self.reset_parameters_bias()

    def _kaiming_uniform(self, out_features, in_features):
        fan_in_mx = mx.array(in_features, dtype=mx.float32)
        zero = mx.array(0.0, dtype=mx.float32)
        six = mx.array(6.0, dtype=mx.float32)

        # bound = sqrt(6.0 / fan_in) if fan_in > 0 else 0.0
        is_positive = mx.greater(fan_in_mx, zero)
        ratio = mx.divide(six, fan_in_mx)
        bound = mx.where(is_positive, mx.sqrt(ratio), zero)

        low = mx.multiply(mx.array(-1.0, dtype=mx.float32), bound)
        high = bound
        return mx.random.uniform(shape=(out_features, in_features), low=low, high=high)

    def set_weights_from_dense(self, dense_weight: mx.array):
        raise NotImplementedError

    def reset_parameters_bias(self):
        if self.bias is not None:
            fan_in_mx = mx.array(self.in_features, dtype=mx.float32)
            zero_mx = mx.array(0.0, dtype=mx.float32)
            one_mx = mx.array(1.0, dtype=mx.float32)

            # fan_in = self.in_features if self.in_features > 0 else 1
            is_positive = mx.greater(fan_in_mx, zero_mx)
            fan_in_safe = mx.where(is_positive, fan_in_mx, one_mx)

            # bound = 1.0 / sqrt(fan_in)
            bound = mx.divide(one_mx, mx.sqrt(fan_in_safe))

            low = mx.multiply(mx.array(-1.0, dtype=mx.float32), bound)
            high = bound

            self.bias = mx.random.uniform(
                shape=self.bias.shape,
                low=low,
                high=high,
            )

    @property
    def saving(self):
        raise NotImplementedError

    def convert_to_dense_weight(self):
        eye = mx.eye(self.in_features, dtype=mx.float32)
        dense_weight = self.forward_matmul(eye).T
        return dense_weight

    def preprocess(self, x):
        in_features = x.shape[-1]
        if in_features < self.in_features_extended:
            # pad_amount = self.in_features_extended - in_features
            in_features_mx = mx.array(in_features, dtype=mx.int32)
            in_features_ext_mx = mx.array(self.in_features_extended, dtype=mx.int32)
            pad_amount = mx.subtract(in_features_ext_mx, in_features_mx)

            zero = mx.array(0, dtype=mx.int32)
            pads = [(zero, zero)] * (x.ndim - 1) + [(zero, pad_amount)]
            x = mx.pad(x, pads)
        return x

    def postprocess(self, output):
        out_features_extended = output.shape[-1]
        if out_features_extended > self.out_features:
            output = output[..., :self.out_features]
        return output

    def forward_matmul(self, x):
        raise NotImplementedError

    def forward(self, x):
        output = self.forward_matmul(x)
        if self.bias is not None:
            bias = self.bias.astype(output.dtype)
            # shape = (1,) * (output.ndim - 1) + (bias.shape[-1],)
            # Build shape list using MLX scalars
            one = mx.array(1, dtype=mx.int32)
            ndim_minus_1 = output.ndim - 1  # Python int for loop
            shape = [one] * ndim_minus_1 + [mx.array(bias.shape[-1], dtype=mx.int32)]
            output = mx.add(output, mx.reshape(bias, shape))
        return output
