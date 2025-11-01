#!/usr/bin/env python
"""
MLX port of core BERT blocks with Monarch mixer sequence mixing.

Implements:
- MLXBertEmbeddings (token + optional position + token type)
- MLX MLP variants (GLU and standard)
- MLXBertLayer (Monarch mixer for sequence mixing + MLP)
- MLXBertEncoder (stack of layers, optional position encodings add-back)

Notes
- This path targets the Monarch mixer configuration (sequence mixing = Hyena/Monarch).
- ALiBi and FlashAttention are only relevant for attention path and are not included here.
- Monarch MLP is not yet implemented; config.use_monarch_mlp will raise NotImplementedError.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

def _get_profile():
    try:
        from .mm_mlx.hyperprofiles_mlx import get_profile as gp  # type: ignore
        return gp()
    except Exception:
        # Fallback when loaded as a loose module: try absolute import with local path
        try:
            import os, sys
            pkg_root = os.path.dirname(__file__)
            if pkg_root not in sys.path:
                sys.path.insert(0, pkg_root)
            from mm_mlx.hyperprofiles_mlx import get_profile as gp2  # type: ignore
            return gp2()
        except Exception:
            class _P:
                layer_norm_strict = False
            return _P()

from src.mm_mlx.monarch_mixer_mlx import MonarchMixerSequenceMixing


class Embedding(nn.Module):
    """Minimal embedding layer when nn.Embedding is not present."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = mx.random.normal((num_embeddings, embedding_dim)) * 0.02

    def __call__(self, input_ids):
        # input_ids: (B, L)
        return mx.take(self.weight, input_ids, axis=0)


def _get_embedding(vocab_size: int, dim: int, padding_idx: Optional[int] = None):
    # Use nn.Embedding if available; MLX Embedding may not accept padding_idx
    if hasattr(nn, 'Embedding'):
        try:
            return nn.Embedding(vocab_size, dim)
        except Exception:
            return Embedding(vocab_size, dim, padding_idx)
    return Embedding(vocab_size, dim, padding_idx)


class MLXBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_positional_encodings = getattr(config, 'use_positional_encodings', False)
        self.word_embeddings = _get_embedding(config.vocab_size, config.hidden_size, getattr(config, 'pad_token_id', None))
        self.token_type_embeddings = _get_embedding(config.type_vocab_size, config.hidden_size)
        if self.use_positional_encodings:
            self.position_embeddings = _get_embedding(config.max_position_embeddings, config.hidden_size)
        if _get_profile().layer_norm_strict:
            self.LayerNorm = LayerNormStrict(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length: int = 0,
        return_position_encodings: bool = False,
    ):
        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError('Must specify either input_ids or inputs_embeds')
        if input_ids is not None:
            input_shape = input_ids.shape
            inputs_embeds = self.word_embeddings(input_ids)
        else:
            input_shape = inputs_embeds.shape[:-1]

        B, L = input_shape[0], input_shape[1]
        if token_type_ids is None:
            token_type_ids = mx.zeros((B, L), dtype=mx.int32)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = None
        if self.use_positional_encodings:
            if position_ids is None:
                position_ids = mx.arange(L, dtype=mx.int32)[None, :]
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        if return_position_encodings:
            return embeddings, position_embeddings
        return embeddings


class MLXBertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Activation via registry
        try:
            from .mm_mlx.activations_mlx import get_activation
        except Exception:
            from mm_mlx.activations_mlx import get_activation
        act_name = getattr(config, 'mlx_mlp_activation', 'gelu_tanh')
        self.act = get_activation(act_name)

        if getattr(config, 'use_monarch_mlp', False):
            from .mm_mlx.blockdiag_linear_mlx import BlockdiagLinear as BDLinear
            self.fc1 = BDLinear(
                config.hidden_size,
                config.intermediate_size,
                bias=False,
                nblocks=getattr(config, 'monarch_mlp_nblocks', 4),
            )
        else:
            self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        if _get_profile().layer_norm_strict:
            self.ln = LayerNormStrict(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.drop = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, x):
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.ln(x + residual)
        return x


class MLXBertGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Activation via registry
        try:
            from .mm_mlx.activations_mlx import get_activation
        except Exception:
            from mm_mlx.activations_mlx import get_activation
        act_name = getattr(config, 'mlx_mlp_activation', 'gelu_tanh')
        self.act = get_activation(act_name)

        if getattr(config, 'use_monarch_mlp', False):
            from .mm_mlx.blockdiag_linear_mlx import BlockdiagLinear as BDLinear
            self.fc_g = BDLinear(
                config.hidden_size,
                2 * config.intermediate_size,
                bias=False,
                nblocks=getattr(config, 'monarch_mlp_nblocks', 4),
            )
        else:
            self.fc_g = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        if _get_profile().layer_norm_strict:
            self.ln = LayerNormStrict(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x):
        residual = x
        h = self.fc_g(x)
        mid = h.shape[-1] // 2
        g, ng = h[:, :, :mid], h[:, :, mid:]
        x = self.act(g) * ng
        x = self.drop(x)
        x = self.fc2(x)
        x = self.ln(x + residual)
        return x


class MLXBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not getattr(config, 'monarch_mixer_sequence_mixing', False):
            raise NotImplementedError('MLX port targets Monarch mixer sequence mixing only')
        self.mixer = MonarchMixerSequenceMixing(
            config.hidden_size,
            l_max=getattr(config, 'long_conv_l_max', config.max_position_embeddings),
            hyena_kernel_lr=getattr(config, 'long_conv_kernel_learning_rate', 1e-3),
            bidirectional=getattr(config, 'bidirectional', True),
            hyena_lr_pos_emb=getattr(config, 'hyena_lr_pos_emb', 1e-5),
            hyena_w=getattr(config, 'hyena_w', 10),
            hyena_w_mod=getattr(config, 'hyena_w_mod', 1),
            hyena_wd=getattr(config, 'hyena_wd', 0.1),
            hyena_emb_dim=getattr(config, 'hyena_emb_dim', 3),
            hyena_filter_dropout=getattr(config, 'hyena_filter_dropout', 0.0),
            hyena_filter_order=getattr(config, 'hyena_filter_order', 64),
            residual_long_conv=getattr(config, 'residual_long_conv', False),
            hyena_training_additions=getattr(config, 'hyena_training_additions', False),
        )
        if getattr(config, 'use_glu_mlp', True):
            self.mlp = MLXBertGLUMLP(config)
        else:
            self.mlp = MLXBertMLP(config)

    def __call__(self, hidden_states):
        # hidden_states: (B, L, H)
        attn_out, _ = self.mixer(hidden_states)
        out = self.mlp(attn_out)
        return out


class MLXBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = MLXBertLayer(config)
        self.layers = [layer] + [MLXBertLayer(config) for _ in range(config.num_hidden_layers - 1)]
        self.use_positional_encodings = getattr(config, 'use_positional_encodings', False)

    def __call__(self, hidden_states, attention_mask=None, output_all_encoded_layers=True, position_encodings=None):
        all_layers = []
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if position_encodings is not None:
                hidden_states = hidden_states + position_encodings
            if output_all_encoded_layers:
                all_layers.append(hidden_states)
        return hidden_states if not output_all_encoded_layers else all_layers


class LayerNormStrict(nn.Module):
    """Torch-like LayerNorm with unbiased=False and eps inside the sqrt.

    Mirrors torch.nn.functional.layer_norm behavior on the last dimension.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((hidden_size,), dtype=mx.float32)
        self.bias = mx.zeros((hidden_size,), dtype=mx.float32)

    def __call__(self, x):
        # x: (..., H)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean(mx.power(x - mean, 2), axis=-1, keepdims=True)
        inv_std = mx.rsqrt(var + mx.array(self.eps, dtype=x.dtype))
        y = (x - mean) * inv_std
        # Affine
        return y * self.weight.reshape(*([1] * (x.ndim - 1)), -1) + self.bias.reshape(*([1] * (x.ndim - 1)), -1)
