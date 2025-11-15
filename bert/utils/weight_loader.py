#!/usr/bin/env python
"""
Weight loader for M2-BERT checkpoints into MLX.

Maps PyTorch checkpoint structure to MLX BertModel hierarchy.
"""

from typing import Dict
import mlx.core as mx


def load_bert_weights(weights_dict: Dict[str, mx.array], model):
    """
    Load BERT weights from checkpoint dict into BertModel.

    Checkpoint structure (HuggingFace/Composer):
        model.bert.embeddings.word_embeddings.weight
        model.bert.embeddings.position_embeddings.weight
        model.bert.embeddings.token_type_embeddings.weight
        model.bert.embeddings.LayerNorm.weight/bias
        model.bert.encoder.layer.{i}.attention.self.query.weight/bias
        model.bert.encoder.layer.{i}.attention.self.key.weight/bias
        model.bert.encoder.layer.{i}.attention.self.value.weight/bias
        model.bert.encoder.layer.{i}.attention.output.dense.weight/bias
        model.bert.encoder.layer.{i}.attention.output.LayerNorm.weight/bias
        model.bert.encoder.layer.{i}.intermediate.dense.weight/bias
        model.bert.encoder.layer.{i}.output.dense.weight/bias
        model.bert.encoder.layer.{i}.output.LayerNorm.weight/bias
        model.bert.pooler.dense.weight/bias

    Args:
        weights_dict: Dictionary of {key: mx.array} from checkpoint
        model: BertModel instance to load weights into
    """
    print(f"Loading {len(weights_dict)} weight tensors into BERT model...")

    # Remove 'model.' prefix if present
    cleaned_weights = {}
    for key, value in weights_dict.items():
        if key.startswith('model.'):
            cleaned_weights[key[6:]] = value
        else:
            cleaned_weights[key] = value

    # Load embedding weights
    load_embeddings(cleaned_weights, model.bert.embeddings)

    # Load encoder layers
    num_layers = len(model.bert.encoder.layer)
    print(f"Loading {num_layers} encoder layers...")
    for i in range(num_layers):
        load_encoder_layer(cleaned_weights, i, model.bert.encoder.layer[i])

    # Load pooler if present
    if hasattr(model.bert, 'pooler') and model.bert.pooler is not None:
        load_pooler(cleaned_weights, model.bert.pooler)

    print(f"\n✅ Successfully loaded all BERT weights")


def load_embeddings(weights: Dict[str, mx.array], embeddings):
    """Load BERT embedding weights."""
    prefix = 'bert.embeddings'

    mappings = {
        f'{prefix}.word_embeddings.weight': ('word_embeddings', 'weight'),
        f'{prefix}.position_embeddings.weight': ('position_embeddings', 'weight'),
        f'{prefix}.token_type_embeddings.weight': ('token_type_embeddings', 'weight'),
        f'{prefix}.LayerNorm.weight': ('LayerNorm', 'weight'),
        f'{prefix}.LayerNorm.bias': ('LayerNorm', 'bias'),
    }

    for ckpt_key, (module, param) in mappings.items():
        if ckpt_key in weights:
            target = getattr(embeddings, module)
            setattr(target, param, weights[ckpt_key])
            print(f"  ✓ {ckpt_key}")
        else:
            print(f"  ⚠ Missing: {ckpt_key}")

    # Load position_ids if present (not a parameter, just a buffer)
    pos_ids_key = f'{prefix}.position_ids'
    if pos_ids_key in weights:
        embeddings.position_ids = weights[pos_ids_key]


def load_encoder_layer(weights: Dict[str, mx.array], layer_idx: int, layer):
    """Load weights for a single BERT encoder layer."""
    prefix = f'bert.encoder.layer.{layer_idx}'

    # Attention weights
    attn_mappings = {
        f'{prefix}.attention.self.query.weight': ('attention', 'self_attn', 'query', 'weight'),
        f'{prefix}.attention.self.query.bias': ('attention', 'self_attn', 'query', 'bias'),
        f'{prefix}.attention.self.key.weight': ('attention', 'self_attn', 'key', 'weight'),
        f'{prefix}.attention.self.key.bias': ('attention', 'self_attn', 'key', 'bias'),
        f'{prefix}.attention.self.value.weight': ('attention', 'self_attn', 'value', 'weight'),
        f'{prefix}.attention.self.value.bias': ('attention', 'self_attn', 'value', 'bias'),
        f'{prefix}.attention.output.dense.weight': ('attention', 'output', 'dense', 'weight'),
        f'{prefix}.attention.output.dense.bias': ('attention', 'output', 'dense', 'bias'),
        f'{prefix}.attention.output.LayerNorm.weight': ('attention', 'output', 'layer_norm', 'weight'),
        f'{prefix}.attention.output.LayerNorm.bias': ('attention', 'output', 'layer_norm', 'bias'),
    }

    for ckpt_key, path in attn_mappings.items():
        if ckpt_key in weights:
            target = layer
            # Navigate to the target module
            for module_name in path[:-1]:
                target = getattr(target, module_name)
            # Set the parameter
            setattr(target, path[-1], weights[ckpt_key])

    # FFN/Intermediate weights
    ffn_mappings = {
        f'{prefix}.intermediate.dense.weight': ('intermediate', 'dense', 'weight'),
        f'{prefix}.intermediate.dense.bias': ('intermediate', 'dense', 'bias'),
        f'{prefix}.output.dense.weight': ('output', 'dense', 'weight'),
        f'{prefix}.output.dense.bias': ('output', 'dense', 'bias'),
        f'{prefix}.output.LayerNorm.weight': ('output', 'layer_norm', 'weight'),
        f'{prefix}.output.LayerNorm.bias': ('output', 'layer_norm', 'bias'),
    }

    for ckpt_key, path in ffn_mappings.items():
        if ckpt_key in weights:
            target = layer
            for module_name in path[:-1]:
                target = getattr(target, module_name)
            setattr(target, path[-1], weights[ckpt_key])


def load_pooler(weights: Dict[str, mx.array], pooler):
    """Load BERT pooler weights."""
    prefix = 'bert.pooler'

    mappings = {
        f'{prefix}.dense.weight': ('dense', 'weight'),
        f'{prefix}.dense.bias': ('dense', 'bias'),
    }

    for ckpt_key, (module, param) in mappings.items():
        if ckpt_key in weights:
            target = getattr(pooler, module)
            setattr(target, param, weights[ckpt_key])
            print(f"  ✓ {ckpt_key}")
