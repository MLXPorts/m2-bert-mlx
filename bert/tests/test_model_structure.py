#!/usr/bin/env python3
"""Test script to understand BertForSequenceClassification structure."""
import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '../../utils')

from configuration_bert import BertConfig
from bert_layers import BertForSequenceClassification
import yaml

# Load config
with open('../yamls/embeddings/m2-bert-80M-32k-retrieval.yaml') as f:
    cfg = yaml.safe_load(f)

model_cfg = cfg['model']['model_config']
model_cfg['num_labels'] = 2
model_cfg['use_cls_token'] = True
model_cfg['gather_sentence_embeddings'] = False
model_cfg['use_positional_encodings'] = False
model_cfg['expand_positional_embeddings'] = False

# Coerce string values to proper types
def _coerce(val):
    if isinstance(val, str):
        if val.strip().startswith('${') and val.strip().endswith('}'):
            key = val.strip()[2:-1]
            return cfg.get(key, val)
        if val.lower() in ('true','false'):
            return val.lower() == 'true'
        if val.isdigit():
            return int(val)
        try:
            return float(val)
        except ValueError:
            return val
    return val

for k,v in list(model_cfg.items()):
    model_cfg[k] = _coerce(v)

# Ensure long_conv_l_max resolves to int
if isinstance(model_cfg.get('long_conv_l_max'), str):
    model_cfg['long_conv_l_max'] = int(cfg.get('max_seq_len', 8192))

config = BertConfig(**model_cfg)
config.max_position_embeddings = _coerce(cfg.get('max_seq_len', 8192))

# Add optional attributes
config.classifier_dropout = None
config.performing_BEIR_evaluation = False
config.expand_positional_embeddings = False
config.sequence_token_planting = False

# Create model
print("Creating model...")
model = BertForSequenceClassification(config)

# Check structure
print('\nModel structure:')
print(f'  model.bert type: {type(model.bert).__name__}')
print(f'  model.bert.encoder type: {type(model.bert.encoder).__name__}')
print(f'  model.bert.encoder.layer type: {type(model.bert.encoder.layer).__name__}')
print(f'  Length of encoder.layer: {len(model.bert.encoder.layer)}')

# Check first layer
layer_0 = model.bert.encoder.layer[0]
print(f'  First layer type: {type(layer_0).__name__}')

# Try navigation path for key: encoder.layer.0.mlp.wo.weight
print('\n--- Testing navigation path ---')
print('Target key: encoder.layer.0.mlp.wo.weight')
print('After stripping "bert.": encoder.layer.0.mlp.wo.weight')

obj = model.bert
print(f'\nStarting at: model.bert ({type(obj).__name__})')

obj = obj.encoder
print(f'  -> encoder ({type(obj).__name__})')

obj = obj.layer
print(f'  -> layer ({type(obj).__name__}, len={len(obj)})')

obj = obj[0]
print(f'  -> [0] ({type(obj).__name__})')

print(f'  -> has "mlp": {hasattr(obj, "mlp")}')
if hasattr(obj, 'mlp'):
    obj = obj.mlp
    print(f'  -> mlp ({type(obj).__name__})')

    print(f'  -> has "wo": {hasattr(obj, "wo")}')
    if hasattr(obj, 'wo'):
        obj = obj.wo
        print(f'  -> wo ({type(obj).__name__})')

        print(f'  -> has "weight": {hasattr(obj, "weight")}')
        if hasattr(obj, 'weight'):
            print(f'  -> weight shape: {obj.weight.shape}')
            print('\n✓ Navigation successful!')
        else:
            print('\n✗ Failed: wo has no "weight" attribute')
    else:
        print('\n✗ Failed: mlp has no "wo" attribute')
else:
    print('\n✗ Failed: layer[0] has no "mlp" attribute')

# Show what layer[0] actually has
print('\n--- Attributes of layer[0] ---')
attrs = [a for a in dir(layer_0) if not a.startswith('_')]
for attr in attrs[:20]:
    print(f'  {attr}')
if len(attrs) > 20:
    print(f'  ... and {len(attrs) - 20} more')

