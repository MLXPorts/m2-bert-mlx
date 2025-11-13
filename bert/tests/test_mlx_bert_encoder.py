#!/usr/bin/env python
import importlib.util
import numpy as np
import os

import mlx.core as mx

THIS_DIR = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
SRC_DIR = os.path.join(PKG_ROOT, 'src')

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

class Cfg:
    def __init__(self):
        self.vocab_size = 5000
        self.type_vocab_size = 2
        self.pad_token_id = 0
        self.hidden_size = 128
        self.intermediate_size = 256
        self.layer_norm_eps = 1e-5
        self.hidden_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.num_hidden_layers = 2
        self.use_positional_encodings = True
        self.monarch_mixer_sequence_mixing = True
        self.residual_long_conv = False
        self.bidirectional = True
        self.hyena_filter_dropout = 0.0
        self.hyena_filter_order = 32
        self.hyena_training_additions = False
        self.use_glu_mlp = True

enc_mod = load('bert_layers_mlx', os.path.join(SRC_DIR, 'bert_layers.py'))

def main():
    cfg = Cfg()
    emb = enc_mod.MLXBertEmbeddings(cfg)
    enc = enc_mod.MLXBertEncoder(cfg)
    B, L, H = 2, 64, 128
    input_ids = mx.array(np.random.randint(0, cfg.vocab_size, size=(B, L), dtype=np.int32))
    token_type_ids = mx.zeros((B, L), dtype=mx.int32)
    x = emb(input_ids=input_ids, token_type_ids=token_type_ids)
    y = enc(x, attention_mask=mx.ones((B, L)))
    if isinstance(y, list):
        y = y[-1]
    print('MLX BertEncoder output:', y.shape)

if __name__ == '__main__':
    main()

