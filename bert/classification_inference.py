#!/usr/bin/env python3
"""
M2-BERT MLX Classification Inference

Provides an API similar to embeddings_inference for sequence classification.
Supports:
  - Single sentence classification
  - Sentence pair classification
  - Batch inference with dynamic padding
  - Softmax probability output (or raw logits)

Uses BertForSequenceClassification defined in bert/src/bert_layers.py.
Assumes weights already converted to safetensors via existing loader path.
"""
from pathlib import Path
from typing import List, Dict, Optional, Union
import sys
import yaml
import mlx.core as mx
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'bert' / 'src'
UTILS_DIR = PROJECT_ROOT / 'utils'
for p in (PROJECT_ROOT, SRC_DIR, UTILS_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Import modules directly now that src directory is on path (avoid src.__init__ usage)
import configuration_bert
import bert_layers
from pytorch_loader import load_pytorch_bin

BertConfig = configuration_bert.BertConfig
BertForSequenceClassification = bert_layers.BertForSequenceClassification

# Minimal task label mappings (extendable)
TASK_NUM_LABELS = {
    'sst2': 2,
    'cola': 2,
    'mrpc': 2,
    'qqp': 2,
    'rte': 2,
    'qnli': 2,
    'mnli': 3,
    'stsb': 1,  # regression
}

class M2BertClassifier:
    def __init__(
        self,
        checkpoint: Union[str, Path],
        yaml_path: Union[str, Path],
        task: str = 'sst2',
        device: Optional[str] = None,
        return_probabilities: bool = True,
        allow_missing_head: bool = True,
        init_missing_head_mode: str = 'zero',  # 'zero' or 'random'
    ):
        self.task = task.lower()
        self.return_probabilities = return_probabilities
        if self.task not in TASK_NUM_LABELS:
            raise ValueError(f"Unsupported task '{task}'. Extend TASK_NUM_LABELS to include it.")
        self.num_labels = TASK_NUM_LABELS[self.task]

        # Load YAML config (reuse existing pattern)
        with open(yaml_path, 'r') as f:
            cfg_all = yaml.safe_load(f)
        model_cfg = cfg_all['model']['model_config']
        # Force classification-relevant overrides
        model_cfg['num_labels'] = self.num_labels
        model_cfg['gather_sentence_embeddings'] = False
        model_cfg['use_cls_token'] = True
        # Ensure large max position embeddings but allow dynamic sequence lengths
        evaluation_max_seq_len = cfg_all.get('evaluation_max_seq_len', 8192)

        # Normalize numeric and boolean fields possibly represented as strings or placeholders
        def _coerce(val):
            if isinstance(val, str):
                # placeholder referencing max_seq_len
                if val.strip().startswith('${') and val.strip().endswith('}'):  # e.g. ${max_seq_len}
                    key = val.strip()[2:-1]
                    return cfg_all.get(key, val)
                # boolean strings
                if val.lower() in ('true','false'):
                    return val.lower() == 'true'
                # purely digits
                if val.isdigit():
                    return int(val)
                # float
                try:
                    return float(val)
                except ValueError:
                    return val
            return val
        for k,v in list(model_cfg.items()):
            model_cfg[k] = _coerce(v)
        # Ensure long_conv_l_max resolves to int
        if isinstance(model_cfg.get('long_conv_l_max'), str):
            # fallback to max_seq_len numeric
            model_cfg['long_conv_l_max'] = int(cfg_all.get('max_seq_len', 8192))

        # Build config
        config = BertConfig(**model_cfg)
        config.max_position_embeddings = _coerce(cfg_all.get('max_seq_len', evaluation_max_seq_len))
        self.config = config

        # Force-disable positional expansion for classification inference (retrieval checkpoint already expanded)
        config.expand_positional_embeddings = False
        config.use_positional_encodings = False
        config.sequence_token_planting = False

        # Ensure optional classifier config fields exist
        opt_defaults = {
            'classifier_dropout': None,
            'performing_BEIR_evaluation': False,
            'expand_positional_embeddings': False,
            'sequence_token_planting': False,
            'gather_sentence_embeddings': False,
            'use_cls_token': True,
        }
        for k,v in opt_defaults.items():
            if not hasattr(config, k):
                setattr(config, k, v)

        # Load tokenizer (same as embeddings)
        tokenizer_name = cfg_all['model'].get('tokenizer_name', 'bert-base-uncased')
        tokenizer_path = Path.home() / f".cache/huggingface/hub/models--{tokenizer_name}/snapshots"
        # Prefer existing tokenizer.json from retrieval checkpoint path if available
        # Fallback to HF local cache by name
        # We'll require user to have tokenizer.json present next to checkpoint
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_dir():
            # try snapshot style: look for pytorch_model.bin if directory given
            bin_path = checkpoint_path / 'pytorch_model.bin'
            if bin_path.exists():
                checkpoint_path = bin_path
            else:
                # try to locate tokenizer.json in same snapshot directory
                snap_tok = checkpoint_path / 'tokenizer.json'
                if snap_tok.exists():
                    self.tokenizer = Tokenizer.from_file(str(snap_tok))
                else:
                    # fallback: raise explicit message
                    raise FileNotFoundError(f"tokenizer.json not found in checkpoint dir {checkpoint_path}")
        else:
            # If a file was provided, assume standard retrieval layout
            tok_guess = checkpoint_path.parent / 'tokenizer.json'
            if tok_guess.exists():
                self.tokenizer = Tokenizer.from_file(str(tok_guess))
            else:
                raise FileNotFoundError(f"Could not locate tokenizer.json near {checkpoint_path}")

        # Load weights following embeddings_inference.py pattern
        print(f"Loading weights from: {checkpoint_path}")
        checkpoint_raw = load_pytorch_bin(checkpoint_path)

        # Extract model weights from checkpoint structure (Composer format)
        if 'state' in checkpoint_raw and 'model' in checkpoint_raw['state']:
            state_dict = checkpoint_raw['state']['model']
            del checkpoint_raw['state']
            del checkpoint_raw
        elif 'model' in checkpoint_raw:
            state_dict = checkpoint_raw['model']
            del checkpoint_raw
        else:
            state_dict = checkpoint_raw
            del checkpoint_raw

        print(f"Loaded {len(state_dict)} tensors from checkpoint")

        # Instantiate model
        self.model = BertForSequenceClassification(self.config)

        # Prepare weights list (clean keys)
        weights_list = []
        for key, value in state_dict.items():
            # Clean key following embeddings_inference pattern
            clean_key = key.replace("module.", "")  # DataParallel wrapper
            clean_key = clean_key.replace("model.", "", 1)  # Composer wrapper
            clean_key = clean_key.replace("bert.", "", 1)  # BertForSequenceClassification wrapper

            weights_list.append((clean_key, value))

        # Clear state_dict to free memory before loading
        del state_dict
        import gc
        gc.collect()

        # Load into model using load_weights (strict=False to skip missing/extra keys)
        print(f"Loading {len(weights_list)} weights into model...")
        self.model.load_weights(weights_list, strict=False)

        # Clear weights_list after loading
        del weights_list
        gc.collect()

        print(f"✓ Weights loaded successfully")

        # Handle missing classifier head - check if classifier exists and has proper weights
        try:
            if not hasattr(self.model.classifier, 'weight') or self.model.classifier.weight.size == 0:
                head_missing = True
            else:
                head_missing = False
        except:
            head_missing = True

        # Handle missing classifier head
        if head_missing and allow_missing_head:
            import mlx.nn as nn
            print(f'[INFO] Initializing new classifier head: {init_missing_head_mode}')
            self.model.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
            if init_missing_head_mode == 'zero':
                self.model.classifier.weight = mx.zeros((self.num_labels, self.config.hidden_size))
                self.model.classifier.bias = mx.zeros((self.num_labels,))
        elif head_missing:
            raise RuntimeError('Classifier head missing and allow_missing_head=False')

        self.label_names = self._default_label_names()
        print(f'✅ M2BertClassifier ready. Task: {self.task}, Num labels: {self.num_labels}')

    def _default_label_names(self) -> List[str]:
        if self.task == 'sst2':
            return ['negative', 'positive']
        if self.task == 'cola':
            return ['unacceptable', 'acceptable']
        if self.task in ['mrpc', 'qqp']:
            return ['not_paraphrase', 'paraphrase']
        if self.task in ['rte', 'qnli']:
            return ['not_entailment', 'entailment']
        if self.task == 'mnli':
            return ['entailment', 'neutral', 'contradiction']
        if self.task == 'stsb':
            return ['similarity']  # regression
        return [f'label_{i}' for i in range(self.num_labels)]

    def _softmax(self, logits: mx.array) -> mx.array:
        # Numerically stable softmax
        maxv = mx.max(logits, axis=1, keepdims=True)
        exps = mx.exp(logits - maxv)
        return exps / mx.sum(exps, axis=1, keepdims=True)

    def _prepare_batch(self, texts_a: List[str], texts_b: Optional[List[str]] = None) -> Dict[str, mx.array]:
        # Tokenize individually, dynamic padding to longest in batch
        encodings = []
        pair = texts_b is not None
        if pair and len(texts_a) != len(texts_b):
            raise ValueError("texts_a and texts_b must have same length for sentence pair tasks")
        for i, ta in enumerate(texts_a):
            if pair:
                tb = texts_b[i]
                # Manually construct pair sequence: [CLS] A [SEP] B [SEP]
                enc_a = self.tokenizer.encode(ta, add_special_tokens=False)
                enc_b = self.tokenizer.encode(tb, add_special_tokens=False)
                ids = [101] + enc_a.ids + [102] + enc_b.ids + [102]
                mask = [1] * len(ids)
            else:
                enc = self.tokenizer.encode(ta, add_special_tokens=True)
                ids = enc.ids
                mask = enc.attention_mask
            encodings.append((ids, mask))
        max_len = max(len(ids) for ids, _ in encodings)
        padded_ids = []
        padded_mask = []
        for ids, mask in encodings:
            if len(ids) < max_len:
                pad_len = max_len - len(ids)
                ids = ids + [0] * pad_len
                mask = mask + [0] * pad_len
            padded_ids.append(ids)
            padded_mask.append(mask)
        input_ids = mx.array(padded_ids, dtype=mx.int32)
        attention_mask = mx.array(padded_mask, dtype=mx.int32)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def classify(self, texts: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        if isinstance(texts, str):
            batch = [texts]
        else:
            batch = texts
        batch_tensors = self._prepare_batch(batch)
        outputs = self.model(
            input_ids=batch_tensors['input_ids'],
            attention_mask=batch_tensors['attention_mask'],
            return_dict=True,
        )
        logits = outputs.logits
        if self.num_labels == 1:
            # Regression task
            preds = logits.squeeze().tolist()
            wrapped = [{"label": self.label_names[0], "value": p} for p in (preds if isinstance(preds, list) else [preds])]
            return wrapped if isinstance(texts, list) else wrapped[0]
        if self.return_probabilities:
            probs = self._softmax(logits)
            results = []
            for p in probs.tolist():
                max_idx = max(range(len(p)), key=lambda i: p[i])
                results.append({
                    "label_index": max_idx,
                    "label": self.label_names[max_idx],
                    "probabilities": {self.label_names[i]: p[i] for i in range(len(p))},
                    "confidence": p[max_idx]
                })
        else:
            logits_list = logits.tolist()
            results = []
            for row in logits_list:
                max_idx = max(range(len(row)), key=lambda i: row[i])
                results.append({
                    "label_index": max_idx,
                    "label": self.label_names[max_idx],
                    "logits": {self.label_names[i]: row[i] for i in range(len(row))},
                    "score": row[max_idx]
                })
        return results if isinstance(texts, list) else results[0]

    def classify_pairs(self, pairs: List[tuple]) -> List[Dict]:
        texts_a = [a for a,_ in pairs]
        texts_b = [b for _,b in pairs]
        batch_tensors = self._prepare_batch(texts_a, texts_b)
        outputs = self.model(
            input_ids=batch_tensors['input_ids'],
            attention_mask=batch_tensors['attention_mask'],
            return_dict=True,
        )
        logits = outputs.logits
        if self.num_labels == 1:
            preds = logits.squeeze().tolist()
            return [{"label": self.label_names[0], "value": p} for p in (preds if isinstance(preds, list) else [preds])]
        probs = self._softmax(logits)
        results = []
        for p in probs.tolist():
            max_idx = max(range(len(p)), key=lambda i: p[i])
            results.append({
                "label_index": max_idx,
                "label": self.label_names[max_idx],
                "probabilities": {self.label_names[i]: p[i] for i in range(len(p))},
                "confidence": p[max_idx]
            })
        return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='M2-BERT MLX Classification Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .bin or cached safetensors checkpoint')
    parser.add_argument('--yaml', type=str, required=True, help='Path to model YAML config')
    parser.add_argument('--task', type=str, default='sst2')
    parser.add_argument('--text', type=str, help='Single text to classify')
    parser.add_argument('--text_pair', type=str, help='Optional second text for pair classification')
    parser.add_argument('--raw_logits', action='store_true', help='Return raw logits instead of probabilities')
    args = parser.parse_args()

    clf = M2BertClassifier(
        checkpoint=args.checkpoint,
        yaml_path=args.yaml,
        task=args.task,
        return_probabilities=not args.raw_logits,
    )

    if args.text and args.text_pair:
        res = clf.classify_pairs([(args.text, args.text_pair)])
        print(res[0])
    elif args.text:
        res = clf.classify(args.text)
        print(res)
    else:
        print("Provide --text (and optionally --text_pair).")
