# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Top-level package for the MosaicML BERT modules.

This module mirrors the structure of the original MosaicML examples repo but
relaxes optional dependencies so that numerics/parity tooling can import the
core building blocks without requiring Composer or GPU-only packages.
"""

from __future__ import annotations

import os
import sys
from typing import Callable

import torch

# Ensure relative imports (bert_layers, etc.) resolve regardless of CWD
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Base components available without extra deps
from bert_layers import (  # noqa: E402
    BertEmbeddings,
    BertEncoder,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertGatedLinearUnitMLP,
    BertLayer,
    BertLMPredictionHead,
    BertModel,
    BertOnlyMLMHead,
    BertOnlyNSPHead,
    BertPooler,
    BertPredictionHeadTransform,
    BertSelfOutput,
    BertUnpadAttention,
    BertUnpadSelfAttention,
)
from bert_padding import (  # noqa: E402
    IndexFirstAxis,
    IndexPutFirstAxis,
    index_first_axis,
    index_put_first_axis,
    pad_input,
    unpad_input,
    unpad_input_only,
)
from configuration_bert import BertConfig  # noqa: E402

# Flash attention helpers are optional (only when CUDA is available)
if torch.cuda.is_available():  # pragma: no cover - CUDA optional
    from flash_attn_triton import (  # type: ignore # noqa: E402
        flash_attn_func as flash_attn_func_bert,
        flash_attn_qkvpacked_func as flash_attn_qkvpacked_func_bert,
    )
else:  # pragma: no cover - CUDA optional
    flash_attn_func_bert = None  # type: ignore
    flash_attn_qkvpacked_func_bert = None  # type: ignore

# HuggingFace helpers are optional but usually present
try:  # pragma: no cover - optional dep
    from hf_bert import create_hf_bert_classification, create_hf_bert_mlm  # noqa: E402
except ImportError:  # pragma: no cover - optional dep
    create_hf_bert_classification = None
    create_hf_bert_mlm = None


def _missing_optional(name: str, exc: Exception) -> Callable:
    def _fn(*args, **kwargs):
        raise ImportError(
            f"{name} is unavailable because optional dependencies are missing. "
            "Install the Composer requirements (pip install -r requirements-cpu.txt)."
        ) from exc

    return _fn


try:  # pragma: no cover - optional dep
    from create_bert import (  # noqa: E402
        create_bert_classification as _create_mosaic_classification,
        create_bert_mlm as _create_mosaic_mlm,
    )
except ImportError as exc:  # pragma: no cover - optional dep
    create_bert_classification = _missing_optional('create_bert_classification', exc)
    create_bert_mlm = _missing_optional('create_bert_mlm', exc)
    create_mosaic_bert_classification = create_bert_classification
    create_mosaic_bert_mlm = create_bert_mlm
else:
    create_bert_classification = _create_mosaic_classification
    create_bert_mlm = _create_mosaic_mlm
    # Backwards-compatible aliases
    create_mosaic_bert_classification = _create_mosaic_classification
    create_mosaic_bert_mlm = _create_mosaic_mlm

__all__ = [
    'BertConfig',
    'BertEmbeddings',
    'BertEncoder',
    'BertForMaskedLM',
    'BertForSequenceClassification',
    'BertGatedLinearUnitMLP',
    'BertLayer',
    'BertLMPredictionHead',
    'BertModel',
    'BertOnlyMLMHead',
    'BertOnlyNSPHead',
    'BertPooler',
    'BertPredictionHeadTransform',
    'BertSelfOutput',
    'BertUnpadAttention',
    'BertUnpadSelfAttention',
    'IndexFirstAxis',
    'IndexPutFirstAxis',
    'index_first_axis',
    'index_put_first_axis',
    'pad_input',
    'unpad_input',
    'unpad_input_only',
    'create_bert_classification',
    'create_bert_mlm',
    'create_hf_bert_classification',
    'create_hf_bert_mlm',
    'create_mosaic_bert_classification',
    'create_mosaic_bert_mlm',
]
