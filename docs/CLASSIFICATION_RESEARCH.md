# M2-BERT MLX Classification Implementation - Research & Requirements

## Executive Summary

To fully implement classification functionality for M2-BERT MLX, we need to:
1. **Create inference pipeline** for BertForSequenceClassification
2. **Support multiple classification tasks** (GLUE benchmark, sentiment analysis, etc.)
3. **Handle different model variants** (80M, 110M, 260M, 341M)
4. **Implement evaluation metrics** for each task type
5. **Support fine-tuned checkpoints** from HuggingFace
6. **Create unified API** similar to embeddings_inference.py

## Architecture Analysis

### Existing Components ✅

1. **BertForSequenceClassification** (`bert/src/bert_layers.py:1299`)
   - Already implemented in MLX
   - Structure: BertModel + Dropout + Linear classifier
   - Supports `num_labels` configuration
   - Has `from_composer()` classmethod for loading checkpoints
   
2. **Weight Loading Infrastructure** ✅
   - Pure Python/MLX PyTorch unpickler
   - Safetensors caching
   - BFloat16 support
   - Dynamic padding

3. **BertModel** ✅
   - Fully functional with FFT convolution
   - Handles variable-length sequences
   - Memory-optimized

### Missing Components ❌

1. **Classification Inference Script**
   - No equivalent of `embed_text.py` for classification
   - Need unified API for different tasks
   
2. **Task-Specific Processors**
   - GLUE task configurations
   - Input preprocessing for each task
   - Output postprocessing (softmax, argmax, etc.)

3. **Evaluation Framework**
   - Metrics computation (accuracy, F1, Matthews correlation, etc.)
   - Task-specific evaluation logic
   - Batch processing for test sets

4. **Model Loading for Classification**
   - Adaptation of weight loader for classifier head
   - Handle different checkpoint formats (Composer, HuggingFace)
   - Support for multiple model sizes

## Available Models

### HuggingFace Models with Classification Support

From HuggingFace Hub search:

#### Text Classification Models:
1. **danfu09/m2-bert-80M** - text-classification
2. **danfu09/m2-bert-110M** - text-classification  
3. **danfu09/m2-bert-260m** - text-classification
4. **danfu09/m2-bert-341m** - text-classification

#### Retrieval/Embedding Models (also support classification):
5. **togethercomputer/m2-bert-80M-8k-retrieval** - text-classification, sentence-similarity
6. **togethercomputer/m2-bert-80M-32k-retrieval** - text-classification, sentence-similarity (ALREADY CACHED)
7. **togethercomputer/m2-bert-80M-2k-retrieval** - text-classification, sentence-similarity

Note: The 32k-retrieval model we have cached already supports classification!

## GLUE Benchmark Tasks

The General Language Understanding Evaluation (GLUE) benchmark consists of 9 tasks:

### Task Types & Metrics

1. **CoLA** (Corpus of Linguistic Acceptability)
   - Task: Binary classification (grammatical vs ungrammatical)
   - Metric: Matthews Correlation Coefficient (MCC)
   - Single sentence input

2. **SST-2** (Stanford Sentiment Treebank)
   - Task: Binary sentiment classification (positive/negative)
   - Metric: Accuracy
   - Single sentence input

3. **MRPC** (Microsoft Research Paraphrase Corpus)
   - Task: Paraphrase detection (binary)
   - Metric: F1 & Accuracy
   - Sentence pair input

4. **QQP** (Quora Question Pairs)
   - Task: Duplicate question detection (binary)
   - Metric: F1 & Accuracy
   - Sentence pair input

5. **STS-B** (Semantic Textual Similarity Benchmark)
   - Task: Regression (similarity score 0-5)
   - Metric: Pearson & Spearman correlation
   - Sentence pair input

6. **MNLI** (Multi-Genre Natural Language Inference)
   - Task: 3-way classification (entailment/contradiction/neutral)
   - Metric: Accuracy (matched & mismatched)
   - Sentence pair input

7. **QNLI** (Question Natural Language Inference)
   - Task: Binary classification (entailment/not entailment)
   - Metric: Accuracy
   - Sentence pair input

8. **RTE** (Recognizing Textual Entailment)
   - Task: Binary classification (entailment/not entailment)
   - Metric: Accuracy
   - Sentence pair input

9. **WNLI** (Winograd Natural Language Inference)
   - Task: Binary classification
   - Metric: Accuracy
   - Sentence pair input (typically excluded from GLUE due to issues)

### Input Formats

**Single Sentence Tasks:** CoLA, SST-2
- Input: `[CLS] sentence [SEP]`
- Tokens: One text sequence

**Sentence Pair Tasks:** MRPC, QQP, STS-B, MNLI, QNLI, RTE, WNLI
- Input: `[CLS] sentence_a [SEP] sentence_b [SEP]`
- Tokens: Two text sequences separated by [SEP]

## Existing Implementation (PyTorch/Composer)

### Current Training Pipeline

From `bert/glue.py`:
```python
TASK_NAME_TO_CLASS = {
    'mnli': finetuning_jobs_module.MNLIJob,
    'rte': finetuning_jobs_module.RTEJob,
    'mrpc': finetuning_jobs_module.MRPCJob,
    'qnli': finetuning_jobs_module.QNLIJob,
    'qqp': finetuning_jobs_module.QQPJob,
    'sst2': finetuning_jobs_module.SST2Job,
    'stsb': finetuning_jobs_module.STSBJob,
    'cola': finetuning_jobs_module.COLAJob,
}

TASK_NAME_TO_NUM_LABELS = {
    'mnli': 3,
    'rte': 2,
    'mrpc': 2,
    'qnli': 2,
    'qqp': 2,
    'sst2': 2,
    'stsb': 1,  # Regression
    'cola': 2,
}
```

### Checkpoint Structure

From `BertForSequenceClassification.from_composer()`:
- Composer format: `checkpoint['state']['model']`
- BEIR evaluation format: `checkpoint` with 'model.' prefix
- Contains both BERT weights and classifier head

### Special Requirements

1. **Position Embedding Expansion**
   - For long context models (>512 tokens)
   - Repeats base embeddings to reach target length
   - Handled in `from_composer()` method

2. **Sequence Token Planting**
   - Inserts [CLS] tokens at regular intervals
   - For very long sequences (>512)
   - Optional via `config.sequence_token_planting`

3. **Input Validation**
   - Must match `config.max_position_embeddings`
   - Attention mask must match input length
   - NaN checking in outputs

## Implementation Plan

### Phase 1: Basic Classification Inference ✅ (Easiest)

**Goal:** Single text classification with existing cached model

**Components:**
1. **classify_text.py** - Simple inference script
   - Load model (BertForSequenceClassification)
   - Tokenize input text
   - Run forward pass
   - Return class probabilities + predicted label
   
2. **classification_inference.py** - Inference class
   - Similar to M2_BERT_Encoder
   - Support both single sentence and sentence pairs
   - Handle different num_labels
   - Dynamic padding

**Dependencies:**
- Existing weight loader ✅
- BertForSequenceClassification ✅
- Tokenizer ✅

**Estimated Time:** 2-3 hours

### Phase 2: GLUE Task Support 🔧 (Moderate)

**Goal:** Full GLUE benchmark evaluation

**Components:**
1. **glue_config.py** - Task configurations
   - Define task types, metrics, num_labels
   - Input/output processors for each task
   
2. **glue_metrics.py** - Metric implementations
   - Accuracy, F1, MCC, Pearson/Spearman correlation
   - Pure Python/MLX implementations
   
3. **glue_evaluator.py** - Evaluation pipeline
   - Load task datasets
   - Batch processing
   - Metric computation
   - Results aggregation

**Dependencies:**
- Phase 1 ✅
- Dataset loading library (HuggingFace datasets or custom)
- Metric implementations

**Estimated Time:** 4-6 hours

### Phase 3: Multi-Model Support 🏗️ (Complex)

**Goal:** Support all M2-BERT sizes and checkpoint formats

**Components:**
1. **model_registry.py** - Model configurations
   - Define all available models
   - Map model names to HuggingFace repos
   - Configuration templates
   
2. **checkpoint_adapter.py** - Universal checkpoint loader
   - Handle Composer format
   - Handle HuggingFace format
   - Handle different weight naming schemes
   - Auto-detect model size/config

3. **auto_classifier.py** - Auto-configuration
   - Infer task type from checkpoint
   - Auto-download models
   - Smart defaults

**Dependencies:**
- Phase 1 & 2 ✅
- Extended weight loader
- HuggingFace Hub integration

**Estimated Time:** 6-8 hours

### Phase 4: Advanced Features 🚀 (Extended)

**Goal:** Production-ready classification system

**Components:**
1. **Batch Inference**
   - Efficient batch processing
   - Progress tracking
   - Memory management
   
2. **Model Serving**
   - REST API endpoint
   - Request batching
   - Response caching
   
3. **Fine-tuning Support**
   - Training loop in MLX
   - Gradient computation
   - Optimizer integration
   
4. **Benchmark Suite**
   - Automated GLUE evaluation
   - Performance comparison vs PyTorch
   - Speed/accuracy tradeoffs

**Dependencies:**
- All previous phases ✅
- MLX training infrastructure
- Web framework (FastAPI, Flask)

**Estimated Time:** 8-12 hours

## Technical Considerations

### 1. Memory Management

**Issue:** Classification requires full sequence processing (no chunking)
**Solution:** 
- Use dynamic padding (already implemented)
- Batch size optimization
- Gradient checkpointing for training

### 2. Tokenization

**Issue:** Sentence pair inputs need special handling
**Solution:**
- Use `tokenizer.encode_plus()` with `text_pair` parameter
- Proper [SEP] token placement
- Token type IDs for segment differentiation

### 3. Output Processing

**Different output types:**
- **Binary/Multi-class:** Apply softmax, return probabilities
- **Regression:** Return raw logit value
- **Multi-label:** Apply sigmoid per class

### 4. Checkpoint Compatibility

**Challenge:** Multiple checkpoint formats
**Solutions:**
- Auto-detection based on keys
- Fallback strategies
- Clear error messages

### 5. Evaluation Metrics

**Challenge:** Different metrics for different tasks
**Implementation:**
```python
TASK_TO_METRICS = {
    'cola': ['mcc'],
    'sst2': ['accuracy'],
    'mrpc': ['accuracy', 'f1'],
    'qqp': ['accuracy', 'f1'],
    'stsb': ['pearson', 'spearman'],
    'mnli': ['accuracy'],
    'qnli': ['accuracy'],
    'rte': ['accuracy'],
}
```

## Dependencies & Requirements

### Required Packages

**Core:**
- mlx-core (already installed)
- tokenizers (already installed)

**For GLUE Evaluation:**
- datasets (HuggingFace) - for loading GLUE tasks
- scikit-learn - for metrics (F1, MCC, etc.)
- scipy - for Pearson/Spearman correlation

**Optional:**
- wandb - for logging/tracking
- tqdm - for progress bars (already used)

### Installation
```bash
pip install datasets scikit-learn scipy
```

## API Design

### Classification Inference API

```python
from bert.classification_inference import M2_BERT_Classifier

# Initialize classifier
classifier = M2_BERT_Classifier(
    checkpoint="togethercomputer/m2-bert-80M-32k-retrieval",
    task="sst2",  # or auto-detect
    device="gpu"
)

# Single text classification
result = classifier.classify("This movie is amazing!")
# Output: {'label': 'positive', 'confidence': 0.987}

# Sentence pair classification
result = classifier.classify_pair(
    "The cat sat on the mat.",
    "A feline was resting on a rug."
)
# Output: {'label': 'paraphrase', 'confidence': 0.823}

# Batch classification
results = classifier.classify_batch([
    "Great product!",
    "Terrible service.",
    "Just okay."
])
# Output: [
#   {'label': 'positive', 'confidence': 0.956},
#   {'label': 'negative', 'confidence': 0.891},
#   {'label': 'neutral', 'confidence': 0.623}
# ]
```

### GLUE Evaluation API

```python
from bert.glue_evaluator import GLUEEvaluator

# Initialize evaluator
evaluator = GLUEEvaluator(
    model_name="togethercomputer/m2-bert-80M-32k-retrieval",
    tasks=["sst2", "mrpc", "cola"]  # or "all"
)

# Run evaluation
results = evaluator.evaluate(
    split="validation",
    batch_size=32
)

# Output:
# {
#   'sst2': {'accuracy': 0.934},
#   'mrpc': {'accuracy': 0.856, 'f1': 0.891},
#   'cola': {'mcc': 0.601}
# }

# Generate report
evaluator.generate_report(results, output_path="glue_results.json")
```

## Testing Strategy

### Unit Tests
1. **Model Loading**
   - Load each model size
   - Verify architecture
   - Check parameter counts

2. **Tokenization**
   - Single sentence
   - Sentence pairs
   - Edge cases (empty, very long)

3. **Forward Pass**
   - Correct output shapes
   - No NaN/Inf values
   - Gradient flow (for training)

4. **Metrics**
   - Accuracy calculation
   - F1 score
   - MCC, Pearson, Spearman

### Integration Tests
1. **End-to-End Inference**
   - Load model → tokenize → classify → postprocess
   - Compare with PyTorch reference
   
2. **GLUE Evaluation**
   - Run on small subset
   - Verify metric computation
   - Check result formatting

### Validation Tests
1. **Model Parity**
   - Compare MLX vs PyTorch outputs
   - Tolerance: max error < 1e-4
   
2. **Performance**
   - Throughput (samples/sec)
   - Memory usage
   - Latency

## Success Criteria

### Minimum Viable Product (Phase 1)
- ✅ Load BertForSequenceClassification
- ✅ Classify single text
- ✅ Return probabilities + label
- ✅ Handle basic error cases

### Full GLUE Support (Phase 2)
- ✅ Run all 8 GLUE tasks (excluding WNLI)
- ✅ Compute correct metrics
- ✅ Match PyTorch baseline (within 1%)

### Production Ready (Phase 3+)
- ✅ Support all model sizes
- ✅ Auto-download models
- ✅ Batch inference optimized
- ✅ Documentation complete
- ✅ Test coverage >80%

## Timeline

**Total Estimated Time:** 20-29 hours

**Recommended Approach:**
1. Start with Phase 1 (2-3 hours) - Get basic inference working
2. Validate with existing cached model
3. Proceed to Phase 2 if needed (4-6 hours)
4. Phase 3 & 4 are optional enhancements

**Immediate Next Steps (Phase 1):**
1. Create `bert/classify_text.py` - Simple CLI script (30 min)
2. Create `bert/classification_inference.py` - Inference class (1.5 hours)
3. Test with cached 32k-retrieval model (30 min)
4. Write documentation (30 min)

## References

- M2-BERT Paper: https://arxiv.org/abs/2310.12109
- GLUE Benchmark: https://gluebenchmark.com/
- HuggingFace M2-BERT: https://huggingface.co/models?search=m2-bert
- Original Code: `/Volumes/stuff/Projects/m2/bert/`
- MLX Documentation: https://ml-explore.github.io/mlx/

## Appendix: Existing Code to Leverage

### From embeddings_inference.py
- Model loading pattern
- Tokenization logic
- Dynamic padding
- Batch processing
- Config handling

### From bert_layers.py
- BertForSequenceClassification
- from_composer() loading
- Position embedding expansion
- Input validation
- Output structure

### From glue.py (PyTorch)
- Task definitions
- Num labels mapping
- Job structure
- Metrics logging

## Conclusion

Classification support is highly feasible and can be implemented incrementally:

**Phase 1 (Basic Inference)** requires minimal new code - mostly adapting existing patterns from embeddings_inference.py. We already have a compatible model cached.

**Phase 2 (GLUE)** adds task-specific logic and metrics but builds directly on Phase 1.

**Phase 3 & 4** are enhancements that can be added over time based on actual needs.

**Recommendation:** Proceed with Phase 1 implementation immediately. It's low-risk, high-value, and demonstrates full model capabilities.

