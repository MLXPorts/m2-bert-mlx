# M2-BERT MLX: Status Report

**Date:** 2025-11-12
**Status:** ✅ FFT Kernel Operational | ⚠️ Weight Loading In Progress

---

## Next Steps: Classification Implementation

### Research Complete ✅

Comprehensive research documented in `CLASSIFICATION_RESEARCH.md`:
- **9 GLUE benchmark tasks** analyzed
- **7 compatible M2-BERT models** identified on HuggingFace
- **BertForSequenceClassification** already implemented in MLX
- **Existing 32k-retrieval model** supports classification (already cached!)

### Implementation Phases

#### Phase 1: Basic Classification Inference (IMMEDIATE - 2-3 hours)
**Status:** Ready to implement
**Goal:** Single text classification with existing cached model

**Components to Create:**
1. `bert/classify_text.py` - CLI script for quick testing
2. `bert/classification_inference.py` - Main inference class
   - Similar to `M2_BERT_Encoder` pattern
   - Support single sentence and sentence pairs
   - Handle different `num_labels` configurations
   - Dynamic padding (already implemented)

**What We Have:**
- ✅ BertForSequenceClassification (bert/src/bert_layers.py:1299)
- ✅ Weight loading infrastructure (pure Python/MLX)
- ✅ Tokenizer (BERT tokenizer from HF)
- ✅ Compatible model cached (togethercomputer/m2-bert-80M-32k-retrieval)

**What's Needed:**
- Inference wrapper class
- Output postprocessing (softmax, argmax)
- Simple CLI interface

#### Phase 2: GLUE Task Support (FUTURE - 4-6 hours)
- Task-specific processors
- Metrics implementation (MCC, F1, Pearson, Spearman)
- Evaluation pipeline
- Dataset loading

#### Phase 3: Multi-Model Support (FUTURE - 6-8 hours)
- Support all M2-BERT sizes (80M, 110M, 260M, 341M)
- Auto-download from HuggingFace
- Universal checkpoint adapter
- Model registry

#### Phase 4: Advanced Features (EXTENDED - 8-12 hours)
- Batch inference optimization
- Model serving API
- Fine-tuning support
- Benchmark suite

### Available Models (HuggingFace)

**Text Classification Models:**
- danfu09/m2-bert-80M
- danfu09/m2-bert-110M
- danfu09/m2-bert-260m
- danfu09/m2-bert-341m

**Retrieval Models (also support classification):**
- togethercomputer/m2-bert-80M-8k-retrieval
- **togethercomputer/m2-bert-80M-32k-retrieval** ← CACHED, USE THIS
- togethercomputer/m2-bert-80M-2k-retrieval

### GLUE Benchmark Tasks

**Binary Classification (2 labels):**
- CoLA (linguistic acceptability) - MCC metric
- SST-2 (sentiment) - Accuracy
- MRPC (paraphrase) - F1 & Accuracy
- QQP (duplicate questions) - F1 & Accuracy
- QNLI (question entailment) - Accuracy
- RTE (textual entailment) - Accuracy

**Multi-class Classification (3 labels):**
- MNLI (natural language inference) - Accuracy

**Regression (1 output):**
- STS-B (semantic similarity) - Pearson & Spearman correlation

### Recommended Next Action

**Start Phase 1 immediately:**
1. Create `bert/classification_inference.py` based on `embeddings_inference.py` pattern
2. Create `bert/classify_text.py` CLI script
3. Test with cached 32k-retrieval model
4. Validate outputs
5. Document usage

**Estimated completion:** 2-3 focused hours

This will demonstrate full M2-BERT MLX capabilities (embeddings ✅ + classification ✅) with minimal effort.

---

## Previous Accomplishments

### 1. ✅ Memory Optimization & Weight Loading (Nov 15, 2025)

**MAJOR BREAKTHROUGH:** Solved catastrophic memory usage (256GB → reasonable)

#### Issues Identified & Fixed:
1. **PyTorch Checkpoint Loading**
   - Problem: 4x memory duplication (checkpoint + state_dict + weights_list + model)
   - Solution: Pure Python/MLX unpickler, no torch/numpy dependencies
   - Added safetensors caching for instant subsequent loads
   - Result: First load ~1x model memory, subsequent loads instant

2. **BFloat16 Handling**
   - Problem: BFloat16 tensors misinterpreted as Float32 (wrong element count)
   - Solution: Proper bit-shifting conversion (bf16 << 16 = f32)
   - Result: All dtypes handled correctly

3. **Configuration Mismatch**
   - Problem: YAML had hidden_size=1792 but checkpoint was hidden_size=768
   - Solution: Fixed YAML to match actual checkpoint dimensions
   - Result: Model layers correctly sized

4. **Excessive Static Padding (CRITICAL)**
   - Problem: Always padding to 32768 tokens for every input
     - "Hello world" (5 tokens) → padded to 32768 tokens = 96.2 MB
   - Solution: Dynamic padding to actual batch max length
     - "Hello world" (5 tokens) → padded to 5 tokens = 15 KB
   - Result: **6554x memory reduction** for short inputs

#### Key Discoveries:
- FFT convolution handles variable-length sequences efficiently (no need to pad to max)
- Model supports up to 32k tokens but works perfectly with shorter sequences
- Original code's static padding was for benchmarking, not correctness

#### Files Created/Modified:
- `utils/pytorch_loader.py` - Pure Python/MLX unpickler with safetensors caching
- `bert/embeddings_inference.py` - Dynamic padding implementation
- `bert/yamls/embeddings/m2-bert-80M-32k-retrieval.yaml` - Fixed config
- `test_tokenization_roundtrip.py` - Comprehensive validation test
- `MEMORY_OPTIMIZATION_SUMMARY.md` - Detailed documentation
- `PYTORCH_LOADER_SUMMARY.md` - Loader implementation details

#### Test Results:
```bash
✓ Embeddings working: python3 bert/embed_text.py --text "Test"
✓ Round-trip validation: All texts decode correctly
✓ Dynamic padding: 3→6→6 tokens (not 32768!)
✓ Memory: 15 KB for short inputs (was 96 MB)
✓ Processing: ~6-7 it/s
```

### 2. ✅ FFT Convolutional Kernel - FULLY OPERATIONAL

The custom Metal FFT convolution kernel is fully integrated and tested:

- **Location:** `bert/src/mm_mlx/metal_fft_conv.py`
- **Features:**
  - JIT-compiled Metal kernel with double-double precision
  - In-kernel twiddle generation
  - Hermitian symmetry enforcement
  - Mixed-radix FFT (radix-2, radix-4)
  - Zero Python numerics in compute paths

- **Test Results:**
  - ✅ Accuracy vs PyTorch: max error < 7e-05
  - ✅ All sequence lengths (64-4096) passing
  - ✅ Full model integration verified
  - ✅ No NaN/Inf issues
  - ✅ Comprehensive test suite: `bert/tests/test_fft_kernels_comprehensive.py`

### 2. ✅ GEMM Kernel Integration

- Copied xLSTM GEMM kernels to `bert/src/kernels/gemm_kernels.py`
- Updated `blockdiag_multiply_mlx.py` to use local copy
- No external path dependencies

### 3. ✅ Checkpoint Conversion Infrastructure

Created `bert/src/mm_mlx/checkpoint_utils.py`:
- Converts PyTorch Composer checkpoints to MLX `.npz` format
- Analyzes checkpoint configuration automatically
- Provides `load_pretrained_m2bert()` function

**Successfully converted:** M2-BERT-341M (395.8M parameters)
- Input: `.model/model.pt` (3.8GB)
- Output: `.model/m2bert_341m_mlx.npz` (1.4GB compressed)

### 4. ✅ Model Architecture

MLX implementation in `bert/src/mm_mlx/`:
- `m2bert_model_mlx.py` - Full M2-BERT model
- `hyena_filter_mlx.py` - Hyena filter with FFT kernel
- `monarch_mixer_mlx.py` - Sequence mixing layer
- `monarch_mlp_mlx.py` - Monarch MLP with block-diagonal operations

**Model tested with random weights:** 87.6M parameters, all tests passing

---

## Current Status: Checkpoint Loading

The checkpoint conversion and loading infrastructure is operational:

### ✅ What Works
- Conversion from PyTorch `.pt` to MLX `.npz` format
- Configuration inference from checkpoint
- Name mapping for most parameters
- Forward pass with partially loaded weights
- FFT kernel fully operational in all layers

### ⚠️ Remaining Issue
The MLX model architecture has hardcoded parameters that don't match the 341M checkpoint:
- `hyena_emb_dim`: Model uses 3, checkpoint has 5
- `intermediate_size`: Model calculated incorrectly for this checkpoint
- Some architecture parameters not exposed in constructor

**Load success: 161/720 parameters (22.4%)**

### Solution
The M2BERTModel and MonarchMixerSequenceMixing constructors need to expose additional parameters:
- `hyena_emb_dim`
- Potentially other Hyena-specific parameters

This is a straightforward fix but requires modifying the model constructors and ensuring backward compatibility with existing code.

### ✅ Verification
Despite partial weight loading, the model:
- Runs forward passes successfully
- Produces clean outputs (no NaN/Inf) 
- Uses the FFT kernel in all Hyena operations
- Has correct overall architecture (12 layers, 1792 hidden size, etc.)

---

## Next Steps

### Immediate (Weight Loading)

1. **Create comprehensive name mapping function**
   - Map all PyTorch parameter paths to MLX paths
   - Handle nested structures properly
   - Verify shape compatibility
   - Test with pretrained 341M model

2. **Validate loaded weights**
   - Forward pass with real data
   - Compare outputs to PyTorch reference
   - Verify FFT kernel works with pretrained weights

### Future Enhancements

1. **Extended Precision Mode**
   - Implement double-double butterfly operations
   - Add EP flags to HyperProfile
   - Document precision vs. speed tradeoffs

2. **Backward Pass**
   - Implement gradients for FFT convolution
   - Enable end-to-end training in MLX

3. **Benchmarking**
   - Compare MLX vs PyTorch throughput
   - Test on various Apple Silicon configs
   - Document performance characteristics

---

## Files Modified/Created

### New Files
- `bert/src/kernels/gemm_kernels.py` - GEMM kernel (copied from xLSTM)
- `bert/src/kernels/__init__.py`
- `bert/src/mm_mlx/checkpoint_utils.py` - Checkpoint conversion utilities
- `bert/tests/test_fft_kernels_comprehensive.py` - Comprehensive FFT tests
- `bert/tests/FFT_KERNEL_STATUS.md` - Detailed kernel documentation
- `bert/tests/FFT_KERNEL_QUICKREF.md` - Quick reference guide
- `.model/m2bert_341m_mlx.npz` - Converted checkpoint (1.4GB)

### Modified Files
- `bert/src/mm_mlx/blockdiag_multiply_mlx.py` - Updated to use local GEMM kernel
- `bert/tests/test_fft_precision_ab.py` - Updated to current kernel API

---

## Usage Examples

### Convert Checkpoint
```bash
cd bert
python -m src.mm_mlx.checkpoint_utils convert \
    --checkpoint ../.model/model.pt \
    --output ../.model/m2bert_341m_mlx.npz
```

### Load Pretrained Model (once weight loading is fixed)
```python
from src.mm_mlx.checkpoint_utils import load_pretrained_m2bert
import mlx.core as mx

# Load model
model = load_pretrained_m2bert('../.model/m2bert_341m_mlx.npz')

# Inference
input_ids = mx.array([[101, 2023, 2003, 1037, 3231, 102]])  # "this is a test"
output = model(input_ids)
```

### Run Tests
```bash
cd bert

# FFT kernel tests
python tests/test_fftconv_mlx_vs_torch.py
python tests/test_fft_precision_ab.py
python tests/test_fft_kernels_comprehensive.py

# Model tests (random weights)
python -m src.mm_mlx.m2bert_model_mlx
```

---

## Documentation

- **FFT Kernel:** `bert/tests/FFT_KERNEL_STATUS.md`
- **Quick Ref:** `bert/tests/FFT_KERNEL_QUICKREF.md`
- **Numeric Stability:** `docs/NUMERIC_STABILITY_TORCH_vs_MLX.md`
- **This Report:** `STATUS_REPORT.md`

---

## Summary

✅ **The FFT convolutional kernel is production-ready and fully operational.**

The kernel achieves excellent accuracy (< 7e-05 max error vs PyTorch) and is properly integrated into the M2-BERT model architecture. The model runs successfully with random weights.

⚠️ **Weight loading from pretrained checkpoints needs refinement.**

The parameter name mapping logic needs to be enhanced to handle the complex nested structure of the MLX model. Once this is completed, the 341M pretrained model will be fully functional in MLX with the custom FFT kernel.

**Estimated time to complete weight loading:** 1-2 hours of focused work on parameter path mapping.
