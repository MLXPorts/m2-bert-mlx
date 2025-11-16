# Torch → MLX Porting Inventory (November 2025)

Purpose: Enumerate all remaining PyTorch (and disallowed NumPy) dependencies in the repository and recommend concrete MLX migration strategies or deprecation choices.

---
## Legend
- ✅ Already MLX (or MLX-adapted wrapper)
- 🟡 Needs light rewrite (API surface small, no CUDA/Triton)
- 🟠 Needs moderate rewrite (tensor ops + autograd + linalg)
- 🔴 Heavy / external dependency (CUDA kernels, Triton, Apex, torchmetrics, streaming dataset)
- ⚫ Candidate for removal (unused, obsolete, superseded by MLX code)

---
## High-Level Summary
Remaining PyTorch usage clusters:
1. Custom CUDA / Triton fused kernels (flash attention, fused dense, butterfly, blockdiag operations)
2. Dataset + data loader utilities (StreamingTextDataset, torch.utils.data, collators)
3. Metric libraries (torchmetrics)
4. Weight conversion and checkpoint ingestion (minimal torch.load in download scripts)
5. Autograd Functions for padding/indexing (bert_padding.py)
6. Numerical linear algebra (SVD in blockdiag_butterfly_projection)
7. Triton-based flash attention (flash_attn_triton.py)
8. Scattered test parity scripts (tests referencing torch for comparison)
9. Numpy usage for intermediate conversions (contrary to project rule to avoid numpy)

MLX Replacements / Status:
- Core model forward path: already MLX (bert_layers, monarch mixer, hyena filter, etc.) ✅
- Weight loading: custom pure Python (pytorch_loader + safetensors loader) ✅
- Classification & embeddings inference: fully MLX ✅

---
## Detailed File Inventory

### 1. ops / low-level kernel dependent modules
| File | Status | Key Torch APIs | Recommendation |
|------|--------|---------------|----------------|
| `bert/src/ops/blockdiag_butterfly_projection.py` | 🟠 | `torch.linalg.svd`, complex tensors, FFT | Replace with MLX SVD or implement approximate rank-1 factor via power iteration (if MLX lacks SVD). Keep an isolated adapter. |
| `bert/src/ops/blockdiag_butterfly_einsum.py` | 🔴 | Likely torch einsum and tensor shaping | Re-implement using MLX `mx.einsum` (if available) or manual reductions. |
| `bert/src/ops/blockdiag_multiply.py` | 🔴 | Autograd, fused multiply kernels, AMP decorators | Provide pure MLX fallback; drop AMP-specific annotations (MLX handles dtype casting differently). |
| `bert/src/ops/fused_dense.py` | 🔴 | Apex fused kernels, custom autograd, `torch.cuda.amp` | Replace with standard MLX linear + bias; evaluate performance; only keep if measurable gain. Mark original as deprecated. |
| `bert/src/ops/fused_softmax_dropout.py` | 🔴 | CUDA fused kernels | Implement MLX softmax + dropout composition; profile. |
| `bert/src/ops/low_rank.py` | 🟡 | Torch SVD (similar to butterfly) | Factorization: port to MLX; if not critical, inline simplified low-rank projection using iterative methods. |
| `bert/src/ops/bert_flashattention.py` | 🔴 | Torch tensor ops + FlashAttention style logic | Leverage existing Monarch Mixer / Hyena convolution path; remove or gate behind a flag. |
| `bert/src/flash_attn_triton.py` | 🔴 | Triton kernels, torch-specific memory layout | Consider complete removal; substitute with existing MLX attention or mixer ops. |

### 2. Data / Dataloaders
| File | Status | Key Dependencies | Recommendation |
|------|--------|------------------|----------------|
| `bert/src/text_data.py` | 🔴 | `streaming` lib, `torch.utils.data`, numpy, tokenizer HF | Build MLX-native iterable dataset wrapper. Replace torch tensors with `mx.array`; remove torch-specific collator logic (cumsum on torch) -> use `mx.cumsum`. |
| `bert/src/convert_dataset.py` | 🟠 | Torch + numpy for converting binary -> tokens | Inline MLX byte → int64 conversion using `mx.array(list_of_ints)`; avoid numpy intermediate. |
| `bert/src/glue/finetuning_jobs.py` | 🔴 | Torch DataLoader, optim (AdamW), metrics | For MLX classification fine-tuning: implement MLX training loop; replace metrics with manual implementations (accuracy, Spearman, MCC). |

### 3. Metrics & Optimizers
| File | Status | Torch Dep | Recommendation |
|------|--------|-----------|----------------|
| `bert/src/create_bert.py` | 🔴 | torchmetrics, AdamW | Provide MLX accuracy/MSE/Spearman/MCC functions; use MLX optimizers or custom step functions. |
| `bert/src/hf_bert.py` | 🔴 | Same set as above | Consolidate with `create_bert.py`; unify config; remove duplicate metric logic. |
| `bert/main.py` | 🟠 | Inline AdamW import usage | Replace with MLX optimizer or simple weight decay step manual update. |

### 4. Padding / Indexing Autograd Functions
| File | Status | Torch Dep | Recommendation |
|------|--------|-----------|----------------|
| `bert/src/ops/bert_padding.py` | 🟠 | Custom `torch.autograd.Function` | Re-implement pad/unpad using MLX advanced indexing & scatter/gather primitives (or simpler loop). Validate performance; assert parity with current tests. |

### 5. Weight Download / Conversion
| File | Status | Torch Dep | Recommendation |
|------|--------|-----------|----------------|
| `bert/download_weights.py` | 🟡 | `torch.load`, numpy for saving | For `.bin` only, keep minimal torch shim isolated; prefer converting once to safetensors via existing `pytorch_loader` logic. Remove numpy by using `mx.save_safetensors`. |

### 6. Tests
Multiple test files import torch for parity comparisons:
- `bert/tests/test_mlx_operations.py`
- `bert/tests/test_fftconv_mlx_vs_torch.py`
- `bert/tests/test_mlx_monarch_parity.py`
- etc.

Recommendation: Keep torch in test scope only (optional extra dependency) or add conditional skip: `if not torch_available: skip`. Provide MLX reference baselines to reduce reliance on torch execution.

### 7. Numpy Usage Inventory (Disallowed)
Common patterns:
- Conversion: `np.frombuffer(...)` (dataset ingest)
- Reshape / math: `np.prod`, `np.array` for saving

Migration:
- Replace `np.frombuffer` with Python `memoryview` + list comprehension or `array('l')` → `mx.array`.
- Use `math.prod` (Python 3.8+) or manual loop for dimension products.
- For saving: use `mx.save_safetensors` or plain binary chunk serialization when appropriate.

---
## Prioritized Migration Plan

### Phase 0 (Safety & Isolation)
- Create `legacy_torch/` folder; move heavy torch-dependent modules there (flash_attn_triton, fused_dense, butterfly) behind feature flag.
- Add runtime capability check: `USE_TORCH_LEGACY=0` to disable automatically.

### Phase 1 (Core Training Loop Parity)
- Implement MLX dataset iterator replacing `StreamingTextDataset` (minimal: yield token id batches from local shards).
- Implement MLX metrics (accuracy, MSE, Spearman, MCC) in `bert/src/metrics_mlx.py`.
- Provide MLX AdamW optimizer (or manual step) in `bert/src/optim/adamw_mlx.py`.

### Phase 2 (Ops Simplification)
- Replace `bert_padding.py` custom autograd with pure MLX version (no autograd function needed if we can express gather/scatter via `mx.take` + `mx.zeros` + assignment semantics).
- Remove `fused_dense.py` usage unless benchmarking proves >10% speed benefit. Use standard MLX Linear.

### Phase 3 (Advanced Kernels)
- Decide strategic value of butterfly projection & blockdiag operations. If kept, implement approximate SVD using power iterations in MLX (sufficient for rank-1). Document numeric deviation.
- Swap any remaining torch FFT calls with MLX or implement small Metal FFT wrapper already existing under `mlx_ops/kernels/metal_fft_conv`.

### Phase 4 (Testing & Deprecation)
- Update tests: add MLX-only versions; mark torch-parity tests with `@requires_torch`.
- Provide summary in `MLX_OPS_STATUS.md` referencing removed torch modules.

### Phase 5 (Full Numpy Elimination)
- Replace all `np.frombuffer` and `np.savez` usage; unify on `mx.save_safetensors`.
- Add lint rule (extend `emberlint_strict.py`) to forbid `import numpy` in non-test modules.

---
## Immediate Low-Risk Targets (Can Patch Quickly)
1. `download_weights.py`: Remove numpy conversion; save safetensors directly.
2. `convert_dataset.py`: Replace `np.frombuffer` path with MLX array construction.
3. `text_data.py`: Replace torch cumsum, eq, cat, from_numpy with MLX equivalents; drop DataLoader usage (provide simple generator). 
4. `bert_padding.py`: Replace custom autograd with functional pad/unpad.

---
## MLX API Mapping Cheat Sheet
| Torch | MLX Replacement |
|-------|-----------------|
| `torch.from_numpy(arr)` | `mx.array(arr.tolist())` (or better: build directly from bytes) |
| `torch.eye(n, dtype=complex)` | Build real & imaginary separately → `mx.array` stacked |
| `torch.cumsum(x, dim)` | `mx.cumsum(x, axis=dim)` |
| `torch.eq(a,b)` | `(a == b)` returns bool array |
| `torch.cat(tensors, dim)` | `mx.concatenate(tensors, axis=dim)` |
| `torch.gather` | Use indexing via `mx.take` or slice composition |
| `torch.scatter` | Construct zeros + assign or create mask & blend |
| `torch.linalg.svd(M)` | Not yet in MLX core (as of Nov 2025). Implement power iteration or randomized SVD manually |
| `torch.fft.fft` | Use existing Metal FFT kernels (mlx_ops) or implement Cooley–Tukey in Python if small |
| `torchmetrics.*` | Manual implementations (accuracy, confusion matrix, Spearman, MCC) |

---
## Risk Notes
- Removing fused kernels may reduce throughput. Benchmark before deprecating for production.
- Approximating SVD can change numerical stability of butterfly projection; test on representative inputs.
- Flash attention removal must be offset by optimized Monarch Mixer path; confirm memory footprint and latency.

---
## Suggested Next Actions
1. Implement MLX padding/unpadding replacement (bert_padding) and remove `torch.autograd.Function` usage.
2. Port `text_data.py` tokenization path to MLX arrays (batch dynamic length). 
3. Introduce `metrics_mlx.py` and switch training scripts off torchmetrics.
4. Add lint rule to block future `import torch` in non-legacy modules.

---
## Tracking
Add a section in `MLX_OPS_STATUS.md` referencing this inventory and mark progress per file.

---
## Appendix: Quick Detection Script (optional)
Run:
```bash
grep -R "import torch" -n bert/src | cut -d: -f1 | sort -u
```
Then verify each file is either in `legacy_torch/` or fully ported.

---
Prepared for: M2-BERT MLX Port
Author: Automated assistant based on repository scan
Date: 2025-11-15

