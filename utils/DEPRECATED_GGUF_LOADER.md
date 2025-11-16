# Deprecated: gguf_loader.py

The file `utils/gguf_loader.py` has been deprecated.

Rationale:
- MLX `mx.load` now supports GGUF directly.
- Maintaining partial, unimplemented GGUF parsing code adds noise.
- Configuration inference is handled via `infer_config_from_safetensors` or fallback to `config.json`.

Actions Taken (2025-11-15):
- Removed import and usage from `infer_config_from_checkpoint.py`.
- Kept the original file for reference; can be deleted once all downstream references are confirmed absent.

Next Steps:
- Delete `utils/gguf_loader.py` after verifying no external tooling expects it.
- Add lint rule to prevent reintroduction of GGUF manual parsing unless MLX API changes.

