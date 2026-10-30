# Paired prior-state probe v1

Frozen protocol: `docs/research/rescue_state_v1_protocol.md`, commit `7c60503` before model inference. Catalog: `artifacts/rescue_state_catalog_v1`.

Two machines, physical GPU1 on each. Each model compares baseline and label-only state_boundaries on the same machine and checkpoint. Both use Transformers4.51.3; local base uses torch2.8.0, remote GRPO uses torch2.7.1. Timing across models is not a hardware-controlled comparison.

Preflight contains scripted reference and FINISH-only trajectories. Model runs contain all twelve trajectories per arm, manifests with source/checkpoint hashes, and original benchmark exports. Runtime caches and weights are excluded.

Recompute:

```bash
python scripts/analyze_rescue_state.py --catalog artifacts/rescue_state_catalog_v1 \
  --runs artifacts/rescue_state_v1/model_runs --output /tmp/rescue_state_analysis.json
```

Strict scoring is prescribed ordered calls and arguments plus successful termination. Extra calls can be valid yet fail this score. Download/translation use offline stubs; force=true is an argument contract, not an actual network refresh. These are six request pairs in two families, not twelve independent benchmark samples. No model selection or final-test evaluation is performed.
