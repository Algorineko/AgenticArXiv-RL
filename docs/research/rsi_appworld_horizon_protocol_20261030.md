# Exposed-train horizon diagnostic

Frozen before invocation: same first two AppWorld train scenarios, same mini API model, temperature 0, output cap 1024, native official checker and maximum four tool calls per response. Use the isolated original AgenticArxiv ToolRegistry adapter, validated against 39 past observations and four terminal verdicts without differences.

Change: maximum model steps 12 → 24, fixed `configs/rsi/bounded_discovery_24.txt` system instruction. Native environment remains at 48 interactions. Global caps: 50 API attempts including retries, 600,000 input and 50,000 output accounted tokens, 15 seconds minimum request spacing. Unknown usage remains conservatively reserved. No GPU or final-pool access.

This is a posthoc development diagnostic, not equal-budget evidence, not an RSI candidate, and not a fresh held-out evaluation. Both prompt and horizon differ from prior runs, so success cannot isolate either cause. Count all truncation, over-parallel responses and failures unchanged. Protected traces/source snapshots stay outside Git; publish only hashes and aggregates.
