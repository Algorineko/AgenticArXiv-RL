# P0 CPU/API calibration protocol

Frozen before research API inference. This is development calibration, not an RSI comparison.

- Eight generated development tasks: four inventory reservation/dispatch cases and four table aggregation cases, including latest-record semantics.
- Two complete passes of the unchanged `doubao-seed-2.0-mini` baseline; temperature 0 does not imply deterministic API responses.
- Per episode: six model calls maximum; four tool calls per response maximum; 512 generated tokens per call. Native function calls only; no generated Python execution.
- Whole run cap: 100 attempted API requests, 300,000 accounted input tokens and 50,000 accounted output tokens. Before each request reserve a conservative input estimate and maximum output. Unknown usage, including timeouts, is charged to the reservation. Provider overhead cannot be guaranteed by a local estimate; observed overruns stop the run.
- HTTP transient retry once at most, counted as an additional request. Non-transient errors stop. Existing output directories are refused.
- Primary metric: independently verified final goal state. Normal assistant termination, generation truncation, tool errors, and budget stops are reported separately.
- Every task must pass a deterministic reference strategy; no-op and invalid-state/output controls must fail. Table references use Python aggregation while the goal oracle uses SQLite. Inventory oracle checks final state including unrelated stock.
- Record task catalog, source digests, protocol, all API attempts and tool traces. Credentials are supplied only via `RSI_API_KEY` and are removed from logged content if echoed.
- Do not optimize prompts or select models on these repeated runs. If baseline completion exceeds 85% or remains below 20%, revise only the development calibration in a separately versioned protocol before designing the independent meta-train/dev/test problem pools.
- A usable P0 does not establish novelty or recursive improvement. P1 requires a separate frozen protocol with experience-matched fixed control, fixed outer meta-optimizer and recursive meta-optimizer.

Implementation: `scripts/rsi_cpu_tasks.py`, `scripts/rsi_api_pilot.py`.
Run output: `artifacts/rsi_p0_v1/`.
