# AppWorld external train probe v1

Freeze before model inference. This is a two-scenario development diagnostic, not a benchmark score or RSI comparison.

- Engine: AppWorld `0.1.3.post1`, isolated Python `3.11.18`, official `data-0.1.0.bundle` downloaded and unpacked into the external runtime cache. Encrypted source hashes and full dependency versions are recorded by the engine preflight.
- Selection: `load_task_ids('train', num_tasks_per_scenario=1)[:2]` in the pinned engine's existing order. Do not inspect/evaluate test or dev scenarios. Hash the selected IDs in public metadata; protected task content stays in the runtime directory.
- Agent: `doubao-seed-2.0-mini`, temperature 0, 1,024 maximum generated tokens per response, twelve model steps per scenario, at most four public tool calls per response. Both scenarios start from clean task state.
- Tool surface: application discovery, API listing, exact API documentation, and a JSON-argument API proxy. Identifiers are validated and model-generated source is never executed. The host generates only a fixed `apis.app.method(**literal_arguments)` expression within AppWorld's engine restrictions.
- Preflight passed: discovery and API documentation; unknown tools and invalid identifiers rejected; initial untouched train state fails the official task goal. This establishes engine readiness, not a healthy model capability baseline.
- Network: fifteen-second request spacing, one bounded retry on transient errors; per-experiment caps thirty attempted requests, 200,000 accounted input and 30,000 accounted output tokens. Unknown provider usage is charged conservatively, as in P0. This is a separate frozen external probe budget; P0 already completed within its own caps.
- Metric: official final task success. Report model steps, native calls, tool errors, truncation and stop reasons separately. No prompt/model selection from these two outcomes and no claim that two cases calibrate the full benchmark.
- The host loads minimal ground truth only for the official final evaluator. No hidden checks, solutions or reference answers are supplied to the model. Its input is the allowed task instruction, supervisor identity and public tool observations.
- Protected instructions/API descriptions and raw API/tool traces remain outside Git under the runtime cache; research Git artifacts contain only source code, source digests and aggregate/hash metadata. Respect the official protected-content distribution requirements when later sharing traces.
- Do not train model weights, run RL/TTT, modify harnesses or evolve environments during this diagnostic. Subsequent interventions require a separate protocol with suitable controls and a task-family split.

Engine/pre-proxy artifacts: `artifacts/rsi_appworld_preflight_v1/`.
Public aggregate result: `artifacts/rsi_appworld_train_probe_v1/summary.json`.
