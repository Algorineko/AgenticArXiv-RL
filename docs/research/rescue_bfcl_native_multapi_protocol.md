# Balanced native multi-API capacity diagnostic (frozen before inference)

## Question

Does the limited 7B advantage on small filesystem-only development cases recur when each task requires coordinating two simulated APIs? This is a capacity/interface diagnostic between public, untrained Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct checkpoints. It does not test training, RSI, or a causal effect of parameter count.

## Frozen roster

The task list contains two scenarios from each of the 16 ordered API pairs present among the eight pinned BFCL v4 multi-turn simulator classes, for 32 scenarios and 129 user turns. For each pair, select the first two source-order scenarios with no more than five sequential gold calls in any user turn. The IDs and per-pair selection are in `artifacts/rescue_bfcl_native_multapi_v1/protocol.json`. All IDs are new to this project's earlier model inference. Since the benchmark and checkpoints are public, overlap with pretraining data cannot be excluded.

The reference runner requires one model call after the final tool call to finish the turn. Therefore the roster's five-call-per-turn ceiling leaves that finish step within the six-call runtime budget. A pre-inference reference execution passed all 32 scenarios and 129 turns. An earlier gold-only preflight attempt stopped at mixed-type dictionary-key serialization; no model inference occurred in that attempt. The final preflight canonicalizes dictionary keys with type tags and completed before the protocol was frozen.

## Execution and scoring

Use the official BFCL native classes, function documentation, and state/response checkers pinned by the source, integration, and full-backend manifests. Instantiate only the two APIs declared for each scenario. Tool calls are dispatched through a literal AST parser and public-method whitelist; model output is never evaluated with Python `eval`. Tool observations and state hashes are stored per turn.

Both models use the Qwen-native tools chat template, greedy decoding, seed 45, at most 384 generated tokens per model call, and at most six model calls per user turn. Model order is 1.5B then 7B. Each arm runs only when physical GPU1 has at least 30 GB free; there is no training and no prompt search. The machine-readable protocol pins all model weights, evaluators, source manifests, and replay analyzer hashes.

After inference, independently re-execute every action and recompute response/state checks. Report full scenario and turn success, parse errors, tool errors, exhausted turns, and replay status. This is not a full BFCL leaderboard score: it is a balanced 32-scenario diagnostic over local simulators. No live messages, transactions, bookings, or other external actions occur. Do not infer a causal scale law from two checkpoints.
