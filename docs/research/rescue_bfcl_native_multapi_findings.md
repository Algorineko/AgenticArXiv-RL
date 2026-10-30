# Balanced native multi-API diagnostic findings

## Result

The frozen protocol evaluated Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct on 32 BFCL v4 scenarios, with two scenarios from each of 16 ordered API-pair strata and 129 user turns total. The roster passed 32/32 scenarios and 129/129 turns under the reference policy before model inference.

| Model | Complete scenarios | Successful turns | Parse-error steps | Tool-error calls | Exhausted turns | Model steps |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 2/32 | 37/129 | 2 | 39 | 0 | 240 |
| Qwen2.5-7B-Instruct | 9/32 | 65/129 | 1 | 69 | 7 | 334 |

All tool observations and native state/response checks were independently re-executed by a post hoc replay-auditor revision. The audit reports its hash and the originally frozen auditor hash in `analysis.json`. The first frozen auditor version did not replay out-of-schema tool names as the runtime did; it rejected the trace before writing an aggregate. The corrected auditor reproduces those safe tool-error observations and then verifies the remaining traces. No inference was changed or repeated.

The paired scenario outcomes were 7 completed only by the 7B model, 0 only by the 1.5B model, 2 by both, and 23 by neither. Success by API-pair stratum:

| API pair | 1.5B | 7B |
|---|---:|---:|
| TwitterAPI + GorillaFileSystem | 0/2 | 0/2 |
| TicketAPI + GorillaFileSystem | 0/2 | 1/2 |
| MessageAPI + GorillaFileSystem | 0/2 | 0/2 |
| GorillaFileSystem + MathAPI | 0/2 | 0/2 |
| MessageAPI + VehicleControlAPI | 0/2 | 0/2 |
| TwitterAPI + VehicleControlAPI | 0/2 | 1/2 |
| TicketAPI + VehicleControlAPI | 0/2 | 0/2 |
| VehicleControlAPI + MathAPI | 0/2 | 0/2 |
| MessageAPI + TradingBot | 1/2 | 1/2 |
| TradingBot + TicketAPI | 0/2 | 1/2 |
| TwitterAPI + TradingBot | 0/2 | 1/2 |
| TradingBot + MathAPI | 0/2 | 1/2 |
| TwitterAPI + TravelAPI | 0/2 | 0/2 |
| TicketAPI + TravelAPI | 0/2 | 1/2 |
| MessageAPI + TravelAPI | 0/2 | 1/2 |
| MathAPI + TravelAPI | 1/2 | 1/2 |

## Interpretation

The direction seen on the earlier filesystem-only subset recurs across a balanced sample of two-API tasks: the 7B checkpoint completed more scenarios and turns than the 1.5B checkpoint. This weakens the explanation that the earlier gap was unique to filesystem tasks. It does not identify a causal parameter-count effect: model size, checkpoint identity and training history change together. The 7B model also used more generation steps (334 vs. 240), made more tool attempts (255 vs. 118), and had more tool errors (69 vs. 39); the protocol fixes per-call token and per-turn step ceilings, not total realized compute.

Most scenarios remain unsolved even by the larger checkpoint. At least one checkpoint completes 9/32 scenarios, and 23/32 are completed by neither. Failures occur across most API pairs, so this diagnostic does not support one isolated API integration as the sole cause. These are public benchmark tasks and public checkpoints; pretraining overlap is unknown. The APIs are local simulators, not live services, and these scores are not official BFCL leaderboard results.

The 32-case result provides a stronger capacity/interface calibration than the earlier four- and eight-case filesystem subsets. It still does not test SFT or RL, does not repair the 0/24 short-SFT workflow result, and offers no evidence for RSI. It supports a practical decision: test training interventions only after a healthy base passes a sufficiently broad multi-step benchmark, then use a genuinely new final lineage.

## Reproducibility

- Frozen protocol and source manifests: `artifacts/rescue_bfcl_native_multapi_v1/`.
- Model traces and manifests: `artifacts/rescue_bfcl_native_multapi_v1/model_runs/{base_1p5,base_7b}/`.
- Independent results: `artifacts/rescue_bfcl_native_multapi_v1/analysis.json`.
- Runner and post hoc replay auditor: `scripts/rescue_bfcl_native_multapi.py` and `scripts/analyze_rescue_bfcl_native_multapi_v2.py`.
- The post hoc analyzer independently re-executes each recorded function action, compares saved observations, and recomputes the official native state and response checks. No additional model inference is used.
