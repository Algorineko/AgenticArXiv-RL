# BFCL filesystem native capacity diagnostic

Frozen roster of eight pure-filesystem cases, 26 user turns. Official native tool interface and checker; reference policy passed 8/8 before model evaluation. These IDs were not in earlier native model-inference rosters, but a post hoc lineage audit found that all 12 first-turn prompts across the prior calibration and this roster exactly match rows in the earlier BFCL v3 source snapshot. This is not an independent task-pool diagnostic or leaderboard score.

| Model | Scenarios | User turns | Parse-error steps | Tool-error calls | Forced stops |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 0/8 | 2/26 | 0 | 12 | 0 |
| Qwen2.5-7B-Instruct | 3/8 | 14/26 | 1 | 25 | 4 |

All model observations and native state/response checks were independently replayed. This is a capacity/interface diagnostic only; no model was trained, no RSI effect was tested, and the two model sizes are not a controlled causal scale intervention.
