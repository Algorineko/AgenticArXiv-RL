# Benchmark Degenerate-policy Baselines

Task set: `expanded` | seed: `42` | training step: `100`

These synthetic policies never execute tools. `finish_rate` only means the trajectory ended in `FINISH`; it is intentionally reported separately from tool-path accuracy and reward.

| Policy | Tasks | Samples | Mean reward | Reward std | Reward range | Finish rate | Exact tool path | Mean arg score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 62 | 62 | 1.000 | 0.000 | [1.000, 1.000] | 100.0% | 100.0% | 1.000 |
| always_finish | 62 | 62 | -0.126 | 0.334 | [-0.225, 1.000] | 100.0% | 8.1% | 0.000 |
| always_search | 62 | 62 | 0.005 | 0.289 | [-0.235, 0.667] | 100.0% | 22.6% | 0.084 |
| random_tool | 62 | 1240 | -0.064 | 0.262 | [-0.235, 1.000] | 100.0% | 15.2% | 0.046 |
| wrong_args | 62 | 62 | 0.450 | 0.174 | [0.375, 1.000] | 100.0% | 100.0% | 0.066 |

Health check: **PASS** (minimum reference gap: `0.300`)

Per-category check: **PASS** (minimum gap: `0.300`). The aggregate check above averages over the whole task set, so a single leaky category is diluted by the rest.

## Highest-scoring non-reference trajectories

| Policy | Task | Category | Samples | Mean reward | Max reward | Exact tool path |
|---|---|---|---:|---:|---:|---:|
| always_finish | chain_ai5_dl2_tr2 | long_chain | 1 | -0.225 | -0.225 | 0.0% |
| always_finish | chain_cl5_dl3_tr3 | long_chain | 1 | -0.225 | -0.225 | 0.0% |
| always_finish | chain_cv5_cache_dl_tr_cache | long_chain | 1 | -0.225 | -0.225 | 0.0% |
| always_finish | chain_lg10_dl_tr_last | long_chain | 1 | -0.225 | -0.225 | 0.0% |
| always_finish | chain_ro5_dl_three | long_chain | 1 | -0.225 | -0.225 | 0.0% |
| always_search | search_CL_7d_5 | search | 1 | 0.667 | 0.667 | 100.0% |
| always_search | search_CR_7d_5 | search | 1 | 0.667 | 0.667 | 100.0% |
| always_search | constraint_search_no_file | constraint | 1 | 0.575 | 0.575 | 100.0% |
| always_search | search_AI_1d_3 | search | 1 | 0.508 | 0.508 | 100.0% |
| always_search | search_AI_30d_25 | search | 1 | 0.508 | 0.508 | 100.0% |
| random_tool | ref_ctrl_null_translate | ref_form | 20 | 0.135 | 0.375 | 60.0% |
| random_tool | multi_ai3_tr1 | composite | 20 | 0.122 | 0.308 | 0.0% |
| random_tool | multi_cr5_cache1 | composite | 20 | 0.075 | 0.308 | 0.0% |
| random_tool | state_dl_then_cache | state | 20 | 0.065 | 0.375 | 0.0% |
| random_tool | opt_service | optional | 20 | 0.055 | 0.575 | 35.0% |
| wrong_args | search_CL_7d_5 | search | 1 | 0.667 | 0.667 | 100.0% |
| wrong_args | search_CR_7d_5 | search | 1 | 0.667 | 0.667 | 100.0% |
| wrong_args | constraint_search_no_file | constraint | 1 | 0.575 | 0.575 | 100.0% |
| wrong_args | multi_cr5_cache1 | composite | 1 | 0.508 | 0.508 | 100.0% |
| wrong_args | multi_ai5_dl1_tr | composite | 1 | 0.464 | 0.464 | 100.0% |
