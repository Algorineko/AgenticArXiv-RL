# Frozen development screen v1

24 development instances in six related task families; one deterministic trial per arm. Promotion and final model evaluations remain closed.

| Arm | Strict | False finish | Total tokens | Rollout seconds |
|---|---:|---:|---:|---:|
| base_baseline | 9/24 | 2 | 220726 | 377.398 |
| grpo_baseline | 12/24 | 12 | 94928 | 120.802 |
| base_completion_check | 0/24 | 1 | 363594 | 635.7 |
| grpo_completion_check | 11/24 | 13 | 95184 | 119.228 |

## Family counts

| Family | Base | GRPO | Base + check | GRPO + check |
|---|---:|---:|---:|---:|
| search | 4/4 | 4/4 | 0/4 | 4/4 |
| download | 0/4 | 0/4 | 0/4 | 0/4 |
| translate | 0/4 | 0/4 | 0/4 | 0/4 |
| cache | 4/4 | 0/4 | 0/4 | 0/4 |
| search_download | 1/4 | 4/4 | 0/4 | 4/4 |
| download_translate | 0/4 | 4/4 | 0/4 | 3/4 |

## Paired prompt changes

- base_prompt_change: 0 fixed, 9 regressed; 0 correct in both, 15 wrong in both.
- grpo_prompt_change: 0 fixed, 1 regressed; 11 correct in both, 12 wrong in both.

## Interpretation limits

- No confidence/significance claim from correlated parameter variants
- Base versus SFT+GRPO pipeline does not isolate an RL effect
- Strict scoring requires the prescribed action chain, not an independent goal-state verifier
- Paper-disjoint task targets do not restrict tool access to the other catalog categories
- No autonomous evolution, reasoning, PDF-quality, or online-service claim

Raw task counts are descriptive. Family and parameter dependence prohibit treating the 24 instances as 24 independent research replications. Rollout time excludes initial weight hashing/loading; shared-server timing is not a controlled throughput comparison.
