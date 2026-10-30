# Research rescue archive

- [Infrastructure and BFCL audit checkpoint](rescue_20261029.md): strict-score fixes and synthetic engineering calibration.
- [Frozen catalog-v1 protocol](rescue_catalog_v1_protocol.md): recorded before model evaluation, protocol commit `582cdff`.
- [Catalog-v1 findings and limits](rescue_catalog_v1_findings.md): four complete development arms, negative prompt result, and next falsifiable hypothesis.

Raw artifacts are under `artifacts/rescue_20261029/`, `artifacts/rescue_catalog_v1/`, and `artifacts/rescue_screen_v1/`. These are development/replay diagnostics; no RSI or broad scientific-assistant capability claim is made.

- [Paired prior-state protocol](rescue_state_v1_protocol.md): frozen before inference at `7c60503`; two independent GPU1 jobs, no multi-GPU communication.
- [Paired prior-state findings](rescue_state_v1_findings.md): 48 complete trajectories; state sensitivity reproduced, label-only change insufficient.
- [Request receipt protocol](rescue_receipt_v1_protocol.md) and [negative findings](rescue_receipt_v1_findings.md): rule-only ceiling and failed static contract baseline.
- [Published SFT control protocol](rescue_sft_state_protocol.md): distinguish pre-existing SFT behavior from unsupported RL attribution.
- [Setup alignment protocol](rescue_prompt_alignment_protocol.md) and [findings](rescue_prompt_alignment_findings.md): static presentation changes, one-shot promotion, and published SFT control.
- [BFCL protocol audit](rescue_bfcl_protocol_audit.md): derivative static chain prediction versus official executable multi-step evaluation.
- [New paired SFT feasibility protocol](rescue_sft_pair_protocol.md): train-only data, fixed 16 optimizer steps, and planned cross-deployment.
- [New SFT development and final findings](rescue_sft_pair_findings.md): primitive success and 0/24 final composition regressions; the original final split is now used.
- [Updated nearby work and claim boundaries](rescue_related_work_update_20261030.md): SEAL, Trajectory2Gym, OpenAgent, and Environment Tuning.
- [Native backend calibration findings](rescue_bfcl_native_calibration_findings.md): nine 1.5B interface/model arms, 7B filesystem calibration, independent observation replay, and full 200-scenario gold backend preflight. These are development checks, not official leaderboard scores. The [post hoc lineage audit](../../artifacts/rescue_bfcl_lineage_v1/lineage_audit.json) found exact first-turn prompt overlap for all 12 filesystem cases and 22/32 multi-API cases against the earlier v3 source pool.
- [7B 原生接口容量校准](rescue_bfcl_native_7b_findings.md): 同四个已用开发场景中 1/4 完整场景、6/14 用户轮；容量提供有限改善但不足以支撑 RSI 或方法效果结论。
- [Remaining native filesystem capacity diagnostic](rescue_bfcl_native_remaining_findings.md): eight unused scenarios, 26 turns, two public base checkpoints; findings are diagnostic only.
- [Balanced multi-API capacity protocol](rescue_bfcl_native_multapi_protocol.md) and [findings](rescue_bfcl_native_multapi_findings.md): 32 frozen scenarios across all 16 two-API strata; 7B completes 9/32 vs. 1.5B 2/32, with independent trace replay. The [post hoc lineage erratum](rescue_bfcl_multapi_lineage_erratum_2026-10-30.md) records overlap with the earlier v3 source-task pool.
- [Evidence-bounded manuscript draft](rescue_writeup_draft_2026-10-30.md): updated with the balanced two-API results and their limits.
