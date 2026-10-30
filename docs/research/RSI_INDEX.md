# API-first RSI research

Status: CPU calibration saturated (16/16); two exposed AppWorld train scenarios remain 0/2 under both baseline and a static discovery control. Reference solutions pass 2/2. No RSI effectiveness result yet.

- [Resource-grounded research proposal](rsi_api_research_proposal_20261030.md): experience-controlled recursive improvement, fixed-meta optimization controls, CPU tasks, budget and stopping rules.
- [Sanitized API resource probe](../../artifacts/rsi_api_preflight_v1/resource_probe.json): five bounded chat requests; one successful native tool response and four unavailable-model errors. These are connectivity checks.
- [Prior research archive manifest](../../artifacts/rsi_api_preflight_v1/prior_research_archive.json): 751 tracked artifacts, scripts and documents at `67ac4f5cecdbba708aa7b424ea00aef9e33904df`, tagged `archive/rescue-audit-20261030`.
- [Historical rescue research index](INDEX.md): previous negative results and protocol audits. Its used task pools must not be presented as fresh RSI validation.
- [P0 frozen calibration protocol](rsi_p0_protocol_20261030.md) and [transport-only v2 continuation](rsi_p0_v2_rate_limit_protocol_20261030.md): independent CPU goal checks, source snapshots, capped API usage and unchanged baseline repeats.
- [Limited RL, environment curriculum and TTT extension](rsi_model_harness_environment_extension_20261030.md): separates controller learning, LLM parameter training and API-side adaptation, with component controls.
- [P0 actual findings](rsi_p0_findings_20261030.md): complete baseline, 48 independently replayed observations, RPM limits, combined resource accounting, and the decision to seek a non-saturated task suite.

Pilot entry point: `python3 scripts/rsi_api_pilot.py --output <new-directory>` with `RSI_API_KEY` in the environment. Offline preflight adds `--validate-only`; replay with `scripts/analyze_rsi_p0.py`. Runtime credentials never enter the artifacts.

Current worktree: `AgenticArXiv-RSI`; branch: `research/rsi-api-20261030`.
Large ignored checkpoints and local credential files are outside the Git archive. The original rescue worktree remains intact.

- [Framework reuse, actual progress and publication positioning](rsi_framework_reuse_and_status_20261030.md).
- [Prior research archive and recovery](../archive/rescue_archive_20261030.md).

- [Framework registry integration audit](../../artifacts/rsi_registry_integration_v1/audit.json): 39 exposed-train observations and four verdicts replayed through the original ToolRegistry, no differences.
- [24-step development diagnostic protocol](rsi_appworld_horizon_protocol_20261030.md): separate larger-budget calibration, no RSI effect claim.

- [Host clock correction](rsi_host_clock_erratum_20261030.md): historical AppWorld request timestamps and durations use simulated time; counts, token accounting and replayed goal verdicts remain usable.
- [Strong fixed improver audit](rsi_strong_baseline_audit_20261030.md): GEPA and existing meta-optimization support constrain novelty claims.
- [ID-only pool provenance audit](../../artifacts/rsi_appworld_pool_v2/audit.json): unchanged membership, builder and registry digests, no task content accessed.
