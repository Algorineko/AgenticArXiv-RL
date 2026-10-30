# Current generator versus runtime setup rendering

At pre-ablation commit f8ebf37, scripts/generate_sft_data.py executes declared setup in generate_deterministic_trajectories, but calls build_sft_samples_from_history with only task text, tools, current trajectory and lineage. The builder passes only accumulated current actions into get_react_prompt.history. It has no initial_history parameter. BenchmarkRunner instead calls agent.run(initial_history=build_visible_setup_context(task_def)); BaseAgent prepends that state before the identical current history.

This is an executable code-path mismatch, not proof of the exact historical training input used for the published weights. Their original training JSONL is missing. The published SFT manifest identifies a two-epoch, full-parameter run with data SHA256 66079e9015780f8e1a76837999df8f9a37b77d5a284e587affb22d84d0dfc793. We do not have matching JSONL bytes and cannot reconstruct that provenance by assumption.

The proposed generator correction adds optional initial_history to the sample builder and supplies actual declared setup state in deterministic generation. For legacy depends_on fallback search, rendered state describes the actual AI/7days/5papers search. Existing callers without initial_history remain unchanged. The optional LLM-generation path is outside this correction and has not been validated for stateful setup; do not claim the whole data pipeline is repaired.

New tests compare first-decision prompt bytes with runtime get_react_prompt(initial_history), ensure future gold actions are absent, ensure later history is retained, and exercise real deterministic setup persistence. No released checkpoint is retrained as part of this correction.
