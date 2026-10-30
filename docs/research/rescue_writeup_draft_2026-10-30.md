# Manuscript reframing draft (evidence-bounded)

## Candidate title

**When Narrow Tool-Use Gains Fail to Transfer: A Failure Audit of Multi-Step Scientific-Agent Workflows**

## Abstract

Tool-using agents can improve on narrow development tasks while failing to complete longer workflows. We audit this failure mode in a scientific-literature assistant using frozen, target-disjoint task pools and executable tool checks. Two short, paired supervised fine-tuning (SFT) runs trained on the same 64 decision examples for 16 optimizer steps reached 12/12 and 6/12 on a narrow development probe, respectively. On a one-shot 24-task workflow evaluation, both trained checkpoints completed 0 tasks, while the untuned 1.5B instruction model completed the requested operation sequence on 10 tasks (8 under the stricter scorer). Excluding four tasks with a prompt-contract conflict leaves 0/20 for each trained checkpoint and 9/20 for the untuned model. This result is specific to one short recipe and does not establish that SFT generally harms compositionality.

We then replace a static BFCL tool-chain proxy with the pinned native execution backend and checker from BFCL v4. On eight filesystem scenarios not used in earlier native inference, the 1.5B and 7B checkpoints completed 0/8 and 3/8 scenarios, respectively. On a balanced set of 32 scenarios spanning all 16 ordered pairs of simulated APIs, they completed 2/32 and 9/32 scenarios, or 37/129 and 65/129 user turns. A post hoc lineage audit found that all 12 filesystem cases and 22/32 multi-API cases have first-turn prompts that exactly match rows in the 200-row BFCL v3 source snapshot used in the earlier static protocol audit. The native results therefore form a cross-version backend audit on an overlapping source-task pool, not independent or held-out replication. Every recorded tool observation and native state/response check was independently replayed. The 7B checkpoint performs better on these samples, but model identity and size change together, realized generation steps differ, and most tasks still fail. Together, the results show that narrow probe success is an unsafe promotion criterion for workflow agents and that benchmark protocol validity must precede algorithm attribution. They do not demonstrate a new training method or recursive self-improvement (RSI).

## 1. Introduction (draft)

Scientific-paper assistants combine operations that are individually simple but depend on state across turns: locating a paper, downloading or translating it, and then acting on a previously created artifact. A model can produce plausible tool calls on isolated requests and still terminate before a workflow is complete. This makes aggregate success on short development probes a weak basis for deciding whether a training intervention improves an agent.

We investigate this problem through a rescue audit of an existing agentic-RL study. The audit began after historical environment snapshots and training examples could not be recovered. We therefore distinguish historical claims from new experiments: the new results use a frozen public-source catalog and newly recorded trajectories; they are not retrospective reproductions of the original run. We also separate executable protocol evidence from research-method evidence. A valid tool receipt can establish that a requested operation occurred in a simulator, but it cannot establish that a translation is faithful or that a paper was understood.

The audit produced two findings. First, a narrow development score did not predict workflow completion after short SFT: both paired checkpoints scored below the untuned model on the one-shot workflow evaluation. The result is a negative observation about this data and recipe, not a universal claim about SFT. Second, replacing a custom text-action interface with the model's native tool interface removed most formatting failures but did not remove execution errors. On filesystem scenarios not previously run through this native adapter, a public 7B checkpoint completed more tasks than a 1.5B checkpoint, yet both retained substantial failures; all 12 filesystem prompts overlap with the earlier v3 source pool.

Our contribution is consequently an evidence audit rather than a new agent-learning algorithm. We document how prompt-contract conflicts, static scoring shortcuts, narrow probes, and unsupported historical assets constrain causal claims. We provide replayable artifacts and identify a stopping rule: a future method claim must beat a simple workflow-regression baseline on tasks fixed before training and evaluation. We do not claim RSI, autonomous environment evolution, or broad scientific-assistant capability.

## 2. Related work and scope

Recent work already studies agent and environment co-adaptation. [SEAL](https://arxiv.org/abs/2605.24426) uses verified failure diagnosis to inform both policy learning and changes to the training interface. Our experiments contain no autonomous interface evolution or learned improvement loop, so we do not claim a new co-evolution method.

[Trajectory2Gym](https://arxiv.org/abs/2610.33568) emphasizes compositional task coverage and supervision grounded in evidence visible to the learner, with evaluation across multiple model sizes and tool-use settings. These ingredients are close to the data-coverage and state-presentation questions audited here. We therefore treat visible-state controls and multi-step regression sets as necessary baselines, not as novel contributions.

[OpenAgent](https://arxiv.org/abs/2607.01084) studies tool-use fragility under shifts in queries, actions, observations, and domains. [Environment Tuning](https://arxiv.org/abs/2510.10197) analyzes static-trajectory overfitting and training with environment feedback. These works make broad claims such as “static SFT is brittle” or “environment feedback is better” insufficient as novelty claims for this study.

Our narrower contribution is an auditable negative case study: a short paired SFT recipe that improves a narrow probe but fails a one-shot workflow evaluation, alongside a correction from static tool-chain scoring to pinned native multi-API execution. The resulting evidence changes what can safely be concluded about this project; it does not yet identify an intervention that outperforms these prior directions.

## 3. Experimental record

### 3.1 Scientific-literature workflow tasks

We froze a source catalog and split target papers across train, development, promotion, and final pools. The one-shot final pool contains 24 tasks from six composition families. It was evaluated once; it is now used and must not be described as an untouched test set. Four tasks were later identified as conflicting with the frozen prompt, which instructed the agent to finish after translation even when the requested workflow also required a later cache operation. We preserve the original score and report the 20-task exclusion as sensitivity analysis.

The paired feasibility run built 96 training trajectories and 208 decision rows per arm. Hidden-state and visible-state arms shared task IDs, row order, labels, and the selected 64 training rows. Each arm started from the same public 1.5B base and received 16 AdamW steps with seed 123. The run is intentionally short and is not a healthy, full-epoch SFT recipe. The two arms differ in whether prior setup text is shown during training; the runtime prompt remained fixed for the primary comparison. No final or promotion task was used for training.

The primary workflow metric requires both the requested operation evidence and an explicit FINISH. A stricter score additionally checks the frozen operation sequence. Tool-call traces are retained so that extra valid calls are distinguishable from missing requested work.

### 3.2 Native executable calibration

For external protocol calibration, we pinned the official Gorilla/BFCL source at commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` and used its native filesystem backend and state/response checkers. Model actions were parsed from the Qwen chat template's native tool-call format and dispatched through a restricted public-function interface. Model text was never passed to Python `eval`. A reference policy first replayed the selected gold trajectories; an independent analyzer then re-executed every recorded call and recomputed native checks.

The filesystem roster contains eight scenarios, IDs `multi_turn_base_10`, `_12`, `_16`, `_25`, `_26`, `_29`, `_37`, and `_38`, totaling 26 user turns. These IDs were not used in our earlier native model inference. The four earlier native calibration scenarios were excluded, as was `_39` because one gold turn exceeded the evaluator's six sequential reference-step limit. All 12 filesystem cases across the two native rosters have first-turn prompts that exactly match rows in the earlier v3 source snapshot. This is a small diagnostic subset, not a full BFCL score or independent benchmark.

The two pretrained checkpoints used the same native tool interface, greedy decoding, seed 45, 384 generated tokens per model call, and six model calls per user turn. This comparison changes checkpoint size and model identity together; it is descriptive and does not isolate a causal scale effect. No model was trained for this comparison.

### 3.3 Balanced two-API scenarios

To check whether the filesystem-only pattern extends beyond one API family, we selected two eligible scenarios from each of the 16 ordered API pairs represented among the eight pinned classes. Eligibility required at most five sequential reference calls in each turn, leaving one call within the six-call budget for the agent to finish. The frozen roster contains 32 scenarios and 129 turns. A reference policy passed all 32 scenarios before model inference. The selection was deterministic by source order, not a random sample. A post hoc comparison with the earlier 200-row BFCL v3 source snapshot found exact normalized first-turn prompt matches for 22 cases. The roster is therefore source-overlapping and not an independent or held-out replication; the historical model-inference overlap cannot be determined because prior static-evaluation traces are unavailable.

## 4. Results

### 4.1 Narrow SFT gains did not predict workflow completion

On the 12-task development probe, the hidden-state SFT checkpoint scored 12/12 under both visible- and hidden-state runtime prompts. The visible-state SFT checkpoint scored 6/12 under either runtime prompt. All six lost tasks were explicit force-download requests, while translation requests succeeded. The public two-epoch SFT checkpoint scored 0/12 under the normal prompt and 6/12 when setup text was hidden; the untuned base scored 8/12. These development scores show strong sensitivity to training source and interface, but they do not establish a general advantage for hiding state.

On the one-shot 24-task workflow evaluation, the untuned base completed the requested operation sequence on 10/24 tasks and passed the stricter score on 8/24. Both new 16-step SFT checkpoints scored 0/24 on the primary metric and 0/24 under the strict score. Each SFT checkpoint stopped with FINISH before the requested workflow was complete on all 24 tasks. Excluding the four prompt-conflict tasks leaves the base at 9/20 and both SFT checkpoints at 0/20. A post hoc audit also found failures on operation chains represented in the training families. These observations reject the use of the narrow probe as a sufficient promotion gate for this recipe; one seed and one short recipe cannot support a general statement about SFT.

### 4.2 Native tools reduced format errors but not multi-step failures

On the eight filesystem scenarios not previously run through this native adapter, the 1.5B checkpoint completed 0/8 scenarios and 2/26 user turns. The 7B checkpoint completed 3/8 scenarios and 14/26 user turns. The 1.5B run had no parse errors, 12 tool-error attempts, and no exhausted user-turn budgets. The 7B run had one parse-error step, 25 tool-error attempts, and four exhausted user-turn budgets. All observations and native checks were independently replayed from the recorded actions.

On the balanced two-API roster, the 1.5B checkpoint completed 2/32 scenarios and 37/129 turns; the 7B checkpoint completed 9/32 scenarios and 65/129 turns. Seven scenarios were completed only by the 7B checkpoint, none only by the 1.5B checkpoint, two by both, and 23 by neither. The 1.5B run had two parse-error steps, 39 tool-error attempts, and no exhausted turns. The 7B run had one parse-error step, 69 tool-error attempts, and seven exhausted turns. A post hoc analyzer independently re-executed every recorded action and recomputed all native checks. The first frozen analyzer rejected unknown function names instead of reproducing the runtime's safe error response; a versioned replay correction was applied without changing the frozen protocol or model traces.

The 7B checkpoint completed more cases in this native cross-protocol audit, but the roster overlaps substantially with the earlier BFCL v3 source pool. It is not independent confirmation that this direction generalizes to new tasks. The 7B checkpoint used more model steps (334 vs. 240) and function attempts (255 vs. 118). Thus, these outcomes support only a descriptive checkpoint difference on this source-overlapping sample, not a causal scale estimate or an equal-compute comparison. Most two-API scenarios remain unsolved by both checkpoints.

A separate backend preflight replayed all 200 original multi-turn scenarios, 734 user turns, and 1,142 logical reference calls. Two independent backend instances reproduced the native state and response checks in every scenario. This validates the replay harness and reference execution; it is not a 200-task model evaluation.
### 4.3 Static improvements did not rescue the method claim

Rule-based request compilation achieved 12/12 on its narrow probe but had no learned policy component. A static receipt-contract prompt reduced the untuned model's development score from 8/12 to 7/12 and the GRPO checkpoint's from 3/12 to 0/12. State-presentation changes yielded mixed results between models and failed to improve the promotion set for the untuned model. These checks make simple contract and prompt explanations testable, but none constitutes RSI or an autonomous improvement loop.

## 5. Discussion

The results expose a mismatch between the unit optimized in the short development probe and the unit required by a workflow: completing the requested sequence while preserving state and stopping only after completion. A checkpoint can score highly on isolated operations while failing every composition in the final set. The proper consequence is to strengthen promotion tests and per-turn regression checks before scaling training.

The native BFCL results add a cross-protocol execution observation. On a 32-scenario set across 16 API pairs, the 7B model completes more tasks and turns than the 1.5B model, but 22 cases have first-turn prompts that match the earlier v3 source snapshot and both checkpoints fail 23 cases. The result is not independent of the earlier task pool and does not identify a causal parameter-count effect. The public benchmark and checkpoints may overlap in pretraining, and the APIs are local simulations.

We find no evidence that RSI is necessary. No model in this audit autonomously changed its environment, selected a useful modification, or demonstrated sustained self-improvement. The failed SFT results also do not motivate adding more recursive layers: a simpler workflow-regression baseline has not yet been ruled out. The defensible next question is narrower: can a predeclared diagnostic predict workflow regressions and select an intervention that beats ordinary multi-step regression training on genuinely new tasks?

## 6. Limitations and claim boundaries

The original historical data snapshot and training JSONL were not available, so the new experiments do not reproduce the original training run. The short SFT comparison uses one seed, 16 optimizer steps, and overlapping task-family templates between training and development; it is a feasibility result. The one-shot workflow final set has been used and includes a known prompt conflict, though the conflict-exclusion sensitivity retains the main negative result.

The BFCL evaluation covers eight filesystem-only scenarios and 32 balanced two-API scenarios from a public benchmark. All 12 filesystem prompts and 22/32 multi-API first-turn prompts overlap exactly with the earlier v3 source-task pool; historical traces do not establish which source rows were previously used for model inference. Pretraining contamination is also unknown. The 7B-versus-1.5B comparison is deterministic and changes more than parameter count; realized generation steps also differ. The checker and simulator establish task-state and response consistency for the modeled API; they do not establish real PDF quality, translation fidelity, or live-service reliability. We report neither a full BFCL leaderboard score nor a broad scientific-agent score.

Accordingly, the evidence supports a reproducibility and failure-analysis paper or a bounded negative result. It does not yet support a new optimization method, general SFT harm, causal scaling law, or RSI claim. A stronger methods paper requires a healthy multi-step baseline, an independently frozen task lineage, and prospective improvement over a simple regression control.

## 7. Decision gate for further method work

Continue the method line only if a new, pre-registered experiment satisfies all of the following: (i) a healthy base/interface passes a nontrivial but unsaturated workflow benchmark; (ii) training and evaluation tasks are separated by lineage, not only instance ID; (iii) each checkpoint is gated on multi-step regressions and task completion, not only primitive probes; and (iv) the proposed diagnostic or intervention beats an ordinary multi-step regression baseline on a new validation set, followed by a still-unseen final set. Otherwise, publish the audit/reproducibility assets as such and stop the RSI/method expansion.
