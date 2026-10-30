# Manuscript reframing draft (evidence-bounded)

## Candidate title

**When Narrow Tool-Use Gains Fail to Transfer: A Failure Audit of Multi-Step Scientific-Agent Workflows**

## Abstract

Tool-using agents can improve on narrow development tasks while failing to complete longer workflows. We audit this failure mode in a scientific-literature assistant using frozen, target-disjoint task pools and executable tool checks. Two short, paired supervised fine-tuning (SFT) runs trained on the same 64 decision examples for 16 optimizer steps reached 12/12 and 6/12 on a narrow development probe, respectively. On a one-shot 24-task workflow evaluation, both trained checkpoints completed 0 tasks, while the untuned 1.5B instruction model completed the requested operation sequence on 10 tasks (8 under the stricter scorer). Excluding four tasks with a prompt-contract conflict leaves 0/20 for each trained checkpoint and 9/20 for the untuned model. This result is specific to one short recipe and does not establish that SFT generally harms compositionality.

We then replace a static tool-chain proxy with the pinned native execution backend and checker from BFCL v4 for filesystem tasks. On eight source scenarios not previously used for inference, the 1.5B and 7B instruction checkpoints completed 0/8 and 3/8 scenarios, respectively, and 2/26 and 14/26 user turns. Every recorded tool observation and native state/response check was independently replayed. The larger checkpoint improves over this small diagnostic set, but the comparison is not a controlled estimate of model-scale effects, and neither model solves most scenarios. Together, the results show that narrow probe success is an unsafe promotion criterion for workflow agents and that benchmark protocol validity must precede algorithm attribution. They do not demonstrate a new training method or recursive self-improvement (RSI). We release protocols, traces, scorer audits, and backend replay artifacts to support a bounded negative result and a more discriminating follow-up study.

## 1. Introduction (draft)

Scientific-paper assistants combine operations that are individually simple but depend on state across turns: locating a paper, downloading or translating it, and then acting on a previously created artifact. A model can produce plausible tool calls on isolated requests and still terminate before a workflow is complete. This makes aggregate success on short development probes a weak basis for deciding whether a training intervention improves an agent.

We investigate this problem through a rescue audit of an existing agentic-RL study. The audit began after historical environment snapshots and training examples could not be recovered. We therefore distinguish historical claims from new experiments: the new results use a frozen public-source catalog and newly recorded trajectories; they are not retrospective reproductions of the original run. We also separate executable protocol evidence from research-method evidence. A valid tool receipt can establish that a requested operation occurred in a simulator, but it cannot establish that a translation is faithful or that a paper was understood.

The audit produced two findings. First, a narrow development score did not predict workflow completion after short SFT: both paired checkpoints scored below the untuned model on the one-shot workflow evaluation. The result is a negative observation about this data and recipe, not a universal claim about SFT. Second, replacing a custom text-action interface with the model's native tool interface removed most formatting failures but did not remove execution errors. On unused filesystem scenarios, a public 7B checkpoint completed more tasks than a 1.5B checkpoint, yet both retained substantial failures.

Our contribution is consequently an evidence audit rather than a new agent-learning algorithm. We document how prompt-contract conflicts, static scoring shortcuts, narrow probes, and unsupported historical assets constrain causal claims. We provide replayable artifacts and identify a stopping rule: a future method claim must beat a simple workflow-regression baseline on tasks fixed before training and evaluation. We do not claim RSI, autonomous environment evolution, or broad scientific-assistant capability.

## 2. Experimental record

### 2.1 Scientific-literature workflow tasks

We froze a source catalog and split target papers across train, development, promotion, and final pools. The one-shot final pool contains 24 tasks from six composition families. It was evaluated once; it is now used and must not be described as an untouched test set. Four tasks were later identified as conflicting with the frozen prompt, which instructed the agent to finish after translation even when the requested workflow also required a later cache operation. We preserve the original score and report the 20-task exclusion as sensitivity analysis.

The paired feasibility run built 96 training trajectories and 208 decision rows per arm. Hidden-state and visible-state arms shared task IDs, row order, labels, and the selected 64 training rows. Each arm started from the same public 1.5B base and received 16 AdamW steps with seed 123. The run is intentionally short and is not a healthy, full-epoch SFT recipe. The two arms differ in whether prior setup text is shown during training; the runtime prompt remained fixed for the primary comparison. No final or promotion task was used for training.

The primary workflow metric requires both the requested operation evidence and an explicit FINISH. A stricter score additionally checks the frozen operation sequence. Tool-call traces are retained so that extra valid calls are distinguishable from missing requested work.

### 2.2 Native executable calibration

For external protocol calibration, we pinned the official Gorilla/BFCL source at commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` and used its native filesystem backend and state/response checkers. Model actions were parsed from the Qwen chat template's native tool-call format and dispatched through a restricted public-function interface. Model text was never passed to Python `eval`. A reference policy first replayed the selected gold trajectories; an independent analyzer then re-executed every recorded call and recomputed native checks.

The primary new roster contains eight filesystem-only scenarios, IDs `multi_turn_base_10`, `_12`, `_16`, `_25`, `_26`, `_29`, `_37`, and `_38`, totaling 26 user turns. These IDs were not used in our earlier model inference. The four earlier calibration scenarios were excluded, as was `_39` because one gold turn exceeded the evaluator's six sequential reference-step limit. Thus the set is held out from this study's prior inference, while possible overlap with public pretraining data remains unknown. This is a small diagnostic subset, not a full BFCL score.

The two pretrained checkpoints used the same native tool interface, greedy decoding, seed 45, 384 generated tokens per model call, and six model calls per user turn. This comparison changes checkpoint size and model identity together; it is descriptive and does not isolate a causal scale effect. No model was trained for this comparison.

## 3. Results

### 3.1 Narrow SFT gains did not predict workflow completion

On the 12-task development probe, the hidden-state SFT checkpoint scored 12/12 under both visible- and hidden-state runtime prompts. The visible-state SFT checkpoint scored 6/12 under either runtime prompt. All six lost tasks were explicit force-download requests, while translation requests succeeded. The public two-epoch SFT checkpoint scored 0/12 under the normal prompt and 6/12 when setup text was hidden; the untuned base scored 8/12. These development scores show strong sensitivity to training source and interface, but they do not establish a general advantage for hiding state.

On the one-shot 24-task workflow evaluation, the untuned base completed the requested operation sequence on 10/24 tasks and passed the stricter score on 8/24. Both new 16-step SFT checkpoints scored 0/24 on the primary metric and 0/24 under the strict score. Each SFT checkpoint stopped with FINISH before the requested workflow was complete on all 24 tasks. Excluding the four prompt-conflict tasks leaves the base at 9/20 and both SFT checkpoints at 0/20. A post hoc audit also found failures on operation chains represented in the training families. These observations reject the use of the narrow probe as a sufficient promotion gate for this recipe; one seed and one short recipe cannot support a general statement about SFT.

### 3.2 Native tools reduced format errors but not multi-step failures

On the eight unused filesystem scenarios, the 1.5B checkpoint completed 0/8 scenarios and 2/26 user turns. The 7B checkpoint completed 3/8 scenarios and 14/26 user turns. The 1.5B run had no parse errors, 12 tool-error attempts, and no exhausted user-turn budgets. The 7B run had one parse-error step, 25 tool-error attempts, and four exhausted user-turn budgets. All observations and native checks were independently replayed from the recorded actions.

The same direction appeared in the earlier four-scenario calibration set: 7B completed 1/4 scenarios and 6/14 turns, compared with 0/4 and 2/14 for 1.5B. Those four cases were already exposed during development, so this agreement is descriptive consistency rather than a second independent replication. The new eight-case set supplies the more relevant held-out-from-our-work evidence, but remains small and restricted to one simulated API family.

A separate backend preflight replayed all 200 original multi-turn scenarios, 734 user turns, and 1,142 logical reference calls. Two independent backend instances reproduced the native state and response checks in every scenario. This validates the replay harness and reference execution; it is not a 200-task model evaluation.

### 3.3 Static improvements did not rescue the method claim

Rule-based request compilation achieved 12/12 on its narrow probe but had no learned policy component. A static receipt-contract prompt reduced the untuned model's development score from 8/12 to 7/12 and the GRPO checkpoint's from 3/12 to 0/12. State-presentation changes yielded mixed results between models and failed to improve the promotion set for the untuned model. These checks make simple contract and prompt explanations testable, but none constitutes RSI or an autonomous improvement loop.

## 4. Discussion

The results expose a mismatch between the unit optimized in the short development probe and the unit required by a workflow: completing the requested sequence while preserving state and stopping only after completion. A checkpoint can score highly on isolated operations while failing every composition in the final set. The proper consequence is to strengthen promotion tests and per-turn regression checks before scaling training.

The native BFCL comparison adds a separate capacity observation. The 7B model performs better than the 1.5B model on the small unused roster, yet 5/8 scenarios and 12/26 turns still fail. This suggests that model choice can affect tool-use performance, but it does not prove that size alone caused the difference or that more parameters solve the workflow problem. Both checkpoints are public instruction models; their training histories differ, and the sample is too small for broad capability claims.

We find no evidence that RSI is necessary. No model in this audit autonomously changed its environment, selected a useful modification, or demonstrated sustained self-improvement. The failed SFT results also do not motivate adding more recursive layers: a simpler workflow-regression baseline has not yet been ruled out. The defensible next question is narrower: can a predeclared diagnostic predict workflow regressions and select an intervention that beats ordinary multi-step regression training on genuinely new tasks?

## 5. Limitations and claim boundaries

The original historical data snapshot and training JSONL were not available, so the new experiments do not reproduce the original training run. The short SFT comparison uses one seed, 16 optimizer steps, and overlapping task-family templates between training and development; it is a feasibility result. The one-shot workflow final set has been used and includes a known prompt conflict, though the conflict-exclusion sensitivity retains the main negative result.

The BFCL evaluation covers eight filesystem-only scenarios from a public benchmark. They are unused by our prior inference, but pretrained-data contamination is unknown. The 7B-versus-1.5B comparison is not randomized and changes more than parameter count. The checker and simulator establish task-state and response consistency for the modeled API; they do not establish real PDF quality, translation fidelity, or live-service reliability. We report neither a full BFCL leaderboard score nor a broad scientific-agent score.

Accordingly, the evidence supports a reproducibility and failure-analysis paper or a bounded negative result. It does not yet support a new optimization method, general SFT harm, causal scaling law, or RSI claim. A stronger methods paper requires a healthy multi-step baseline, an independently frozen task lineage, and prospective improvement over a simple regression control.

## 6. Decision gate for further method work

Continue the method line only if a new, pre-registered experiment satisfies all of the following: (i) a healthy base/interface passes a nontrivial but unsaturated workflow benchmark; (ii) training and evaluation tasks are separated by lineage, not only instance ID; (iii) each checkpoint is gated on multi-step regressions and task completion, not only primitive probes; and (iv) the proposed diagnostic or intervention beats an ordinary multi-step regression baseline on a new validation set, followed by a still-unseen final set. Otherwise, publish the audit/reproducibility assets as such and stop the RSI/method expansion.
