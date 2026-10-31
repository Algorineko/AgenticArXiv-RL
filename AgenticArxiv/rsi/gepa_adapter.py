"""Adapter to pinned upstream GEPA, with a caller-owned bounded episode runner.

This module does not implement or replace GEPA's optimizer or reflection prompt.
All live solver and reflection calls must share one accounting API instance.
"""
from copy import deepcopy
from .improver import MAX_TEXT_CHARS


class NativeTraceGEPAAdapter:
    propose_new_texts = None

    def __init__(self, run_episode, allowed_task_ids):
        self.run_episode = run_episode
        self.allowed_task_ids = frozenset(allowed_task_ids)
        if not self.allowed_task_ids:
            raise ValueError("Explicit train/selection task allowlist required")
        self.evaluated_episodes = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch
        if (set(candidate) != {"solver_harness"} or not isinstance(candidate["solver_harness"], str)
                or not candidate["solver_harness"].strip() or len(candidate["solver_harness"]) > MAX_TEXT_CHARS):
            raise ValueError("Only solver_harness text is mutable")
        # Validate the entire requested batch before executing anything.
        for task in batch:
            if (not isinstance(task, dict) or set(task) != {"id", "prompt", "domain"}
                    or task["id"] not in self.allowed_task_ids):
                raise ValueError("Task not in the explicit training/selection allowlist")
        traces, outputs, scores = [], [], []
        for task in batch:
            trace = self.run_episode(deepcopy(task), candidate["solver_harness"])
            if trace["task_id"] != task["id"] or type(trace["verdict"]["success"]) is not bool:
                raise ValueError("Episode identity or independent verdict is invalid")
            self.evaluated_episodes += 1
            outputs.append({"task_id": task["id"], "stop": trace["stop"]})
            scores.append(float(trace["verdict"]["success"]))
            if capture_traces:
                traces.append({"task_prompt": task["prompt"], "trace": deepcopy(trace)})
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=traces if capture_traces else None,
                               num_metric_calls=len(batch))

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        if components_to_update != ["solver_harness"] or eval_batch.trajectories is None:
            raise ValueError("Captured solver traces required; no other components may change")
        if not (len(eval_batch.trajectories) == len(eval_batch.outputs) == len(eval_batch.scores)):
            raise ValueError("Misaligned reflection batch")
        records = []
        for item, output, score in zip(eval_batch.trajectories, eval_batch.outputs, eval_batch.scores):
            trace = item["trace"]
            if trace["task_id"] not in self.allowed_task_ids or trace["task_id"] != output["task_id"]:
                raise ValueError("Reflection trace outside the task allowlist")
            records.append({"Inputs": {"task": item["task_prompt"]},
                            "Generated Outputs": deepcopy(trace["steps"]),
                            "Feedback": {"independent_goal_success": bool(score), "stop": trace["stop"]}})
        return {"solver_harness": records}


class BudgetedReflectionLM:
    """Pass upstream reflection prompts unchanged through shared API accounting."""
    def __init__(self, api):
        self.api = api
        self.calls = 0

    def __call__(self, prompt):
        if not isinstance(prompt, str):
            raise ValueError("Only text reflection is enabled")
        self.calls += 1
        response = self.api.chat([{"role": "user", "content": prompt}], [],
                                 f"gepa-reflection:{self.calls}")
        choices = response.get("choices") or []
        if len(choices) != 1 or choices[0].get("finish_reason") == "length":
            raise RuntimeError("GEPA reflection missing or truncated; do not accept partial text")
        text = (choices[0].get("message") or {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("GEPA reflection returned no instruction text")
        return text
