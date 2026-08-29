"""Multi-turn on-policy distillation for ReAct tool-use trajectories.

TRL's :class:`GKDTrainer` samples one continuation from the initial prompt and
assumes that every supervised token is contiguous after that prompt.  Agentic
rollouts violate both assumptions: tool observations are inserted between
assistant turns, and those environment tokens must be context-only.  This
module keeps GKD's teacher/student reverse-KL objective while replacing its
single-turn sampler and contiguous loss mask.

The implementation deliberately uses textual ReAct actions, matching the
project's inference and GRPO paths.  Each sample owns an independent
``AgenticArxivMultiTurnEnv`` instance.  Prompt and observation tokens receive
label ``-100``; only student-generated assistant tokens participate in the
distillation loss.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch

from rl.grpo_reward import TOOL_ERROR_PREFIX, _thought_of, parse_react_action


IGNORE_INDEX = -100


@dataclass
class MultiTurnOPDRollout:
    """One fully materialized trajectory and its non-contiguous loss mask."""

    input_ids: List[int]
    labels: List[int]
    history: List[Dict[str, Any]]
    termination_type: str
    turn_count: int
    assistant_tokens: int
    observation_tokens: int


@dataclass
class MultiTurnOPDBatch:
    """Padded tensors plus auditable trajectories for one training batch."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    trajectories: List[Dict[str, Any]]
    metrics: Dict[str, float]


class MultiTurnOPDCollator:
    """Left-pad prompt-only rows while preserving task ids for environment setup."""

    def __init__(self, pad_token_id: int):
        if pad_token_id is None:
            raise ValueError("Multi-turn OPD requires a pad_token_id")
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty multi-turn OPD batch")
        sequences = [list(feature["input_ids"]) for feature in features]
        width = max(len(sequence) for sequence in sequences)
        prompts, masks = [], []
        for sequence in sequences:
            padding = width - len(sequence)
            prompts.append([self.pad_token_id] * padding + sequence)
            masks.append([0] * padding + [1] * len(sequence))
        return {
            "prompts": torch.tensor(prompts, dtype=torch.long),
            "prompt_attention_mask": torch.tensor(masks, dtype=torch.long),
            "task_ids": [str(feature.get("task_id", "")) for feature in features],
        }


def validate_tokenizer_compatibility(student_tokenizer: Any, teacher_tokenizer: Any) -> None:
    """Fail before training when token ids do not represent the same strings.

    Reverse-KL compares the full student and teacher distributions at each
    vocabulary index.  Equal vocabulary *sizes* are insufficient: index 42
    must refer to the same token in both models.
    """

    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    if student_vocab != teacher_vocab:
        only_student = len(set(student_vocab) - set(teacher_vocab))
        only_teacher = len(set(teacher_vocab) - set(student_vocab))
        remapped = sum(
            1 for token, token_id in student_vocab.items()
            if token in teacher_vocab and teacher_vocab[token] != token_id
        )
        raise ValueError(
            "Multi-turn OPD requires identical teacher/student token-to-id vocabularies "
            f"(student={len(student_vocab)}, teacher={len(teacher_vocab)}, "
            f"student_only={only_student}, teacher_only={only_teacher}, remapped={remapped})."
        )

    special_names = (
        "pad_token_id",
        "eos_token_id",
        "bos_token_id",
        "unk_token_id",
    )
    mismatches = [
        name for name in special_names
        if getattr(student_tokenizer, name, None) != getattr(teacher_tokenizer, name, None)
    ]
    if mismatches:
        raise ValueError(
            "Teacher/student special token ids differ: " + ", ".join(mismatches)
        )


def _dispatch_tool(environment: Any, action: Mapping[str, Any]) -> Any:
    name = str(action["name"])
    args = dict(action.get("args") or {})
    method = getattr(environment, name, None)
    if method is None or not callable(method):
        raise ValueError(f"Unknown tool: {name}")
    return method(**args)


def _dependency_setup_steps(
    task: Mapping[str, Any], tasks_by_id: Mapping[str, Mapping[str, Any]]
) -> List[Mapping[str, Any]]:
    """Resolve setup calls without leaking the current task's gold action."""

    explicit = task.get("setup") or []
    if explicit:
        return list(explicit)

    chain: List[Mapping[str, Any]] = []
    current = task.get("depends_on")
    seen = set()
    while current:
        if current in seen:
            raise ValueError(f"Dependency cycle while preparing task {task.get('id')}: {current}")
        seen.add(current)
        dependency = tasks_by_id.get(str(current))
        if dependency is None:
            raise ValueError(f"Missing dependency task {current!r} for {task.get('id')!r}")
        chain.append(dependency)
        current = dependency.get("depends_on")

    steps: List[Mapping[str, Any]] = []
    for dependency in reversed(chain):
        expected_tools = dependency.get("expected_tools") or []
        expected_args = dependency.get("expected_tool_args") or []
        for index, name in enumerate(expected_tools):
            args = expected_args[index] if index < len(expected_args) else {}
            steps.append({"name": name, "args": dict(args or {})})
    return steps


def prepare_rollout_environment(
    environment: Any,
    task_id: str,
    tasks_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reset and seed exactly the state declared by a benchmark task."""

    environment.reset(task_id=task_id)
    task = tasks_by_id.get(task_id)
    if task is None:
        return
    for setup_action in _dependency_setup_steps(task, tasks_by_id):
        _dispatch_tool(environment, setup_action)


def _observation_suffix_ids(tokenizer: Any, observation: str, max_body_tokens: int) -> List[int]:
    prefix = list(tokenizer("\nObservation: ", add_special_tokens=False)["input_ids"])
    body = list(tokenizer(str(observation), add_special_tokens=False)["input_ids"])
    suffix = list(tokenizer("\nThought:", add_special_tokens=False)["input_ids"])
    return prefix + body[: max(0, int(max_body_tokens))] + suffix


def run_multiturn_opd_rollouts(
    *,
    prompt_ids: Sequence[Sequence[int]],
    task_ids: Sequence[str],
    tokenizer: Any,
    environment_factory: Callable[[], Any],
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    generate_turn: Callable[[Sequence[Sequence[int]]], Sequence[Sequence[int]]],
    max_turns: int,
    max_observation_tokens: int,
) -> List[MultiTurnOPDRollout]:
    """Run independent student-policy trajectories and build token-level masks."""

    if len(prompt_ids) != len(task_ids):
        raise ValueError("prompt_ids and task_ids must have the same length")
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")

    sequences = [list(ids) for ids in prompt_ids]
    labels = [[IGNORE_INDEX] * len(ids) for ids in sequences]
    histories: List[List[Dict[str, Any]]] = [[] for _ in sequences]
    assistant_counts = [0] * len(sequences)
    observation_counts = [0] * len(sequences)
    terminations = ["INCOMPLETE"] * len(sequences)
    environments = [environment_factory() for _ in sequences]
    for environment, task_id in zip(environments, task_ids):
        prepare_rollout_environment(environment, str(task_id), tasks_by_id)

    active = list(range(len(sequences)))
    for turn_index in range(max_turns):
        if not active:
            break
        generated_batch = list(generate_turn([sequences[index] for index in active]))
        if len(generated_batch) != len(active):
            raise RuntimeError(
                f"generate_turn returned {len(generated_batch)} rows for {len(active)} active rollouts"
            )

        next_active: List[int] = []
        for generated, index in zip(generated_batch, active):
            generated_ids = [int(token_id) for token_id in generated]
            sequences[index].extend(generated_ids)
            labels[index].extend(generated_ids)
            assistant_counts[index] += len(generated_ids)

            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
            kind, action = parse_react_action(completion)
            thought = _thought_of(completion)
            if kind == "finish":
                histories[index].append({
                    "thought": thought,
                    "action": "FINISH",
                    "observation": "Task completed",
                })
                terminations[index] = "FINISH"
                continue
            if kind == "parse_error":
                histories[index].append({
                    "thought": thought,
                    "action": "PARSE_ERROR",
                    "observation": "Unable to parse Action",
                    "parse_failed": True,
                })
                terminations[index] = "PARSE_ERROR"
                continue

            try:
                result = _dispatch_tool(environments[index], action)
                observation = str(result)[:4000]
            except Exception as exc:  # noqa: BLE001
                observation = f"{TOOL_ERROR_PREFIX}{exc}"

            histories[index].append({
                "thought": thought,
                "action": json.dumps(action, ensure_ascii=False),
                "observation": observation,
            })

            if turn_index + 1 < max_turns:
                suffix_ids = _observation_suffix_ids(
                    tokenizer, observation, max_observation_tokens
                )
                sequences[index].extend(suffix_ids)
                labels[index].extend([IGNORE_INDEX] * len(suffix_ids))
                observation_counts[index] += len(suffix_ids)
                next_active.append(index)

        active = next_active

    for index in active:
        histories[index].append({
            "thought": "Maximum number of turns reached",
            "action": "FORCE_STOP",
            "observation": "Turn limit",
        })
        terminations[index] = "FORCE_STOP"
    # A tool call on the final allowed turn is also a force stop, but it is not
    # placed in ``next_active`` because there is no following observation turn.
    for index, termination in enumerate(terminations):
        if termination == "INCOMPLETE":
            histories[index].append({
                "thought": "Maximum number of turns reached",
                "action": "FORCE_STOP",
                "observation": "Turn limit",
            })
            terminations[index] = "FORCE_STOP"

    return [
        MultiTurnOPDRollout(
            input_ids=sequences[index],
            labels=labels[index],
            history=histories[index],
            termination_type=terminations[index],
            turn_count=sum(
                1 for step in histories[index] if step.get("action") != "FORCE_STOP"
            ),
            assistant_tokens=assistant_counts[index],
            observation_tokens=observation_counts[index],
        )
        for index in range(len(sequences))
    ]


def pad_multiturn_opd_rollouts(
    rollouts: Sequence[MultiTurnOPDRollout], pad_token_id: int, device: torch.device
) -> MultiTurnOPDBatch:
    """Right-pad complete trajectories without confusing EOS with padding."""

    if not rollouts:
        raise ValueError("Cannot pad an empty rollout batch")
    width = max(len(rollout.input_ids) for rollout in rollouts)
    input_rows, attention_rows, label_rows = [], [], []
    for rollout in rollouts:
        if len(rollout.input_ids) != len(rollout.labels):
            raise ValueError("Every rollout must have one label per input token")
        padding = width - len(rollout.input_ids)
        input_rows.append(rollout.input_ids + [pad_token_id] * padding)
        attention_rows.append([1] * len(rollout.input_ids) + [0] * padding)
        label_rows.append(rollout.labels + [IGNORE_INDEX] * padding)

    trajectories = [
        {
            "history": rollout.history,
            "timing": {},
            "token_usage": {},
            "iteration_count": rollout.turn_count,
        }
        for rollout in rollouts
    ]
    metrics = {
        "opd/turns": mean(rollout.turn_count for rollout in rollouts),
        "opd/finished_rate": mean(
            1.0 if rollout.termination_type == "FINISH" else 0.0 for rollout in rollouts
        ),
        "opd/parse_error_rate": mean(
            1.0 if rollout.termination_type == "PARSE_ERROR" else 0.0 for rollout in rollouts
        ),
        "opd/force_stop_rate": mean(
            1.0 if rollout.termination_type == "FORCE_STOP" else 0.0 for rollout in rollouts
        ),
        "opd/assistant_tokens": mean(rollout.assistant_tokens for rollout in rollouts),
        "opd/observation_tokens": mean(rollout.observation_tokens for rollout in rollouts),
    }
    return MultiTurnOPDBatch(
        input_ids=torch.tensor(input_rows, dtype=torch.long, device=device),
        attention_mask=torch.tensor(attention_rows, dtype=torch.long, device=device),
        labels=torch.tensor(label_rows, dtype=torch.long, device=device),
        trajectories=trajectories,
        metrics=metrics,
    )


def make_multiturn_gkd_trainer(gkd_trainer_class: type) -> type:
    """Create a GKD subclass without hard-coding TRL's experimental import path."""

    class MultiTurnGKDTrainer(gkd_trainer_class):
        def __init__(
            self,
            *args: Any,
            environment_factory: Callable[[], Any],
            tasks_by_id: Mapping[str, Mapping[str, Any]],
            max_turns: int = 4,
            max_observation_tokens: int = 256,
            max_sequence_length: Optional[int] = None,
            **kwargs: Any,
        ) -> None:
            self.environment_factory = environment_factory
            self.tasks_by_id = dict(tasks_by_id)
            self.max_turns = max(1, int(max_turns))
            self.max_observation_tokens = max(0, int(max_observation_tokens))
            self.max_sequence_length = max_sequence_length
            self._pending_multiturn_metrics: List[Dict[str, float]] = []
            self.latest_trajectory_results: List[Dict[str, Any]] = []
            super().__init__(*args, **kwargs)

        @staticmethod
        def _unpad_prompts(inputs: Mapping[str, Any]) -> List[List[int]]:
            prompts = inputs["prompts"]
            masks = inputs["prompt_attention_mask"]
            return [
                row[mask.bool()].detach().cpu().tolist()
                for row, mask in zip(prompts, masks)
            ]

        def _generate_turn(self, model: Any, sequences: Sequence[Sequence[int]]) -> List[List[int]]:
            if not sequences:
                return []
            device = next(model.parameters()).device
            width = max(len(sequence) for sequence in sequences)
            if self.max_sequence_length is not None:
                remaining = min(self.max_sequence_length - len(sequence) for sequence in sequences)
                if remaining <= 0:
                    return [[] for _ in sequences]
            else:
                remaining = self.generation_config.max_new_tokens
            max_new_tokens = min(int(self.generation_config.max_new_tokens), int(remaining))

            input_rows, mask_rows = [], []
            for sequence in sequences:
                padding = width - len(sequence)
                input_rows.append([self.processing_class.pad_token_id] * padding + list(sequence))
                mask_rows.append([0] * padding + [1] * len(sequence))
            input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
            attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
            generation_config = copy.deepcopy(self.generation_config)
            generation_config.max_new_tokens = max_new_tokens
            generation_config.return_dict_in_generate = True
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
            generated = outputs.sequences[:, width:]
            eos = generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) if eos is not None else set()
            rows: List[List[int]] = []
            for tensor_row in generated:
                row: List[int] = []
                for token_id in tensor_row.detach().cpu().tolist():
                    row.append(int(token_id))
                    if token_id in eos_ids:
                        break
                if not eos_ids:
                    while row and row[-1] == self.processing_class.pad_token_id:
                        row.pop()
                rows.append(row)
            return rows

        def training_step(self, model: Any, inputs: Dict[str, Any], num_items_in_batch: Any = None):
            try:
                from trl.models.utils import unwrap_model_for_generation
            except ImportError:  # pragma: no cover - compatibility with older TRL
                from trl.models import unwrap_model_for_generation

            prompt_ids = self._unpad_prompts(inputs)
            task_ids = [str(task_id) for task_id in inputs.get("task_ids", [""] * len(prompt_ids))]
            with unwrap_model_for_generation(
                model,
                self.accelerator,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model:
                rollouts = run_multiturn_opd_rollouts(
                    prompt_ids=prompt_ids,
                    task_ids=task_ids,
                    tokenizer=self.processing_class,
                    environment_factory=self.environment_factory,
                    tasks_by_id=self.tasks_by_id,
                    generate_turn=lambda sequences: self._generate_turn(unwrapped_model, sequences),
                    max_turns=self.max_turns,
                    max_observation_tokens=self.max_observation_tokens,
                )

            batch = pad_multiturn_opd_rollouts(
                rollouts,
                pad_token_id=self.processing_class.pad_token_id,
                device=inputs["prompts"].device,
            )
            if int((batch.labels != IGNORE_INDEX).sum()) == 0:
                raise RuntimeError("Multi-turn OPD generated no assistant tokens; refusing a zero-signal step")
            inputs["input_ids"] = batch.input_ids
            inputs["attention_mask"] = batch.attention_mask
            inputs["labels"] = batch.labels
            self.latest_trajectory_results = batch.trajectories
            self._pending_multiturn_metrics.append(batch.metrics)

            # Skip GKDTrainer.training_step: it would replace this trajectory
            # with a fresh single-turn continuation.  The next class in the MRO
            # is SFTTrainer/Trainer, which calls our compute_loss below.
            return super(gkd_trainer_class, self).training_step(
                model, inputs, num_items_in_batch
            )

        def compute_loss(self, model: Any, inputs: Mapping[str, Any], return_outputs=False, num_items_in_batch=None):
            student_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )
            self.teacher_model.eval()
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                )
            if student_outputs.logits.shape[-1] != teacher_outputs.logits.shape[-1]:
                raise RuntimeError(
                    "Teacher/student vocabulary dimensions differ during multi-turn OPD: "
                    f"{student_outputs.logits.shape[-1]} vs {teacher_outputs.logits.shape[-1]}"
                )

            shifted_labels = inputs["labels"][:, 1:]
            if not bool((shifted_labels != IGNORE_INDEX).any()):
                raise RuntimeError("Multi-turn OPD loss mask contains no assistant targets")
            loss = self.generalized_jsd_loss(
                student_logits=student_outputs.logits[:, :-1, :],
                teacher_logits=teacher_outputs.logits[:, :-1, :],
                labels=shifted_labels,
                beta=self.beta,
            )
            return (loss, student_outputs) if return_outputs else loss

        def log(self, logs: Dict[str, float], *args: Any, **kwargs: Any):
            if self._pending_multiturn_metrics:
                keys = self._pending_multiturn_metrics[0]
                for key in keys:
                    logs.setdefault(
                        key,
                        mean(metrics[key] for metrics in self._pending_multiturn_metrics),
                    )
                self._pending_multiturn_metrics.clear()
            return super().log(logs, *args, **kwargs)

    MultiTurnGKDTrainer.__name__ = "MultiTurnGKDTrainer"
    MultiTurnGKDTrainer.__qualname__ = "MultiTurnGKDTrainer"
    return MultiTurnGKDTrainer
