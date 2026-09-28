"""DAPO dynamic sampling without depending on TRL private trainer methods.

The public ``rollout_func`` contract in TRL 0.29 lets a rollout return extra
per-sample fields.  This module uses that contract to discard prompt groups
whose rewards have no useful within-group variance, sample *different*
prompts, and return a full replacement batch.  The trainer's private
``_generate_and_score_completions`` method is deliberately left untouched.

The buffer and prompt pool are dependency-free so their correctness can be
tested on CPU.  Only the runtime distributed guard imports torch, and only
when it is called.
"""

from __future__ import annotations

import copy
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


class DynamicSamplingError(RuntimeError):
    """Base class for dynamic-sampling failures."""


class DynamicSamplingExhausted(DynamicSamplingError):
    """Raised instead of returning a partial batch."""


class DynamicSamplingDistributedError(DynamicSamplingError):
    """Raised when the single-process implementation is launched distributed."""


def reward_standard_deviation(rewards: Sequence[float]) -> float:
    """Return the sample standard deviation used by TRL's GRPO grouping.

    TRL calls ``torch.std`` on each prompt group, whose default correction is
    one.  Matching that definition makes a configured floating-point threshold
    mean the same thing in the filter and in the trainer's reward metrics.
    """

    if len(rewards) < 2:
        raise ValueError("dynamic sampling requires at least two rewards per prompt")
    values = [float(value) for value in rewards]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"dynamic sampling rewards must be finite, got {values!r}")
    mean = math.fsum(values) / len(values)
    squared_error = math.fsum((value - mean) ** 2 for value in values)
    return math.sqrt(squared_error / (len(values) - 1))


@dataclass
class DynamicSample:
    """One rollout kept as an atomic record so parallel fields cannot drift."""

    prompt_ids: list[int]
    completion_ids: list[int]
    env_mask: list[int]
    task_id: str
    trajectory: Mapping[str, Any]
    reward: float
    logprobs: Any = None

    def validate(self) -> None:
        if len(self.completion_ids) != len(self.env_mask):
            raise DynamicSamplingError(
                "completion_ids and env_mask lost alignment for "
                f"task_id={self.task_id!r}: {len(self.completion_ids)} != {len(self.env_mask)}"
            )
        trajectory_task_id = str(self.trajectory.get("task_id") or "")
        if trajectory_task_id and trajectory_task_id != self.task_id:
            raise DynamicSamplingError(
                "task_id and trajectory lost alignment: "
                f"task_id={self.task_id!r}, trajectory.task_id={trajectory_task_id!r}"
            )


@dataclass(frozen=True)
class DynamicSamplingStats:
    target_groups: int
    accepted_groups: int
    rejected_groups: int
    resample_attempts: int
    reward_std_threshold: float


class DynamicSamplingBuffer:
    """Pure group-level variance filter with an exact-size accepted buffer."""

    def __init__(
        self,
        *,
        target_groups: int,
        group_size: int,
        reward_std_threshold: float = 0.0,
    ) -> None:
        if target_groups < 0:
            raise ValueError("target_groups cannot be negative")
        if group_size < 2:
            raise ValueError("group_size must be at least 2 for dynamic sampling")
        threshold = float(reward_std_threshold)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("reward_std_threshold must be a finite non-negative number")
        self.target_groups = int(target_groups)
        self.group_size = int(group_size)
        self.reward_std_threshold = threshold
        self._accepted: list[tuple[DynamicSample, ...]] = []
        self.rejected_groups = 0

    @property
    def accepted_groups(self) -> int:
        return len(self._accepted)

    @property
    def missing_groups(self) -> int:
        return self.target_groups - self.accepted_groups

    @property
    def complete(self) -> bool:
        return self.missing_groups == 0

    def add_group(self, samples: Sequence[DynamicSample]) -> bool:
        """Validate and accept a group iff its sample std is above threshold."""

        if self.complete:
            raise DynamicSamplingError("cannot add a group after the accepted buffer is full")
        if len(samples) != self.group_size:
            raise DynamicSamplingError(
                f"rollout group has {len(samples)} samples; expected {self.group_size}"
            )
        group = tuple(samples)
        for sample in group:
            sample.validate()
        task_ids = {sample.task_id for sample in group}
        if len(task_ids) != 1:
            raise DynamicSamplingError(
                f"one reward group contains multiple task_ids: {sorted(task_ids)!r}"
            )
        std = reward_standard_deviation([sample.reward for sample in group])
        # A threshold is intentionally strict: equality has no margin and is
        # rejected.  This also makes threshold=0 exactly implement DAPO's
        # non-zero-variance rule.
        if std <= self.reward_std_threshold:
            self.rejected_groups += 1
            return False
        self._accepted.append(group)
        return True

    def samples(self) -> list[DynamicSample]:
        """Return a full flat batch, never a silently truncated one."""

        if not self.complete:
            raise DynamicSamplingExhausted(
                "dynamic sampling buffer is incomplete: "
                f"accepted_groups={self.accepted_groups}, target_groups={self.target_groups}"
            )
        flattened = [sample for group in self._accepted for sample in group]
        expected = self.target_groups * self.group_size
        if len(flattened) != expected:  # defensive invariant
            raise DynamicSamplingError(
                f"dynamic sampling output cardinality mismatch: {len(flattened)} != {expected}"
            )
        return flattened


def _task_id_from_prompt(prompt: Any) -> str:
    if isinstance(prompt, Mapping):
        direct = prompt.get("task_id") or prompt.get("_task_id")
        if direct:
            return str(direct)
        if "prompt" in prompt:
            return _task_id_from_prompt(prompt["prompt"])
    if isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
        found = {
            str(item.get("_task_id") or item.get("task_id"))
            for item in prompt
            if isinstance(item, Mapping) and (item.get("_task_id") or item.get("task_id"))
        }
        if len(found) == 1:
            return found.pop()
        if len(found) > 1:
            raise DynamicSamplingError(
                f"prompt contains conflicting hidden task ids: {sorted(found)!r}"
            )
    raise DynamicSamplingError(
        "dynamic sampling requires every prompt to carry a stable task_id; "
        "build prompts with build_prompt_dataset()"
    )


class DeterministicPromptPool:
    """Seeded replacement source that honors a per-batch exclusion set."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], *, seed: int) -> None:
        if not rows:
            raise ValueError("dynamic sampling prompt pool cannot be empty")
        self._rows = [copy.deepcopy(dict(row)) for row in rows]
        identities = [_task_id_from_prompt(row) for row in self._rows]
        if len(set(identities)) != len(identities):
            raise ValueError("dynamic sampling prompt pool contains duplicate task_ids")
        self._identities = identities
        self._rng = random.Random(int(seed))

    def take(self, excluded_task_ids: Iterable[str]) -> Mapping[str, Any]:
        excluded = {str(task_id) for task_id in excluded_task_ids}
        eligible = [
            index for index, task_id in enumerate(self._identities)
            if task_id not in excluded
        ]
        if not eligible:
            raise DynamicSamplingExhausted(
                "dynamic sampling prompt pool has no untried prompt left in this batch"
            )
        index = self._rng.choice(eligible)
        return copy.deepcopy(self._rows[index])


def assert_dynamic_sampling_single_process(trainer: Any = None) -> None:
    """Fail loudly when this first implementation is launched distributed."""

    detected: list[str] = []
    raw_world_size = os.environ.get("WORLD_SIZE")
    if raw_world_size:
        try:
            if int(raw_world_size) > 1:
                detected.append(f"WORLD_SIZE={raw_world_size}")
        except ValueError:
            detected.append(f"invalid WORLD_SIZE={raw_world_size!r}")

    if trainer is not None:
        accelerator = getattr(trainer, "accelerator", None)
        num_processes = getattr(accelerator, "num_processes", 1)
        if int(num_processes or 1) > 1:
            detected.append(f"accelerator.num_processes={num_processes}")

    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size > 1:
                detected.append(f"torch.distributed.world_size={world_size}")
    except (ImportError, RuntimeError):
        # The environment variable and accelerator checks still cover normal
        # launch paths.  A broken optional torch import should not hide them.
        pass

    if detected:
        details = ", ".join(detected)
        raise DynamicSamplingDistributedError(
            "DAPO dynamic sampling currently supports single-process training only; "
            f"distributed launch detected ({details}). Disable --dynamic_sampling "
            "or launch one process."
        )


def _required_sequence(output: Mapping[str, Any], key: str, expected: int) -> Sequence[Any]:
    value = output.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DynamicSamplingError(f"rollout output {key!r} must be a sequence")
    if len(value) != expected:
        raise DynamicSamplingError(
            f"rollout output {key!r} has {len(value)} items; expected {expected}"
        )
    return value


class DynamicSamplingRollout:
    """Public-API rollout wrapper implementing DAPO dynamic sampling."""

    def __init__(
        self,
        *,
        rollout_func: Callable[[Sequence[Any], Any], Mapping[str, Any]],
        prompt_rows: Sequence[Mapping[str, Any]],
        preview_reward_func: Callable[..., Sequence[float]],
        reward_std_threshold: float = 0.0,
        max_resample_attempts: int = 32,
        seed: int = 42,
    ) -> None:
        if max_resample_attempts < 0:
            raise ValueError("max_resample_attempts cannot be negative")
        threshold = float(reward_std_threshold)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("reward_std_threshold must be a finite non-negative number")
        self.rollout_func = rollout_func
        self.preview_reward_func = preview_reward_func
        self.reward_std_threshold = threshold
        self.max_resample_attempts = int(max_resample_attempts)
        self.prompt_pool = DeterministicPromptPool(prompt_rows, seed=seed)
        self.last_stats: Optional[DynamicSamplingStats] = None

    @staticmethod
    def _group_size(trainer: Any) -> int:
        training = bool(getattr(getattr(trainer, "model", None), "training", True))
        name = "num_generations" if training else "num_generations_eval"
        value = int(getattr(trainer, name, 0) or 0)
        if value < 2:
            raise DynamicSamplingError(
                f"dynamic sampling requires {name} >= 2, got {value}"
            )
        return value

    def _run_candidates(
        self,
        prompts: Sequence[Any],
        task_ids: Sequence[str],
        trainer: Any,
        group_size: int,
    ) -> list[list[DynamicSample]]:
        expected = len(prompts)
        if expected != len(task_ids) or expected % group_size:
            raise DynamicSamplingError(
                "candidate prompts/task_ids must be aligned complete reward groups"
            )
        output = self.rollout_func(list(prompts), trainer)
        prompt_ids = _required_sequence(output, "prompt_ids", expected)
        completion_ids = _required_sequence(output, "completion_ids", expected)
        env_masks = _required_sequence(output, "env_mask", expected)
        trajectories = _required_sequence(output, "trajectory_results", expected)

        logprobs_value = output.get("logprobs")
        if logprobs_value is None:
            logprobs = [None] * expected
        else:
            logprobs = list(_required_sequence(output, "logprobs", expected))

        state = getattr(trainer, "state", None)
        rewards = list(self.preview_reward_func(
            completions=[None] * expected,
            task_id=list(task_ids),
            trainer_state=state,
            trajectory_results=list(trajectories),
        ))
        if len(rewards) != expected:
            raise DynamicSamplingError(
                f"preview reward returned {len(rewards)} values; expected {expected}"
            )

        samples: list[DynamicSample] = []
        for index in range(expected):
            trajectory = trajectories[index]
            if not isinstance(trajectory, Mapping):
                raise DynamicSamplingError(
                    f"trajectory_results[{index}] must be a mapping"
                )
            sample = DynamicSample(
                prompt_ids=list(prompt_ids[index]),
                completion_ids=list(completion_ids[index]),
                env_mask=list(env_masks[index]),
                task_id=str(task_ids[index]),
                trajectory=trajectory,
                reward=float(rewards[index]),
                logprobs=logprobs[index],
            )
            sample.validate()
            samples.append(sample)
        return [
            samples[index:index + group_size]
            for index in range(0, expected, group_size)
        ]

    @staticmethod
    def _validate_prompt_groups(
        prompts: Sequence[Any], group_size: int
    ) -> list[str]:
        task_ids = [_task_id_from_prompt(prompt) for prompt in prompts]
        for start in range(0, len(task_ids), group_size):
            group_ids = task_ids[start:start + group_size]
            if len(set(group_ids)) != 1:
                raise DynamicSamplingError(
                    "TRL prompt repetitions are not contiguous/aligned: "
                    f"group={group_ids!r}"
                )
        return task_ids

    def __call__(self, prompts: Sequence[Any], trainer: Any) -> dict[str, Any]:
        assert_dynamic_sampling_single_process(trainer)
        group_size = self._group_size(trainer)
        if not prompts:
            raise DynamicSamplingError("dynamic sampling received an empty prompt batch")
        if len(prompts) % group_size:
            raise DynamicSamplingError(
                f"prompt count {len(prompts)} is not divisible by group size {group_size}"
            )

        task_ids = self._validate_prompt_groups(prompts, group_size)
        target_groups = len(prompts) // group_size
        buffer = DynamicSamplingBuffer(
            target_groups=target_groups,
            group_size=group_size,
            reward_std_threshold=self.reward_std_threshold,
        )
        attempted_task_ids = set(task_ids)

        for group in self._run_candidates(
            prompts, task_ids, trainer, group_size
        ):
            buffer.add_group(group)

        resample_attempts = 0
        while not buffer.complete:
            remaining_budget = self.max_resample_attempts - resample_attempts
            if remaining_budget <= 0:
                raise DynamicSamplingExhausted(
                    "DAPO dynamic sampling exhausted max_resample_attempts="
                    f"{self.max_resample_attempts}: accepted_groups={buffer.accepted_groups}/"
                    f"{target_groups}, rejected_groups={buffer.rejected_groups}; "
                    "refusing to return a partial batch"
                )

            # Refill all currently missing slots together.  This keeps real GPU
            # generation batched while each candidate still consumes one clear
            # resample attempt.
            candidate_count = min(buffer.missing_groups, remaining_budget)
            replacement_prompts: list[Any] = []
            replacement_task_ids: list[str] = []
            pool_error: Optional[DynamicSamplingExhausted] = None
            for _ in range(candidate_count):
                try:
                    row = self.prompt_pool.take(attempted_task_ids)
                except DynamicSamplingExhausted as exc:
                    pool_error = exc
                    break
                task_id = _task_id_from_prompt(row)
                attempted_task_ids.add(task_id)
                prompt = row.get("prompt") if isinstance(row, Mapping) else None
                if prompt is None:
                    raise DynamicSamplingError(
                        f"replacement row for task_id={task_id!r} has no 'prompt' field"
                    )
                replacement_prompts.extend(
                    copy.deepcopy(prompt) for _ in range(group_size)
                )
                replacement_task_ids.extend([task_id] * group_size)

            replacement_groups = len(replacement_prompts) // group_size
            if replacement_groups == 0:
                raise DynamicSamplingExhausted(
                    "DAPO dynamic sampling cannot fill the accepted batch: "
                    f"accepted_groups={buffer.accepted_groups}/{target_groups}, "
                    f"rejected_groups={buffer.rejected_groups}, "
                    f"resample_attempts={resample_attempts}/"
                    f"{self.max_resample_attempts}; {pool_error}"
                ) from pool_error

            resample_attempts += replacement_groups
            for group in self._run_candidates(
                replacement_prompts,
                replacement_task_ids,
                trainer,
                group_size,
            ):
                buffer.add_group(group)

        accepted = buffer.samples()
        if len(accepted) != len(prompts):
            raise DynamicSamplingError(
                f"dynamic sampling changed batch size: {len(prompts)} -> {len(accepted)}"
            )

        self.last_stats = DynamicSamplingStats(
            target_groups=target_groups,
            accepted_groups=buffer.accepted_groups,
            rejected_groups=buffer.rejected_groups,
            resample_attempts=resample_attempts,
            reward_std_threshold=self.reward_std_threshold,
        )
        stats = {
            "target_groups": self.last_stats.target_groups,
            "accepted_groups": self.last_stats.accepted_groups,
            "rejected_groups": self.last_stats.rejected_groups,
            "resample_attempts": self.last_stats.resample_attempts,
            "reward_std_threshold": self.last_stats.reward_std_threshold,
        }

        if all(sample.logprobs is None for sample in accepted):
            accepted_logprobs: Any = None
        elif any(sample.logprobs is None for sample in accepted):
            raise DynamicSamplingError(
                "accepted rollout mixes missing and present logprobs"
            )
        else:
            accepted_logprobs = [sample.logprobs for sample in accepted]

        return {
            "prompt_ids": [sample.prompt_ids for sample in accepted],
            "completion_ids": [sample.completion_ids for sample in accepted],
            "logprobs": accepted_logprobs,
            "env_mask": [sample.env_mask for sample in accepted],
            # These public extra fields overwrite stale dataset metadata before
            # TRL invokes the formal reward function.
            "task_id": [sample.task_id for sample in accepted],
            "trajectory_results": [sample.trajectory for sample in accepted],
            "dynamic_sampling_stats": stats,
        }


def make_dynamic_sampling_rollout_func(
    rollout_func: Callable[[Sequence[Any], Any], Mapping[str, Any]],
    *,
    prompt_rows: Sequence[Mapping[str, Any]],
    preview_reward_func: Callable[..., Sequence[float]],
    reward_std_threshold: float = 0.0,
    max_resample_attempts: int = 32,
    seed: int = 42,
) -> DynamicSamplingRollout:
    """Build the composable rollout wrapper used by ``train_grpo.py``."""

    return DynamicSamplingRollout(
        rollout_func=rollout_func,
        prompt_rows=prompt_rows,
        preview_reward_func=preview_reward_func,
        reward_std_threshold=reward_std_threshold,
        max_resample_attempts=max_resample_attempts,
        seed=seed,
    )
