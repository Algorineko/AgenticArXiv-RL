"""可在基础环境检查的停止策略基类。

安装训练依赖时使用原 Transformers 回调；缺依赖时只允许纯策略检查，
实际训练仍在入口检查并导入训练库，绝不替换训练器或模型。
"""

from pathlib import Path

from agentic_arxiv.training.control import (
    CONTROL_VERSION, STATUS_FILE, atomic_control_json, read_stop_request,
)

try:
    from transformers import TrainerCallback
except ImportError:
    class TrainerCallback:
        """无训练依赖时供纯停止策略使用，不提供训练执行能力。"""


class ManagedStopCallback(TrainerCallback):
    """Consume requests only at a shared optimizer boundary and stop after commit.

    The checkpoint adapter must finish its transaction before ``on_save``. This
    callback never initiates a collective from a microbatch, a rollout, or an
    exception handler. Its status is training-side evidence, not proof of exit.
    """

    def __init__(self, control_dir, run_id, attempt_id, *, distributed=None,
                 checkpoint_committed, protection_reason=lambda: None):
        self.control_dir = Path(control_dir)
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.distributed = distributed
        self.checkpoint_committed = checkpoint_committed
        self.protection_reason = protection_reason
        self.user_stopped = False
        self.reason = None
        self.checkpoint = None
        self._pending_step = None
        self._request = None
        self._failed = False

    def _main(self, operation):
        return self.distributed.run_on_main(operation) if self.distributed is not None else operation()

    def _status(self, status, step, *, reason=None, checkpoint=None):
        # Exception handlers call this directly, never through _main.
        if self.distributed is not None and not self.distributed.is_main:
            return
        atomic_control_json(self.control_dir / STATUS_FILE, {
            "schema_version": CONTROL_VERSION, "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "request_id": self._request["request_id"] if self._request else None,
            "status": status, "global_step": step, "reason": reason,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
        })

    def _priority_reason(self, state, control):
        reason = self.protection_reason()
        if reason:
            return str(reason)
        max_steps = getattr(state, "max_steps", 0)
        if max_steps and max_steps > 0 and state.global_step >= max_steps:
            return "max_steps_or_epochs_completed"
        if control.should_training_stop and not self.user_stopped:
            return "trainer_stopped_early"
        return None

    def on_step_end(self, args, state, control, **kwargs):
        if self._failed or self._pending_step is not None or self.user_stopped:
            return control
        try:
            self._request = self._main(lambda: read_stop_request(
                self.control_dir, self.run_id, self.attempt_id))
            if self._request is None:
                return control
            reason = self._main(lambda: self._priority_reason(state, control))
            if reason:
                self.reason = reason
                self._main(lambda: self._status("superseded", state.global_step, reason=reason))
                return control
            self._pending_step = state.global_step
            self._main(lambda: self._status("checkpoint_pending", state.global_step))
            control.should_save = True
            return control
        except BaseException as exc:
            self.mark_failed(exc, state.global_step)
            raise

    def on_save(self, args, state, control, **kwargs):
        if self._failed or self._pending_step is None:
            return control
        try:
            reason = self._main(lambda: self._priority_reason(state, control))
            if reason:
                self.reason = reason
                self._main(lambda: self._status("superseded", state.global_step, reason=reason))
                self._pending_step = None
                return control
            if state.global_step != self._pending_step:
                raise RuntimeError("安全停止保存步数与请求边界不一致")
            checkpoint = self._main(lambda: self._committed_checkpoint(state.global_step))
            self._main(lambda: self._status("user_safe_stop", state.global_step,
                                           reason="user_safe_stop", checkpoint=checkpoint))
            self.checkpoint = checkpoint
            self.reason = "user_safe_stop"
            self.user_stopped = True
            self._pending_step = None
            control.should_training_stop = True
            return control
        except BaseException as exc:
            self.mark_failed(exc, state.global_step)
            raise

    def _committed_checkpoint(self, step):
        checkpoint = self.checkpoint_committed(step)
        if checkpoint is None or Path(checkpoint).name != f"checkpoint-{step}":
            raise RuntimeError("安全停止未收到当前优化步的完整 checkpoint 提交确认")
        return str(checkpoint)

    def mark_failed(self, error, step):
        """Best-effort local failure evidence, deliberately without communication."""
        self._failed = True
        self.user_stopped = False
        self.reason = f"training_exception:{type(error).__name__}"
        try:
            self._status("failed", step, reason=self.reason)
        except Exception:
            pass
