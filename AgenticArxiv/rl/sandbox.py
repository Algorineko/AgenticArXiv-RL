"""Per-rollout state snapshots and restoration.

The RL environment is intentionally stateful: searches seed paper references,
downloads update cache state, and translation creates task handles.  A fresh
object per GRPO generation protects the common path, but benchmark runners and
custom rollout functions may reuse an environment.  ``RolloutSandbox`` gives
both paths the same reset contract: capture the initial state once, then restore
it before every independent trajectory.
"""

from __future__ import annotations

import copy
import shutil
import weakref
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, Mapping, Optional, Union


@dataclass
class _FileSnapshot:
    """Baseline file contents as bytes or paths to temporary disk copies."""

    roots: tuple[Path, ...]
    files: Dict[Path, Union[bytes, Path]]
    backup_dir: Optional[TemporaryDirectory] = None

    def __post_init__(self) -> None:
        """Release temporary copies when the snapshot is no longer referenced."""
        if self.backup_dir is not None:
            weakref.finalize(self, self.backup_dir.cleanup)


class RolloutSandbox:
    """Snapshot and restore mutable rollout state.

    Components can implement ``capture_state``/``restore_state``.  For small
    test doubles, a normal ``__dict__`` is supported as a fallback.  File roots
    are always explicit; reset only removes/restores files below those roots,
    so it cannot affect unrelated workspace files. Large baseline files use
    temporary disk backups, released with the sandbox, instead of heap storage.
    """

    def __init__(
        self,
        *components: Any,
        file_roots: Iterable[Path] = (),
    ) -> None:
        self.components = tuple(component for component in components if component is not None)
        self._component_states = [self._capture_component(component) for component in self.components]
        self._files = self._capture_files(file_roots)

    @staticmethod
    def _capture_component(component: Any) -> Any:
        capture = getattr(component, "capture_state", None)
        if callable(capture):
            return ("protocol", copy.deepcopy(capture()))
        if hasattr(component, "__dict__"):
            return ("dict", copy.deepcopy(component.__dict__))
        raise TypeError(
            f"Sandbox component {type(component).__name__} must implement "
            "capture_state/restore_state or expose __dict__"
        )

    @staticmethod
    def _capture_files(roots: Iterable[Path]) -> _FileSnapshot:
        normalized = tuple(Path(root).resolve() for root in roots if root is not None)
        files: Dict[Path, Union[bytes, Path]] = {}
        backup_dir: Optional[TemporaryDirectory] = None
        for root in normalized:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    resolved = path.resolve()
                    if backup_dir is not None and resolved.is_relative_to(Path(backup_dir.name).resolve()):
                        continue
                    if resolved in files:
                        continue
                    # Keep small offline stubs in memory, but never omit a
                    # baseline PDF: reset would mistake it for a new artifact.
                    if path.stat().st_size <= 2_000_000:
                        files[resolved] = path.read_bytes()
                    else:
                        if backup_dir is None:
                            backup_dir = TemporaryDirectory(prefix="aa-rl-sandbox-")
                        backup = Path(backup_dir.name) / str(len(files))
                        shutil.copyfile(path, backup)
                        files[resolved] = backup
        return _FileSnapshot(roots=normalized, files=files, backup_dir=backup_dir)

    @staticmethod
    def _restore_component(component: Any, state: Any) -> None:
        kind, payload = state
        if kind == "protocol":
            restore = getattr(component, "restore_state", None)
            if not callable(restore):
                raise TypeError(f"{type(component).__name__} lost restore_state")
            restore(copy.deepcopy(payload))
            return
        component.__dict__.clear()
        component.__dict__.update(copy.deepcopy(payload))

    def _restore_files(self) -> None:
        baseline = self._files.files
        backup_root = (
            Path(self._files.backup_dir.name).resolve()
            if self._files.backup_dir is not None else None
        )
        for root in self._files.roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*"), reverse=True):
                if backup_root is not None and path.resolve().is_relative_to(backup_root):
                    continue
                if path.is_file() and not path.is_symlink() and path.resolve() not in baseline:
                    path.unlink()
        for path, content in baseline.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, Path):
                shutil.copyfile(content, path)
            else:
                path.write_bytes(content)

    def reset(self) -> None:
        """Restore every component and explicitly sandboxed artifact root."""
        for component, state in zip(self.components, self._component_states):
            self._restore_component(component, state)
        self._restore_files()

    def describe(self) -> Mapping[str, int]:
        """Small diagnostic payload for rollout logs and tests."""
        return {
            "components": len(self.components),
            "file_roots": len(self._files.roots),
            "tracked_files": len(self._files.files),
        }
