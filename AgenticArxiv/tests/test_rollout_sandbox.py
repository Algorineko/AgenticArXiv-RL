"""Rollout sandbox isolation tests (CPU-only, no network or model required)."""

import tempfile
import unittest
from pathlib import Path

from rl.sandbox import RolloutSandbox


class TestArtifactRootsAreIsolatedTest(unittest.TestCase):
    """跑测试不能删掉仓库里的产物目录。

    `RolloutSandbox` 的 reset 会把产物根恢复到构造时的基线，也就是删掉基线里
    没有的文件。`AgenticArxivMultiTurnEnv` 默认拿 `settings.*_path` 当这些根，
    而测试会实例化它 —— 于是「pytest 顺手清空刚下载好的快照 PDF」是一个真实
    发生过的破坏性行为（一次 `rl.build_snapshot` 要下载几百个 PDF）。
    `tests/conftest.py` 把产物根改指临时目录来杜绝它，这个测试守住那条契约。
    """

    def setUp(self):
        from config import settings

        self.settings = settings
        self.repo_root = Path(__file__).resolve().parents[2]

    def test_artifact_paths_live_outside_the_repository(self):
        for name in (
            "pdf_raw_path",
            "pdf_translated_path",
            "figures_path",
            "pdf_cache_path",
        ):
            with self.subTest(setting=name):
                value = Path(getattr(self.settings, name)).resolve()
                self.assertFalse(
                    self.repo_root in value.parents or value == self.repo_root,
                    f"tests/conftest.py 没有把 {name} 隔离到临时目录: {value}",
                )

    def test_multiturn_env_sandboxes_only_isolated_roots(self):
        from rl.multiturn_env import AgenticArxivMultiTurnEnv

        roots = AgenticArxivMultiTurnEnv(snapshot_path=None)._artifact_roots()
        self.assertTrue(roots)
        for root in roots:
            with self.subTest(root=str(root)):
                self.assertFalse(
                    self.repo_root in Path(root).resolve().parents,
                    f"rollout sandbox 会重置仓库内的目录: {root}",
                )


class _Component:
    def __init__(self):
        self.state = {"papers": [], "active": None}

    def capture_state(self):
        return dict(self.state)

    def restore_state(self, state):
        self.state = dict(state)


class RolloutSandboxTest(unittest.TestCase):
    def test_reset_restores_component_and_explicit_artifact_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.txt"
            baseline.write_text("before", encoding="utf-8")
            component = _Component()
            sandbox = RolloutSandbox(component, file_roots=(root,))

            component.state["papers"].append("paper-1")
            component.state["active"] = "paper-1"
            baseline.write_text("mutated", encoding="utf-8")
            (root / "leaked.txt").write_text("should disappear", encoding="utf-8")

            sandbox.reset()

            self.assertEqual(component.state, {"papers": [], "active": None})
            self.assertEqual(baseline.read_text(encoding="utf-8"), "before")
            self.assertFalse((root / "leaked.txt").exists())

    def test_repeated_reset_is_idempotent(self):
        component = _Component()
        sandbox = RolloutSandbox(component)
        component.state["active"] = "x"
        sandbox.reset()
        sandbox.reset()
        self.assertEqual(component.state, {"papers": [], "active": None})


if __name__ == "__main__":
    unittest.main()
