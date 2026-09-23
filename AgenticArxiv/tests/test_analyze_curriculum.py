"""奖励课程分析工具的分段与判定逻辑。

工具本身是给 README TODO P1「课程定档」提供证据用的：把一次训练切在课程窗口
上，逐分量看压制期与全权重期的差别。切错了或者把「权重放开」误读成「策略变
强」，结论就会反。
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

try:
    import tensorboard  # noqa: F401

    HAVE_TENSORBOARD = True
except ImportError:  # pragma: no cover - 可选依赖
    HAVE_TENSORBOARD = False

from analyze_curriculum import (  # noqa: E402
    CURRICULUM_SCALED,
    DIAGNOSTICS,
    WEIGHTED_COMPONENTS,
    _mean,
    _split,
)


class SplitTest(unittest.TestCase):
    def test_split_is_halF_open_on_the_boundary(self):
        series = [(0, 1.0), (29, 2.0), (30, 3.0), (31, 4.0)]
        early, late = _split(series, 30)
        self.assertEqual(early, [1.0, 2.0])
        self.assertEqual(late, [3.0, 4.0])

    def test_empty_side_is_tolerated(self):
        early, late = _split([(5, 1.0)], 30)
        self.assertEqual(early, [1.0])
        self.assertEqual(late, [])

    def test_mean_of_nothing_is_none_not_zero(self):
        """区分「没有数据」与「均值恰好是 0」——后者会被读成一种结论。"""
        self.assertIsNone(_mean([]))
        self.assertEqual(_mean([0.0]), 0.0)


class TagPrefixTest(unittest.TestCase):
    """TRL 落盘的 tag 带 `train/` 前缀，工具必须两种都认。

    只认裸 tag 时的表现不是报错而是「没有找到 reward_components/* 曲线」——
    数据明明在文件里，工具却让使用者去加 --report_to tensorboard，把人引到
    错误的方向。
    """

    def test_bare_and_prefixed_tags_both_resolve(self):
        from analyze_curriculum import _TAG_PREFIXES

        self.assertIn("", _TAG_PREFIXES)
        self.assertIn("train/", _TAG_PREFIXES)

    def test_prefixed_tag_is_found_when_only_that_exists(self):
        from unittest.mock import MagicMock

        from analyze_curriculum import _scalar_series

        accumulator = MagicMock()

        def scalars(tag):
            if tag == "train/reward_components/tool":
                return [MagicMock(step=0, value=0.5)]
            raise KeyError(tag)

        accumulator.Scalars.side_effect = scalars

        series = _scalar_series(accumulator, "reward_components/tool")
        self.assertEqual(series, [(0, 0.5)])

    def test_missing_tag_yields_empty_not_an_exception(self):
        from unittest.mock import MagicMock

        from analyze_curriculum import _scalar_series

        accumulator = MagicMock()
        accumulator.Scalars.side_effect = KeyError("nope")
        self.assertEqual(_scalar_series(accumulator, "reward_components/tool"), [])


class ComponentGroupsTest(unittest.TestCase):
    def test_only_three_components_are_curriculum_scaled(self):
        self.assertEqual(set(CURRICULUM_SCALED), {"tool", "argument", "outcome"})
        for name in CURRICULUM_SCALED:
            with self.subTest(component=name):
                self.assertIn(name, WEIGHTED_COMPONENTS)

    def test_diagnostics_are_not_part_of_the_weighted_average(self):
        self.assertEqual(set(DIAGNOSTICS) & set(WEIGHTED_COMPONENTS), set())


@unittest.skipUnless(HAVE_TENSORBOARD, "分析工具依赖可选的 tensorboard")
class EndToEndTest(unittest.TestCase):
    def _write_run(self, root: Path, *, ramp: bool = True, steps: int = 60):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(root))
        for step in range(steps):
            after = step >= 30
            for name, early, late in (
                ("format", 0.2, 0.9),
                ("tool", 0.1, 0.5),
                ("argument", 0.05, 0.4),
                ("process", 0.2, 0.7),
                ("outcome", 0.1, 0.6),
            ):
                value = late if (after and ramp) else early
                writer.add_scalar(f"reward_components/{name}", value, step)
            writer.add_scalar(
                "reward_weights/tool", 1.0 if (after and ramp) else 1.0 / 3, step
            )
            writer.add_scalar("reward", 0.6 if after else 0.3, step)
        writer.close()

    def test_reports_the_ramp_when_the_curriculum_bit(self):
        from analyze_curriculum import analyze

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            self._write_run(root)
            import io
            import contextlib

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = analyze(root, 30)

        self.assertEqual(code, 0)
        output = buffer.getvalue()
        self.assertIn("reward_weights/tool", output)
        self.assertIn("1.0", output)
        for name in CURRICULUM_SCALED:
            with self.subTest(component=name):
                self.assertIn(f"`{name}` 全权重期", output)
        self.assertNotIn("权重全程没有变化", output)

    def test_flags_a_curriculum_that_never_ramped(self):
        from analyze_curriculum import analyze

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "flat"
            self._write_run(root, ramp=False)
            import io
            import contextlib

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                analyze(root, 30)

        self.assertIn("权重全程没有变化", buffer.getvalue())

    def test_missing_scalars_is_a_clear_error(self):
        from analyze_curriculum import analyze

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "empty"
            root.mkdir()
            with self.assertRaises(SystemExit):
                analyze(root, 30)


if __name__ == "__main__":
    unittest.main()
