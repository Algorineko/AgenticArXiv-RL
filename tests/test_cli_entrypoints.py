"""安装后的唯一入口回归，替代旧命名空间的双目录路径注入断言。"""

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from agentic_arxiv.cli import HELP_COMMANDS, _operation_kwargs, parse_args
from agentic_arxiv.config import read_default_mainline_config


def _environment():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


class InstalledEntrypointTest(unittest.TestCase):
    def setUp(self):
        self.distribution = importlib.metadata.distribution("agentic-arxiv")
        self.console = Path(sys.executable).parent / "agentic-arxiv"
        self.assertTrue(self.console.is_file(), "测试环境必须实际安装 console script")

    def _help(self, command, cwd, *, console=False):
        prefix = [str(self.console)] if console else [sys.executable, "-I", "-m", "agentic_arxiv"]
        proc = subprocess.run(prefix + list(command) + ["--help"], cwd=cwd,
                              env=_environment(), capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("usage:", proc.stdout)
        return proc.stdout

    def test_module_help_from_two_non_source_directories(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory in (first, second):
                for command in HELP_COMMANDS:
                    with self.subTest(directory=directory, command=command):
                        self._help(command, directory)

    def test_console_help_uses_same_installed_implementation(self):
        entrypoint = next(entry for entry in self.distribution.entry_points if entry.name == "agentic-arxiv")
        self.assertEqual(entrypoint.value, "agentic_arxiv.cli:main")
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory in (first, second):
                for command in (("evaluate", "run"), ("evaluate", "baselines"), ("evaluate", "rescore")):
                    with self.subTest(directory=directory, command=command):
                        self.assertEqual(self._help(command, directory, console=True),
                                         self._help(command, directory))

    def test_help_does_not_import_operations_or_read_credentials(self):
        probe = """
import sys
from pathlib import Path
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = Path(args[0]).name
        if name in ('.env', 'log.txt'):
            raise AssertionError('帮助读取凭据或日志: ' + str(args[0]))
sys.addaudithook(audit)
from agentic_arxiv.cli import main
try:
    main(sys.argv[1:])
except SystemExit as exc:
    assert exc.code == 0
for module in sys.modules:
    assert not module.startswith(('torch', 'transformers', 'trl', 'datasets',
        'agentic_arxiv.runtime', 'agentic_arxiv.training', 'agentic_arxiv.evaluation',
        'agentic_arxiv.data_pipeline', 'agentic_arxiv.llm', 'agentic_arxiv.tools')), module
"""
        with tempfile.TemporaryDirectory() as directory:
            for command in HELP_COMMANDS:
                proc = subprocess.run([sys.executable, "-I", "-c", probe, *command, "--help"],
                                      cwd=directory, env=_environment(), capture_output=True,
                                      text=True, timeout=120)
                with self.subTest(command=command):
                    self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            self.assertEqual(list(Path(directory).iterdir()), [])
            # 显式配置也是实际操作输入，帮助不能先访问它。
            self._help(("train", "grpo", "--config", "missing.json"), directory)

    def test_missing_offline_snapshot_reports_selected_workspace(self):
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run([sys.executable, "-I", "-m", "agentic_arxiv", "run", "--offline",
                                   "--workspace", workspace], cwd=cwd, env=_environment(),
                                  capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            expected = Path(workspace).resolve() / "data/mock_arxiv_snapshot.json"
            self.assertIn(str(expected), proc.stderr)
            self.assertEqual(list(Path(workspace).iterdir()), [])


class ArgumentContractTest(unittest.TestCase):
    def test_workspace_at_each_command_level(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for prefix in (("--workspace", directory, "data", "sft", "mix"),
                           ("data", "--workspace", directory, "sft", "mix"),
                           ("data", "sft", "--workspace", directory, "mix"),
                           ("data", "sft", "mix", "--workspace", directory)):
                with self.subTest(argv=prefix):
                    _, args = parse_args(prefix)
                    self.assertEqual(args.workspace, root)
                    self.assertEqual(args.output, str(root / "data/sft/sft_react_messages_v3_train_mix.jsonl"))
                    self.assertNotIn("sft_data_command", _operation_kwargs(args))

    def test_operation_modes_and_delegated_defaults(self):
        _, run = parse_args(("run",))
        _, rollout = parse_args(("evaluate", "rollout"))
        _, online = parse_args(("evaluate", "rollout", "--no-offline"))
        _, grpo = parse_args(("train", "grpo"))
        self.assertFalse(run.offline)
        self.assertTrue(rollout.offline)
        self.assertFalse(online.offline)
        for name in ("loss_type", "epsilon_high", "mask_truncated_completions",
                     "importance_sampling_level", "scale_rewards"):
            self.assertIsNone(getattr(grpo, name), name)
        _, explicit = parse_args(("train", "grpo", "--no-mask-truncated-completions", "--no-qlora"))
        self.assertIs(explicit.mask_truncated_completions, False)
        self.assertIs(explicit.qlora, False)

    def test_package_config_explicit_config_and_cli_precedence(self):
        payload = json.loads(read_default_mainline_config())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            payload["sft"]["defaults"]["epochs"] = payload["sft"]["defaults"]["epochs"] + 1
            config = root / "config.json"
            config.write_text(json.dumps(payload), encoding="utf-8")
            _, configured = parse_args(("--workspace", str(root), "train", "sft", "--config", "config.json"))
            self.assertEqual(configured.epochs, payload["sft"]["defaults"]["epochs"])
            _, explicit = parse_args(("train", "sft", "--workspace", str(root), "--config", "config.json", "--epochs", "1"))
            self.assertEqual(explicit.epochs, 1)
            self.assertEqual(explicit.config_path, str(config))

    def test_split_and_local_model_paths_use_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, args = parse_args(("train", "grpo", "--workspace", str(root),
                                  "--split", "data/splits/v7_86.json:train", "--model", "outputs/sft/final"))
            self.assertEqual(args.split, f"{root}/data/splits/v7_86.json:train")
            self.assertEqual(args.model, str(root / "outputs/sft/final"))
            _, local = parse_args(("evaluate", "rollout", "--workspace", str(root), "--model", "models/policy"))
            self.assertEqual(local.model, str(root / "models/policy"))

    def test_dpo_arguments_preserve_workspace_and_explicit_precedence(self):
        payload = json.loads(read_default_mainline_config())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            payload["dpo"]["defaults"]["epochs"] = 7
            path = root / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            _, configured = parse_args(("train", "dpo", "--workspace", str(root), "--config", "config.json"))
            self.assertEqual(configured.epochs, 7)
            _, args = parse_args(("train", "dpo", "--workspace", str(root), "--config", "config.json",
                                  "--epochs", "2", "--inspect-only", "--no-verify"))
            self.assertEqual(args.epochs, 2)
            self.assertTrue(args.inspect_only)
            self.assertFalse(args.verify)
            self.assertEqual(args.data, str(root / "data/dpo/dpo_react_messages_v3_train.jsonl"))
            self.assertEqual(args.model, str(root / "outputs/sft_qlora/final"))
            _, generated = parse_args(("data", "dpo", "generate", "--workspace", str(root),
                                       "--snapshot", "data/snapshot.json", "--split", "data/splits/v7_86.json:train"))
            self.assertEqual(generated.snapshot, str(root / "data/snapshot.json"))
            self.assertEqual(generated.split, f"{root}/data/splits/v7_86.json:train")
            self.assertEqual(generated.num_trials, 5)
            self.assertNotIn("dpo_data_command", _operation_kwargs(generated))

    def test_dpo_capability_checks_do_not_require_train_for_generation(self):
        from unittest.mock import patch
        from agentic_arxiv.cli import _check_capabilities

        for command, expected in ((("data", "dpo", "generate"), "local"), (("train", "dpo"), "train")):
            _, args = parse_args(command)
            with patch("agentic_arxiv.cli._require_capability") as require:
                _check_capabilities(args)
            require.assert_called_once_with(expected)


if __name__ == "__main__":
    unittest.main()
