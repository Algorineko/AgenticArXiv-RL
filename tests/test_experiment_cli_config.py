"""管理 CLI/config 工程回归；仅构造控制元数据，不训练或伪造研究输入。"""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from agentic_arxiv.cli import HELP_COMMANDS, main, parse_args
from agentic_arxiv.config import read_default_mainline_config
from agentic_arxiv.experiments.config import load_profile, load_recipe, parse_overrides, resolve
from agentic_arxiv.runtime.react_contract import protocol_metadata


def profile(root):
    root = Path(root).resolve()
    return {"schema_version": 1, "workspace": str(root), "run_root": "managed",
            "input_roots": ["inputs"], "cache_root": "cache", "image": "local/test@sha256:" + "1" * 64,
            "uid": 1000, "gid": 1000, "gpus": ["GPU-a", "GPU-b"], "shm_size": "16g"}


def recipe(group="M1", stage="sft"):
    return {"schema_version": 1, "group": group, "stage": stage, "training_config": None,
            "inputs": {"base_model": None, "data": None},
            "training_overrides": {"defaults": {"seed": 17}, "trainer_parameters": {}},
            "checkpoint": {"save_steps": 50, "save_total_limit": 3},
            "evaluation": {"kind": "benchmark", "config": None},
            "budget": {"max_steps": None, "comparison_group": None, "accounting": "gpu_hours", "strict_match": False,
                       "costs": {"preference_generation_gpu_hours": None, "dpo_gpu_hours": None, "grpo_gpu_hours": None}}}


class ExperimentConfigurationTest(unittest.TestCase):
    def test_steps_and_epoch_budget_modes_are_explicit_and_compatible(self):
        for group, stage in (("M1", "sft"), ("M2", "dpo"), ("M3", "grpo")):
            selected = recipe(group, stage)
            selected["budget"]["max_steps"] = 12
            selected["training_overrides"]["defaults"]["max_steps"] = 12
            resolved = resolve(profile("/tmp"), selected)
            self.assertNotIn("epochs", resolved["recipe"]["budget"])
            selected["budget"]["epochs"] = None
            self.assertEqual(resolve(profile("/tmp"), selected)["training"]["defaults"]["max_steps"], 12)
            selected["budget"].update(max_steps=None, epochs=2)
            selected["training_overrides"]["defaults"].update(max_steps=-1, epochs=2)
            resolved = resolve(profile("/tmp"), selected)
            self.assertEqual(resolved["training"]["defaults"]["max_steps"], -1)
            self.assertEqual(resolved["training"]["defaults"]["epochs"], 2)

    def test_budget_conflicts_and_wrong_types_are_rejected_before_execution(self):
        for key, value in (("epochs", True), ("epochs", 1.5), ("epochs", 0), ("epochs", "2"),
                           ("max_steps", True), ("max_steps", 0), ("max_steps", 2.5)):
            selected = recipe()
            selected["budget"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                resolve(profile("/tmp"), selected)
        for budget, defaults in (({"max_steps": 12, "epochs": 2}, {"max_steps": 12, "epochs": 2}),
                                 ({"max_steps": 12}, {"max_steps": 11}),
                                 ({"epochs": 2}, {"max_steps": 12, "epochs": 2}),
                                 ({"epochs": 2}, {"max_steps": -1, "epochs": 3})):
            selected = recipe()
            selected["budget"].update(budget)
            selected["training_overrides"]["defaults"].update(defaults)
            with self.subTest(budget=budget, defaults=defaults), self.assertRaises(ValueError):
                resolve(profile("/tmp"), selected)

    def test_resource_aliases_rejected_before_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "real").mkdir()
            (root / "alias").symlink_to(root / "real", target_is_directory=True)
            for key in ("workspace", "run_root", "cache_root", "input_roots"):
                payload = profile(root)
                payload[key] = [str(root / "alias")] if key == "input_roots" else str(root / "alias")
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "符号链接|别名"):
                    resolve(payload, recipe())

    def test_three_layers_strict_validation_and_no_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = json.loads(read_default_mainline_config())
            config["sft"]["defaults"]["seed"] = 3
            (root / "mainline.json").write_text(json.dumps(config), encoding="utf-8")
            selected = recipe()
            selected["training_config"] = "mainline.json"
            before = copy.deepcopy(selected)
            resolved = resolve(profile(root), selected, parse_overrides(seed=29, gpus="GPU-b,GPU-a"), root)
            self.assertEqual(resolved["training"]["defaults"]["seed"], 29)
            self.assertEqual(resolved["sources"]["training.defaults.seed"], "explicit")
            self.assertEqual(resolved["overrides"]["gpus"], ["GPU-b", "GPU-a"])
            self.assertEqual(resolved["profile"]["run_root"], str(root / "managed"))
            self.assertEqual(selected, before)
            self.assertEqual(len(resolved["fingerprint"]), 64)
            self.assertEqual(resolve(profile(root), recipe())["training"]["defaults"]["seed"], 17)

    def test_unknown_fields_wrong_types_and_resource_algorithm_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            for key, value in (("lr", 0.1), ("privileged", True), ("uid", True), ("gid", 0),
                               ("image", "repo:latest"), ("schema_version", True)):
                payload = profile(directory)
                payload[key] = value
                with self.subTest(profile_field=key), self.assertRaises(ValueError):
                    resolve(payload, recipe())
            for overrides in ({"defaults": {"seed": True}}, {"defaults": {"nonexistent": 1}},
                              {"defaults": {"model": "unrecorded"}}, {"gpus": ["GPU-c"]},
                              {"gpus": ["GPU-a", "GPU-a"]}, {"trainer_parameters": {"logging_steps": "1"}}):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    resolve(profile(directory), recipe(), overrides)

    def test_nested_unknown_fields_and_ambiguous_config_rejected(self):
        for field, value in (("inputs", {"newest": "latest"}), ("evaluation", {"model": "wrong"}),
                             ("checkpoint", {"save_steps": 0, "save_total_limit": 2}),
                             ("evaluation", {"config": "fixed.json", "seed": 1}),
                             ("inputs", {"model": {"kind": "run", "run_id": "r", "attempt_id": "latest", "artifact": "final"}})):
            selected = recipe()
            selected[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                resolve(profile("/tmp"), selected)

    def test_load_rejects_duplicate_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for raw in ('{"schema_version":1,"schema_version":1}', '{"value":NaN}'):
                path.write_text(raw, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_profile(path)
            payload = recipe()
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_recipe(path), payload)

    def test_original_config_still_valid_and_grpo_algorithm_guard_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            selected = recipe("M3", "grpo")
            selected["training_overrides"]["defaults"]["loss_type"] = "other"
            with self.assertRaises(ValueError):
                resolve(profile(directory), selected)
            selected["training_overrides"]["defaults"]["loss_type"] = None
            resolved = resolve(profile(directory), selected)
            self.assertIsNone(resolved["training"]["defaults"]["loss_type"])
            self.assertEqual(resolved["training"]["defaults"]["save_steps"], 50)

    def test_m0_has_no_training_and_overrides_cannot_enable_it(self):
        selected = recipe("M0", "evaluate")
        selected["training_overrides"]["defaults"] = {}
        selected["checkpoint"] = None
        self.assertIsNone(resolve(profile("/tmp"), selected)["training"])
        with self.assertRaises(ValueError):
            resolve(profile("/tmp"), selected, parse_overrides(seed=5))

    def test_duplicate_cli_overrides_rejected(self):
        for values, seed in ((("defaults.seed=1", "defaults.seed=2"), None), (("defaults.seed=1",), 3), (("seed=1",), None)):
            with self.assertRaises(ValueError):
                parse_overrides(values, seed=seed)

    def test_verification_reference_is_explicit_and_strict(self):
        selected = recipe()
        selected["inputs"]["model"] = {
            "kind": "registration", "registration_id": "formal-sft",
            "verification_ref": {"run_id": "stage-check", "attempt_id": "a1"},
        }
        resolve(profile("/tmp"), selected)
        for field, value in (("attempt_id", "latest"), ("extra", "unexpected"), ("run_id", [])):
            modified = copy.deepcopy(selected)
            modified["inputs"]["model"]["verification_ref"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve(profile("/tmp"), modified)


class ExperimentCommandTest(unittest.TestCase):
    def test_invalid_artifact_registration_returns_json_input_error(self):
        for invalid, message in (("manifest", "缺少必要 training_manifest"),
                                 ("stage", "不支持的训练产物 stage"),
                                 ("weights", "权重或分片索引缺失"),
                                 ("base_weights", "权重或分片索引缺失")):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                base, model = root / "inputs" / "base", root / "inputs" / "model"
                for path in (base, model):
                    path.mkdir(parents=True)
                    (path / "config.json").write_text("{}", encoding="utf-8")
                    # 仅测试文件存在性合同，不构造或加载模型参数。
                    (path / "model.safetensors").touch()
                manifest = model / "training_manifest.json"
                manifest.write_text(json.dumps({
                    "stage": "grpo_full" if invalid == "stage" else "sft_full",
                    "base_model": str(base),
                    "protocol_metadata": {**protocol_metadata(), "chat_template_hash": "test-template"},
                }), encoding="utf-8")
                if invalid == "manifest":
                    manifest.unlink()
                elif invalid in {"weights", "base_weights"}:
                    ((model if invalid == "weights" else base) / "model.safetensors").unlink()
                filename = root / "profile.json"
                filename.write_text(json.dumps(profile(root)), encoding="utf-8")
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    code = main(["experiment", "register", "M1", "--model", str(model),
                                 "--profile", str(filename), "--json"])
                payload = json.loads(stream.getvalue())
                self.assertEqual(code, 2)
                self.assertEqual(payload["exit_code"], 2)
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["operation"], "register")
                self.assertEqual(payload["error"]["type"], "ValueError")
                self.assertIn(message, payload["error"]["message"])
                self.assertFalse((root / "managed").exists())

    def test_base_evaluation_missing_weights_returns_json_input_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            model = root / "inputs" / "base"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (root / "profile.json").write_text(json.dumps(profile(root)), encoding="utf-8")
            (root / "evaluation.json").write_text(json.dumps({"offline": True,
                "snapshot": str(Path(__file__).parent / "fixtures/t4_snapshot_for_t5.json")}), encoding="utf-8")
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = main(["experiment", "evaluate", str(model), "--kind", "benchmark",
                             "--evaluation-config", str(root / "evaluation.json"),
                             "--profile", str(root / "profile.json"), "--gpus", "GPU-a", "--json"])
            payload = json.loads(stream.getvalue())
            self.assertEqual(code, 2)
            self.assertEqual(payload["exit_code"], 2)
            self.assertFalse(payload["ok"])
            self.assertIn("权重或分片索引缺失", payload["error"]["message"])
            self.assertFalse((root / "managed").exists())

    def test_managed_benchmark_rejects_network_or_implicit_snapshot_without_side_effects(self):
        for config in ({}, {"offline": False}, {"offline": True}):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                (root / "profile.json").write_text(json.dumps(profile(root)), encoding="utf-8")
                (root / "evaluation.json").write_text(json.dumps(config), encoding="utf-8")
                stream = io.StringIO()
                with patch("agentic_arxiv.experiments.docker.DockerRuntime.call",
                           side_effect=AssertionError("invalid plan must not contact Docker")) as docker:
                    with contextlib.redirect_stdout(stream):
                        code = main(["experiment", "evaluate", str(root / "inputs" / "base"),
                                     "--evaluation-config", str(root / "evaluation.json"),
                                     "--profile", str(root / "profile.json"), "--gpus", "GPU-a", "--json"])
                payload = json.loads(stream.getvalue())
                self.assertEqual(code, 2)
                self.assertFalse(payload["ok"])
                self.assertIn("managed benchmark requires", payload["error"]["message"])
                docker.assert_not_called()
                self.assertFalse((root / "managed").exists())

    def test_real_readonly_list_does_not_create_root_or_import_training(self):
        probe = '''
import sys
from pathlib import Path
def audit(event,args):
    if event == 'open' and isinstance(args[0], (str, bytes)) and Path(args[0]).name == '.env':
        raise AssertionError('credential read')
sys.addaudithook(audit)
from agentic_arxiv.cli import main
assert main(sys.argv[1:]) == 0
assert not any(name.startswith(('torch','transformers','trl','datasets','agentic_arxiv.training')) for name in sys.modules)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filename = root / "server.json"
            filename.write_text(json.dumps(profile(root)), encoding="utf-8")
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            result = subprocess.run([sys.executable, "-I", "-c", probe, "experiment", "list", "--profile", str(filename), "--json"],
                                    cwd=directory, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["data"], {"runs": []})
            self.assertFalse((root / "managed").exists())

    def test_every_help_is_lightweight_and_does_not_read_env(self):
        probe = '''
import sys
from pathlib import Path
def audit(event,args):
    if event == 'open' and isinstance(args[0], (str, bytes)) and Path(args[0]).name == '.env':
        raise AssertionError('credential read')
sys.addaudithook(audit)
from agentic_arxiv.cli import main
try: main(sys.argv[1:])
except SystemExit as exc: assert exc.code == 0
assert not any(name.startswith(('torch','transformers','trl','datasets','agentic_arxiv.training','agentic_arxiv.experiments.service')) for name in sys.modules)
'''
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            for command in HELP_COMMANDS:
                result = subprocess.run([sys.executable, "-I", "-c", probe, *command, "--help"],
                                        cwd=directory, env=env, text=True, capture_output=True, timeout=30)
                with self.subTest(command=command):
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_json_result_scope_and_exit_codes(self):
        service = ModuleType("agentic_arxiv.experiments.service")
        class StubService:
            def __init__(self, profile, workspace=None):
                self.profile = profile
            def handle(self, operation, args, resolved=None):
                return {"execution": "running", "training": "not_started"}
        service.ExperimentService = StubService
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profile.json").write_text(json.dumps(profile(root)), encoding="utf-8")
            stream = io.StringIO()
            with patch.dict(sys.modules, {service.__name__: service}), patch("agentic_arxiv.config.initialize_operation", side_effect=AssertionError(".env")), contextlib.redirect_stdout(stream):
                code = main(["experiment", "list", "--profile", "profile.json", "--workspace", directory, "--json"])
            payload = json.loads(stream.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["completion_scope"], "management_operation_only")
            self.assertEqual(payload["data"]["training"], "not_started")
            stream = io.StringIO()
            with patch.dict(sys.modules, {service.__name__: service}), contextlib.redirect_stdout(stream):
                code = main(["experiment", "list", "--profile", "missing.json", "--workspace", directory, "--json"])
            self.assertEqual(code, 4)
            self.assertFalse(json.loads(stream.getvalue())["ok"])

    def test_cli_rejects_resource_symlinks_before_service_receives_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "server.json").write_text(json.dumps(profile(root)), encoding="utf-8")
            (root / "real").mkdir()
            (root / "alias").symlink_to(root / "real", target_is_directory=True)
            commands = (
                ["register", "M1", "--model", "alias"],
                ["resume", "r1", "--checkpoint", "alias"],
                ["evaluate", "alias", "--evaluation-config", "eval.json", "--gpus", "GPU-a"],
                ["evaluate", '{"kind":"path","path":"alias"}', "--evaluation-config", "eval.json", "--gpus", "GPU-a"],
            )
            for command in commands:
                stream = io.StringIO()
                with self.subTest(command=command), patch("agentic_arxiv.experiments.service.ExperimentService.handle", side_effect=AssertionError("unsafe service call")), contextlib.redirect_stdout(stream):
                    code = main(["experiment", *command, "--profile", str(root / "server.json"), "--json"])
                self.assertEqual(code, 2)
                self.assertIn("符号链接", json.loads(stream.getvalue())["error"]["message"])

    def test_command_arguments_registered(self):
        for command in (
            ["check", "m1.json", "--set", "defaults.seed=9"],
            ["register", "M1", "--model", "inputs/sft"], ["start", "m1.json", "--gpus", "0"],
            ["list"], ["status", "r1"], ["logs", "r1", "--rank", "0", "--tail", "10"],
            ["stop", "r1", "--force"], ["resume", "r1", "--checkpoint", "c1"],
            ["evaluate", "inputs/base", "--kind", "stage", "--evaluation-config", "eval.json", "--gpus", "0"],
        ):
            _, args = parse_args(["experiment", *command, "--profile", "server.json"])
            self.assertEqual(args._operation, f"experiment.{command[0]}")

    def test_documented_commands_only_use_registered_parameters(self):
        document = Path(__file__).resolve().parents[1] / "docs" / "experiment-cli.md"
        for line in document.read_text(encoding="utf-8").splitlines():
            if not line.startswith("agentic-arxiv "):
                continue
            argv = shlex.split(line)[1:]
            with self.subTest(command=line), contextlib.redirect_stdout(io.StringIO()):
                if "--help" in argv:
                    with self.assertRaises(SystemExit) as stopped:
                        parse_args(argv)
                    self.assertEqual(stopped.exception.code, 0)
                else:
                    parse_args(argv)


if __name__ == "__main__":
    unittest.main()
