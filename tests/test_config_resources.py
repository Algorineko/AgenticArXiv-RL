"""Installation resources and operation configuration use no research fixtures."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agentic_arxiv import config


DEFAULT_CONFIG_SHA256 = "f3c958144123a522254657cf036bf004291af77dccf4a72d0bfecfa09183d883"
ORIGINAL_SUBTREE_SHA256 = {
    "source_baseline": "ef52e47d179881ec68d5572b1612e9995361f48fd4b11537acffb7e2710933d8",
    "sft": "5ec1d374da6e45887561170b89afe264b537b9526c1852ce0b7d281b0c91d9ee",
    "grpo": "00ed5989f0a88bff44bd061ef0f85066c430130f9ed192a458eb9b7f13dacefd",
}


class ConfigResourceTests(unittest.TestCase):
    def test_current_resource_fingerprint_and_original_stage_subtrees(self):
        content = config.read_default_mainline_config()
        self.assertEqual(hashlib.sha256(content).hexdigest(), DEFAULT_CONFIG_SHA256)
        payload = json.loads(content)
        self.assertEqual(payload["schema_version"], 1)
        for stage, expected in ORIGINAL_SUBTREE_SHA256.items():
            canonical = json.dumps(payload[stage], sort_keys=True, separators=(",", ":")).encode()
            self.assertEqual(hashlib.sha256(canonical).hexdigest(), expected, stage)
        for name in (
            "loss_type", "epsilon_high", "mask_truncated_completions",
            "importance_sampling_level", "scale_rewards",
        ):
            self.assertIsNone(payload["grpo"]["defaults"][name])

    def test_legacy_config_uses_packaged_dpo_defaults_without_writing(self):
        from agentic_arxiv.training.mainline_config import stage_config, stage_config_source

        payload = json.loads(config.read_default_mainline_config())
        expected = payload.pop("dpo")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "legacy.json")
            content = json.dumps(payload).encode()
            path.write_bytes(content)
            self.assertEqual(stage_config("dpo", path), expected)
            self.assertEqual(stage_config("sft", path), payload["sft"])
            self.assertEqual(stage_config("grpo", path), payload["grpo"])
            self.assertEqual(path.read_bytes(), content)
            source = stage_config_source("dpo", path)
            self.assertTrue(source["used_packaged_dpo_defaults"])
            self.assertNotEqual(source["defaults_source"], str(path))

    def test_explicit_dpo_sections_reject_unknown_missing_and_wrong_types(self):
        from agentic_arxiv.training.mainline_config import stage_config

        original = json.loads(config.read_default_mainline_config())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "config.json")
            for group in original["dpo"]:
                for failure in ("missing", "unknown", "type"):
                    payload = json.loads(json.dumps(original))
                    fields = payload["dpo"][group]
                    key = next(iter(fields))
                    if failure == "missing":
                        fields.pop(key)
                    elif failure == "unknown":
                        fields["unknown_dpo_option"] = True
                    else:
                        fields[key] = []
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.subTest(group=group, failure=failure), self.assertRaisesRegex(ValueError, "dpo"):
                        stage_config("dpo", path)

    def test_dpo_design_defaults_are_separate_from_original_stages(self):
        from agentic_arxiv.training.mainline_config import stage_config

        stage = stage_config("dpo")
        defaults = stage["defaults"]
        self.assertEqual((defaults["epochs"], defaults["lr"], defaults["beta"]), (3, 5e-6, 0.1))
        self.assertEqual((defaults["batch_size"], defaults["grad_accum"], defaults["max_length"]), (1, 8, 4096))
        self.assertTrue(defaults["qlora"] and defaults["verify"])
        self.assertEqual(defaults["min_reward"], -0.3)
        generated = stage["generation_defaults"]
        self.assertEqual((generated["num_trials"], generated["temperature"], generated["seed"]), (5, 0.8, 42))
        self.assertEqual((generated["min_reward_gap"], generated["min_task_yield"]), (0.05, 0.25))
        self.assertEqual((generated["max_turns"], generated["max_new_tokens"]), (6, 256))

    def test_root_and_config_import_do_not_load_dotenv_local_stack_or_file_logs(self):
        code = """
import os, pathlib, sys
import agentic_arxiv
assert not {'dotenv', 'loguru', 'torch', 'transformers', 'trl', 'datasets', 'peft'} & set(sys.modules)
from agentic_arxiv import config
assert 'dotenv' not in sys.modules
assert os.environ.get('AGENTIC_ARXIV_IMPORT_SENTINEL') is None
assert config.settings.antigravity_api_key == 'no-token-here'
assert config.settings.pdf2zh_threads == 4
assert not pathlib.Path('output').exists()
"""
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, ".env").write_text(
                "AGENTIC_ARXIV_IMPORT_SENTINEL=must-not-load\n", encoding="utf-8"
            )
            environment = dict(os.environ)
            environment.pop("AGENTIC_ARXIV_IMPORT_SENTINEL", None)
            environment.pop("PYTHONPATH", None)
            environment["LLM_API_KEY"] = "must-not-read-on-import"
            environment["PDF2ZH_THREADS"] = "must-not-parse-on-import"
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=directory, env=environment,
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_explicit_workspace_environment_priority_and_shared_settings_identity(self):
        old_workspace = config.get_workspace()
        shared_settings = config.settings
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
                workspace = Path(directory).resolve()
                (workspace / ".env").write_text(
                    "MODEL=workspace-model\nLLM_BASE_URL=https://workspace.invalid\n"
                    "PDF_RAW_PATH=cache/raw\n", encoding="utf-8"
                )
                os.environ["LLM_BASE_URL"] = "https://environment.invalid"
                configured = config.initialize_operation(workspace)
                self.assertIs(shared_settings, configured)
                self.assertEqual(configured.models.agent_model, "workspace-model")
                self.assertEqual(configured.antigravity_base_url, "https://environment.invalid")
                self.assertEqual(configured.pdf_raw_path, str(workspace / "cache/raw"))
                self.assertEqual(configured.figures_path, str(workspace / "output/pdf_figures"))
                self.assertFalse((workspace / "output").exists())
        finally:
            config.initialize_operation(old_workspace, load_env=False)

    def test_path_resolution_is_independent_of_cwd_and_preserves_absolute_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            relative = Path("data/mock_arxiv_snapshot.json")
            self.assertEqual(config.workspace_path(relative, workspace=workspace), workspace / relative)
            absolute = workspace / "existing" / ".." / "input.json"
            self.assertEqual(config.workspace_path(absolute, workspace="/other"), absolute)
            with self.assertRaisesRegex(ValueError, str(workspace / "outputs/missing")):
                config.resolve_model_source("outputs/missing", local_only=True, workspace=workspace)


if __name__ == "__main__":
    unittest.main()
