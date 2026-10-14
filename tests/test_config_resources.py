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


DEFAULT_CONFIG_SHA256 = "3479911b31a7740321283a30c70f929f1fd8835be9b9974573d2c377a894b34c"


class ConfigResourceTests(unittest.TestCase):
    def test_default_resource_retains_original_bytes(self):
        content = config.read_default_mainline_config()
        self.assertEqual(hashlib.sha256(content).hexdigest(), DEFAULT_CONFIG_SHA256)
        payload = json.loads(content)
        self.assertEqual(payload["schema_version"], 1)
        for name in (
            "loss_type", "epsilon_high", "mask_truncated_completions",
            "importance_sampling_level", "scale_rewards",
        ):
            self.assertIsNone(payload["grpo"]["defaults"][name])

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
