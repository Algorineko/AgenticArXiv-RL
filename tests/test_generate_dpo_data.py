"""只检查既有切分拒绝与空诊断 IO，不生成模型响应或研究数据。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_arxiv.config import initialize_operation
from agentic_arxiv.data_pipeline.generate_dpo_data import (
    batch_paths, generate_dpo_dataset, publish_batch,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN = f"{ROOT / 'data/splits/v2_62.json'}:train"
SNAPSHOT = ROOT / "tests/fixtures/t4_snapshot_for_t5.json"


class DPOInputGuardTest(unittest.TestCase):
    def tearDown(self):
        initialize_operation(ROOT, load_env=False)

    def test_existing_split_rejection_inputs_fail_before_model_or_output(self):
        bad_splits = (None, "train", TRAIN.replace(":train", ":dev"),
                      TRAIN.replace(":train", ":iid_test"),
                      TRAIN.replace(":train", ":ood_test"))
        with tempfile.TemporaryDirectory() as temporary:
            with patch("agentic_arxiv.llm.local.TransformersLLMClient") as loader:
                for split in bad_splits:
                    with self.subTest(split=split), self.assertRaises(SystemExit):
                        generate_dpo_dataset(split=split, workspace=temporary)
                loader.assert_not_called()
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_basic_does_not_use_sft_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "expanded"):
                generate_dpo_dataset(task_set="basic", workspace=temporary)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_missing_snapshot_fails_before_model_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("agentic_arxiv.llm.local.TransformersLLMClient") as loader:
                with self.assertRaisesRegex(FileNotFoundError, "快照"):
                    generate_dpo_dataset(split=TRAIN, workspace=temporary)
                loader.assert_not_called()
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_missing_model_uses_existing_snapshot_but_never_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("agentic_arxiv.llm.local.TransformersLLMClient") as loader:
                with self.assertRaisesRegex(ValueError, "模型路径"):
                    generate_dpo_dataset(split=TRAIN, snapshot=str(SNAPSHOT), workspace=temporary)
                loader.assert_not_called()
            self.assertEqual(list(Path(temporary).iterdir()), [])


class DPOBatchMetadataTest(unittest.TestCase):
    def test_zero_pairs_publish_only_failed_diagnostic_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "metadata-check.jsonl"
            paths = publish_batch(output, [], [], {"status": "failed"}, {})
            self.assertFalse(paths["data"].exists())
            self.assertFalse(paths["manifest"].exists())
            self.assertEqual(paths["traces"].read_text(encoding="utf-8"), "")
            self.assertEqual(json.loads(paths["report"].read_text()), {"status": "failed"})
            self.assertEqual(set(Path(temporary).iterdir()), {paths["traces"], paths["report"]})
            with self.assertRaises(FileExistsError):
                publish_batch(output, [], [], {"status": "failed"}, {})

    def test_existing_companion_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "metadata-check.jsonl"
            paths = batch_paths(output)
            paths["report"].write_text("existing metadata", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                publish_batch(output, [], [], {}, {})
            self.assertEqual(paths["report"].read_text(), "existing metadata")
            self.assertEqual(list(Path(temporary).iterdir()), [paths["report"]])


if __name__ == "__main__":
    unittest.main()
