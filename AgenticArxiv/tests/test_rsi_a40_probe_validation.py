"""Reject inaccurate model provenance and GPU resource claims."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from collect_rsi_a40_model_probe import source_sha, validate


class A40ProbeValidationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"model_id": "Qwen/Qwen3-8B", "model_type": "qwen3",
                         "source_sha256": source_sha("scripts/rsi_prepare_qwen_model.py"),
                         "files": {"config.json": {"bytes": 100, "sha256": "a" * 64},
                                   "model.safetensors": {"bytes": 16_000_000_000, "sha256": "b" * 64}}}
        self.probe = {"probe_source_sha256": source_sha("scripts/rsi_qwen_local_probe.py"),
                      "config_sha256": "a" * 64, "gpu_name": "NVIDIA A40", "physical_gpu": 1,
                      "memory_before_mib": 45400, "max_memory_fraction": 0.60,
                      "peak_allocated_mib": 17000, "peak_reserved_mib": 18000, "gpu_total_memory_mib": 46000}

    def test_correct_fingerprint_and_bounded_single_gpu_are_accepted(self):
        self.assertTrue(all(validate(self.manifest, self.probe).values()))

    def test_wrong_gpu_config_or_memory_limit_is_rejected(self):
        for change in ("gpu", "config", "memory", "source"):
            probe = deepcopy(self.probe)
            if change == "gpu": probe["gpu_name"] = "unexpected accelerator"
            if change == "config": probe["config_sha256"] = "c" * 64
            if change == "memory": probe["peak_reserved_mib"] = 30000
            if change == "source": probe["probe_source_sha256"] = "c" * 64
            self.assertFalse(all(validate(self.manifest, probe).values()))


if __name__ == "__main__":
    unittest.main()
