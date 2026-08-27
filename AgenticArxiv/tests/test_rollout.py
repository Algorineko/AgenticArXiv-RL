"""Tests for rollout metadata helpers.

These tests do not load a real local model.  A lightweight fake preserves the
important ``TransformersLLMClient`` contract: ``model_name`` is the identifier
while ``model`` is the in-memory Hugging Face model object.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.rollout import _get_model_name  # noqa: E402


class TestModelMetadata(unittest.TestCase):
    def test_local_client_uses_serializable_model_name(self):
        class LocalClient:
            model_name = "outputs/sft/final"
            model = object()

        model_name = _get_model_name(LocalClient())

        self.assertEqual(model_name, "outputs/sft/final")
        self.assertEqual(
            json.loads(json.dumps({"model": model_name})),
            {"model": "outputs/sft/final"},
        )

    def test_remote_client_keeps_legacy_string_model(self):
        class RemoteClient:
            model = "gpt-4"

        self.assertEqual(_get_model_name(RemoteClient()), "gpt-4")

    def test_non_serializable_model_object_is_not_recorded(self):
        class UnknownClient:
            model = object()

        self.assertEqual(_get_model_name(UnknownClient()), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
