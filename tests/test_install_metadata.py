"""Check the metadata actually installed by setuptools, not a parallel manifest."""

from importlib import metadata
import unittest

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def requirements_for(extra=""):
    parsed = [Requirement(item) for item in metadata.requires("agentic-arxiv") or ()]
    return {
        canonicalize_name(item.name): item
        for item in parsed
        if item.marker is None or item.marker.evaluate({"extra": extra})
    }


class InstallMetadataTests(unittest.TestCase):
    def test_base_dependencies_keep_original_constraints_and_exclude_local_stack(self):
        base = requirements_for()
        self.assertEqual(set(base), {
            "arxiv", "requests", "python-dotenv", "loguru", "pydantic", "pymupdf", "pillow",
        })
        self.assertEqual(str(base["pydantic"].specifier), ">=2.0")
        self.assertEqual(str(base["pymupdf"].specifier), ">=1.24.0")

    def test_local_does_not_install_training_components(self):
        local = requirements_for("local")
        self.assertEqual(set(local) - set(requirements_for()), {
            "torch", "transformers", "accelerate", "peft",
        })
        self.assertEqual(str(local["torch"].specifier), ">=2.0.0")
        self.assertEqual(str(local["transformers"].specifier), ">=4.45.0")
        self.assertEqual(str(local["accelerate"].specifier), ">=0.25.0")
        self.assertEqual(str(local["peft"].specifier), ">=0.17.1")

    def test_train_includes_matching_local_constraints_and_original_guards(self):
        local = requirements_for("local")
        train = requirements_for("train")
        for name, requirement in local.items():
            self.assertIn(name, train)
            self.assertEqual(str(requirement.specifier), str(train[name].specifier))
        self.assertEqual(set(train) - set(local), {
            "trl", "datasets", "bitsandbytes", "packaging",
        })
        for name, constraint in {
            "trl": ">=0.28.0", "datasets": ">=2.14.0",
            "peft": ">=0.17.1", "bitsandbytes": ">=0.45.5",
        }.items():
            self.assertEqual(str(train[name].specifier), constraint)

    def test_optional_capabilities_and_dev_do_not_implicitly_install_train(self):
        base = set(requirements_for())
        self.assertEqual(set(requirements_for("translate")) - base, {"pdf2zh"})
        self.assertEqual(set(requirements_for("logging")) - base, {"tensorboard"})
        self.assertEqual(set(requirements_for("dev")) - base, {
            "pytest", "pytest-cov", "flake8", "build", "packaging",
        })
        for extra in ("", "local", "train", "translate", "logging", "dev"):
            self.assertNotIn("fire", requirements_for(extra))

    def test_single_console_entry_and_distribution_excludes_research_assets(self):
        distribution = metadata.distribution("agentic-arxiv")
        commands = [item for item in distribution.entry_points if item.group == "console_scripts"]
        self.assertEqual([(item.name, item.value) for item in commands], [
            ("agentic-arxiv", "agentic_arxiv.cli:main"),
        ])
        for file in distribution.files or ():
            self.assertFalse(str(file).startswith(("data/", "artifacts/", "models/", "docs/archive/")))


if __name__ == "__main__":
    unittest.main()
