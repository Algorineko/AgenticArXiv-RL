"""Benchmark CLI entrypoints must be runnable from the repository root.

The README runs every command from the repository root
(``python -m AgenticArxiv.benchmark.<module>``), while benchmark/readme.md uses
``python -m benchmark.<module>`` from inside ``AgenticArxiv/``. Each entrypoint
carries a small ``sys.path`` shim so both styles work; this test guards that
no entrypoint silently loses it (rescore_traces did, see the linked issue).
"""

import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACKAGE_DIR = os.path.join(REPO_ROOT, "AgenticArxiv")

BENCHMARK_ENTRYPOINTS = ("run_benchmark", "run_baselines", "rescore_traces")


def _run_help(module: str, cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


class BenchmarkEntrypointImportTest(unittest.TestCase):
    def test_help_from_repo_root(self):
        for name in BENCHMARK_ENTRYPOINTS:
            with self.subTest(module=name):
                proc = _run_help(f"AgenticArxiv.benchmark.{name}", REPO_ROOT)
                self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
                self.assertIn("usage:", proc.stdout)

    def test_help_from_package_dir(self):
        for name in BENCHMARK_ENTRYPOINTS:
            with self.subTest(module=name):
                proc = _run_help(f"benchmark.{name}", PACKAGE_DIR)
                self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
                self.assertIn("usage:", proc.stdout)


if __name__ == "__main__":
    unittest.main()
