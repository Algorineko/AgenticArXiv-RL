"""Snapshot defaults must match training paths regardless of the caller's cwd."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from rl import build_snapshot  # noqa: E402


class SnapshotPathTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        self.expected = REPO_ROOT / "data" / "mock_arxiv_snapshot.json"

    def test_defaults_are_independent_of_working_directory(self):
        for cwd in (REPO_ROOT, PACKAGE_ROOT, Path(self.tmpdir.name)):
            with self.subTest(cwd=cwd), \
                 mock.patch.object(build_snapshot, "MockArxivEnv") as env_cls, \
                 redirect_stdout(io.StringIO()):
                os.chdir(cwd)
                build_snapshot.build(pin_references=False)
                self.assertEqual(
                    env_cls.call_args.kwargs["snapshot_path"].resolve(),
                    self.expected,
                )
                with mock.patch.object(sys, "argv", ["build_snapshot", "--no-pin-references"]):
                    build_snapshot.main()
                self.assertEqual(
                    env_cls.call_args.kwargs["snapshot_path"].resolve(),
                    self.expected,
                )

    def test_cli_explicit_paths_are_preserved(self):
        os.chdir(self.tmpdir.name)
        for path in ("custom/snapshot.json", str(Path.cwd() / "absolute.json")):
            with self.subTest(path=path), \
                 mock.patch.object(sys, "argv", ["build_snapshot", "--snapshot", path]), \
                 mock.patch.object(build_snapshot, "build") as build:
                build_snapshot.main()
                self.assertEqual(build.call_args.kwargs["snapshot_path"], path)


if __name__ == "__main__":
    unittest.main()
