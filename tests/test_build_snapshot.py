"""Snapshot defaults must match training paths regardless of the caller's cwd."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]

from agentic_arxiv.data_pipeline import build_snapshot  # noqa: E402
from agentic_arxiv.cli import main as cli_main


class SnapshotPathTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        self.expected = REPO_ROOT / "data" / "mock_arxiv_snapshot.json"

    def test_same_workspace_defaults_are_independent_of_working_directory(self):
        for cwd in (REPO_ROOT, REPO_ROOT / "src", Path(self.tmpdir.name)):
            with self.subTest(cwd=cwd), \
                 mock.patch.object(build_snapshot, "MockArxivEnv") as env_cls, \
                 redirect_stdout(io.StringIO()):
                os.chdir(cwd)
                build_snapshot.build(pin_references=False, workspace=REPO_ROOT)
                self.assertEqual(
                    env_cls.call_args.kwargs["snapshot_path"].resolve(),
                    self.expected,
                )
                cli_main(["--workspace", str(REPO_ROOT), "data", "snapshot", "--no-pin-references"])
                self.assertEqual(
                    env_cls.call_args.kwargs["snapshot_path"].resolve(),
                    self.expected,
                )

    def test_cli_explicit_paths_are_preserved(self):
        os.chdir(self.tmpdir.name)
        for path in ("custom/snapshot.json", str(Path.cwd() / "absolute.json")):
            with self.subTest(path=path), \
                 mock.patch.object(build_snapshot, "build") as build:
                cli_main(["data", "snapshot", "--snapshot", path])
                self.assertEqual(
                    Path(build.call_args.kwargs["snapshot_path"]),
                    (Path.cwd() / path).resolve(),
                )

    def test_translated_pages_are_recorded_by_default(self):
        """读译文任务引用每池第 1 篇；默认快照不录它们，这些任务就无法离线回放。"""
        with mock.patch.object(build_snapshot, "build") as build:
            cli_main(["data", "snapshot"])
        self.assertEqual(build.call_args.kwargs["translate_max_ref"], 1)


if __name__ == "__main__":
    unittest.main()
