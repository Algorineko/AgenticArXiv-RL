"""Opt-in, offline backfill from a paper-ID -> completed mono-PDF manifest."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "AgenticArxiv"))
os.environ.setdefault("STORE_BACKEND", "memory")

from models.schemas import Paper, TranslateAsset
from models.store import store, use_memory_store
from rl.env import MockArxivEnv
from tools.bootstrap import register_all_tools


def backfill_translated_content(snapshot_path: Path, translations: Mapping[str, Path]) -> int:
    """Record complete normalized documents while preserving existing snapshot entries."""
    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)
    register_all_tools()
    backend = use_memory_store(reset=False)
    baseline = backend.capture_state()
    environment = MockArxivEnv(snapshot_path, mode="record")
    try:
        for paper_id, path in translations.items():
            store.set_last_papers("__translated_backfill__", [Paper(id=paper_id, title=paper_id)])
            store.upsert_translate_asset(TranslateAsset(
                paper_id=paper_id, status="READY", output_mono_path=str(path)))
            environment.execute_tool("get_translated_paper_content", {
                "session_id": "__translated_backfill__", "ref": 1})
        environment.save_snapshot()
    finally:
        backend.restore_state(baseline)
    return len(translations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True,
                        help="JSON object mapping paper IDs to completed mono PDF paths")
    args = parser.parse_args()
    manifest = json.loads(args.translations.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not all(
        isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip()
        for key, value in manifest.items()
    ):
        parser.error("translations must map nonempty paper IDs to PDF paths")
    paths = {key: (args.translations.parent / value).resolve() for key, value in manifest.items()}
    count = backfill_translated_content(args.snapshot, paths)
    print(f"Recorded {count} translated documents in {args.snapshot}")


if __name__ == "__main__":
    main()
