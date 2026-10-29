"""Catalog isolation, deterministic generation and fail-closed frozen inputs."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmark.rescue_catalog import ASPECTS, build_catalog, parse_atom, sha256, verify_catalog


def pools():
    return {a: [{"id": f"2401.{j * 100 + i:05d}v1", "title": f"fixture {a} {i}"}
                for i in range(1, 9)] for j, a in enumerate(ASPECTS)}


def write_catalog(path):
    snapshot, tasks, splits = build_catalog(pools())
    for name, value in (("snapshot.json", snapshot), ("tasks.json", tasks), ("splits.json", splits)):
        (path / name).write_text(json.dumps(value))
    manifest = {"kind": "real_metadata_replay_screen_v1",
                "files": {p.name: sha256(p) for p in path.iterdir()}}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_deterministic_paper_and_composition_separation():
    first = build_catalog(pools())
    assert first == build_catalog(pools())
    _, tasks, splits = first
    assert {k: len(v) for k, v in splits["split"].items()} == {
        "train": 12, "search_dev": 24, "promotion_val": 12, "final_test": 24}
    dev = {x["rescue_family"] for x in tasks if x["rescue_split"] == "search_dev"}
    final = {x["rescue_family"] for x in tasks if x["rescue_split"] == "final_test"}
    assert not dev.intersection(final)
    assert all(len(t["expected_tools"]) + 1 <= t["max_iterations"] for t in tasks)


def test_versioned_duplicate_papers_do_not_cross_splits():
    p = pools()
    p["LG"].insert(0, {"id": p["AI"][0]["id"].replace("v1", "v2"), "title": "cross-list"})
    _, _, splits = build_catalog(p)
    assert len(splits["paper_ids_by_split"]["search_dev"]) == 8


def test_insufficient_distinct_papers_fails():
    p = pools()
    p["LG"] = p["AI"]
    with pytest.raises(ValueError, match="paper-disjoint"):
        build_catalog(p)


def test_modified_frozen_file_is_rejected(tmp_path):
    write_catalog(tmp_path)
    verify_catalog(tmp_path)
    (tmp_path / "tasks.json").write_text("[]")
    with pytest.raises(ValueError, match="hash/path"):
        verify_catalog(tmp_path)


def test_even_rehashed_overlap_is_rejected(tmp_path):
    manifest = write_catalog(tmp_path)
    f = tmp_path / "splits.json"
    splits = json.loads(f.read_text())
    splits["paper_ids_by_split"]["final_test"][0] = splits["paper_ids_by_split"]["train"][0]
    f.write_text(json.dumps(splits))
    manifest["files"]["splits.json"] = sha256(f)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="paper split"):
        verify_catalog(tmp_path)


def test_atom_error_cannot_become_a_paper():
    with pytest.raises(ValueError, match="error/non-paper"):
        parse_atom('<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>error</id></entry></feed>')


def test_model_entry_point_cannot_score_final_test(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/rescue_replay_eval.py"
    r = subprocess.run([sys.executable, str(script), "--catalog", str(tmp_path),
        "--catalog-sha256", "unused", "--split", "final_test", "--policy", "model",
        "--output", str(tmp_path / "output")], text=True, capture_output=True)
    assert r.returncode == 2
    assert "reserved for scripted reference" in r.stderr
    assert not (tmp_path / "output").exists()
