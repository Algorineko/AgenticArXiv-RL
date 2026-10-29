"""Record arXiv Atom metadata, then freeze a small paper-disjoint replay screen."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AgenticArxiv"))
from benchmark.rescue_catalog import ASPECTS, build_catalog, parse_atom, sha256, verify_catalog


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "feeds").mkdir()
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=7)).strftime("%Y%m%d0000")
    end = now.strftime("%Y%m%d2359")
    pools, requests_record = {}, []
    for index, aspect in enumerate(ASPECTS):
        if index:
            time.sleep(3.1)
        params = {"search_query": f"cat:cs.{aspect} AND submittedDate:[{start} TO {end}]",
                  "start": 0, "max_results": 24, "sortBy": "submittedDate", "sortOrder": "descending"}
        response = requests.get("https://export.arxiv.org/api/query", params=params,
                                headers={"User-Agent": "AgenticArXiv-reproducible-catalog/1.0"}, timeout=45)
        response.raise_for_status()
        raw = response.content
        (args.output / "feeds" / f"{aspect}.atom").write_bytes(raw)
        pools[aspect] = parse_atom(raw)
        requests_record.append({"aspect": aspect, "url": response.url,
                                "records": len(pools[aspect]), "recorded_at": datetime.now(timezone.utc).isoformat()})
        print(aspect, len(pools[aspect]), flush=True)
    snapshot, tasks, splits = build_catalog(pools)
    for name, data in (("snapshot.json", snapshot), ("tasks.json", tasks), ("splits.json", splits)):
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    files = {str(f.relative_to(args.output)): sha256(f) for f in sorted(args.output.rglob("*")) if f.is_file()}
    manifest = {"kind": "real_metadata_replay_screen_v1", "recorded_at": now.isoformat(),
                "source": "arXiv Atom API", "requests": requests_record, "files": files,
                "n_tasks": len(tasks), "counts": {k: len(v) for k, v in splits["split"].items()},
                "limitations": ["Small development screen, not a publication benchmark",
                                "Real metadata; PDF download and translation remain offline stubs",
                                "Replay days is an argument contract, not a new online temporal query",
                                "Train/dev/promotion share primitive families; only final holds new compositions",
                                "Paper-disjoint categories confound domain with split; no broad OOD claim"],
                "builder_sha256": sha256(__file__),
                "task_generator_sha256": sha256(ROOT / "AgenticArxiv/benchmark/rescue_catalog.py")}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    verify_catalog(args.output)
    print(json.dumps({"manifest_sha256": sha256(args.output / "manifest.json"), "counts": manifest["counts"]}))


if __name__ == "__main__":
    main()
