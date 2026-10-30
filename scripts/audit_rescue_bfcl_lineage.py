"""Compare a frozen native BFCL roster with the project's earlier v3 source snapshot."""
import argparse
import hashlib
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pyarrow.parquet as pq


def norm(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.findall(r"\w+", text))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old-parquet", type=Path, required=True)
    p.add_argument("--v4-data", type=Path, required=True, help="Pinned BFCL data directory")
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    old_table = pq.read_table(a.old_parquet)
    old_questions = [
        " ".join(m["content"] for m in row if m["role"] == "user")
        for row in old_table["messages"].to_pylist()
    ]
    qmap = {
        row["id"]: row
        for row in map(json.loads, (a.v4_data / "BFCL_v4_multi_turn_base.json").read_text().splitlines())
    }
    protocol = json.loads(a.protocol.read_text())
    cases = []
    for case_id in protocol["case_ids"]:
        turns = qmap[case_id]["question"]
        user_texts = [m["content"] for turn in turns for m in turn if m["role"] == "user"]
        first, all_text = norm(user_texts[0]), norm(" ".join(user_texts))
        first_matches = [i for i, old in enumerate(old_questions) if norm(old) == first]
        full_matches = [i for i, old in enumerate(old_questions) if norm(old) == all_text]
        scores = [SequenceMatcher(None, all_text, norm(old)).ratio() for old in old_questions]
        best_row = max(range(len(scores)), key=scores.__getitem__)
        cases.append({
            "case_id": case_id,
            "turns": len(turns),
            "exact_first_question_v3_rows": first_matches,
            "exact_full_dialogue_v3_rows": full_matches,
            "best_full_dialogue_v3_row": best_row,
            "best_full_dialogue_similarity": round(scores[best_row], 6),
        })
    result = {
        "kind": "posthoc_bfcl_v3_v4_task_lineage_audit_v1",
        "protocol_sha256": sha(a.protocol),
        "old_source": {
            "sha256": sha(a.old_parquet),
            "rows": len(old_questions),
            "note": "Earlier BFCL v3 static-source snapshot; text overlap does not establish that a model was previously run on each matched row.",
        },
        "normalization": "Unicode NFKC, lowercase, then retain word-character sequences joined by single spaces",
        "counts": {
            "cases": len(cases),
            "cases_with_exact_first_question_match": sum(bool(c["exact_first_question_v3_rows"]) for c in cases),
            "cases_with_exact_full_dialogue_match": sum(bool(c["exact_full_dialogue_v3_rows"]) for c in cases),
        },
        "cases": cases,
        "interpretation": "Roster overlaps the earlier BFCL source-task pool. Prior model-inference overlap cannot be determined from this source-text comparison; historical traces were unavailable for rescoring.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
