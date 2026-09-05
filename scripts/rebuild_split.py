#!/usr/bin/env python3
"""重新生成 / 校验固化的训练-留出切分 `data/splits/v1.json`。

背景
----
`data/splits/v1.json` 是手工固化、防漂移的切分基准（train / iid_test / ood_test），
训练与评测必须引用同一份文件，阶段间数字才可比。问题是它没有生成入口：
任务集一扩（例如新增 `search_kw_*`），文件就悄悄漏掉新任务，任何代码都不会报错，
只有 `tests/test_splits.py::PinnedV1SplitTest` 能抓到。

本脚本补上这个缺口：
  * `--rebuild`   用 `make_split` 全量重建切分并写回 v1.json；
                  重建后旧任务的归属可能随随机重排移动，脚本会逐项列出变化，便于审计。
  * 默认模式      只校验现有 v1.json 是否与当前任务集一致（62 全覆盖、无重叠、
                  seed/rates 可复现、模板不跨 train/ood），只读不改文件。

用法
----
    # 校验（CI / 日常）：不符则退出码非 0
    python scripts/rebuild_split.py --check

    # 重建（任务集变更后手动执行一次）
    python scripts/rebuild_split.py --rebuild

    # 重建并输出详细 diff
    python scripts/rebuild_split.py --rebuild --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))

from benchmark.splits import DEFAULT_SPLIT_PATH, make_split, template_key  # noqa: E402
from benchmark.tasks_expanded import EXPANDED_TASKS  # noqa: E402


def load_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"切分文件不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assignment(payload: Dict[str, Any]) -> Dict[str, str]:
    """{task_id: split_name}，便于对比两次切分的归属差异。"""
    out: Dict[str, str] = {}
    for name, ids in payload["split"].items():
        for tid in ids:
            out[tid] = name
    return out


def validate(payload: Dict[str, Any], tasks: List[Dict[str, Any]], path: Path) -> List[str]:
    """返回校验中发现的问题列表；空即通过。"""
    problems: List[str] = []
    ids = {t["id"] for t in tasks}
    split = payload["split"]

    assigned = [tid for bucket in split.values() for tid in bucket]
    if len(assigned) != len(set(assigned)):
        dupes = sorted(t for t in set(assigned) if assigned.count(t) > 1)
        problems.append(f"有任务被切到多份里: {dupes}")

    missing = ids - set(assigned)
    if missing:
        problems.append(f"任务集有但切分缺失（共 {len(ids)} 条）: {sorted(missing)}")
    extra = set(assigned) - ids
    if extra:
        problems.append(f"切分里有任务集不存在: {sorted(extra)}")

    # 模板不得跨 train 与 ood_test：ood 是「完全没见过的形态」，若出现在 train
    # 就不是 ood 而是换皮记忆。train 与 iid_test 同模板是合法的——iid 的定义就是
    # 「同模板、未见过实例」，见 benchmark/tasks_expanded.py 与 splits.py 的注释。
    by_id = {t["id"]: t for t in tasks}
    keys = {
        name: {template_key(by_id[tid]) for tid in bucket if tid in by_id}
        for name, bucket in split.items()
    }
    straddle = sorted(keys["train"] & keys["ood_test"])
    if straddle:
        problems.append(f"模板同时出现在 train 与 ood_test: {straddle}")

    # seed / rates 的可复现性由测试覆盖（PinnedV1SplitTest），这里只要求文件自带这些字段
    for field in ("seed", "ood_keys", "rates"):
        if field not in payload:
            problems.append(f"文件缺少 {field!r} 字段，无法复现切分")

    return problems


def rebuild(tasks: List[Dict[str, Any]], old: Dict[str, Any], verbose: bool) -> Dict[str, Any]:
    ood_keys = [tuple(k) for k in old.get("ood_keys", [])]
    new_split = make_split(
        tasks,
        ood_keys=ood_keys,
        seed=old.get("seed", 0),
        rates=old.get("rates"),
    )
    payload = {
        "split": new_split,
        "rates": old.get("rates", {}),
        "ood_keys": [list(k) for k in ood_keys],
        "seed": old.get("seed", 0),
    }

    if verbose and old.get("split"):
        old_assign = assignment(old)
        new_assign = assignment(payload)
        moved = sorted(
            t for t in old_assign if old_assign[t] != new_assign.get(t)
        )
        added = sorted(set(new_assign) - set(old_assign))
        if moved:
            print("[rebuild] 旧任务归属发生变化:")
            for t in moved:
                print(f"    {t}: {old_assign[t]} -> {new_assign[t]}")
        if added:
            print(f"[rebuild] 新增任务 {len(added)} 条: {added}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="校验或重建固化切分 data/splits/v1.json")
    parser.add_argument("--check", action="store_true", help="只校验现有文件（只读）")
    parser.add_argument("--rebuild", action="store_true", help="全量重建并写回文件")
    parser.add_argument("--verbose", action="store_true", help="输出新旧归属 diff")
    args = parser.parse_args()

    old = load_payload(DEFAULT_SPLIT_PATH)
    problems = validate(old, EXPANDED_TASKS, DEFAULT_SPLIT_PATH)

    if args.rebuild:
        payload = rebuild(EXPANDED_TASKS, old, args.verbose)
        problems_after = validate(payload, EXPANDED_TASKS, DEFAULT_SPLIT_PATH)
        if problems_after:
            print("[rebuild] 重建后仍不合规:")
            for p in problems_after:
                print("    -", p)
            sys.exit(1)
        DEFAULT_SPLIT_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        total = sum(len(v) for v in payload["split"].values())
        print(f"[rebuild] 已重建 {DEFAULT_SPLIT_PATH}（{total} 条任务）")
        return

    if problems:
        print(f"[check] 切分与任务集不一致，共 {len(EXPANDED_TASKS)} 条任务，发现问题如下:")
        for p in problems:
            print("    -", p)
        print("    → 任务集变更后需运行 `python scripts/rebuild_split.py --rebuild`")
        sys.exit(1)
    total = sum(len(v) for v in old["split"].values())
    print(f"[check] OK：切分覆盖全部 {total} 条任务，模板不跨侧")


if __name__ == "__main__":
    main()
