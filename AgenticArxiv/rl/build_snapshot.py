"""生成 MockArxivEnv 快照（唯一需要联网的一步）。

跑一次即可，之后所有 rollout / 训练都能完全离线、确定性复现：

    python -m rl.build_snapshot                       # 默认写仓库内 data/mock_arxiv_snapshot.json
    python -m rl.build_snapshot --aspects AI LG CL CV --max_results 30

设计说明
--------
不再依赖 benchmark.runner（原实现调用了并不存在的 run_single_benchmark），
而是直接驱动检索工具，为每个 aspect 记录一个"论文池"。
rollout 时 MockArxivEnv 会按 aspect 取池、按 max_results 切片，
从而对任意参数组合都能给出合理返回，而不是精确 key 命中才行。
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 快照生成阶段不需要数据库
os.environ.setdefault("STORE_BACKEND", "memory")

import tools.arxiv_tool  # noqa: F401  触发工具注册
import tools.pdf_download_tool  # noqa: F401
import tools.paper_content_tool  # noqa: F401
from models.schemas import Paper
from models.store import store
from rl.env import MockArxivEnv

DEFAULT_SNAPSHOT = str(
    Path(__file__).resolve().parents[2] / "data" / "mock_arxiv_snapshot.json"
)

# 覆盖 benchmark/rl 任务集里出现过的所有方向
DEFAULT_ASPECTS = ["*", "AI", "LG", "CL", "CV", "RO", "CR"]
DEFAULT_KEYWORD_QUERIES = [
    "all:agentic reinforcement learning",
    "all:large language model",
    "all:retrieval augmented generation",
]

# Some benchmark tasks resolve a paper by a fixed title fragment or arXiv ID.
# A rolling "last N days" snapshot eventually drops those papers and silently
# turns valid trajectories into environment failures.  Keep the small set of
# benchmark anchors at the front of the relevant pools so that max_results=5
# setup calls can always resolve them.  The order is intentional: it preserves
# the ambiguous short-fragment counterexamples documented in tasks_expanded.py.
REFERENCE_POOL_IDS = {
    "AI": ["2608.14539v1", "2608.14530v1", "2608.14528v1"],
    "CV": ["2608.14546v1", "2608.14543v1", "2608.14539v1"],
}


def _base_id(value: str) -> str:
    """Return an arXiv identifier without its version suffix."""
    import re

    return re.sub(r"v\d+$", "", value)


def _pin_reference_papers(env: MockArxivEnv) -> None:
    """Fetch and prepend the fixed papers required by offline benchmark tasks."""
    import arxiv
    from tools.arxiv_tool import _paper_info

    wanted = list(dict.fromkeys(
        paper_id
        for ids in REFERENCE_POOL_IDS.values()
        for paper_id in ids
    ))
    client = arxiv.Client()
    results = client.results(arxiv.Search(id_list=wanted, max_results=len(wanted)))
    fetched = {_base_id(paper["id"]): paper for paper in map(_paper_info, results)}

    missing = [paper_id for paper_id in wanted if _base_id(paper_id) not in fetched]
    if missing:
        raise RuntimeError(f"无法获取基准锚点论文: {missing}")

    pools = env.snapshot.get("get_recently_submitted_cs_papers", {})
    for aspect, ids in REFERENCE_POOL_IDS.items():
        entry = next(
            (item for item in pools.values()
             if str(item.get("args", {}).get("aspect")) == aspect),
            None,
        )
        if entry is None:
            raise RuntimeError(f"快照缺少 aspect={aspect} 的论文池")

        original = list(entry.get("result") or [])
        pinned_bases = {_base_id(paper_id) for paper_id in ids}
        pinned = [fetched[_base_id(paper_id)] for paper_id in ids]
        remainder = [
            paper for paper in original
            if _base_id(str(paper.get("id", ""))) not in pinned_bases
        ]
        # Preserve the requested pool size after adding anchors.
        entry["result"] = (pinned + remainder)[:len(original)]


def _validate_reference_pools(env: MockArxivEnv) -> None:
    """Fail fast when fixed benchmark references cannot resolve in top-5 pools."""
    requirements = {
        "AI": ["2608.14528", "Learning State", "State Across", "Marionette", "Handover"],
        "CV": ["2608.14543", "Image Restoration", "An Uncertainty-Aware"],
    }
    pools = env.snapshot.get("get_recently_submitted_cs_papers", {})
    failures = []
    for aspect, needles in requirements.items():
        entry = next(
            (item for item in pools.values()
             if str(item.get("args", {}).get("aspect")) == aspect),
            None,
        )
        papers = list((entry or {}).get("result") or [])[:5]
        haystacks = [
            f"{paper.get('id', '')} {paper.get('title', '')}".lower()
            for paper in papers
        ]
        for needle in needles:
            if not any(needle.lower() in value for value in haystacks):
                failures.append(f"aspect={aspect} missing={needle!r}")
    if failures:
        raise RuntimeError("快照与固定评测任务不一致: " + "; ".join(failures))



def _snapshot_paper_content(env: MockArxivEnv) -> tuple[int, int]:
    """Download each unique snapshotted paper once and pre-extract readable sections."""
    unique = {}
    for tool_name in ("get_recently_submitted_cs_papers", "search_arxiv_papers"):
        for entry in env.snapshot.get(tool_name, {}).values():
            for item in entry.get("result") or []:
                if isinstance(item, dict) and item.get("id") and not item.get("_offline_fallback"):
                    unique[str(item["id"])] = item

    session_id = "__snapshot_paper_content__"
    ok = fail = 0
    for item in unique.values():
        clean = {k: v for k, v in item.items() if not str(k).startswith("_")}
        try:
            paper = Paper(**clean)
            store.set_last_papers(session_id, [paper])
            env.execute_tool("download_arxiv_pdf", {"session_id": session_id, "ref": 1})
            # Full text is guaranteed; named sections are best-effort because
            # papers use heterogeneous headings. Missing sections stay a
            # deterministic tool error rather than fabricated content.
            env.execute_tool("get_paper_content", {"session_id": session_id, "ref": 1})
            for section in ("abstract", "method", "result", "conclusion"):
                try:
                    env.execute_tool(
                        "get_paper_content",
                        {"session_id": session_id, "ref": 1, "section": section},
                    )
                except (ValueError, RuntimeError):
                    pass
            ok += 1
        except Exception as exc:
            print(f"  [WARN] content paper={item.get('id')} → {exc}")
            fail += 1
    return ok, fail


def build(
    snapshot_path: str = DEFAULT_SNAPSHOT,
    aspects=None,
    keyword_queries=None,
    max_results: int = 50,
    days: int = 30,
    pin_references: bool = True,
    allow_partial: bool = False,
) -> None:
    aspects = list(aspects or DEFAULT_ASPECTS)
    keyword_queries = list(keyword_queries or DEFAULT_KEYWORD_QUERIES)
    path = Path(snapshot_path)
    env = MockArxivEnv(snapshot_path=path, mode="record", offline_download=False)

    print(f"生成 MockEnv 快照 → {path}")
    print(f"  aspects={aspects} keyword_queries={keyword_queries}")
    print(f"  max_results={max_results} days={days}")

    ok, fail = 0, 0
    for aspect in aspects:
        try:
            papers = env.execute_tool(
                "get_recently_submitted_cs_papers",
                {
                    "aspect": aspect,
                    "days": days,
                    "max_results": max_results,
                    "save_to_file": False,
                },
            )
            print(f"  [OK]   aspect={aspect:<3} → {len(papers)} 篇")
            ok += 1
        except Exception as e:
            print(f"  [FAIL] aspect={aspect:<3} → {e}")
            fail += 1

    for query in keyword_queries:
        try:
            papers = env.execute_tool(
                "search_arxiv_papers",
                {
                    "query": query,
                    "days": days,
                    "max_results": max_results,
                },
            )
            print(f"  [OK]   query={query!r} → {len(papers)} 篇")
            ok += 1
        except Exception as e:
            print(f"  [FAIL] query={query!r} → {e}")
            fail += 1

    if fail and not allow_partial:
        raise RuntimeError(
            f"快照构建失败：{fail}/{ok + fail} 个查询未成功；"
            "为避免写出不完整数据，未保存文件。"
        )

    if pin_references:
        print("  固定并校验 benchmark 锚点论文")
        _pin_reference_papers(env)
        _validate_reference_pools(env)

    print("  预下载并抽取快照论文文本")
    content_ok, content_fail = _snapshot_paper_content(env)
    print(f"  content: {content_ok} 成功 / {content_fail} 失败")

    env.save_snapshot()
    total = sum(len(v) for v in env.snapshot.values())
    print(f"\n完成：{ok} 个 aspect 成功、{fail} 个失败，共 {total} 条快照记录")
    print(f"快照文件: {path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="生成 MockArxivEnv 快照")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--aspects", nargs="+", default=None)
    parser.add_argument("--keyword-queries", nargs="+", default=None)
    parser.add_argument("--max_results", type=int, default=50)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--no-pin-references",
        action="store_true",
        help="不向 AI/CV 池固定 benchmark 所需论文（仅用于原始数据研究）",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许部分查询失败后仍保存快照（训练/正式评测不建议）",
    )
    args = parser.parse_args()

    build(
        snapshot_path=args.snapshot,
        aspects=args.aspects,
        keyword_queries=args.keyword_queries,
        max_results=args.max_results,
        days=args.days,
        pin_references=not args.no_pin_references,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
