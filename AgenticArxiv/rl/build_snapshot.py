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

from tools.bootstrap import require_all_tools
from models.schemas import Paper
from models.store import store
from rl.env import MockArxivEnv
from tools.figure_analysis_tool import FIGURE_QUESTIONS
from tools.paper_summary_tool import MAX_WORDS_BUCKETS, SUMMARY_STYLES

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



def _unique_snapshot_papers(env: MockArxivEnv, max_ref: int = 0) -> dict:
    """Collect every distinct paper referenced by a recorded search pool.

    ``max_ref`` caps how deep into each pool to go.  Search results themselves
    are free (no PDF transfer), but pre-extracting text and figures means
    downloading one PDF per paper, and that transfer is the whole cost of a
    snapshot build.  On a slow or flaky route to arXiv, prefetching all ~400
    papers of a default build can take hours, while the task templates only ever
    address the first few entries of a pool (``ref`` ordinals).  ``0`` keeps
    every paper, which is still the default so existing snapshots are unchanged.
    """
    unique = {}
    for tool_name in ("get_recently_submitted_cs_papers", "search_arxiv_papers"):
        for entry in env.snapshot.get(tool_name, {}).values():
            for item in (entry.get("result") or [])[: (max_ref or None)]:
                if isinstance(item, dict) and item.get("id") and not item.get("_offline_fallback"):
                    unique[str(item["id"])] = item
    return unique


#: 预取的整体墙钟预算。单次请求的超时只管「两次读到字节之间的间隔」，而 arXiv
#: 会间歇性让连接彻底悬住 —— 实测有请求卡在 SSL 读上二十多分钟，逐请求超时并未
#: 生效。没有整体预算时，一次网络抽风就能把快照构建无限期拖住，而快照是整条
#: 离线流水线的入口。
DEFAULT_PREFETCH_BUDGET_S = 900


def _prefetch_pdfs(
    papers: dict,
    workers: int,
    budget_seconds: int = DEFAULT_PREFETCH_BUDGET_S,
    downloader=None,
) -> tuple[int, int, int]:
    """Transfer the snapshot's PDFs concurrently before the serial content pass.

    PDF transfer is pure network wait and takes the overwhelming majority of
    snapshot build time, so it parallelises almost perfectly.  Only the HTTP
    transfer and the file write happen in the pool: the in-memory store is not
    thread-safe, and the serial pass below updates it anyway when it finds each
    file already on disk.  Returns ``(downloaded, cached, failed)``.

    The whole stage is bounded by ``budget_seconds``: anything still pending when
    the budget runs out is counted as failed and left to the serial pass, which
    retries it with its own (shorter) path.  Snapshot construction therefore
    always reaches the content phase instead of hanging on one bad connection.
    """
    from concurrent.futures import (
        ThreadPoolExecutor,
        TimeoutError as FuturesTimeoutError,
        as_completed,
    )

    from config import settings
    from tools.pdf_download_tool import _fallback_pdf_url
    from utils.pdf_downloader import (
        download_pdf as default_downloader,
        normalize_arxiv_pdf_url,
        safe_filename,
    )

    fetch_pdf = downloader or default_downloader

    jobs = []
    for paper_id, item in papers.items():
        url = normalize_arxiv_pdf_url(
            item.get("pdf_url") or _fallback_pdf_url(paper_id)
        )
        jobs.append(
            (
                paper_id,
                url,
                os.path.join(settings.pdf_raw_path, safe_filename(paper_id) + ".pdf"),
            )
        )

    os.makedirs(settings.pdf_raw_path, exist_ok=True)
    counters = {"downloaded": 0, "cached": 0, "failed": 0}
    errors: list[str] = []

    cached_jobs = []
    pending_jobs = []
    for job in jobs:
        path = job[2]
        if os.path.exists(path) and os.path.getsize(path) > 0:
            cached_jobs.append(job)
        else:
            pending_jobs.append(job)
    counters["cached"] = len(cached_jobs)

    def fetch(job):
        paper_id, url, path = job
        fetch_pdf(url, path)

    # 不用 with：线程池的 __exit__ 走 shutdown(wait=True)，而卡在 SSL 读上的
    # 工作线程杀不掉，等于把「超时预算」架空。这里显式 shutdown(wait=False)，
    # 让主流程带着已完成的结果继续走内容阶段。注意 Python 3.10 里
    # concurrent.futures.TimeoutError 与内建 TimeoutError 还不是同一个类，
    # 必须按前者捕获。
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    futures = {pool.submit(fetch, job): job for job in pending_jobs}
    try:
        try:
            for future in as_completed(futures, timeout=budget_seconds):
                job = futures[future]
                try:
                    future.result()
                    counters["downloaded"] += 1
                except Exception as exc:  # noqa: BLE001
                    counters["failed"] += 1
                    errors.append(f"{job[0]}: {exc}")
        except FuturesTimeoutError:
            stuck = [job[0] for future, job in futures.items() if not future.done()]
            counters["failed"] += len(stuck)
            errors.append(
                f"预取超时（{budget_seconds}s）：仍有 {len(stuck)} 篇未完成，"
                "交给串行阶段重试"
            )
            for future in futures:
                future.cancel()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    for message in errors[:10]:
        print(f"  [WARN] 预取失败 {message}")
    if len(errors) > 10:
        print(f"  [WARN] ... 另有 {len(errors) - 10} 个预取失败")

    return counters["downloaded"], counters["cached"], counters["failed"]


def _snapshot_paper_content(env: MockArxivEnv, max_ref: int = 0) -> tuple[int, int]:
    """Download each unique snapshotted paper once and pre-extract readable sections."""
    unique = _unique_snapshot_papers(env, max_ref=max_ref)

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
            # T3 summaries: the offline key space is exactly
            # styles × word-budget buckets, so recording that full grid here
            # makes every valid policy action replayable.  A paper that cannot
            # be summarised (no usable sections) is a deterministic tool error
            # at rollout time too, so it is not recorded.
            for style in SUMMARY_STYLES:
                for max_words in MAX_WORDS_BUCKETS:
                    try:
                        env.execute_tool(
                            "summarize_paper",
                            {
                                "session_id": session_id,
                                "ref": 1,
                                "style": style,
                                "max_words": max_words,
                            },
                        )
                    except (ValueError, RuntimeError):
                        pass
            # T4 figure extraction.  Recorded for every paper, including the
            # empty result: "this paper has no embedded raster figures" is a
            # fact the tool reports rather than an error, and offline replay
            # has to be able to answer it too.
            figures = 0
            try:
                extraction = env.execute_tool(
                    "extract_paper_figures",
                    {"session_id": session_id, "ref": 1},
                )
                figures = int(extraction.get("count") or 0)
            except (ValueError, RuntimeError):
                pass
            # T5 图表分析。question 是枚举、figure_no 有上界，所以每篇论文的
            # 键空间正好是「实际抽出的图数 × 问法数」，可以一次录满；任何合法
            # 动作在回放时都能命中。抽不出图的论文不录——它在环境里本来就会以
            # 确定性的「图号越界」报错。
            for figure_no in range(1, figures + 1):
                for question in FIGURE_QUESTIONS:
                    try:
                        env.execute_tool(
                            "analyze_figure",
                            {
                                "session_id": session_id,
                                "ref": 1,
                                "figure_no": figure_no,
                                "question": question,
                            },
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
    content_workers: int = 8,
    content_max_ref: int = 0,
    prefetch_budget: int = DEFAULT_PREFETCH_BUDGET_S,
    skip_prefetch: bool = False,
) -> None:
    # 快照要覆盖全部工具，否则回放时缺哪一类工具就少哪一类的记录，
    # 而缺的那部分会以「replay 模式下快照缺失」的形式在训练时才炸出来。
    require_all_tools("快照构建")

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

    unique_papers = _unique_snapshot_papers(env, max_ref=content_max_ref)
    scope = f"每池前 {content_max_ref} 篇" if content_max_ref else "全部论文"
    if skip_prefetch:
        # PDF 已经在磁盘上时（例如上一次构建下完但中途失败），并行预取这一步
        # 只是重复劳动，而且 arXiv 的间歇性挂起恰好发生在这一阶段。直接进内容
        # 阶段：它自己会为缺的论文走带重试的串行下载。
        print(f"  跳过预取（--skip-prefetch），直接进入内容阶段（{len(unique_papers)} 篇）")
    else:
        print(f"  预取 {len(unique_papers)} 篇论文的 PDF（{scope}，并发 {content_workers}）")
        fetched, cached, failed = _prefetch_pdfs(
            unique_papers, content_workers, budget_seconds=prefetch_budget
        )
        print(f"  PDF: 新下载 {fetched} / 已缓存 {cached} / 失败 {failed}")

    print("  抽取快照论文文本与图表")
    content_ok, content_fail = _snapshot_paper_content(env, max_ref=content_max_ref)
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
    parser.add_argument(
        "--content-workers",
        type=int,
        default=8,
        help="并发预取 PDF 的线程数；PDF 传输是快照构建的耗时大头",
    )
    parser.add_argument(
        "--skip-prefetch",
        action="store_true",
        help="跳过并行预取，直接进入内容抽取阶段。PDF 已在本地（上次构建下完但"
             "中途失败）时用，可绕开 arXiv 在预取阶段的间歇性挂起",
    )
    parser.add_argument(
        "--prefetch-budget",
        type=int,
        default=DEFAULT_PREFETCH_BUDGET_S,
        help="预取阶段的整体墙钟预算（秒）。arXiv 会间歇性让连接彻底悬住，"
             "逐请求超时管不住；超预算的论文记为失败并交给串行阶段重试",
    )
    parser.add_argument(
        "--content-max-ref",
        type=int,
        default=0,
        help="只为每个论文池的前 N 篇预取正文与图表（0=全部）。搜索池本身不受影响，"
             "所以检索类任务看到的论文数量不变。网络到 arXiv 慢或抖动时，"
             "把范围压到任务真正会用到的 ref 上可以省掉绝大部分下载",
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
        content_workers=args.content_workers,
        content_max_ref=args.content_max_ref,
        prefetch_budget=args.prefetch_budget,
        skip_prefetch=args.skip_prefetch,
    )


if __name__ == "__main__":
    main()
