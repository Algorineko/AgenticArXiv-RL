"""生成 MockArxivEnv 快照（唯一需要联网的一步）。

跑一次即可，之后所有 rollout / 训练都能完全离线、确定性复现：

    python -m rl.build_snapshot                       # 默认写 ../data/mock_arxiv_snapshot.json
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
from rl.env import MockArxivEnv

# 覆盖 benchmark/rl 任务集里出现过的所有方向
DEFAULT_ASPECTS = ["*", "AI", "LG", "CL", "CV", "RO", "CR"]
DEFAULT_KEYWORD_QUERIES = [
    "all:agentic reinforcement learning",
    "all:large language model",
    "all:retrieval augmented generation",
]


def build(
    snapshot_path: str = "../data/mock_arxiv_snapshot.json",
    aspects=None,
    keyword_queries=None,
    max_results: int = 50,
    days: int = 30,
) -> None:
    aspects = list(aspects or DEFAULT_ASPECTS)
    keyword_queries = list(keyword_queries or DEFAULT_KEYWORD_QUERIES)
    path = Path(snapshot_path)
    env = MockArxivEnv(snapshot_path=path, mode="record")

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

    env.save_snapshot()
    total = sum(len(v) for v in env.snapshot.values())
    print(f"\n完成：{ok} 个 aspect 成功、{fail} 个失败，共 {total} 条快照记录")
    print(f"快照文件: {path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="生成 MockArxivEnv 快照")
    parser.add_argument("--snapshot", default="../data/mock_arxiv_snapshot.json")
    parser.add_argument("--aspects", nargs="+", default=None)
    parser.add_argument("--keyword-queries", nargs="+", default=None)
    parser.add_argument("--max_results", type=int, default=50)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    build(
        snapshot_path=args.snapshot,
        aspects=args.aspects,
        keyword_queries=args.keyword_queries,
        max_results=args.max_results,
        days=args.days,
    )


if __name__ == "__main__":
    main()
