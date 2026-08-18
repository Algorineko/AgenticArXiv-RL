#!/usr/bin/env python3
"""MockArxivEnv 测试：验证快照回放、参数语义与离线下载桩。

运行：
    cd AgenticArxiv
    python -m rl.build_snapshot        # 先生成快照（需联网，一次即可）
    python tests/test_mock_env.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("STORE_BACKEND", "memory")

import tools.arxiv_tool  # noqa: F401,E402
import tools.cache_status_tool  # noqa: F401,E402
import tools.pdf_download_tool  # noqa: F401,E402

from agents.agent_engine import ReActAgent  # noqa: E402
from agents.side_effects import LocalSideEffectManager  # noqa: E402
from models.schemas import Paper  # noqa: E402
from models.store import store  # noqa: E402
from rl.env import MockArxivEnv  # noqa: E402

SNAPSHOT = Path(__file__).resolve().parent.parent.parent / "data" / "mock_arxiv_snapshot.json"

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main():
    if not SNAPSHOT.exists():
        print(f"快照不存在: {SNAPSHOT}\n请先运行: python -m rl.build_snapshot")
        return 1

    print("=== 快照回放（不触网）===")
    env = MockArxivEnv(snapshot_path=SNAPSHOT, mode="replay")
    papers = env.execute_tool(
        "get_recently_submitted_cs_papers",
        {"aspect": "AI", "days": 7, "max_results": 5, "session_id": "t"},
    )
    check("检索命中快照且零真实调用",
          len(papers) == 5 and env.stats["real_calls"] == 0, env.describe())

    print("\n=== 参数语义（按 max_results 切片，而非精确 key 命中）===")
    n3 = len(env.execute_tool("get_recently_submitted_cs_papers",
                              {"aspect": "AI", "days": 7, "max_results": 3}))
    n9 = len(env.execute_tool("get_recently_submitted_cs_papers",
                              {"aspect": "AI", "days": 3, "max_results": 9}))
    check("未记录过的参数组合也能返回正确数量", n3 == 3 and n9 == 9, f"{n3} / {n9}")

    print("\n=== aspect 归一化 ===")
    a = env.execute_tool("get_recently_submitted_cs_papers", {"aspect": "AI", "max_results": 2})
    b = env.execute_tool("get_recently_submitted_cs_papers", {"aspect": "cs.AI", "max_results": 2})
    check("cs.AI 与 AI 取到同一论文池", [p["id"] for p in a] == [p["id"] for p in b])

    print("\n=== 离线下载桩（不发 HTTP）===")
    store.set_last_papers("t", [Paper(**p) for p in papers])
    before = env.stats["real_calls"]
    res = env.execute_tool("download_arxiv_pdf", {"session_id": "t", "ref": 1})
    check("下载返回 READY 且未产生真实调用",
          res["status"] == "READY" and env.stats["real_calls"] == before,
          f"paper_id={res['paper_id']}")
    check("下载后缓存状态可查",
          env.execute_tool("get_paper_cache_status",
                           {"session_id": "t", "ref": 1})["pdf_ready"] is True)

    print("\n=== env 注入 Agent 后接管工具执行 ===")
    env2 = MockArxivEnv(snapshot_path=SNAPSHOT, mode="replay")
    agent = ReActAgent(llm_client=None, side_effect_mgr=LocalSideEffectManager(), env=env2)
    obs = agent._execute_with_side_effects(
        {"name": "get_recently_submitted_cs_papers",
         "args": {"aspect": "CV", "days": 7, "max_results": 3}}
    )
    check("Agent 的工具调用走到了 env",
          "成功获取 3 篇论文" in obs and env2.stats["hit"] >= 1, env2.describe())

    print("\n" + "=" * 56)
    print(f"通过 {len(PASSED)} / {len(PASSED) + len(FAILED)}")
    if FAILED:
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
