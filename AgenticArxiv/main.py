"""普通文本 ReAct 命令；工具执行模式必须明确，不依赖服务或数据库。"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent_engine import ReActAgent
from rl.multiturn_env import AgenticArxivMultiTurnEnv
from utils.llm_client import TransformersLLMClient, get_env_llm_client


def _deliver_result(result, artifact_paths, destination):
    """交付本次结果和自有文件，原轨迹路径通过导出映射追溯。"""
    for step in result["history"]:
        if step.get("parsed_action") or step.get("error"):
            print(step.get("visible_observation", step.get("observation", "")))
    if result["history"] and result["history"][-1].get("thought"):
        print(result["history"][-1]["thought"])

    exported = {}
    for name, root in artifact_paths.items():
        root = Path(root)
        for source in sorted(root.rglob("*")):
            # 只交付本 episode 的普通文件，不追随指向共享输入的链接。
            if source.is_symlink() or not source.is_file():
                continue
            target = destination / "artifacts" / name / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            exported[str(source)] = str(target)

    result_path = destination / "result.json"
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump({**result, "artifact_exports": exported}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"结果已保存: {result_path}")
    for path in exported.values():
        print(f"文件已保存: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="AgenticArxiv 文本 ReAct")
    parser.add_argument(
        "task", nargs="?",
        default="获取最近7天内机器学习领域（ML）的最新论文,最多获取15篇",
        help="当前任务文本",
    )
    parser.add_argument("--offline", action="store_true", help="只回放已有快照，不联网")
    parser.add_argument("--snapshot", help="离线快照路径")
    parser.add_argument("--llm-backend", choices=("api", "transformers"), default="api")
    parser.add_argument("--model", help="API 模型名，或本地 transformers 模型目录")
    parser.add_argument("--max-iterations", type=int, default=6, help="轮数预算，含 FINISH")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/runs"),
        help="结果根目录，每次运行创建独立子目录（默认 output/runs）",
    )
    args = parser.parse_args(argv)
    if args.max_iterations < 1:
        parser.error("--max-iterations 必须至少为 1，FINISH 也占用一轮")
    if args.snapshot and not args.offline:
        parser.error("--snapshot 必须与 --offline 一起使用")
    snapshot = None
    if args.offline:
        snapshot = Path(args.snapshot) if args.snapshot else (
            Path(__file__).resolve().parents[1] / "data" / "mock_arxiv_snapshot.json"
        )
        if not snapshot.exists():
            parser.error(f"离线模式需要已有快照: {snapshot}")
    if args.llm_backend == "transformers":
        if not args.model or not Path(args.model).exists():
            parser.error("transformers 后端需要存在的 --model 本地模型目录")
        client = TransformersLLMClient(args.model)
    else:
        client = get_env_llm_client()
    environment = AgenticArxivMultiTurnEnv(
        snapshot, mode="replay" if args.offline else "live"
    )
    agent = ReActAgent(client, env=environment, max_iterations=args.max_iterations)
    try:
        output_root = args.output_dir.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        destination = Path(mkdtemp(prefix="run-", dir=output_root))
        result = agent.run(args.task, agent_model=args.model)
        print(f"交互结束: {result['termination_reason']}")
        _deliver_result(result, environment.artifact_paths, destination)
        return 0 if result["termination_reason"] == "FINISH" else 1
    finally:
        agent.close()


if __name__ == "__main__":
    sys.exit(main())
