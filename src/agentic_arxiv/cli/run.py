"""普通 ReAct 的组件组装与清理前结果交付。"""

from pathlib import Path
from tempfile import mkdtemp


def _deliver_result(result, artifact_paths, destination):
    from agentic_arxiv.provenance import export_episode_artifacts, write_result

    for step in result["history"]:
        if step.get("parsed_action") or step.get("error"):
            print(step.get("visible_observation", step.get("observation", "")))
    if result["history"] and result["history"][-1].get("thought"):
        print(result["history"][-1]["thought"])
    exported = export_episode_artifacts(artifact_paths, destination)
    result_path = destination / "result.json"
    write_result(result, result_path, artifact_exports=exported)
    print(f"结果已保存: {result_path}")
    for path in exported.values():
        print(f"文件已保存: {path}")


def execute(args):
    from agentic_arxiv.llm.api import get_env_llm_client
    from agentic_arxiv.runtime.agent_engine import ReActAgent
    from agentic_arxiv.runtime.multiturn_env import AgenticArxivMultiTurnEnv

    if args.llm_backend == "transformers":
        from agentic_arxiv.llm.local import TransformersLLMClient

        client = TransformersLLMClient(args.model)
    else:
        client = get_env_llm_client()
    environment = AgenticArxivMultiTurnEnv(
        args.snapshot if args.offline else None,
        mode="replay" if args.offline else "live",
    )
    agent = ReActAgent(client, env=environment, max_iterations=args.max_iterations)
    try:
        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        destination = Path(mkdtemp(prefix="run-", dir=output_root))
        result = agent.run(args.task, agent_model=args.model)
        print(f"交互结束: {result['termination_reason']}")
        _deliver_result(result, environment.artifact_paths, destination)
        return 0 if result["termination_reason"] == "FINISH" else 1
    finally:
        agent.close()
