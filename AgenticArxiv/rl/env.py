"""RL 环境封装

MockArxivEnv: 快照回放环境（用于快速 rollout）
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from tools.tool_registry import registry


class MockArxivEnv:
    """快照回放环境（用于快速 rollout）

    工作原理：
    1. 预先运行真实工具，记录输入→输出映射到 snapshot
    2. rollout 时从 snapshot 查询，命中则直接返回
    3. 未命中则回退到真实工具执行（并可选记录到 snapshot）

    适用场景：
    - 快速 rollout（避免真实 arxiv API 调用）
    - 确定性重放（同样输入保证同样输出）
    - 离线训练（不依赖外部网络）
    """

    def __init__(self, snapshot_path: Optional[Path] = None):
        """
        Args:
            snapshot_path: 快照文件路径（JSON 格式）
        """
        self.snapshot_path = snapshot_path
        self.snapshot: Dict[str, Dict[str, Any]] = {}

        if snapshot_path and snapshot_path.exists():
            with open(snapshot_path, "r", encoding="utf-8") as f:
                self.snapshot = json.load(f)

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """执行工具（优先从快照查询）

        Args:
            tool_name: 工具名称
            args: 工具参数

        Returns:
            工具执行结果（字符串）
        """
        # 构造 key（参数排序后拼接）
        key = self._make_key(args)
        tool_data = self.snapshot.get(tool_name, {})

        if key in tool_data:
            # 命中快照，直接返回
            return tool_data[key]["result"]

        # 未命中，回退到真实工具
        result = registry.execute_tool(tool_name, args)

        # 可选：记录到 snapshot（供下次使用）
        # self._add_to_snapshot(tool_name, key, args, result)

        return result

    def _make_key(self, args: Dict[str, Any]) -> str:
        """构造参数 key（排序后拼接）

        Args:
            args: 工具参数

        Returns:
            key 字符串（如 "cs.AI|7|5"）
        """
        sorted_keys = sorted(args.keys())
        return "|".join(str(args.get(k, "")) for k in sorted_keys)

    def _add_to_snapshot(
        self, tool_name: str, key: str, args: Dict[str, Any], result: str
    ) -> None:
        """添加到快照（供下次使用）

        Args:
            tool_name: 工具名称
            key: 参数 key
            args: 工具参数
            result: 工具结果
        """
        if tool_name not in self.snapshot:
            self.snapshot[tool_name] = {}

        self.snapshot[tool_name][key] = {"args": args, "result": result}

    def save_snapshot(self) -> None:
        """保存快照到文件"""
        if not self.snapshot_path:
            raise ValueError("snapshot_path not set")

        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.snapshot_path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot, f, ensure_ascii=False, indent=2)


def generate_snapshot_from_benchmark():
    """从 benchmark tasks 生成初始快照

    运行所有 benchmark 任务，记录工具调用到快照文件。
    供后续 rollout 快速回放使用。
    """
    from benchmark.tasks import get_all_tasks
    from benchmark.runner import run_single_benchmark

    snapshot_path = Path("data/mock_arxiv_snapshot.json")
    env = MockArxivEnv(snapshot_path)

    print("📸 生成 MockEnv 快照...")

    for task_def in get_all_tasks():
        print(f"  运行任务: {task_def['id']}")
        try:
            # 运行一次真实任务，触发工具调用
            result = run_single_benchmark(
                task_def, agent_type="regex", trial=0, session_id="snapshot_gen"
            )
            # 工具调用会通过 registry 执行，已被 env 记录
        except Exception as e:
            print(f"    ⚠️  任务执行失败: {e}")

    env.save_snapshot()
    print(f"✅ 快照已保存: {snapshot_path}")
