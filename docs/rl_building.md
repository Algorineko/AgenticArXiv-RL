# AgenticArXiv-RL 改造计划

> **目标**：将 AgenticArXiv 重构为可训练的 Agentic RL 环境，支持 SFT/DPO/GRPO 渐进式训练  
> **强制约束**：环境管理必须采用 `.venv`，不可用 conda

---

## 📌 背景与目标

### 现状
- **AgenticArXiv** 是一个完善的 ReAct Agent + arXiv 论文管理系统，支持 3 种 Agent 架构（ReAct/MCP/Skill）
- 已有完整的工具层（arxiv搜索/PDF下载/翻译/缓存查询）、数据层（MySQL 7表）、前后端分离架构（FastAPI + Vue3）
- 已有 `benchmark/` 测试套件（`TaskMetrics`、`BENCHMARK_TASKS`），用于对比三种 Agent 模式的性能和准确性

### 目标
1. **改造为 RL 训练环境**：把 Agent 执行循环改造成可记录 trajectory、可打分（verifiable reward）、可回放的 RL 环境
2. **支持渐进式训练路径**：SFT（监督微调）→ DPO（直接偏好优化）→ GRPO（在线 RL）
3. **复用 verifiable reward 组件**：基于 `benchmark/metrics.py` 的 `TaskMetrics`（`tool_call_accurate`、`task_completed`、`parse_failures` 等）构建奖励函数
4. **精简工程**：去掉 FastAPI/Vue3/MySQL 等重依赖，专注离线 RL 训练（rollout + reward + SFT/DPO/GRPO）
5. **轻量级定位**：作为学习 demo，不涉及重量级 PPO（需 value model）

### 训练路径规划
```
阶段0: 环境准备（.venv + trajectory 记录）
  ↓
阶段1: SFT（Supervised Fine-Tuning）
  - 用 benchmark/tasks 生成 SFT 数据集（expert demonstrations）
  - 训练模型学会基本的工具调用格式
  ↓
阶段2: DPO（Direct Preference Optimization）
  - 从 SFT 模型 rollout，收集 chosen/rejected 对
  - 训练模型偏好正确的工具选择和参数填写
  ↓
阶段3: GRPO（Group Relative Policy Optimization）
  - 用 verifiable reward 在线训练
  - 适合小模型（无需 critic/value model）
  - 作为学习 demo 的最终目标
```

---

## 🔍 保留 vs 精简 分析矩阵

### ✅ 保留并复用（RL 核心组件）

| 模块 | 文件/目录 | 为什么保留 | RL 中的角色 |
|------|---------|----------|-----------|
| **Agent 核心** | `agents/base_agent.py` | 通用 ReAct 执行循环（L92-162） | **rollout 循环**（需解耦副作用） |
| | `agents/agent_engine.py` | ReActAgent 实现 + 正则解析 | **RL 策略（policy）** |
| | `agents/prompt_templates.py` | ReAct prompt 模板 | prompt 构建（RL 安全，无副作用） |
| | `agents/context_manager.py` | `ReactStep` dataclass | trajectory 数据结构（可复用） |
| **工具层** | `tools/tool_registry.py` | 工具注册表 | **动作空间枚举**（`list_tools()`） |
| | `tools/*.py` (4个工具) | arxiv/download/translate/cache | **状态转移函数**（action → observation） |
| **Reward 来源** | `benchmark/metrics.py` | ⭐ `TaskMetrics`、`_check_tool_sequence`、`_count_parse_failures` 等 | **verifiable reward 的现成实现** |
| | `benchmark/tasks.py` | `BENCHMARK_TASKS`（含 `expected_tools` ground truth） | **RL 训练/eval 任务集种子** |
| | `benchmark/runner.py` | 执行 agent.run() 的循环逻辑 | rollout 循环参考（需改为 RL 风格） |
| **LLM 客户端** | `utils/llm_client.py` | `chat_completions()` 调用 | **策略 LLM 调用**（后续可换 vLLM） |
| **数据模型** | `models/schemas.py` | Pydantic 模型（Paper、PdfAsset 等） | state 的数据结构（去 ORM） |
| **工具库** | `utils/logger.py` | 日志 | 训练日志 |
| **数据快照** | `data/*.csv` | arXiv 论文数据快照 | **MockEnv 的快照数据源**（快速 rollout） |
| | `output/recent_cs_papers.txt` | 工具输出样本 | eval/badcase 样本 |

### 🔄 重构（保留但需改造）

| 模块 | 现状问题 | 改造方案 |
|------|---------|---------|
| **base_agent.py** | 副作用硬编码（DB/SSE/translate/store 写）L192-277 | **解耦为可注入接口**：`SideEffectManager` 抽象类，默认 `NoOpSideEffectManager`；MySQL 版实现为 `MySQLSideEffectManager`（可选依赖） |
| **存储层** | MySQL 7表 + `init_db()` 硬绑定 | **改为 JSONL 文件化**：`traces/*.jsonl` 存 trajectory；`eval/eval_cases.jsonl` 存任务集；去掉 pymysql 依赖 |
| **reward 计算** | 分散在 `benchmark/metrics.py` 中 | **提炼为 `rl/reward.py`**：`RewardCalculator` 类，复用 `_check_tool_sequence` 等函数，返回 `{step_reward, final_reward, metrics}` |
| **解析错误处理** | `agent_engine.py:L121` 解析失败返回 `(thought, None)` 被当成 FINISH | **计入惩罚**：在 reward 中加 `parse_failure_penalty = -0.2`，并记录到 trajectory 的 `parse_failed: bool` 字段 |

### 🗄️ 归档（archive/ 不删除，但不在 RL 路径中）

| 模块 | 为什么归档 | 位置 |
|------|----------|------|
| **mcp_protocol/** | RL 只用 ReAct(regex)，不需要跨进程 MCP | `archive/mcp_protocol/` |
| **skill_cli/** | RL 只用 ReAct(regex)，不需要 CLI 方案 | `archive/skill_cli/` |
| **api/** | FastAPI 应用，RL 不需要 HTTP API | `archive/api/` |
| **services/** | `event_bus`、`runtime`、`log_service`（SSE/异步翻译） | `archive/services/` |
| **AgenticArxivWeb/** | Vue3 前端，RL 不需要 | `archive/AgenticArxivWeb/` |
| **bin/** | 启动脚本（依赖 FastAPI + 前端） | `archive/bin/` |
| **Makefile 前端部分** | `npm install` 等 | 去掉或注释 |

### ❌ 删除/忽略（不保留）

| 模块 | 为什么删除 |
|------|----------|
| **models/db.py** | MySQL engine + `init_db()`，JSONL 后不需要 |
| **models/orm.py** | SQLAlchemy ORM 7表，JSONL 后不需要 |
| **models/store.py** | 依赖 MySQL 的业务逻辑，需用 `MockStore`/`JSONLStore` 替代 |
| **requirements.txt 中** | `fastapi`、`uvicorn`、`pymysql`、`sqlalchemy`、`pdf2zh`（训练不需要真翻译） |

---

## 🏗️ 目标目录结构（RL 专用）

```
AgenticArXiv-RL/                     # 新仓库名（建议大写 RL）
├─ AgenticArxiv/                     # Python 包（保持原名）
│  ├─ agents/                        # 保留（解耦副作用）
│  │  ├─ base_agent.py              # 通用 ReAct 循环（重构副作用接口）
│  │  ├─ agent_engine.py            # ReActAgent（RL 策略）
│  │  ├─ prompt_templates.py        # prompt 模板
│  │  └─ context_manager.py         # ReactStep
│  ├─ tools/                         # 保留（动作空间）
│  │  ├─ tool_registry.py
│  │  ├─ arxiv_tool.py
│  │  ├─ pdf_download_tool.py
│  │  ├─ pdf_translate_tool.py
│  │  └─ cache_status_tool.py
│  ├─ benchmark/                     # 保留（reward 来源）
│  │  ├─ metrics.py                 # ⭐ verifiable reward 组件
│  │  ├─ tasks.py                   # ⭐ 任务集种子
│  │  └─ runner.py                  # 参考
│  ├─ rl/                            # ⭐ 新增（RL 核心）
│  │  ├─ __init__.py
│  │  ├─ env.py                     # RLEnv 类（MDP 封装）+ MockArxivEnv
│  │  ├─ policy.py                  # 策略包装（LLM → action）
│  │  ├─ reward.py                  # RewardCalculator（复用 benchmark metrics）
│  │  ├─ trajectory.py              # Trajectory 数据类 + JSONL 读写
│  │  ├─ rollout.py                 # rollout 循环（agent.run 的 RL 版）
│  │  ├─ tasks.py                   # 从 benchmark/tasks 扩展的 RL 任务集
│  │  ├─ train_sft.py               # ⭐ SFT 训练脚本
│  │  ├─ train_dpo.py               # ⭐ DPO 训练脚本
│  │  ├─ train_grpo.py              # ⭐ GRPO 训练脚本
│  │  └─ train_ppo.py               # ⭐ PPO 训练脚本（可选）
│  ├─ utils/                         # 保留
│  │  ├─ llm_client.py              # LLM 调用
│  │  └─ logger.py                  # 日志
│  ├─ models/                        # 精简
│  │  └─ schemas.py                 # Pydantic 模型（去 ORM）
│  ├─ config.py                      # 环境变量配置
│  └─ requirements.txt               # 重写（见下）
├─ traces/                           # ⭐ 新增（trajectory 存储）
│  ├─ train/                         # 训练轨迹
│  │  └─ rollout_YYYYMMDD_HHMMSS.jsonl
│  └─ eval/                          # 评估轨迹
│     └─ eval_YYYYMMDD_HHMMSS.jsonl
├─ data/                             # 保留（快照数据）
│  ├─ sft/                           # ⭐ SFT 数据集
│  │  └─ sft_train.jsonl
│  ├─ dpo/                           # ⭐ DPO 数据集
│  │  └─ dpo_train.jsonl
│  ├─ raw_data.csv                  # arXiv 论文样本
│  └─ mock_arxiv_snapshot.json       # ⭐ 新增（MockEnv 快照）
├─ eval/                             # ⭐ 新增（评估集）
│  ├─ eval_cases.jsonl              # 测试任务（从 benchmark/tasks 扩展）
│  └─ badcase_replay.py             # badcase 回放脚本
├─ archive/                          # 归档
│  ├─ mcp_protocol/
│  ├─ skill_cli/
│  ├─ api/
│  ├─ services/
│  ├─ AgenticArxivWeb/
│  └─ bin/
├─ docs/
│  ├─ rl_building.md                # 本文档
│  └─ metric_stats.md               # 保留（原 benchmark 统计）
├─ .venv/                            # ⭐ Python 虚拟环境（强制）
├─ .env.example                      # 环境变量模板
├─ README.md                         # 更新（RL 版）
└─ Makefile                          # 精简（去前端）
```

---

## 🎯 RL 环境 MDP 设计

### State（状态）
- **任务描述**：用户请求（如"检索最近7天AI论文"）
- **对话历史**：`List[{thought, action, observation}]`（已执行的步骤）
- **工具结果**：上一步工具返回的数据（论文列表、PDF状态等）
- **会话状态**：`session_id`、`last_active_paper_id`（来自 store，或 mock 中固定）

**数据结构**（JSON）：
```json
{
  "task": "检索最近7天内人工智能(cs.AI)方向的论文，最多5篇",
  "history": [
    {
      "step": 1,
      "thought": "需要调用 arxiv 搜索工具",
      "action": "{\"name\":\"get_recently_submitted_cs_papers\",\"arguments\":{\"aspect\":\"cs.AI\",\"days\":7,\"max_results\":5}}",
      "observation": "成功获取5篇论文..."
    }
  ],
  "session_context": {
    "session_id": "rl_train_001",
    "last_papers_count": 5
  }
}
```

### Action（动作）
- **动作空间**：4 个工具（来自 `tool_registry.list_tools()`）
  1. `get_recently_submitted_cs_papers(aspect, days, max_results)`
  2. `download_arxiv_pdf(ref, session_id)`
  3. `translate_arxiv_pdf(ref, session_id)`
  4. `get_paper_cache_status(ref, session_id)`
- **FINISH**：特殊动作，表示任务完成
- **动作表示**：JSON 字符串（与原 ReAct 一致）

**示例**：
```json
{
  "name": "get_recently_submitted_cs_papers",
  "arguments": {
    "aspect": "cs.AI",
    "days": 7,
    "max_results": 5
  }
}
```

### Reward（奖励）—— ⭐ 复用 benchmark/metrics.py 的 verifiable 项

**基于 `TaskMetrics` 的 reward 设计**：

| 维度 | 来源 | 奖励规则 | 值 |
|------|------|---------|-----|
| **任务成功** | `task_completed` (L95) | 正常 FINISH | +1.0 |
| **工具调用准确** | `tool_call_accurate` (L100, `_check_tool_sequence`) | 调用顺序匹配 `expected_tools` | +0.5 |
| **解析错误** | `parse_failures` (L102, `_count_parse_failures`) | 每次解析失败 | -0.2 |
| **工具执行失败** | `tool_exec_failures` (L103, `_count_tool_failures`) | 每次工具报错 | -0.3 |
| **超时** | `termination_type` == "FORCE_STOP" | 达到 max_iterations 未完成 | -0.5 |
| **错误终止** | `termination_type` == "ERROR" | 执行异常 | -1.0 |
| **不必要调用** | 额外逻辑 | 调用了不在 `expected_tools` 中的工具 | -0.1 |

**总奖励公式**（伪代码，见 `rl/reward.py`）：
```python
def compute_reward(trajectory, task_def, metrics: TaskMetrics) -> float:
    reward = 0.0
    if metrics.task_completed:
        reward += 1.0
    if metrics.tool_call_accurate:
        reward += 0.5
    reward -= 0.2 * metrics.parse_failures
    reward -= 0.3 * metrics.tool_exec_failures
    if metrics.termination_type == "FORCE_STOP":
        reward -= 0.5
    elif metrics.termination_type == "ERROR":
        reward -= 1.0
    # 不必要调用惩罚
    extra_calls = len(metrics.tool_call_sequence) - len(metrics.expected_tools)
    if extra_calls > 0:
        reward -= 0.1 * extra_calls
    return reward
```

**关键**：这些奖励都是 **可验证的**（verifiable），无需人类标注 → 对应 RLVR（Reinforcement Learning with Verifiable Reward）框架。

### Transition（状态转移）
- **工具执行**：`registry.execute_tool(action_name, args) → observation`
- **状态更新**：`history.append({thought, action, observation})`
- **终止条件**：
  1. `action_dict is None` → FINISH（正常终止）
  2. `iteration >= max_iterations` → FORCE_STOP（超时）
  3. 工具执行抛异常 → ERROR（错误终止）

---

## 🛠️ 渐进式训练路径（4阶段）

### 阶段0：环境准备（Step 0-3）

**目标**：搭建 RL 基础设施（.venv、trajectory 记录、MockEnv）

#### Step 0：.venv 环境搭建（强制）

**创建虚拟环境**：
```bash
cd /Users/dev/projects/AgenticArXiv
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows
```

**重写 `AgenticArxiv/requirements.txt`**（去重依赖，分层）：

**核心依赖**（必装）：
```txt
# LLM & RL
torch>=2.0.0
transformers>=4.35.0
trl>=0.8.0                # TRL (SFT/DPO/GRPO/PPO)
datasets>=2.14.0
accelerate>=0.25.0

# Agent 核心
arxiv
requests
python-dotenv
loguru
pydantic>=2.0

# 工具
fire                      # CLI（用于 eval/rollout 脚本）
```

**可选依赖**（训练时不需要）：
```txt
# 仅 eval/demo 时需要（放到 requirements-extra.txt）
pdf2zh                    # PDF 翻译（训练时用 mock）
```

**安装**：
```bash
pip install -r AgenticArxiv/requirements.txt
```

#### Step 1：解耦副作用（base_agent.py 重构）

**目标**：把 `base_agent.py` 中的副作用（DB/SSE/translate/store 写）抽到可注入接口，离线训练用 no-op。

**新增接口**（`agents/side_effects.py`）：
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class SideEffectManager(ABC):
    """副作用管理器抽象接口（DB/SSE/store/translate）"""
    
    @abstractmethod
    def create_chat_log(self, session_id: str, msg_id: str, role: str, content: str, model: str) -> None:
        pass
    
    @abstractmethod
    def save_agent_step(self, session_id: str, msg_id: str, step_index: int, step_data: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def publish_sse(self, session_id: str, event_data: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def enqueue_translate(self, paper_id: int, session_id: str, **kwargs) -> None:
        pass
    
    @abstractmethod
    def set_last_papers(self, session_id: str, papers: list) -> None:
        pass
    
    @abstractmethod
    def set_last_active_paper_id(self, session_id: str, paper_id: int) -> None:
        pass


class NoOpSideEffectManager(SideEffectManager):
    """无操作实现（用于离线 RL 训练）"""
    def create_chat_log(self, *args, **kwargs): pass
    def save_agent_step(self, *args, **kwargs): pass
    def publish_sse(self, *args, **kwargs): pass
    def enqueue_translate(self, *args, **kwargs): pass
    def set_last_papers(self, *args, **kwargs): pass
    def set_last_active_paper_id(self, *args, **kwargs): pass
```

#### Step 2：JSONL trajectory 记录

**新增 `rl/trajectory.py`**：
```python
# AgenticArxiv/rl/trajectory.py
"""Trajectory 数据类 + JSONL 读写"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path

@dataclass
class TrajectoryStep:
    step: int
    thought: str
    action: str  # JSON 字符串 或 "FINISH"
    observation: str
    llm_latency_ms: int
    tool_latency_ms: int
    parse_failed: bool = False

@dataclass
class Trajectory:
    task_id: str
    task: str
    session_id: str
    steps: List[TrajectoryStep]
    final_reward: float
    metrics: Dict[str, Any]  # TaskMetrics 的字典形式
    timestamp: str

def save_trajectory(traj: Trajectory, filepath: Path):
    """保存单条 trajectory 到 JSONL（追加模式）"""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(traj), ensure_ascii=False) + "\n")

def load_trajectories(filepath: Path) -> List[Trajectory]:
    """从 JSONL 加载 trajectories"""
    with open(filepath, "r", encoding="utf-8") as f:
        return [Trajectory(**json.loads(line)) for line in f]
```

#### Step 3：MockArxivEnv 快照

**新增 `rl/env.py`**：
```python
# AgenticArxiv/rl/env.py
"""RL 环境封装"""

import json
from pathlib import Path
from typing import Dict, Any
from tools.tool_registry import registry

class MockArxivEnv:
    """快照回放环境（用于快速 rollout）"""
    def __init__(self, snapshot_path: Path):
        with open(snapshot_path) as f:
            self.snapshot = json.load(f)
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        # 构造 key（如 "cs.AI|7|5"）
        key = "|".join(str(args.get(k, "")) for k in sorted(args.keys()))
        tool_data = self.snapshot.get(tool_name, {})
        if key in tool_data:
            return tool_data[key]["result"]
        # fallback：调用真实工具（并记录到 snapshot）
        return registry.execute_tool(tool_name, args)
```

---

### 阶段1：SFT（Supervised Fine-Tuning）

**目标**：让模型学会基本的工具调用格式和正确的工具选择。

#### 数据准备

**从 `benchmark/tasks.py` 生成 expert demonstrations**：

```python
# scripts/generate_sft_data.py
"""从 benchmark tasks 生成 SFT 训练数据"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "AgenticArxiv"))

from benchmark.tasks import get_all_tasks
from utils.llm_client import get_env_llm_client
from agents.agent_engine import ReActAgent
from agents.side_effects import NoOpSideEffectManager
import json

def generate_sft_dataset():
    """执行所有 benchmark tasks，收集成功的 trajectories 作为 expert demos"""
    llm_client = get_env_llm_client()
    agent = ReActAgent(llm_client, side_effect_mgr=NoOpSideEffectManager())
    
    sft_data = []
    for task_def in get_all_tasks():
        result = agent.run(task_def["task"])
        
        # 只保留成功的 trajectories
        if result["iteration_count"] > 0 and result["final_observation"]:
            # 转为 SFT 格式（每一步作为一条训练样本）
            for i, step in enumerate(result["history"]):
                sft_data.append({
                    "messages": [
                        {"role": "system", "content": "你是一个 arXiv 论文检索 Agent..."},
                        {"role": "user", "content": task_def["task"]},
                        {"role": "assistant", "content": step["action"]}  # 只学习 action
                    ]
                })
    
    # 保存到 JSONL
    output_path = Path("data/sft/sft_train.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ SFT 数据生成完成：{len(sft_data)} 条样本 → {output_path}")

if __name__ == "__main__":
    generate_sft_dataset()
```

#### SFT 训练

**新增 `rl/train_sft.py`**：

```python
# AgenticArxiv/rl/train_sft.py
"""SFT 训练脚本（使用 TRL SFTTrainer）"""

from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main():
    # 1. 加载模型
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 2. 加载 SFT 数据集
    train_dataset = load_dataset("json", data_files="data/sft/sft_train.jsonl", split="train")
    
    # 3. 配置 SFT
    config = SFTConfig(
        output_dir="./outputs/sft",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_seq_length=1024,
        logging_steps=10,
        save_steps=100,
    )
    
    # 4. 训练
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    
    # 5. 保存
    trainer.save_model("./outputs/sft/final")
    print("✅ SFT 训练完成")

if __name__ == "__main__":
    main()
```

---

### 阶段2：DPO（Direct Preference Optimization）

**目标**：让模型偏好正确的工具选择和参数填写，拒绝错误的路由。

#### 数据准备

**从 SFT 模型 rollout，收集 chosen/rejected 对**：

```python
# scripts/generate_dpo_data.py
"""从 SFT 模型 rollout 生成 DPO 数据集"""

def generate_dpo_dataset():
    """
    策略：
    1. 用 SFT 模型对每个 task rollout 多次（如 5 次）
    2. 按 reward 排序，reward 最高的作为 chosen，最低的作为 rejected
    3. 构造 DPO 格式：{prompt, chosen, rejected}
    """
    sft_model = load_model("./outputs/sft/final")
    dpo_data = []
    
    for task_def in get_all_tasks():
        rollouts = []
        for _ in range(5):
            result = agent.run(task_def["task"], model=sft_model)
            reward, metrics = reward_calc.compute_reward(task_def, result)
            rollouts.append({"result": result, "reward": reward})
        
        # 排序
        rollouts.sort(key=lambda x: x["reward"], reverse=True)
        chosen = rollouts[0]["result"]["history"][-1]["action"]  # 最优动作
        rejected = rollouts[-1]["result"]["history"][-1]["action"]  # 最差动作
        
        dpo_data.append({
            "prompt": task_def["task"],
            "chosen": chosen,
            "rejected": rejected,
        })
    
    # 保存
    with open("data/dpo/dpo_train.jsonl", "w") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ DPO 数据生成完成：{len(dpo_data)} 条样本")
```

#### DPO 训练

**新增 `rl/train_dpo.py`**：

```python
# AgenticArxiv/rl/train_dpo.py
"""DPO 训练脚本（使用 TRL DPOTrainer）"""

from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def main():
    model_name = "./outputs/sft/final"  # 从 SFT 模型继续
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)  # reference model
    
    train_dataset = load_dataset("json", data_files="data/dpo/dpo_train.jsonl", split="train")
    
    config = DPOConfig(
        output_dir="./outputs/dpo",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=5e-6,
        beta=0.1,  # DPO 温度系数
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model("./outputs/dpo/final")
    print("✅ DPO 训练完成")

if __name__ == "__main__":
    main()
```

---

### 阶段3：GRPO（Group Relative Policy Optimization）

**目标**：用 verifiable reward 在线训练，无需 value model。

**新增 `rl/reward.py`**（复用 benchmark/metrics.py）：

```python
# AgenticArxiv/rl/reward.py
"""奖励计算器（复用 benchmark/metrics.py 的 verifiable 组件）"""

from benchmark.metrics import TaskMetrics, compute_metrics

class RewardCalculator:
    """基于 verifiable metrics 的奖励计算"""
    
    def __init__(self):
        self.weights = {
            "task_completed": 1.0,
            "tool_call_accurate": 0.5,
            "parse_failure_penalty": -0.2,
            "tool_exec_failure_penalty": -0.3,
            "force_stop_penalty": -0.5,
            "error_penalty": -1.0,
            "unnecessary_call_penalty": -0.1,
        }
    
    def compute_reward(self, task_def: dict, result: dict, agent_type: str = "regex", trial: int = 0) -> tuple[float, TaskMetrics]:
        """计算 reward + 返回 TaskMetrics"""
        metrics = compute_metrics(task_def, result, agent_type, trial, session_id="rl")
        
        reward = 0.0
        if metrics.task_completed:
            reward += self.weights["task_completed"]
        if metrics.tool_call_accurate:
            reward += self.weights["tool_call_accurate"]
        reward += self.weights["parse_failure_penalty"] * metrics.parse_failures
        reward += self.weights["tool_exec_failure_penalty"] * metrics.tool_exec_failures
        
        if metrics.termination_type == "FORCE_STOP":
            reward += self.weights["force_stop_penalty"]
        elif metrics.termination_type == "ERROR":
            reward += self.weights["error_penalty"]
        
        # 不必要调用惩罚
        extra_calls = len(metrics.tool_call_sequence) - len(metrics.expected_tools)
        if extra_calls > 0:
            reward += self.weights["unnecessary_call_penalty"] * extra_calls
        
        return reward, metrics
```

**新增 `rl/train_grpo.py`**：

```python
# AgenticArxiv/rl/train_grpo.py
"""GRPO 训练脚本（使用 TRL GRPOTrainer）"""

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from rl.reward import RewardCalculator

def main():
    model_name = "./outputs/dpo/final"  # 从 DPO 模型继续
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 定义 reward_model
    reward_calc = RewardCalculator()
    
    def reward_fn(responses, prompts):
        # TODO: 调用 RewardCalculator.compute_reward()
        # 返回 List[float]
        pass
    
    config = GRPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        num_sample_generations=4,  # 每个 prompt 采样 4 条 trajectory
    )
    
    # trainer = GRPOTrainer(...)
    # trainer.train()
    print("⚠️ GRPO 训练脚本骨架已就绪，需完成 TODO")

if __name__ == "__main__":
    main()
```

---

### 阶段4：PPO（可选，对比实验）

**目标**：使用 PPO 训练，对比 GRPO 效果（需要额外 value model）。

**新增 `rl/train_ppo.py`**：

```python
# AgenticArxiv/rl/train_ppo.py
"""PPO 训练脚本（使用 TRL PPOTrainer）"""

from trl import PPOConfig, PPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "./outputs/dpo/final"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # TODO: 需要额外训练一个 value model
    # value_model = AutoModelForSequenceClassification.from_pretrained(...)
    
    config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        num_train_epochs=3,
    )
    
    # trainer = PPOTrainer(...)
    print("⚠️ PPO 训练脚本骨架已就绪，需完成 TODO")

if __name__ == "__main__":
    main()
```

---

## ✅ 验证（改造完成后检查）

### 1. 环境验证
```bash
cd /Users/dev/projects/AgenticArXiv
source .venv/bin/activate
python -c "import torch, transformers, trl; print('✅ RL 依赖安装成功')"
```

### 2. Rollout 跑通
```bash
cd AgenticArxiv
python -m rl.rollout search_01 ../traces/train/
# 期望输出: ✅ Task search_01 rollout 完成, Reward: 1.50
```

### 3. SFT 数据生成
```bash
python scripts/generate_sft_data.py
# 期望输出: ✅ SFT 数据生成完成：N 条样本 → data/sft/sft_train.jsonl
```

### 4. Git 远程正确
```bash
git remote -v | grep AgenticArXiv-RL
# 期望输出: origin git@github.com:Algorineko/AgenticArXiv-RL.git
```

---

## 📅 时间规划（对应完整学习路径）

| 周数 | 阶段 | 核心任务 | 产出 |
|------|------|---------|------|
| Week 1-2 | 阶段0 | 环境准备（.venv/trajectory/MockEnv） | rollout 跑通 |
| Week 3-4 | 阶段1 | SFT（expert demos + TRL SFTTrainer） | SFT 模型 |
| Week 5-6 | 阶段2 | DPO（rollout + chosen/rejected + DPOTrainer） | DPO 模型 |
| Week 7-9 | 阶段3 | GRPO（verifiable reward + GRPOTrainer） | GRPO 模型 + 指标对比 |
| Week 10（可选） | 阶段4 | PPO（value model + PPOTrainer） | PPO 模型 + GRPO vs PPO 对比 |

---

## 📖 参考资料

### TRL 官方文档
- [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [DPOTrainer](https://huggingface.co/docs/trl/en/dpo_trainer)
- [GRPOTrainer](https://huggingface.co/docs/trl/en/grpo_trainer)
- [PPOTrainer](https://huggingface.co/docs/trl/en/ppo_trainer)

### 关键论文
- **InstructGPT** (OpenAI, 2022)：RLHF 三阶段（SFT → RM → PPO）
- **DPO** (Stanford, 2023)：直接偏好优化，无需 reward model
- **GRPO** (相关工作)：组内相对优势估计，无需 value model

---

## 🎯 下一步行动

### 今天（立即执行）
- [ ] 创建 `.venv` 环境：`python3 -m venv .venv && source .venv/bin/activate`
- [ ] 重写 `requirements.txt`（去 FastAPI/MySQL，加 torch/trl）
- [ ] 安装依赖：`pip install -r AgenticArxiv/requirements.txt`
- [ ] 运行 Git 新仓命令

### 本周（阶段0）
- [ ] 实现 `agents/side_effects.py`（`NoOpSideEffectManager`）
- [ ] 实现 `rl/trajectory.py`（JSONL 读写）
- [ ] 实现 `rl/env.py`（`MockArxivEnv`）
- [ ] 实现 `rl/reward.py`（复用 benchmark/metrics.py）

### Week 3-4（阶段1：SFT）
- [ ] 生成 SFT 数据集（`scripts/generate_sft_data.py`）
- [ ] 完善 `rl/train_sft.py`
- [ ] 跑通 SFT 训练

### Week 5-6（阶段2：DPO）
- [ ] 生成 DPO 数据集（`scripts/generate_dpo_data.py`）
- [ ] 完善 `rl/train_dpo.py`
- [ ] 跑通 DPO 训练

### Week 7-9（阶段3：GRPO）
- [ ] 完善 `rl/train_grpo.py`
- [ ] 跑通 GRPO 训练
- [ ] 指标监控（wandb / tensorboard）
- [ ] 超参调优、reward hacking 排查

