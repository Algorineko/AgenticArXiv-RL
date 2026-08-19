<p align="center">
  <a href="README.md">🇨🇳 中文</a> | <a href="README.en.md">🇬🇧 English</a> | <a href="README.es-ES.md">🇪🇸 Español</a>
</p>

# AgenticArXiv-RL — Agentic RL 训练环境

> **基于 ReAct Agent + arXiv 工具的 Agentic RL 训练环境**  
> 支持 SFT/DPO/GRPO/PPO 渐进式训练路径，用于研究 LLM Agent 强化学习

---

## 🎯 项目定位

将 arXiv 论文检索/下载/翻译任务改造为**可训练的强化学习环境**，专注于：

1. **Verifiable Reward**：基于规则化奖励（工具调用准确度、任务完成度、解析错误等），无需人类标注
2. **渐进式训练**：SFT（监督微调）→ DPO（直接偏好优化）→ GRPO（组内相对策略优化）→ PPO（近端策略优化）
3. **轻量级工程**：纯 Python + JSONL 存储，无需 MySQL/FastAPI/前端，专注离线训练

**非目标**：生产级 arXiv 应用、Web UI、实时翻译服务（这些功能已归档到 `archive/`）

---

## 🚀 快速开始

### 前置要求

- Python 3.9+
- LLM API（支持 OpenAI API 格式，如 Claude、Gemini、Qwen 等）
- 使用 `.venv` 虚拟环境

### 1️⃣ 克隆项目

```bash
git clone https://github.com/Algorineko/AgenticArXiv-RL.git
cd AgenticArXiv-RL
```

### 2️⃣ 环境配置

**创建虚拟环境**：
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

**安装依赖**：
```bash
pip install -r AgenticArxiv/requirements.txt
```

**配置 LLM API**：
```bash
cat > AgenticArxiv/.env << 'EOF'
# LLM API 配置
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
MODEL=gpt-4-turbo

# 可选：PDF 路径配置
PDF_RAW_PATH=./output/pdf_raw
PDF_TRANSLATED_PATH=./output/pdf_translated
EOF
```

### 3️⃣ 测试 Rollout

```bash
python -m AgenticArxiv.rl.rollout search_01 traces/train/
```

**期望输出**：
```
✅ Task search_01 rollout 完成
   Reward: 1.50
   Metrics: task_completed=True, tool_call_accurate=True
   Trajectory 保存至: traces/train/rollout_20260621_150000.jsonl
```

---

## 📚 核心概念

### MDP 设计

| 维度 | 定义 |
|------|------|
| **State** | 任务描述 + 对话历史 + 工具结果 |
| **Action** | 4 个工具（arxiv搜索/下载/翻译/缓存查询）+ FINISH |
| **Reward** | Verifiable（任务成功+1.0、工具准确+0.5、解析错误-0.2 等） |
| **Transition** | `execute_tool(action) → observation` |

### 动作空间（4 个工具）

1. `get_recently_submitted_cs_papers(aspect, days, max_results)` — 搜索 arXiv 论文
2. `download_arxiv_pdf(ref, session_id)` — 下载 PDF
3. `translate_arxiv_pdf(ref, session_id)` — 翻译 PDF
4. `get_paper_cache_status(ref, session_id)` — 查询缓存状态

### Verifiable Reward 组件

| 维度 | 奖励 | 来源 |
|------|------|------|
| 任务成功 | +1.0 | `task_completed` |
| 工具调用准确 | +0.5 | `tool_call_accurate`（`_check_tool_sequence`） |
| 解析错误 | -0.2 | `parse_failures` |
| 工具执行失败 | -0.3 | `tool_exec_failures` |
| 超时 | -0.5 | `termination_type == "FORCE_STOP"` |
| 错误终止 | -1.0 | `termination_type == "ERROR"` |

**关键**：所有奖励都是 **可验证的**（rule-based），无需人类标注 → 对应 RLVR（Reinforcement Learning with Verifiable Reward）框架。

---

## 🛠️ 训练路径（SFT → DPO → GRPO）

### 阶段1：SFT（Supervised Fine-Tuning）

**目标**：让模型学会基本的工具调用格式。

**步骤**：
1. 生成 expert demonstrations：
   ```bash
   python scripts/generate_sft_data.py
   ```
2. 训练：
   ```bash
   python -m AgenticArxiv.rl.train_sft
   ```
3. 产出：`./outputs/sft/final` 模型

**数据格式**（`data/sft/sft_train.jsonl`）：
```json
{
  "messages": [
    {"role": "system", "content": "你是 arXiv 论文检索 Agent..."},
    {"role": "user", "content": "检索最近7天AI论文"},
    {"role": "assistant", "content": "{\"name\":\"get_recently_submitted_cs_papers\",\"arguments\":{...}}"}
  ]
}
```

---

### 阶段2：DPO（Direct Preference Optimization）

**目标**：让模型偏好正确的工具选择，拒绝错误路由。

**步骤**：
1. 用 SFT 模型 rollout，收集 chosen/rejected 对：
   ```bash
   python scripts/generate_dpo_data.py
   ```
2. 训练：
   ```bash
   python -m AgenticArxiv.rl.train_dpo
   ```
3. 产出：`./outputs/dpo/final` 模型

**数据格式**（`data/dpo/dpo_train.jsonl`）：
```json
{
  "prompt": "检索最近7天AI论文",
  "chosen": "{\"name\":\"get_recently_submitted_cs_papers\",...}",
  "rejected": "{\"name\":\"download_arxiv_pdf\",...}"
}
```

---

### 阶段3：GRPO（Group Relative Policy Optimization）

**目标**：用 verifiable reward 在线训练，无需 value model。

**步骤**：
```bash
python -m AgenticArxiv.rl.train_grpo
```

**产出**：`./outputs/grpo/final` 模型

**优势**：
- 无需 reward model（DPO 的缺点：无法在线学习）
- 无需 value model（PPO 的缺点：显存开销大）
- 适合小模型（如 Qwen2.5-1.5B）

---

## 📂 目录结构

```
AgenticArXiv-RL/
├─ AgenticArxiv/                     # Python 包
│  ├─ agents/                        # Agent 核心
│  │  ├─ base_agent.py              # 通用 ReAct 循环
│  │  ├─ agent_engine.py            # ReActAgent（RL 策略）
│  │  ├─ prompt_templates.py
│  │  └─ side_effects.py            # 副作用解耦接口
│  ├─ tools/                         # 工具层（动作空间）
│  │  ├─ tool_registry.py
│  │  ├─ arxiv_tool.py
│  │  ├─ pdf_download_tool.py
│  │  ├─ pdf_translate_tool.py
│  │  └─ cache_status_tool.py
│  ├─ benchmark/                     # ⭐ Verifiable Reward 来源
│  │  ├─ metrics.py                 # TaskMetrics、_check_tool_sequence
│  │  ├─ tasks.py                   # BENCHMARK_TASKS（任务集种子）
│  │  └─ runner.py
│  ├─ rl/                            # ⭐ RL 核心
│  │  ├─ env.py                     # RLEnv + MockArxivEnv
│  │  ├─ policy.py
│  │  ├─ reward.py                  # RewardCalculator
│  │  ├─ trajectory.py              # Trajectory + JSONL 读写
│  │  ├─ rollout.py
│  │  ├─ tasks.py
│  │  ├─ train_sft.py               # ⭐ SFT 训练
│  │  ├─ train_dpo.py               # ⭐ DPO 训练
│  │  └─ train_grpo.py              # ⭐ GRPO 训练
│  ├─ utils/
│  │  ├─ llm_client.py
│  │  └─ logger.py
│  └─ requirements.txt
├─ traces/                           # Trajectory 存储（JSONL）
│  ├─ train/
│  └─ eval/
├─ data/
│  ├─ sft/                           # SFT 数据集
│  ├─ dpo/                           # DPO 数据集
│  └─ mock_arxiv_snapshot.json       # MockEnv 快照
├─ eval/
│  ├─ eval_cases.jsonl
│  └─ badcase_replay.py
├─ scripts/
│  ├─ generate_sft_data.py
│  └─ generate_dpo_data.py
├─ archive/                          # 归档（原 Web 应用）
│  ├─ api/
│  ├─ AgenticArxivWeb/
│  ├─ mcp_protocol/
│  └─ skill_cli/
├─ docs/
│  └─ rl_building.md                # 完整改造计划
├─ .venv/                            # Python 虚拟环境
└─ README.md                         # 本文档
```

---

## 🔬 使用示例

### 1. Rollout（收集 trajectory）

```bash
# 单个任务
python -m AgenticArxiv.rl.rollout search_01 traces/train/

# 批量 rollout
python -m AgenticArxiv.rl.rollout --all --output_dir traces/train/
```

### 2. 训练流程（SFT → DPO → GRPO）

```bash
# Step 1: 生成 SFT 数据
python scripts/generate_sft_data.py

# Step 2: SFT 训练
python -m AgenticArxiv.rl.train_sft

# Step 3: 生成 DPO 数据（需要 SFT 模型）
python scripts/generate_dpo_data.py

# Step 4: DPO 训练
python -m AgenticArxiv.rl.train_dpo

# Step 5: GRPO 训练
python -m AgenticArxiv.rl.train_grpo
```

### 3. Reward 计算测试

```python
from rl.reward import RewardCalculator
from benchmark.tasks import get_task_by_id

task_def = get_task_by_id('search_01')
# 构造一个 mock result
result = {
    'history': [
        {'thought': '...', 'action': '...', 'observation': '...'},
        {'thought': '...', 'action': 'FINISH', 'observation': '...'},
    ],
    'timing': {...},
    'token_usage': {...},
    'iteration_count': 2,
}

reward_calc = RewardCalculator()
reward, metrics = reward_calc.compute_reward(task_def, result)
print(f'Reward: {reward:.2f}')  # 期望: ~1.5
```

---

## 🧪 测试任务集

来自 `benchmark/tasks.py`，包含 7 个任务：

| ID | 任务 | 类型 | 预期工具 |
|----|------|------|---------|
| `search_01` | 检索最近7天AI论文 | 搜索 | `get_recently_submitted_cs_papers` |
| `search_02` | 获取最近3天ML论文 | 搜索 | `get_recently_submitted_cs_papers` |
| `search_03` | 搜索最近7天NLP论文 | 搜索 | `get_recently_submitted_cs_papers` |
| `download_01` | 下载第1篇论文PDF | 下载 | `download_arxiv_pdf` |
| `translate_01` | 翻译第1篇论文 | 翻译 | `translate_arxiv_pdf` |
| `cache_01` | 查看第1篇论文缓存状态 | 缓存 | `get_paper_cache_status` |
| `composite_01` | 搜索+下载 | 复合 | `get_recently_submitted_cs_papers`, `download_arxiv_pdf` |

---

## 📊 指标监控

### Reward 曲线

使用 TensorBoard 或 wandb 监控：
```bash
tensorboard --logdir ./outputs/grpo/logs
```

### 关键指标

| 指标 | 说明 | 目标 |
|------|------|------|
| `reward` | 平均奖励 | ↑ 上升 |
| `kl_div` | KL 散度（vs reference model） | ↔ 稳定（不过大） |
| `task_completed_rate` | 任务成功率 | ↑ 上升 |
| `tool_call_accurate_rate` | 工具调用准确率 | ↑ 上升 |
| `parse_failures` | 解析失败次数 | ↓ 下降 |
| `tool_exec_failures` | 工具执行失败次数 | ↓ 下降 |

---

## 🛡️ 依赖说明

**核心依赖**（`requirements.txt`）：
```txt
torch>=2.0.0
transformers>=4.35.0
trl>=0.8.0                # TRL (SFT/DPO/GRPO/PPO)
datasets>=2.14.0
accelerate>=0.25.0
arxiv
requests
python-dotenv
loguru
pydantic>=2.0
fire
```

**不再需要**（已去除）：
- `fastapi`、`uvicorn`（无 Web 服务）
- `sqlalchemy`、`pymysql`（改用 JSONL）
- `pdf2zh`（训练时用 mock）

---

## 🔗 相关资源

### 官方文档
- [TRL 文档](https://huggingface.co/docs/trl/)
- [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [DPOTrainer](https://huggingface.co/docs/trl/en/dpo_trainer)
- [GRPOTrainer](https://huggingface.co/docs/trl/en/grpo_trainer)

### 论文
- **InstructGPT** (OpenAI, 2022)：RLHF 三阶段（SFT → RM → PPO）
- **DPO** (Stanford, 2023)：直接偏好优化
- **RLVR**：Reinforcement Learning with Verifiable Reward

### 原 AgenticArXiv（Web 应用版）
本项目基于 [AgenticArXiv](https://github.com/Algorineko/AgenticArXiv) 改造，原版包含：
- FastAPI 后端 + Vue3 前端
- 三种 Agent 架构（ReAct/MCP/Skill）
- 实时 SSE 推送、MySQL 存储、PDF 翻译服务

这些功能已归档到 `archive/`。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发建议
1. Fork 本仓库
2. 创建 feature 分支：`git checkout -b feature/your-feature`
3. 提交改动：`git commit -m "feat: add your feature"`
4. 推送分支：`git push origin feature/your-feature`
5. 提交 Pull Request

---

## 📄 License

MIT License

---

## 🙋 FAQ

### Q: 与原 AgenticArXiv 的区别？

| 维度 | 原版 AgenticArXiv | 本项目 (AgenticArXiv-RL) |
|------|------------------|-------------------------|
| **定位** | 生产级 arXiv 应用 | RL 训练研究环境 |
| **架构** | FastAPI + Vue3 + MySQL | 纯 Python + JSONL |
| **Agent 模式** | 3 种（ReAct/MCP/Skill） | 仅 ReAct（精简） |
| **核心功能** | 实时翻译、SSE、Web UI | SFT/DPO/GRPO 训练 |
| **依赖** | 重（14+ 包） | 轻（8 核心包） |

### Q: 为什么只保留 ReAct，归档 MCP/Skill？

RL 训练专注单一策略（ReAct 正则解析），MCP/Skill 增加复杂度但不改变核心逻辑。

### Q: 为什么改用 JSONL 而非 MySQL？

- **可移植性**：JSONL 无需数据库依赖
- **轻量级**：更适合 RL 训练的离线场景
- **TRL 兼容**：TRL 数据集直接支持 JSONL

### Q: 为什么选 GRPO 不用 PPO？

GRPO 更适合轻量级学习项目：
- ✅ 无需额外 value model（显存/训练开销更小）
- ✅ 适合小模型（如 Qwen2.5-1.5B）
- ✅ 实现简单，调试容易

PPO 更适合生产级大模型训练（7B+），本项目作为学习 demo 不涉及。

---

**开始你的 Agentic RL 训练之旅！** 🚀
