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

后续命令默认均在仓库根目录 `AgenticArXiv-RL/` 下执行。

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
| **Reward** | 五分量多粒度可验证奖励（format / tool / argument / process / outcome，见下节） |
| **Transition** | `execute_tool(action) → observation`（`MockArxivEnv` 离线快照回放，确定性可复现） |

### 动作空间（4 个工具）

1. `get_recently_submitted_cs_papers(aspect, days, max_results)` — 搜索 arXiv 论文
2. `download_arxiv_pdf(ref, session_id)` — 下载 PDF
3. `translate_arxiv_pdf(ref, session_id)` — 翻译 PDF
4. `get_paper_cache_status(ref, session_id)` — 查询缓存状态

### Verifiable Reward 组件

**多粒度五分量可验证奖励**（`rl/reward.py`，借鉴 LLM-TIR 的分层奖励），每个分量归一化到 `[-1, 1]`，加权求和后除以权重和：

| 分量 | 默认权重 | 信号 |
|------|:---:|------|
| `format`（格式） | 1 | 每一步 action 是否为合法 JSON 工具调用或终止符 |
| `tool`（工具序列） | 3 | 预测与期望工具序列的**顺序感知 LCS-F1**（`benchmark/metrics.py` 严格匹配） |
| `argument`（参数） | 2 | 参数键召回率 × 精确值准确率；任务无 `expected_tool_args` 时自动跳过 |
| `process`（过程） | 1 | 合法步骤加分 − 解析失败 / 执行失败 / 多余调用惩罚 |
| `outcome`（结果） | 3 | 正确完成 +1、工具路径错误的完成 +0.25、强制停止 −0.5、错误 −1 |

**课程学习**：前 30 个训练步将 `tool` / `argument` / `outcome` 权重乘以 1/3（先学 ReAct 结构、后学语义正确性），30 步后全权重生效（`RewardCalculator.schedule`）。

**关键**：所有奖励都是 **可验证的**（rule-based），无需人类标注 → 对应 RLVR（Reinforcement Learning with Verifiable Reward）框架。每条轨迹记录 `reward_components` 分量明细，便于审计与 reward-hacking 排查。

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

该命令直接加载 `outputs/sft/final` 的本地 Hugging Face 模型进行多次采样，
不需要 `LLM_API_KEY`。常用可选参数：

```bash
python scripts/generate_dpo_data.py \
  --model outputs/sft/final \
  --num_rollouts_per_task 8 \
  --temperature 0.8 \
  --seed 42
```

若已生成 `data/mock_arxiv_snapshot.json`，工具调用会自动使用离线回放，
保证数据生成可复现；否则会回退到实时网络。只有奖励差超过
`--min_reward_gap`（默认 0.05）且首个工具动作不同的轨迹才会组成偏好对。
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

**多轮 rollout 与奖励打分**（`rl/grpo_reward.py`）：每轮由当前 policy 生成 ReAct 动作，独立 `MockArxivEnv` 执行工具并把 observation 插回上下文，直到 `FINISH`、解析失败或达到 `--max_turns`。所有 assistant token 进入 GRPO loss，环境 observation token 通过 `env_mask=0` 仅作上下文；完整轨迹再交给五分量 `RewardCalculator` 打分，与 rollout / benchmark 共用同一套标准。

```bash
python -m AgenticArxiv.rl.build_snapshot
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --max_turns 4

# 记录训练曲线（三个阶段同一套参数）
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --report_to tensorboard
tensorboard --logdir outputs/grpo/logs
```

**训练曲线**（`rl/observability.py`）：`--report_to` 取 `none` / `auto` / `tensorboard` / `wandb`（可逗号分隔），三个训练阶段共用。除 TRL 自带的 reward / kl / grad_norm / `frac_reward_zero_std` 外，额外记录：

| 指标组 | 内容 | 为什么单独记 |
|---|---|---|
| `reward_components/*` | format / tool / argument / process / outcome | 各自恒在 `[-1,1]` 且与权重无关 |
| `reward_weights/*` | 当前课程权重 | 课程前 30 步压低 tool/argument/outcome 权重，只看 total 会把「权重放开」误读成「策略退化」 |
| `rollout/*` | turns / finished / parse_error_rate / tool_error_rate | reward 掉下去时区分「策略退化」与「没学会收尾、每次跑满 max_turns」 |

### 训练质量保障（自动校验）

训练链路内置多层自动校验，把「静默训练失败」变成响亮报错：

- **生成长度体检**：训练前校验 `max_completion_length` 是否放得下标准动作，防「永远吐不出完整动作」的零梯度空转
- **零方差守护**：组内奖励方差连续为 0（优势全零）时中止训练并给出修复建议（`RewardVarianceGuard`）
- **Canary 评估**：训练中每 N 步在固定任务上采样评估，性能退化达到阈值连续多次则提前停止（`CanaryCallback`）
- **阶段验证**：每个阶段产出模型须过最低质量阈值——SFT 可解析率 ≥ 0.3、DPO 平均奖励 ≥ −0.3、GRPO 平均奖励 ≥ −0.2（`StageVerifier`，`--no-verify` 可跳过）
- **混合精度自适应**：CUDA 优先 bf16、回退 fp16，CPU / MPS 关闭（`rl/precision.py`）
- **日志后端校验**：`--report_to` 指定的后端没装时在加载模型前就失败，避免训练跑完才发现没有任何曲线（`rl/observability.py`）

### 阶段4：PPO（Proximal Policy Optimization）

**目标**：标准 Actor-Critic 架构在线微调策略与价值网络。

**步骤**：
```bash
python -m AgenticArxiv.rl.train_ppo --model outputs/grpo/final
```

**产出**：`./outputs/ppo/final` 模型

---

## 📂 目录结构

```
AgenticArXiv-RL/
├─ AgenticArxiv/                     # ⭐ Python 包（RL 训练环境）
│  ├─ agents/                        # Agent 核心
│  │  ├─ base_agent.py              # 通用 ReAct 循环
│  │  ├─ agent_engine.py            # ReActAgent（RL 策略）
│  │  ├─ context_manager.py
│  │  ├─ prompt_templates.py
│  │  └─ side_effects.py           # 副作用解耦接口
│  ├─ tools/                         # 工具层（动作空间）
│  │  ├─ tool_registry.py          # 工具注册表
│  │  ├─ arxiv_tool.py             # arXiv 搜索
│  │  ├─ pdf_download_tool.py      # PDF 下载
│  │  ├─ pdf_translate_tool.py     # PDF 翻译
│  │  └─ cache_status_tool.py      # 缓存查询
│  ├─ benchmark/                     # ⭐ Verifiable Reward 来源
│  │  ├─ metrics.py               # TaskMetrics、工具序列严格匹配、参数匹配
│  │  ├─ tasks.py                 # BENCHMARK_TASKS（7 个任务种子）
│  │  ├─ runner.py                 # 基准执行器
│  │  ├─ run_benchmark.py          # 命令行基准入口
│  │  └─ report.py                 # 指标统计报告
│  ├─ rl/                            # ⭐ RL 核心
│  │  ├─ train_sft.py              # ⭐ SFT 训练
│  │  ├─ train_dpo.py              # ⭐ DPO 训练
│  │  ├─ train_grpo.py             # ⭐ GRPO 训练（含训练守卫）
│  │  ├─ train_ppo.py              # ⭐ PPO 训练（Actor-Critic）
│  │  ├─ env.py                    # RLEnv + MockArxivEnv（离线快照环境）
│  │  ├─ reward.py                 # RewardCalculator（五分量可验证奖励 + 课程）
│  │  ├─ grpo_reward.py            # GRPO 奖励适配（单步 completion → 合成轨迹）
│  │  ├─ rollout.py                # 离线 rollout 数据收集
│  │  ├─ trajectory.py             # Trajectory + JSONL 读写
│  │  ├─ build_snapshot.py         # 生成 arXiv 离线快照（唯一联网步骤）
│  │  ├─ canary.py                 # 训练中周期性评估（防退化早停）
│  │  ├─ stage_verifier.py         # 阶段产出模型质量阈值验证
│  │  ├─ precision.py              # 混合精度策略（bf16/fp16/CPU）
│  │  └─ observability.py          # 日志后端 + 奖励分量曲线
│  ├─ models/                        # 存储层（RL 用 store_memory，Web 版用 store_mysql）
│  ├─ services/                      # 副作用服务（event_bus / log / runtime）
│  ├─ api/ · mcp_protocol/ · skill_cli/   # 归档的 Web / MCP / Skill 兼容层
│  ├─ utils/                         # llm_client、logger、PDF 工具
│  ├─ tests/                         # 16 个单元测试（unittest）
│  └─ requirements.txt
├─ scripts/                          # 数据生成
│  ├─ generate_sft_data.py          # 用 LLM API 生成 expert 轨迹
│  └─ generate_dpo_data.py          # 用本地 SFT 模型采样构造偏好对
├─ docs/
│  ├─ rl_building.md               # 完整改造计划
│  ├─ multigranular_rl.md         # 多粒度奖励设计（五分量 + 课程学习）
│  └─ metric_stats.md            # 指标统计方案
├─ data/                             # 数据集（sft/ 与 dpo/ 为 gitignored，需自行生成）
│  ├─ sft/                           # SFT 数据集（JSONL）
│  ├─ dpo/                           # DPO 偏好对（JSONL）
│  └─ mock_arxiv_snapshot.json       # MockEnv 离线快照
├─ traces/                           # Trajectory 存储（JSONL，gitignored）
├─ archive/                          # 归档（原 Web 应用：PDFMathTranslate / arxiv-api / weather-agent）
├─ AgenticArxivWeb/                  # 原 Vue3 前端（已归档）
├─ bin/ · Makefile · Overview.md     # 遗留的 Web 启动脚本与文档（待现代化）
└─ README.md / README.en.md / README.es-ES.md   # 🇨🇳 🇬🇧 🇪🇸 三语说明
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
transformers>=4.45.0
trl>=0.20.0               # TRL (SFT/DPO/GRPO)，已在 0.29.1 上验证
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
## 📝 TODO（开发路线图）

按优先级排列。

### P0 — 近期（填补核心缺口）

- [x] **多轮 Agentic Rollout**：已实现真正的「行动 → 环境反馈 → 再行动」交互采样；每条 generation 使用独立 `MockArxivEnv`，环境 token 以 `env_mask=0` 排除策略 loss，完整 assistant 轨迹参与 GRPO 更新与五分量奖励。
- [x] **训练可观测性**：SFT / DPO / GRPO 统一 `--report_to`（none / auto / tensorboard / wandb），后端未安装时直接报错而非静默不记。除 TRL 自带的 reward / kl / grad_norm / `frac_reward_zero_std` 外，另单独记录 format/tool/argument/process/outcome 五个奖励分量与当前课程权重——课程会在前 30 步改变权重，只看 total reward 分不出「策略在变强」还是「权重表在动」；再加 `rollout/` 下的 turns / finished / parse_error_rate / tool_error_rate 用于定位多轮采样本身的问题。

### P1 — 中期（数据与评测）

- [x] **任务集扩充**：基准集扩到 59 条（`benchmark/tasks_expanded.py`，`--task-set expanded`），涵盖 search / ref_form / composite / state / optional / constraint / long_chain / infeasible 八类；`benchmark/tasks.py` 保留为 8 条冒烟子集。两边统一走 `benchmark/task_spec.py` 的 `TaskSpec`：`expected_tools` 与 `expected_tool_args` 由同一份 `steps` 派生，不再手写两份平行列表漂移。区分度用 `benchmark/run_baselines.py` 的确定性退化策略量化并逐类目卡门槛（`tests/test_reward_discrimination.py`）——修掉参数档的四处漏分后，「无视任务永远搜 cs.AI」在检索类任务上从 0.833 降到 0.446，「本该什么都不做却调了工具」从 +0.165 变成 −0.235。
- [x] **评测口径与切分**：报告从「成功率 + token 均值 + 迭代均值」扩到 `pass^k` 可靠性（tau-bench 口径，逐任务估计再平均；样本不足的任务记为跳过而非 0）、`false_finish`（以 FINISH 结束但期望工具没做全——退化策略实测 `always_finish` 91.5%、`reference` 0%）、`ref_score`（比解析出的 `paper_id` 而非 `ref` 的写法，同时消掉字符串比对的假阳性与假阴性）、代价按成功次数归一（`skill_cli` 从贵 43% 变成贵 99%）与失败形态拆分。任务集按**模板**切成 train/iid_test/ood_test（`benchmark/splits.py`，固化于 `data/splits/v1.json`），`--split` 在 `run_benchmark.py` 与 `train_grpo.py` 两侧接通；`rl_train` 只取成功率中间带，两端的任务组内方差为零、不产生梯度。
- [ ] **eval/ badcase replay**：目录树中的 `eval/`（`eval_cases.jsonl`、`badcase_replay.py`）实际尚不存在，需实现坏例回放闭环。
- [ ] **Reward hacking 排查**：在现有 `RewardVarianceGuard` / `CanaryCallback` 基础上补 reward-hacking 案例库与多粒度权重课程调优。

### P2 — 性能与规模

- [ ] **vLLM 加速采样**：替换 HF generate，提升 rollout 吞吐（多轮 rollout 落地后优先级上升）。
- [ ] **多卡支持**：accelerate / FSDP 配置（依赖已有 accelerate，但当前零配置、单卡单进程）。

### P3 — 长期（算法演进）

- [ ] **DAPO 系改进**：clip-higher、dynamic sampling、overlong filtering、token-level loss（loss/clip 在 TRL 内部，需 fork 或覆写 `compute_loss`）。
- [ ] **异步训练框架**：迁移 verl `fully_async_policy` / AReaL 全异步架构，承接 SAO（见下）。

### 🔭 SAO：下一代异步 Agentic RL 算法

> **SAO（Single-Rollout Asynchronous Optimization，单 rollout 异步优化）** 由清华大学 KEG 实验室提出（2026-07），是 GRPO 在**异步 agentic 训练**场景下的演进方向。核心动机：长程 agent 任务的 rollout 是训练瓶颈，GRPO 的组式采样在异步下会 off-policy、不稳定（典型 <200 步即崩）。
>
> 五个关键技术点：
> 1. **单 rollout 采样**：每个 prompt 只生成一条轨迹、随到随训，替代组式对比；
> 2. **DIS 直接双边重要性采样**：用 rollout 时记录的 token logprob 计算 `r_t = π_θ / π_rollout`，越出信任区间 `[1−ε_l, 1+ε_h]` 的 token **直接掩码为 0**（非 PPO 式单侧 clip）；
> 3. **value model 解耦更新**：策略:value = 1:2 更新频率，value 训练时**冻结注意力层**（只训 MoE 投影层）；
> 4. **skip-observation GAE**：优势只在模型生成的 token 之间传播，跳过环境观察 token，滤除环境噪声。
>
> 效果：稳定训练 ~1000 步，AIME2025 达 **97.3%**（vs GRPO 84.2%），SWE-Bench Verified 29.8%，已用于 GLM-5.2（750B）训练。
>
> 引入路径：先做 P0 多轮 rollout → 引入 skip-observation 掩码与 DIS 双边裁剪 → 迁移 verl `fully_async_policy`（`gen_batch_size=1` / `staleness_threshold` / token 级 TIS 裁剪，与 SAO 思路一致）或 AReaL v1.0 实现全异步 + value model。
>
> 📄 **论文**：[Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning（arXiv:2607.07508）](https://arxiv.org/abs/2607.07508)（清华 KEG，官方代码尚未开源）

---

**开始你的 Agentic RL 训练之旅！** 🚀
