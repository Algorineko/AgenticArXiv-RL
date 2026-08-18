# AgenticArXiv-RL — Agentic RL 训练环境

> 基于 ReAct Agent + arXiv 工具的可验证奖励训练环境，用于研究 LLM Agent 的 SFT / DPO / GRPO 训练流程。

---

## 项目简介

这个仓库把 arXiv 论文检索、下载、翻译等任务，改造成一个适合离线训练的 Agentic RL 环境。

核心目标是：

- 用 **可验证奖励** 代替人工标注
- 用 **ReAct 工具调用轨迹** 作为训练样本
- 支持 **SFT → DPO → GRPO** 的渐进式训练路径
- 保持工程尽量轻量：Python + JSONL + 本地轨迹文件

如果你在找的是原来的 Web 应用、实时翻译服务或前端界面，这些能力已经迁移或归档，不是当前主线。

---

## 当前状态

### 已实现

- `rollout`：执行 benchmark 任务、计算 reward、保存 trajectory
- `RewardCalculator`：支持分层、可审计的 reward 计算
- `generate_sft_data.py`：从成功轨迹生成 SFT 数据
- `train_sft.py`：SFT 训练入口

### 开发中

- `train_grpo.py`：当前是训练骨架，reward 函数仍需补全
- 更完整的在线训练闭环与评估流程

### 已归档

- 原 Web 应用相关能力
- FastAPI / Vue3 / MySQL 体系
- MCP / Skill 等非当前训练主线内容

---

## 快速开始

### 环境要求

- Python 3.9+
- 可用的 LLM API，且兼容 OpenAI API 格式
- 建议使用虚拟环境

### 安装依赖

```bash
git clone https://github.com/Algorineko/AgenticArXiv-RL.git
cd AgenticArXiv-RL
python -m venv .venv
```

#### macOS / Linux

```bash
source .venv/bin/activate
pip install -r AgenticArxiv/requirements.txt
```

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r AgenticArxiv/requirements.txt
```

### 配置环境变量

在 `AgenticArxiv/.env` 中配置 API 信息：

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
MODEL=gpt-4-turbo

PDF_RAW_PATH=./output/pdf_raw
PDF_TRANSLATED_PATH=./output/pdf_translated
```

#### Windows PowerShell 创建 `.env`

```powershell
@'
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
MODEL=gpt-4-turbo
PDF_RAW_PATH=./output/pdf_raw
PDF_TRANSLATED_PATH=./output/pdf_translated
'@ | Set-Content -Encoding utf8 AgenticArxiv/.env
```

### 跑一次 rollout

```bash
cd AgenticArxiv
python -m rl.rollout search_01 ../traces/train/
```

你会得到一个 JSONL trajectory 文件，并看到类似输出（奖励取值范围为 `[-1, 1]`）：

```text
✅ 任务完成
Reward: 0.80
Metrics: task_completed=True, tool_call_accurate=True, parse_failures=0, tool_exec_failures=0
Trajectory 保存至: traces/train/rollout_YYYYMMDD_HHMMSS.jsonl
```

---

## 核心入口

### Rollout

```bash
cd AgenticArxiv
python -m rl.rollout search_01 ../traces/train/
python -m rl.rollout --all ../traces/train/
```

### 生成 SFT 数据

```bash
python scripts/generate_sft_data.py
```

### SFT 训练

```bash
python -m rl.train_sft
```

### GRPO 训练

```bash
python -m rl.train_grpo
```

> 说明：`train_grpo.py` 当前仍是骨架实现，适合先理解训练接口和 reward 接入点。

### 命令执行位置

- `scripts/` 下的命令在仓库根目录执行（如 `python scripts/generate_sft_data.py`）
- `rl.` 开头的命令在 `AgenticArxiv/` 目录内执行（如 `python -m rl.rollout ...`）

---

## 项目设计

### MDP 视角

| 维度 | 定义 |
|------|------|
| State | 任务描述 + 对话历史 + 工具结果 |
| Action | 工具调用 + `FINISH` |
| Reward | 规则化、可验证的奖励信号 |
| Transition | 执行动作得到 observation |

### 工具动作空间

1. `get_recently_submitted_cs_papers(aspect, days, max_results)`
2. `download_arxiv_pdf(ref, session_id)`
3. `translate_arxiv_pdf(ref, session_id)`
4. `get_paper_cache_status(ref, session_id)`

### Reward 设计

当前 reward 不是单一分数，而是多个 component 的组合：

- `format`
- `tool`
- `argument`
- `process`
- `outcome`

这种方式的好处是：

- 更容易调试
- 更容易做 curriculum
- 更适合训练日志分析
- 更符合“可验证奖励”的研究目标

---

## 数据与输出

以下路径由脚本运行时自动创建（不在仓库中）。

### 训练数据

- `data/sft/sft_train.jsonl`：SFT 数据（由 `scripts/generate_sft_data.py` 生成）
- `data/dpo/dpo_train.jsonl`：DPO 数据（由 `scripts/generate_dpo_data.py` 生成）

### 轨迹输出

- `traces/train/`：训练 rollout 轨迹（JSONL）
- `traces/eval/`：评估轨迹

### 训练产物

- `./outputs/sft/final`
- `./outputs/dpo/final`
- `./outputs/grpo/final`

---

## 测试任务集

当前 benchmark 里有 7 个任务：

| ID | 任务 | 类型 | 预期工具 |
|----|------|------|---------|
| `search_01` | 检索最近7天AI论文 | 搜索 | `get_recently_submitted_cs_papers` |
| `search_02` | 获取最近3天ML论文 | 搜索 | `get_recently_submitted_cs_papers` |
| `search_03` | 搜索最近7天NLP论文 | 搜索 | `get_recently_submitted_cs_papers` |
| `download_01` | 下载第1篇论文PDF | 下载 | `download_arxiv_pdf` |
| `translate_01` | 翻译第1篇论文 | 翻译 | `translate_arxiv_pdf` |
| `cache_01` | 查看第1篇论文缓存状态 | 缓存 | `get_paper_cache_status` |
| `composite_01` | 搜索 + 下载 | 复合 | `get_recently_submitted_cs_papers`, `download_arxiv_pdf` |

---

## 依赖说明

训练主线依赖如下（`AgenticArxiv/requirements.txt`）：

```txt
torch>=2.0.0
transformers>=4.35.0
trl>=0.8.0
datasets>=2.14.0
accelerate>=0.25.0
arxiv
requests
python-dotenv
loguru
pydantic>=2.0
fire
```

说明：

- RL 训练路径与 MySQL / FastAPI 解耦，安装上述依赖即可跑通 rollout
- 运行旧版 Web 应用所需的 `fastapi` / `sqlalchemy` / `pymysql` 等依赖已移入 `requirements-extra.txt`，不作为训练主线
- 训练数据生成使用 mock 环境，无需安装 `pdf2zh`

---

## 目录结构概览

```text
AgenticArXiv-RL/
├─ AgenticArxiv/        # 主 Python 包（agents / tools / benchmark / rl / utils ...）
├─ AgenticArxivWeb/     # 旧版前端（已不在训练主线上）
├─ scripts/             # 数据生成脚本（generate_sft_data / generate_dpo_data）
├─ data/                # 数据文件与快照
├─ docs/                # 设计与重构文档
├─ draw/                # 数据可视化脚本
├─ archive/             # 已归档的旧能力（weather-agent、arxiv-api 等）
├─ bin/                 # 旧版启动/停止脚本
├─ benchmark_run.ipynb  # benchmark 运行示例
├─ Makefile             # 旧版工程脚本
└─ README.md
```

---

## 常见问题

### 为什么只保留 ReAct？

因为当前训练重点是稳定的工具调用轨迹与可验证奖励。ReAct 足够表达这类任务，而且更容易做监督数据和 reward 对齐。

### 为什么改用 JSONL？

因为离线训练更需要：

- 可移植
- 易回放
- 易审计
- 易和 TRL / datasets 对接

### 为什么 GRPO 没有写完整？

因为当前代码更像一个研究脚手架：接口已经明确，但 reward_fn 和在线闭环还需要进一步补完。

---

## 贡献

欢迎提交 Issue 和 Pull Request。

建议先按以下顺序验证修改：

1. 跑 `python -m rl.rollout search_01 ../traces/train/`
2. 跑 `python scripts/generate_sft_data.py`
3. 跑 `python -m rl.train_sft`

这样能较快确认修改没有破坏主流程。

---

## License

MIT License
