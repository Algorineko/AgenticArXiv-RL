<p align="center">
  <a href="README.md">🇨🇳 中文</a> | <a href="README.en.md">🇬🇧 English</a> | <a href="README.es-ES.md">🇪🇸 Español</a>
</p>

# AgenticArXiv-RL — Agentic RL 训练环境

> **基于 ReAct Agent + arXiv 工具的 Agentic RL 训练环境**  
> 支持 SFT/DPO/GRPO/PPO 渐进式训练路径，另提供可选的 OPD（on-policy 蒸馏）路线，用于研究 LLM Agent 强化学习

<p align="center">
  <img src="imgs/AgenticArXiv-RL.jpg" alt="AgenticArXiv-RL 项目介绍" width="800"/>
</p>

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

**输出示例（奖励随实际轨迹变化，范围为 `[-1, 1]`）**：
```
✅ Task search_01 rollout 完成
   Reward: 1.00
   Metrics: task_completed=True, tool_call_accurate=True
   Trajectory 保存至: traces/train/rollout_20260621_150000.jsonl
```

---

## 📚 核心概念

### MDP 设计

| 维度 | 定义 |
|------|------|
| **State** | 任务描述 + 对话历史 + 工具结果 |
| **Action** | 8 个工具（arXiv 浏览/关键词检索/下载/翻译/缓存查询/论文阅读/论文总结/图表抽取）+ FINISH |
| **Reward** | 五分量多粒度可验证奖励（format / tool / argument / process / outcome，见下节） |
| **Transition** | `execute_tool(action) → observation`（`MockArxivEnv` 离线快照回放，确定性可复现） |

### 动作空间（8 个工具）

1. `get_recently_submitted_cs_papers(aspect, days, max_results)` — 搜索 arXiv 论文
2. `download_arxiv_pdf(ref, session_id)` — 下载 PDF
3. `translate_arxiv_pdf(ref, session_id)` — 翻译 PDF
4. `get_paper_cache_status(ref, session_id)` — 查询缓存状态
5. `search_arxiv_papers(query, max_results, days=None)` — 按关键词、标题或作者检索
6. `get_paper_content(ref, session_id, section=None)` — 读取已下载论文的摘要或指定章节
7. `summarize_paper(ref, style, max_words)` — 对已下载论文做 env 侧摘要（tldr / structured / bullet）
8. `extract_paper_figures(ref)` — 抽出已下载论文的图表文件与 caption

> 「检索 → 下载 → 阅读 → 总结 → 抽图」这条解读闭环已经打通。图表**分析**（T5，需要 env 侧 VLM）仍是设计稿，见下文「🧰 工具集演进设计」。

### Verifiable Reward 组件

**多粒度可验证奖励**（`rl/reward.py`，借鉴 LLM-TIR 的分层奖励）。核心五个分量归一化到 `[-1, 1]` 并沿用课程权重；另外记录结果质量与效率两个诊断分量，并对严重失败施加不可补偿的 reward 上限：

| 分量 | 默认权重 | 信号 |
|------|:---:|------|
| `format`（格式） | 1 | 每一步 action 是否为合法 JSON 工具调用或终止符 |
| `tool`（工具序列） | 3 | 预测与期望工具序列的**顺序感知 LCS-F1**（`benchmark/metrics.py` 严格匹配） |
| `argument`（参数） | 2 | 参数键召回率 × 精确值准确率；任务无 `expected_tool_args` 时自动跳过 |
| `process`（过程） | 1 | 合法步骤加分 − 解析失败 / 执行失败 / 多余调用惩罚 |
| `outcome`（结果） | 3 | 正确完成 +1、工具路径错误的完成 +0.25、强制停止 −0.5、错误 −1 |

| `result_quality`（结果质量） | 诊断 / gate | 检查工具 observation 是否真实表示成功结果；空结果、回退结果和执行错误会触发负向 gate |
| `efficiency`（效率） | 诊断 / gate | 按任务标准步骤归一化多余调用和失败重试，严重冗余会触发 reward 上限 |

**课程学习**：前 30 个训练步将 `tool` / `argument` / `outcome` 权重乘以 1/3（先学 ReAct 结构、后学语义正确性），30 步后全权重生效（`RewardCalculator.schedule`）。

**非补偿性失败**：解析失败、工具执行失败和未知结果最多得到负奖励；虚假 FINISH 不得靠格式分抵消。单步 GRPO 为了构造完整轨迹而自动补的 `FINISH` 会标记为 `forced_finish`，不会拿到完整终止奖励。

#### T5 图表分析结果的判定

`analyze_figure` 的 observation 可能是工具结果的 Python `repr`，也可能是 JSON。
结果质量检查要求**完整解析**为对象，且 `paper_id` 与 `answer` 都是非空白字符串。
仅出现字段名、缺少字段、值为其他类型或 observation 被截断，均不能证明已有答案。

| observation 示例 | 单步 `result_quality` | 判定 |
| --- | ---: | --- |
| `{'paper_id': '2601.00004v1', 'answer': 'A rising trend.'}` | `+1` | Python `repr`，答案非空 |
| `{"paper_id": "2601.00004v1", "answer": "The caption does not state this."}` | `+1` | JSON，诚实说明 caption 未提供信息 |
| `{'paper_id': '2601.00004v1', 'answer': ''}` | `-1` | 空答案 |
| `{"paper_id": "2601.00004v1", "answer": "   "}` | `-1` | 只有空白 |
| `{'paper_id': '2601.00004v1'}` | `-1` | 缺少 `answer` |
| `{'answer': 'A rising trend.'}` | `-1` | 缺少 `paper_id` |
| `{'paper_id': '2601.00004v1', 'meta': {'answer': 'nested'}}` | `-1` | 答案只在嵌套字段中 |
| `{'paper_id': '2601.00004v1', 'answer': 'cut off` | `-1` | 截断，无法确认完整答案 |

`result_quality` 的课程权重为零，但仍参与安全闸门：任一步 `analyze_figure`
缺少有效答案，总奖励最多为 `-0.75`，不能靠格式分或工具序列分补成正分。
多步任务仍对各步骤结果质量取平均；例如前三步为 `+1`、图表分析为 `-1`，
平均值是 `0.5`，逐步安全检查仍会拦住这条轨迹。
排查时应同时查看完整轨迹、分量和总分。
该规则只检查**是否存在答案**，不判断 VLM 文本的事实正确性，也不改变
`get_paper_content`、`summarize_paper` 等工具的评分语义。
更多细节见[多粒度奖励文档](docs/multigranular_rl.md)。

**关键**：所有奖励都是 **可验证的**（rule-based），无需人类标注 → 对应 RLVR（Reinforcement Learning with Verifiable Reward）框架。每条轨迹记录 `reward_components` 分量明细，便于审计与 reward-hacking 排查。

### Rollout 隔离

`rl/sandbox.py` 中的 `RolloutSandbox` 会在轨迹开始前记录环境、内存 Store 和指定产物目录的基线状态，在下一条轨迹前恢复该状态并清理新建文件。多轮 GRPO、普通 rollout 和离线 benchmark 都使用这一 reset 契约，避免搜索结果、下载缓存、翻译状态和统计计数在轨迹之间串联。

---

## 🛠️ 训练路径（SFT → DPO → GRPO / OPD）

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

> **已发布的权重**（Qwen2.5-1.5B 全参 SFT，2 epoch / 2628 条参数化专家轨迹，loss 0.079）：
> [🤗 ModelScope · AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT)
> 注意该产物**只覆盖前 8 个工具**——`analyze_figure`（T5）是之后加入的。现在可以选择生成 T5 专家数据，但该已发布模型尚未用它训练。

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

# 记录训练曲线（各阶段同一套参数）
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --report_to tensorboard
tensorboard --logdir outputs/grpo/logs

# DAPO 系选项（默认不改变任何值，历史实验仍然可比）
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --dapo
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --epsilon_high 0.2   # 只开 clip-higher

# 多卡（DDP；须用装了训练依赖的那个解释器启动 accelerate）
accelerate launch --config_file configs/accelerate/ddp_2gpu.yaml \
  -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --no-qlora
```

**多卡**：`configs/accelerate/` 提供 DDP 与 FSDP 两份配置。训练脚本会检测 `accelerate launch` 导出的 `LOCAL_RANK`，此时不再钉单卡（钉了反而会让其它 rank 看不到设备）；单进程下仍然钉住，避免 Trainer 退化成 `DataParallel` 而段错误。QLoRA 的 `device_map={"": 0}` 与多进程互斥，检测到就直接报错而不是静默按单卡训练。FSDP 需要 `torch>=2.6`。

**训练曲线**（`rl/observability.py`）：`--report_to` 取 `none` / `auto` / `tensorboard` / `wandb`（可逗号分隔），五个训练阶段（SFT / DPO / GRPO / OPD / PPO）共用。除 TRL 自带的 reward / kl / grad_norm / `frac_reward_zero_std` 外，额外记录：

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

### 阶段 3'：OPD（On-Policy Distillation，可选，与 GRPO 互替）

**目标**：用强教师模型的逐 token 信号蒸馏学生的 ReAct 动作能力，训练全程不依赖奖励。

**步骤**：
```bash
python -m AgenticArxiv.rl.train_opd --model outputs/sft/final --teacher Qwen/Qwen2.5-7B-Instruct

# 多轮 Agentic OPD：真实执行 Action → Observation → Action
python -m AgenticArxiv.rl.train_opd --model outputs/sft/final \
  --teacher Qwen/Qwen2.5-7B-Instruct --max_turns 4 \
  --snapshot data/mock_arxiv_snapshot.json
```

**产出**：`./outputs/opd/final` 模型

**定位**：OPD 是一个**完整的训练范式**而非单一 trick——学生在任务 prompt 上 on-policy 采样，教师对每个 token 给出 logprob，损失取 reverse-KL `D_KL(π_student ‖ π_teacher)`（mode-seeking）。它与 GRPO 是「教师路线 vs 奖励路线」的关系：

| 维度 | GRPO | OPD |
|---|---|---|
| 学习信号 | 五分量可验证奖励（稀疏、轨迹级） | 教师逐 token logprob（稠密） |
| 额外模型 | 无 | 教师模型（需本地权重取 logprob，外部 API 拿不到逐 token 分布） |
| 性能上限 | 可探索超越教师 | 收敛到教师行为 |
| 适用场景 | 有可验证奖励、无教师 | 有强教师、想省 RL 探索成本 |

OPD 也能当 trick 用：RL 前的 warm start、RL 中的 teacher-KL 正则、verl 的 PG-OPD（把 reverse-KL 当奖励做策略梯度）。

**单轮与多轮**：`--max_turns 1`（默认）保留 TRL GKDTrainer 的单段 prompt/completion 行为；`--max_turns > 1` 启用 Agentic OPD。多轮模式为每条轨迹创建独立 `MockArxivEnv`，执行工具后把真实 Observation 写回下一轮上下文；prompt 和 Observation token 的 loss mask 为 `-100`，reverse-KL 只覆盖学生生成的 assistant token。第一版要求教师与学生的 token-to-id vocabulary 完全一致，不兼容时在加载模型后、训练前明确失败。学生与教师同驻显存（1.5B 学生 + 7B 教师仅权重 bf16 约 17GB，训练还需梯度、优化器和 logits 显存；32GB 卡建议使用更小教师或学生）。已在 trl 1.5.1（`trl.experimental.gkd`）上验证；`beta=1.0` 即 reverse-KL 方向（`tests/test_opd.py` 有数值单测锁死）。Canary 与阶段验证沿用 GRPO 同一套——OPD 训练本身无奖励，但产出模型仍要在环境里过关。

**离线对比**：用相同的 snapshot 与 task set 分别运行 SFT、`--max_turns 1`、`--max_turns > 1` 和 GRPO，并比较各输出目录 `final/verification_report.json` 中由 `StageVerifier` 生成的同口径结果；这些 reward 只用于评测，不会进入 OPD loss。

### 阶段4：PPO（Proximal Policy Optimization）— ⚠️ 当前依赖下不可用

**目标**：标准 Actor-Critic 架构在线微调策略与价值网络。

**现状**：TRL 自 0.9 起弃用、并在后续版本**移除**了经典 PPO trainer —— `PPOTrainer` / `PPOConfig` / `AutoModelForCausalLMWithValueHead` 在 `requirements.txt` 允许的 `trl>=0.28.0` 上都不存在，`train_ppo.py` 因此连 import 都过不去。它现在会在 import 阶段给出明确说明，而不是抛一句看不出因果的 ImportError。

要真正跑 PPO，需要按 `trl.experimental.ppo` 的实验 API 重写本脚本（配置项与数据集格式都不同），或把 trl 固定到 0.9 之前。本项目里 PPO 是学习性对照：有可验证奖励走 GRPO、有强教师走 OPD，两者都比 PPO 更省显存。

```bash
# 当前会带着说明直接退出
python -m AgenticArxiv.rl.train_ppo --model outputs/grpo/final
```

**产出**（重写后）：`./outputs/ppo/final` 模型

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
│  │  ├─ arxiv_tool.py             # arXiv 分类/关键词搜索
│  │  ├─ pdf_download_tool.py      # PDF 下载
│  │  ├─ pdf_translate_tool.py     # PDF 翻译
│  │  ├─ cache_status_tool.py      # 缓存查询
│  │  ├─ paper_content_tool.py     # 确定性论文内容读取（T2）
│  │  ├─ paper_summary_tool.py     # env 侧论文摘要（T3）
│  │  └─ paper_figures_tool.py     # 确定性图表抽取（T4）
│  ├─ benchmark/                     # ⭐ Verifiable Reward 来源
│  │  ├─ metrics.py               # TaskMetrics、工具序列严格匹配、参数匹配
│  │  ├─ tasks.py                 # BENCHMARK_TASKS（8 条冒烟任务）
│  │  ├─ tasks_expanded.py        # 扩展任务集（81 条、十三类模板）
│  │  ├─ task_spec.py             # TaskSpec：expected_tools / expected_tool_args 由 steps 统一派生
│  │  ├─ badcases.py              # 坏例用例的判定与回放
│  │  ├─ splits.py                # 模板层 train/iid/ood 切分
│  │  ├─ baselines.py · run_baselines.py  # 确定性退化策略基线（逐类目区分度闸门）
│  │  ├─ runner.py · run_benchmark.py     # 基准执行器与命令行入口
│  │  └─ report.py                # 指标统计报告
│  ├─ rl/                            # ⭐ RL 核心
│  │  ├─ train_sft.py              # ⭐ SFT 训练
│  │  ├─ train_dpo.py              # ⭐ DPO 训练
│  │  ├─ train_grpo.py             # ⭐ GRPO 训练（含训练守卫）
│  │  ├─ train_ppo.py              # ⭐ PPO 训练（Actor-Critic）
│  │  ├─ train_opd.py              # ⭐ OPD 训练（on-policy 蒸馏，与 GRPO 互替）
│  │  ├─ opd_multiturn.py          # 多轮 OPD rollout、Observation mask 与自定义 GKD loss
│  │  ├─ env.py                    # RLEnv + MockArxivEnv（离线快照环境）
│  │  ├─ multiturn_env.py          # TRL 多轮环境适配器（environment_factory，每代独立实例）
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
│  ├─ tests/                         # 单元测试（unittest / pytest）
│  └─ requirements.txt
├─ configs/accelerate/                # 多卡启动配置（ddp_2gpu / fsdp_2gpu）
├─ scripts/                          # 数据生成
│  ├─ generate_sft_data.py          # 用 LLM API 生成 expert 轨迹
│  ├─ generate_parametric_sft_data.py  # 从任务模板参数化派生专家轨迹（无需 API）
│  ├─ augment_sft_data.py           # 语义保持的措辞增强
│  ├─ build_sft_train_mix.py        # 合成可审计的训练混合集 + manifest
│  └─ generate_dpo_data.py          # 用本地 SFT 模型采样构造偏好对
├─ docs/
│  ├─ rl_building.md               # 完整改造计划
│  ├─ multigranular_rl.md         # 多粒度奖励设计（五分量 + 课程学习）
│  └─ metric_stats.md            # 指标统计方案
├─ data/                             # 数据集（sft/ 与 dpo/ 为 gitignored，需自行生成）
│  ├─ sft/                           # SFT 数据集（JSONL）
│  ├─ dpo/                           # DPO 偏好对（JSONL）
│  ├─ splits/                        # 模板级 train/iid/ood 切分（v1 → v3_81）
│  └─ mock_arxiv_snapshot.json       # MockEnv 离线快照
├─ eval/                             # 坏例回放闭环
│  ├─ badcase_replay.py             # 回放 / 捕获 CLI（无需 LLM）
│  ├─ eval_cases.jsonl              # 用例库，兼作 reward hacking 案例库
│  └─ readme.md
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
print(f'Reward: {reward:.2f}')  # 奖励范围：[-1, 1]，具体值取决于轨迹
```

---

## 🧪 任务集与评测

### 冒烟集（`benchmark/tasks.py`，8 条）

| ID | 任务 | 类型 | 预期工具 |
|----|------|------|---------|
| `search_01` | 检索最近7天AI论文 | 搜索 | `get_recently_submitted_cs_papers` |
| `search_02` | 获取最近3天ML论文 | 搜索 | `get_recently_submitted_cs_papers` |
| `search_03` | 搜索最近7天NLP论文 | 搜索 | `get_recently_submitted_cs_papers` |
| `search_04` | 检索最近7天CS全部方向论文（最多10篇） | 搜索 | `get_recently_submitted_cs_papers` |
| `download_01` | 下载第1篇论文PDF | 下载 | `download_arxiv_pdf` |
| `translate_01` | 翻译第1篇论文 | 翻译 | `translate_arxiv_pdf` |
| `cache_01` | 查看第1篇论文缓存状态 | 缓存 | `get_paper_cache_status` |
| `composite_01` | 搜索+下载 | 复合 | `get_recently_submitted_cs_papers`, `download_arxiv_pdf` |

### 扩展集（`benchmark/tasks_expanded.py`，81 条）

`run_benchmark.py --task-set expanded` 启用，涵盖 search / keyword_search / ref_form / composite / state / optional / constraint / long_chain / infeasible / paper_reading / paper_summary / figure_extraction 十二类模板。两套任务统一走 `benchmark/task_spec.py` 的 `TaskSpec`：`expected_tools` 与 `expected_tool_args` 由同一份 `steps` 派生，不出现两份手写列表的漂移。

解读族（`paper_reading` / `paper_summary`）的参数档用显式 `{键: None}` 把「该键必须缺省」变成可判定的要求：`argument_match_score` 只统计期望键的命中率、不多算多余键，不声明 `section` 键的话，「只要 title+abstract」的任务传 `section="method"` 也能拿满参数分。

`figure_extraction` 的 ref 是**对着快照挑过的**：离线快照里那些位置确实有内嵌位图（CV 池前 12 篇都有）。论文有没有位图是快照的属性，换一份快照就要对一遍——`extract_paper_figures` 对没有位图的论文返回 `count: 0` 而不是报错，所以这类任务一旦失配，表现是奖励噪声而不是响亮失败。

### 评测口径与切分

报告从「成功率 + token 均值 + 迭代均值」扩到：

- **`pass^k` 可靠性**：tau-bench 口径，逐任务估计再平均；样本不足的任务记为跳过而非 0
- **`false_finish`**：以 FINISH 结束但期望工具没做全——退化策略实测 `always_finish` 91.5%、`reference` 0%
- **`ref_score`**：比解析出的 `paper_id` 而非 `ref` 的写法，同时消掉字符串比对的假阳性与假阴性
- **代价按成功次数归一**（`skill_cli` 从贵 43% 修正为贵 99%）与失败形态拆分

任务集按**模板**切成 train/iid_test/ood_test（`benchmark/splits.py`，固化于 `data/splits/v1.json`），`--split` 在 `run_benchmark.py` 与 `train_grpo.py` 两侧接通；`rl_train` 只取成功率中间带——两端的任务组内方差为零、不产生梯度。

### 区分度闸门与坏例回放

- **逐类目区分度闸门**（`benchmark/run_baselines.py`）：用确定性退化策略量化奖励区分度，逐类目卡门槛（`tests/test_reward_discrimination.py`）——修掉参数档的四处漏分后，「无视任务永远搜 cs.AI」在检索类任务上从 0.833 降到 0.446，「本该什么都不做却调了工具」从 +0.165 变成 −0.235。
- **坏例回放**（`eval/badcase_replay.py` + `eval/eval_cases.jsonl`）：把单条失败轨迹连同当时的判定冻成永久回归用例；回放只重跑打分器，不需要 LLM / 网络 / 工具，`pytest` 本身就是闸门（`tests/test_badcases.py::ShippedCasesTest`）。用例分 `open`（毛病还在）与 `fixed`（已修，再复现即回归、退出码 1）；`hack/*` 记录退化策略的骗分轨迹并配阈值断言（「这种行为不许拿到 X 分」，#40 与 #47 修掉的两个洞都有用例守着），兼作 reward hacking 案例库——与 `run_baselines.py` 的闸门互补：**闸门看均值、用例钉单条**。`--save-traces` + `capture` 按 `false_finish` / `ref_score` / 工具序列从真实跑的轨迹里挑坏例，而非只挑崩掉的。

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

**核心依赖**（`requirements.txt`，覆盖 rollout / benchmark / 五个训练阶段）：
```txt
torch>=2.0.0
transformers>=4.45.0
trl>=0.28.0               # 下限由多轮 GRPO 决定：0.28.0 起非 vLLM 路径才调用 rollout_func，
                          # 更早版本多轮 rollout 会静默退化；已在 0.29.1 验证（OPD 在 1.5.1 验证）
datasets>=2.14.0
accelerate>=0.25.0
arxiv
requests
python-dotenv
loguru
pydantic>=2.0
fire
```

**可选依赖**（`requirements-extra.txt`，按需安装，核心训练链路不依赖）：
- `pdf2zh` — 真实 PDF 翻译（训练/基准走 mock，跑翻译 eval/demo 时才需要）
- `fastapi` / `uvicorn` / `sqlalchemy` / `pymysql` — 仅运行归档的 Web 版时需要
- `tensorboard`（推荐，零配置离线可用）/ `wandb` — `--report_to` 的训练曲线后端

---

## 🔗 相关资源

### 模型权重
- [AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT) —— 阶段 1 SFT 产物，Qwen2.5-1.5B 全参微调（ModelScope）

### 官方文档
- [TRL 文档](https://huggingface.co/docs/trl/)
- [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [DPOTrainer](https://huggingface.co/docs/trl/en/dpo_trainer)
- [GRPOTrainer](https://huggingface.co/docs/trl/en/grpo_trainer)

### 论文
- **InstructGPT** (OpenAI, 2022)：RLHF 三阶段（SFT → RM → PPO）
- **DPO** (Stanford, 2023)：直接偏好优化
- **RLVR**：Reinforcement Learning with Verifiable Reward
- **On-Policy Distillation** ([Thinking Machines Lab, 2025](https://thinkingmachines.ai/blog/on-policy-distillation/))：OPD 的方法来源（学生 on-policy 采样 + 教师逐 token reverse-KL）
- **GKD / On-Policy Distillation of Language Models** (Agarwal et al., ICLR 2024, [arXiv:2306.13649](https://arxiv.org/abs/2306.13649))：广义 JSD 蒸馏，TRL GKDTrainer 的论文依据
- **Rethinking On-Policy Distillation of Large Language Models** ([arXiv:2604.13016](https://arxiv.org/abs/2604.13016))：对 OPD 配方的独立复现与分析

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
| **依赖** | 重（14+ 包） | 轻（11 核心包 + 可选 extra） |

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

### Q: OPD、SFT、DPO、GRPO 怎么选？

| 场景 | 推荐 |
|------|------|
| 有 expert 演示，先学动作格式 | SFT |
| 有偏好对（正/负轨迹），无在线奖励 | DPO |
| 有可验证奖励，想在线超越基线 | GRPO |
| 有强教师模型，想省 RL 探索成本 | OPD |

四者是互补关系：SFT 是所有路径的起点；OPD 与 GRPO 都在 SFT 之上，前者走教师蒸馏（上限=教师）、后者走奖励优化（可探索超越），两者产出可互为对方的初始化或对照基线。

---
## 🧰 工具集演进设计（增量路线图）

> 本项目的最终目标是**本地部署的轻量 LLM 独立完成 arXiv 论文的检索、下载与分析解读**。T1/T2 已实现；T3–T5 仍是设计稿。

### 现状盘点与缺口

| 工具 | 能力 | 边界 |
|------|------|------|
| `get_recently_submitted_cs_papers` | 子领域 + 时间窗检索 | 只认 `cat:cs.*` + 提交日期，无翻页；摘要截断 200 字符 |
| `search_arxiv_papers` | 关键词/篇名/作者检索 | 无翻页；离线模式对快照外查询返回显式标记的确定性降级结果 |
| `download_arxiv_pdf` | 下载 PDF | — |
| `translate_arxiv_pdf` | pdf2zh 整篇翻译 | 产出是翻译后的 PDF 文件，翻译后的正文不进入模型上下文；依赖可选 extra |
| `get_paper_cache_status` | 查缓存 | — |
| `get_paper_content` | 读取摘要或 method/result/conclusion 章节 | 需要先下载 PDF；确定性抽取，不调用 LLM |
| `summarize_paper` | 按 style/budget 生成摘要 | 需要先下载 PDF；默认走确定性抽取式后端，`local_model` 后端为可选 |
| `extract_paper_figures` | 抽出内嵌图表图片 + caption，返回文件路径 | 需要先下载 PDF；纯向量图（无内嵌位图）会返回 `count: 0` |

三个结论：

1. **「检索」半环已打通**：既支持时间窗浏览，也支持关键词、篇名和作者查找；翻页仍未实现。
2. **「分析解读」闭环已打通**：读内容 → 总结 → 抽图三步都是确定性工具，模型可以独立完成一篇论文的解读链。图表**语义分析**（T5）仍待实现，它需要 env 侧常驻一个 VLM。
3. **动作空间不是越大越好**：策略是 1.5B 量级小模型，每加一个工具都放大工具选择与 JSON 格式的学习负担。新增工具的准入标准是「能开启一类新任务」，而不是「可能有用」——下表 5 个候选里 T1/T2 是关键路径，T3 是主要增量，T4 已实现，T5 可选。

### 建议新增的工具（按依赖顺序）

| 优先级 | 工具 | 设计要点 | 为什么仍是 RLVR 友好 |
|--------|------|----------|----------------------|
| **T1** ✅ | `search_arxiv_papers(query, max_results, days=None)` | 关键词检索，映射 arXiv API 的 `all:` / `ti:` / `au:` 字段；与现有工具并存（时间窗浏览 vs 精确查找是两类任务） | 期望工具/参数照常由 `task_spec.steps` 派生；`MockArxivEnv` 按查询串哈希离线回放，未收录的 query 走**确定性降级**（返回固定子集并在 observation 里显式标注），保证可复现、也防止模型把空结果当检索成功 |
| **T2** ✅ | `get_paper_content(ref, section=None)` | PDF → 纯文本（PyMuPDF），默认返回 title/abstract，可按节取（method / result / conclusion） | 确定性文本抽取，无 LLM 参与；快照预存抽取结果。**它是全部解读类任务的前置件** |
| **T3** ✅ | `summarize_paper(ref, style, max_words)` | 总结论文：**env 侧**生成摘要（输入来自 T2 的文本），返回摘要文本 | 可训练的是「何时调、对哪个 ref 调、style/长度参数对不对」——全部规则可判；摘要质量本身**不进奖励**（见下） |
| **T4** ✅ | `extract_paper_figures(ref)` | 图表可视化准备：抽出图表图片 + caption，返回文件路径列表 | 确定性；验证「ref 正确 + 数量 ≥ 1」 |
| **T5**（可选，多模态） | `analyze_figure(ref, figure_no, question=None)` | 图表分析：env 侧调本地 VLM（如 Qwen2.5-VL）读图回答 | 规则只判「调没调对、参数对不对」；VLM 回答质量不进奖励，避免把第三方模型的噪声写进策略梯度 |

**T3 的摘要后端**：README 原本写的是「env 侧调本地摘要模型」。落地时默认采用**确定性抽取式后端**（按 section 取整句、按词数预算裁剪，不采样、不调模型），原因有三：它让同一条轨迹在任意时刻回放都逐字节一致；它不需要在 `build_snapshot` 之外再引入一个需要权重的前置条件；而奖励只看工具调用决策、不看摘要文字，模型后端对训练信号没有贡献。需要更自然语言的摘要时可用 `SUMMARY_BACKEND=local_model SUMMARY_MODEL_PATH=<本地模型目录>` 切到模型后端（贪心解码，仍然确定性），代价是快照构建阶段要加载权重。

配套任务模板（沿用 `tasks_expanded.py` 的分类，全部由 `task_spec.steps` 声明式派生）：

- `search_kw_*`：关键词检索类（T1）
- `paper_reading`：检索 → 下载 → 读内容（T2），5 条
- `paper_summary`：检索 → 下载 → 总结（T3），5 条
- `long_chain` 的 `chain_ai5_read_then_summary`：读 → 总结串成 4 步链（T2+T3）
- `figure_extraction`：检索 → 下载 → 抽图（T4），4 条
- `analyze_figure`（T5，可选）：检索 → 下载 → 抽图 → 图表分析，仅多模态环境启用

### 训练与评测侧的连带设计

1. **快照扩展**：`build_snapshot.py` 一次跑齐——除搜索结果外，对快照论文**预抽取全文文本与图表文件**；新工具全部离线回放，维持「build_snapshot 是唯一联网步骤」的约定。
2. **奖励零改动**：五分量方案原样复用；`expected_tools` / `expected_tool_args` 从 `steps` 派生，`reward.py` 与课程学习均不需动。
3. **防 reward hacking**：解读类工具天然多出「乱调工具刷 process 分」的面——沿用 `run_baselines.py` 逐类目闸门 + `eval/eval_cases.jsonl` 单例钉死（例如：ref 指向不存在论文时调 `summarize_paper` 必须扣分）。
4. **摘要质量的奖励问题（刻意不做）**：把「摘要写得好不好」变成奖励需要 LLM-as-judge 或 rubric 打分，会引入非确定性奖励与新的 hacking 面。设计上先把总结收敛为**工具调用决策问题**（何时调、对谁调），质量评估留到长期单独立项。
5. **多模态的边界（刻意隔离）**：T5 的 VLM 只活在 env 侧，策略仍是纯文本小模型——动作空间里只有「调不调、怎么问」，看图能力外包给环境。只有策略本身换成多模态模型时，才考虑把图片放进 observation。
6. **硬件门槛**：只有 T5 需要一个 env 侧模型（VLM 约 6GB 量级，量化后更低），它与训练显存互不影响（不进梯度）。T3 默认的抽取式后端不加载任何权重，所以解读闭环（T1–T4）在训练机之外不需要额外硬件。

### 落地顺序

```
T1 关键词检索 ──→ T2 读内容 ──→ T3 总结          （解读闭环，已完成）
                      └────→ T4 抽图 → T5 图分析 （T4 已完成；T5 可选，多模态环境）
```

每落一个工具：扩任务模板 → 重跑 `run_baselines.py` 重新卡各类目区分度门槛 → 重新生成 SFT/DPO 数据 → `eval/eval_cases.jsonl` 补对应用例。T3/T4 落地时同步做了：

- 任务模板：新增 `paper_reading`（5 条）、`paper_summary`（5 条）、`figure_extraction`（4 条）与 4 步解读链 1 条，扩展集 62 → 81 条
- 切分：新增 `data/splits/v3_81.json`（v1/v2 保持不变，历史实验的成功率仍然可比）
- 区分度：`run_baselines.py` 逐类目闸门覆盖新类目（`tests/test_reward_discrimination.py`）
- 坏例：`eval/eval_cases.jsonl` 新增 8 条 `hack/summary-*` / `hack/read-*` / `hack/figure-*` 用例，钉住「预算传错」「跳过下载直接总结/抽图」「多传 section」「抽错论文」等骗分形态（用例库现共 14 条）

---

## 📝 TODO（开发路线图）

### P0 — 工具集扩展（解读闭环）

T1–T4 已实现（见「🧰 工具集演进设计」）：

- [x] **T1 关键词检索** `search_arxiv_papers`：补全「找一篇具体论文」的查找型检索
- [x] **T2 论文阅读** `get_paper_content`：确定性 PDF → 文本与离线快照回放，全部解读类任务的前置件（关键路径）
- [x] **T3 论文总结** `summarize_paper`：env 侧摘要，把「解读」变成可训练的工具调用决策（默认确定性抽取式后端，`SUMMARY_BACKEND=local_model` 可切本地模型）
- [x] **T4 图表抽取** `extract_paper_figures`：确定性抽出内嵌图表与 caption，离线快照回放
- [ ] **T5 图表分析** `analyze_figure`（可选，多模态环境）：工具、环境集成、任务模板、单测和可选的参数化 SFT 派生规则已写好；仍需录制真实快照并生成数据。VLM 只在 env 侧，策略仍是纯文本小模型。
  - ⏳ **快照缺口**：仓库不收录本地离线快照，当前没有可验证的 T5 记录。2026-09-22 构建时 arXiv 对本机限速到 ~5KB/s（34MB 的论文 90 秒只传了 492KB），因此没有完成重建。有可用网络或 PDF 缓存时运行 `python -m AgenticArxiv.rl.build_snapshot --skip-prefetch`；缺失 T5 记录时 replay 会报错。
  - 若已有包含 T4 图表记录的快照，可运行 `python -m AgenticArxiv.rl.backfill_figure_analysis --snapshot data/mock_arxiv_snapshot.json`，直接从 caption 补录默认 `extractive` 后端的 T5 结果，无需重新下载 PDF；VLM 结果仍需用 VLM 录制。
  - 后端：默认 `extractive`（只复用 T4 已抽出的 caption，确定性强、不需要权重）；`FIGURE_ANALYSIS_BACKEND=vlm VLM_MODEL_PATH=<本地 VLM 目录>` 切本地 VLM（贪心解码，答案在构建快照时录制）。
  - **生成 T5 专家数据**（须先准备含 T5 记录的快照）：
    ```bash
    python scripts/generate_parametric_sft_data.py \
      --split-file data/splits/v3_81.json \
      --snapshot data/mock_arxiv_snapshot.json \
      --include-t5
    ```
    默认命令仍生成历史 v2 数据；T5 输出单独写入 `data/sft/sft_v3_t5_parametric_seed.jsonl`。
  - 已知边界：T5 数据和模型都未在本仓库生成或训练；已发布模型尚未学过这个工具。

### P1 — 奖励课程调优

- [x] **多粒度权重课程定档（已有实测数据，结论：SFT 起点的 30 步压制档位不成立）**

  用 `scripts/analyze_curriculum.py` 读了真实训练曲线，做了两臂对照：同一个 SFT 起点（Qwen2.5-1.5B，2 epoch，loss≈0.08），同样的 9 条能产生组内梯度的任务，各 60 步，只差 `--reward_curriculum_steps`。

  | 分量 | 课程 30：压制期 → 全权重期 | 课程 0 |
  |---|---|---|
  | `format` | 0.983 → 1.000 | 0.983 → 1.000 |
  | `process` | 0.937 → 1.000 | 0.934 → 1.000 |
  | `tool` | −0.074 → +0.215 | −0.074 → +0.215 |
  | `argument` | −0.545 → −0.335 | −0.530 → −0.319 |
  | `outcome` | −0.127 → +0.028 | −0.127 → +0.030 |

  两个结论：

  1. **课程的前提在第一步就已经被满足**：`format` 起点 0.983、`process` 0.937 —— 30 步压制存在的理由是「先学会 ReAct 结构」，而 SFT 已经把结构教完了。它保护的是一个已经走过去的阶段。
  2. **两臂结果不可区分**：分量曲线几乎逐点重合。压制 `tool`/`argument`/`outcome` 三十步，既没有让语义学得更好，也没有更差——只是把三十分之三十的步数花在了被压低的信号上。

  建议：**从 SFT 起点的 GRPO 用 `--reward_curriculum_steps 0`**；这个档位留给冷启动 / 基座模型直接开训的场景。诚实的前提说明：9 条任务 × 60 步是小规模运行，差异本身落在噪声里——所以结论不是「课程有害」，而是「它没有挣到那 30 步」。

- [x] **顺带修掉的选任务规则缺陷**：`rl_train` 按 `rates` 取成功率中间带，而 `rates` 是 **base 模型**实测的。SFT 之后难度带整体移位，实测 48 条 train 任务里只剩 **9 条（19%）**还能给出组内梯度；39 条零方差任务中有 **24 条落在「中间带但恒定」**——「中间带 ⇒ 有组内方差」对饱和策略并不成立，它们会以 `frac_reward_zero_std=1` 的形式浪费步数并触发 `RewardVarianceGuard`。换模型后需要重新测量 rates，而不是沿用上一次的档位。

### P2 — 性能与规模

- [x] **多卡支持**：`configs/accelerate/ddp_2gpu.yaml`（DDP）与 `fsdp_2gpu.yaml`（FSDP）已就位，训练脚本在 `accelerate launch` 下自动跳过单卡钉死，QLoRA 多进程启动会显式报错。DDP 已在双卡机器上实测跑通（两 rank 建 NCCL 通信、梯度同步、正常保存）。
  - FSDP 配置需要 `torch>=2.6`：本仓库的 `rl/trl_compat.py` 在 torch<2.6 上只提供单卡占位，那条路径未在本机验证。
  - 用法：`accelerate launch --config_file configs/accelerate/ddp_2gpu.yaml -m AgenticArxiv.rl.train_sft --no-qlora ...`（`accelerate` 必须用装了训练依赖的那个 Python 解释器启动）
- [ ] **vLLM 加速采样**：替换 HF generate，提升多轮 rollout 的采样吞吐。
  - **现状**：TRL 0.29 要求 vLLM 0.10.2–0.12.0，而本机可用的 vLLM 是平台定制版 0.6.2，装在一起会让 `trl.trainer.grpo_trainer` 在 import 阶段就失败。要推进得先拿到与当前 TRL 匹配的平台上 vLLM 构建，属于环境依赖而非代码改动，因此暂缓。

### P3 — 长期（算法演进）

- [x] **DAPO 系改进（TRL 已原生支持的部分）**：`--loss_type`（token-level loss）、`--epsilon_high`（clip-higher）、`--mask_truncated_completions`（overlong filtering）三个开关已接到 `train_grpo.py`，并提供 `--dapo` 预设一次填齐（`loss_type=dapo`、`epsilon_high=0.28`、`mask_truncated_completions=True`、`beta=0`）。默认不改变任何值，历史实验仍然可比。
  - **dynamic sampling（组内零方差重采样）未实现**：TRL 0.29 没有暴露可拦截的生成钩子，覆写 `_generate_and_score_completions` 会随版本漂移。当前的兜底是 `RewardVarianceGuard`（连续零方差即中止训练）加上 `frac_reward_zero_std` 曲线，先把「静默空转」变成可见信号。
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
> 引入路径：多轮 rollout 已落地 → 引入 skip-observation 掩码与 DIS 双边裁剪 → 迁移 verl `fully_async_policy`（`gen_batch_size=1` / `staleness_threshold` / token 级 TIS 裁剪，与 SAO 思路一致）或 AReaL v1.0 实现全异步 + value model。
>
> 📄 **论文**：[Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning（arXiv:2607.07508）](https://arxiv.org/abs/2607.07508)（清华 KEG，官方代码尚未开源）

---

**开始你的 Agentic RL 训练之旅！** 🚀
