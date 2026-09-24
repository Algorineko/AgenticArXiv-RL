# CLI 命令参考

> 版本范围：当前源码树。每条命令下方都写明工作目录、输入、输出、网络/快照
> 与硬件前提。本文是参数参考，不代表这些命令已在本地执行，也不提供训练指标。

## 1. 工作目录和前提

| 命令族 | 从哪里运行 | 典型依赖 |
|---|---|---|
| `python -m benchmark.*` | `AgenticArxiv/` | Python 包、工具注册；API benchmark 还需要 LLM 配置 |
| `python -m rl.*` | `AgenticArxiv/` | Python 包；快照构建需要网络和 arXiv 访问 |
| `python scripts/*.py` | 仓库根目录 | `AgenticArxiv/` 在脚本中加入 `sys.path` |
| `python eval/badcase_replay.py` | 仓库根目录 | 不需要 LLM、网络或模型 |
| `python draw/plot.py` | 仓库根目录 | `draw/` 和报告数据 |

路径约定：

- 从 `AgenticArxiv/` 指向根目录数据时使用 `../data/...`。
- 从仓库根目录运行脚本时使用 `data/...`。
- `--offline` 控制工具环境；`--backend transformers` 只控制 LLM 从哪里加载。
  本地模型也可以访问真实工具，除非同时指定 `--offline`。
- 下面的“需要 GPU”是代码路径的前提，不是对某次实验成功的声明。

## 2. 构建 MockArxivEnv 快照

入口：`AgenticArxiv/rl/build_snapshot.py`。

从 `AgenticArxiv/` 运行：

```bash
python -m rl.build_snapshot
```

该入口是快照流水线中唯一默认需要访问 arXiv 的步骤。它记录搜索池、论文正文、
摘要、图表抽取和图表分析结果；完成后 rollout、重评分和大多数训练路径可以使用
`MockArxivEnv(mode="replay")` 离线运行。

### 2.1 参数

| 参数 | 默认值 | 作用 | 前提 / 备注 |
|---|---|---|---|
| `--snapshot` | `data/mock_arxiv_snapshot.json`（由源码按仓库根计算） | 输出快照路径 | 父目录可自动创建 |
| `--aspects` | `* AI LG CL CV RO CR` | 要抓取的 cs 方向 | `nargs="+"`，一次传多个 |
| `--keyword-queries` | 三个默认查询 | 关键词搜索池 | `nargs="+"`；传入后替换默认列表 |
| `--max_results` | `50` | 每个搜索池的最大结果数 | 不等于内容预取深度 |
| `--days` | `30` | 搜索时间窗口 | 影响搜索记录 key |
| `--no-pin-references` | 关闭 | 不固定 benchmark 锚点论文 | 仅适合原始数据研究，正式评测不建议 |
| `--allow-partial` | 关闭 | 查询失败仍保存部分快照 | 正式训练/评测不建议 |
| `--content-workers` | `8` | PDF 预取线程数 | 只并发下载，不并发修改内存 store |
| `--skip-prefetch` | 关闭 | 跳过并行 PDF 预取 | 适合 PDF 已在磁盘的恢复场景 |
| `--prefetch-budget` | `900` 秒 | 预取阶段整体墙钟预算 | 超时任务交给后续串行阶段重试 |
| `--content-max-ref` | `0` | 每个池只预取前 N 篇正文/图表 | `0` 表示全部；搜索池数量不变 |

示例：

```bash
# 从 AgenticArxiv/ 运行；只覆盖常用方向，限制内容预取深度
python -m rl.build_snapshot \
  --aspects AI LG CL CV \
  --max_results 30 \
  --content-max-ref 5 \
  --snapshot ../data/mock_arxiv_snapshot.json
```

注意：`--no-pin-references` 会跳过固定的 AI/CV benchmark 锚点；这可能让
`ref` 依赖标题或 arXiv id 的任务无法复现。`--allow-partial` 不是“所有失败都
算成功”，只是允许保存不完整输入，后续 replay miss 仍会暴露缺口。

## 3. Benchmark 运行器

入口：`AgenticArxiv/benchmark/run_benchmark.py`。

```bash
cd AgenticArxiv
python -m benchmark.run_benchmark --task-set expanded --offline \
  --split ../data/splits/v3_81.json:iid_test \
  --agents regex --output ../data/bench_v3_iid
```

默认任务集是 `benchmark/tasks.py` 的 8 条基础任务；`--task-set expanded` 才会
读取 81 条扩展任务。默认 agent 是 `regex mcp skill_cli`，默认重复 3 次；如果
为了 Base/SFT/GRPO 的能力对比而不是执行框架对比，README 建议显式用
`--agents regex`。

### 3.1 参数表

| 参数 | 默认值 / 选择 | 作用 |
|---|---|---|
| `--agents` | `regex mcp skill_cli` | 一个或多个 agent 类型 |
| `--repeat` | `3` | 每个任务重复次数 |
| `--tasks` | `None`；源码 choices 为 `search`, `download`, `translate`, `cache`, `composite`, `keyword_search`, `ref_form`, `optional`, `state`, `long_chain`, `constraint`, `infeasible` | 按类别筛选；新解读类别不在这个 choices 列表中 |
| `--task-ids` | `None` | 按任务 id 筛选，优先于 split/category |
| `--output` | 根目录 `data/` | 报告输出目录 |
| `--model` | 配置中的 agent model | API 模型名或本地 transformers 目录 |
| `--backend` | `api`；另有 `transformers` | LLM 后端 |
| `--local-device` | `auto` | transformers 的 `auto` / `cuda` / `cpu` |
| `--local-dtype` | `auto` | `auto` / `float16` / `bfloat16` / `float32` |
| `--seed` | `42` | 本地生成基础随机种子 |
| `--prefix` | `bench_r<timestamp>` | session id 前缀 |
| `--no-thinking` | 关闭 | 向支持的 chat template 传 `enable_thinking=False` |
| `--offline` | 关闭 | 使用 `data/mock_arxiv_snapshot.json` 回放工具 |
| `--snapshot` | `None` | 覆盖默认快照路径 |
| `--save-traces [PATH]` | 不保存 | 保存每条 history；不传 PATH 时写 `<output>/traces.jsonl` |
| `--split [FILE:]NAME` | `None` | 只运行一份切分；必须与 expanded 任务集配合 |
| `--task-set` | `default` | `default` 或 `expanded` |

`--tasks` 的 choices 是 argparse 的真实边界，不要因为 `tasks_expanded.py`
存在 `paper_reading`、`paper_summary`、`figure_extraction`、`figure_analysis`
就写成可用的 `--tasks paper_summary`。要运行这些类别，使用
`--task-set expanded` 后传 `--task-ids`，或使用包含它们的显式 split。

### 3.2 工作模式和输出

| 模式 | 是否访问 LLM | 是否访问真实 arXiv | 适合 |
|---|---|---|---|
| 默认 API | 是 | 是，除非 `--offline` | API 集成或真实工具实验 |
| `--backend transformers` | 是，本地模型 | 由 `--offline` 单独控制 | 本地模型 benchmark |
| `--offline` | 是 | 否；读取 snapshot，下载走离线桩 | 可复现比较、CI 前检查 |
| `--save-traces` | 同上 | 同上 | 后续 badcase capture 或重评分 |

典型输出包括 `summary.json`、`report.md`、`raw_data.csv`，有错误时还会有
`errors.csv`；具体报告文件由 `BenchmarkReport.save_all()` 写出。`traces.jsonl`
只在显式 `--save-traces` 时生成，且是 benchmark trace schema，不是
`rl.trajectory.Trajectory` schema。

## 4. 确定性奖励基线

入口：`AgenticArxiv/benchmark/run_baselines.py`。从 `AgenticArxiv/` 运行：

```bash
python -m benchmark.run_baselines --task-set expanded
python -m benchmark.run_baselines \
  --task-set expanded --seed 42 --random-samples 20 \
  --output ../data/baselines_v3
```

此命令不调用模型、网络或真实工具；它构造 synthetic history，再用同一个
`RewardCalculator` 诊断评分器是否能区分弱策略。当前 `ALL_POLICIES` 有五种：

| policy | 行为 |
|---|---|
| `reference` | 回放任务声明的标准工具链，作为评分参照 |
| `always_finish` | 立即 FINISH |
| `always_search` | 对所有任务调用同一个合法搜索 |
| `random_tool` | 用稳定 seed 选择一次工具调用 |
| `wrong_args` | 保留参考工具名，但故意传错误参数 |

参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--task-set` | `expanded` | `default` 或 `expanded` |
| `--policies` | 全部 policy | 选择要运行的名字 |
| `--seed` | `42` | random_tool 的第一个稳定 seed |
| `--random-samples` | `20` | 连续采样的 seed 数，必须至少为 1 |
| `--min-reference-gap` | `0.3` | 每个弱策略与 reference 的最小平均奖励差 |
| `--min-category-gap` | `0.3` | 每个类别的最小奖励差 |
| `--top` | `5` | 每个弱策略显示的高分任务数 |
| `--training-step` | `100` | 奖励课程所在 step；100 使用完整 correctness 权重 |
| `--output` | `None` | 写 `baseline_report.json` 和 `.md` 的目录 |

健康检查不通过时入口返回退出码 1。synthetic baseline 只诊断评分敏感度，
不测量 LLM 质量，不应在 PR 或论文中写成模型成功率。

## 5. 轨迹重评分

入口：`AgenticArxiv/benchmark/rescore_traces.py`。它不加载模型，只用已有
history、summary 中的计时/token 和指定任务池重新计算 metrics。

```bash
cd AgenticArxiv
python -m benchmark.rescore_traces \
  --traces ../data/bench_v3/traces.jsonl \
  --summary ../data/bench_v3/summary.json \
  --output ../data/bench_v3_rescored \
  --snapshot ../data/mock_arxiv_snapshot.json \
  --task-set expanded \
  --split ../data/splits/v3_81.json:iid_test
```

| 参数 | 必需 | 含义 |
|---|---|---|
| `--traces` | 是 | `run_benchmark --save-traces` 产生的 JSONL |
| `--summary` | 是 | 对应 benchmark 的 `summary.json` |
| `--output` | 是 | 新报告输出目录 |
| `--snapshot` | 否 | 默认 `data/mock_arxiv_snapshot.json`（从 `AgenticArxiv/` 看） |
| `--task-set` | 否 | `default` 或 `expanded`，默认 expanded |
| `--split` | 否 | 进一步限制任务池 |

如果 trace 中的 task id 不属于选定任务池，命令会提示并跳过；全部跳过或
没有可重算 metrics 时退出。重评分只能改变评分口径，不能补回原 trace 没有
记录的模型输出、工具执行时间或快照结果。

## 6. Badcase 回放与捕获

入口：`eval/badcase_replay.py`，从仓库根目录运行。回放不需要 LLM、网络或工具，
它只重新执行评分器。

```bash
# 默认回放 eval/eval_cases.jsonl
python eval/badcase_replay.py

# 显式回放并打印条件
python eval/badcase_replay.py replay \
  --cases eval/eval_cases.jsonl --task-set expanded \
  --training-step 100 --verbose

# 先跑 benchmark，再从 trace 中挑坏例；capture 需要 traces JSONL
python eval/badcase_replay.py capture \
  --traces data/bench_v3/traces.jsonl \
  --cases eval/eval_cases.jsonl \
  --source bench_v3 --dry-run
```

全局参数：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--cases` | `eval/eval_cases.jsonl` | 用例库路径 |
| `--task-set` | `expanded` | 用例对应的任务池 |
| `--training-step` | `100` | 奖励课程 step；比较用例时必须固定 |
| `-v` / `--verbose` | 关闭 | 打印 reproduction 条件和说明 |

`capture` 子命令额外需要 `--traces`，可选 `--source` 和 `--dry-run`。状态
`open` 表示问题仍复现；修复后回放显示 `newly_fixed`，确认后才应把用例改成
`fixed`；一个已 fixed 用例再次出现就是 regression，入口返回 1。

## 7. SFT 数据生成

### 7.1 基础或 expanded 专家样本

入口：根目录 `scripts/generate_sft_data.py`。

```bash
python scripts/generate_sft_data.py \
  --task_set expanded \
  --split data/splits/v3_81.json:train \
  --snapshot data/mock_arxiv_snapshot.json \
  --output data/sft/sft_v3_train.jsonl
```

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--output` | `data/sft/sft_train.jsonl` | 输出 JSONL |
| `--snapshot` | `data/mock_arxiv_snapshot.json`（存在时） | 离线快照；不存在则进入 auto 环境 |
| `--use_llm` | 关闭 | 改用环境配置的 LLM 专家；默认确定性专家 |
| `--task_set` | `basic` | `basic` 或 `expanded` |
| `--split` | `None` | expanded 必须传版本化 `PATH:train` |

expanded 生成器拒绝裸 split、全量任务以及任何留出 split；确定性专家轨迹
失败或不满足严格成功也不会写入样本。

### 7.2 参数化 seed

入口：`scripts/generate_parametric_sft_data.py`。

```bash
python scripts/generate_parametric_sft_data.py \
  --split-file data/splits/v2_62.json \
  --snapshot data/mock_arxiv_snapshot.json \
  --output data/sft/sft_v2_parametric_seed.jsonl
```

| 参数 | 默认值 |
|---|---|
| `--split-file` | `data/splits/v2_62.json` |
| `--snapshot` | `data/mock_arxiv_snapshot.json` |
| `--output` | `data/sft/sft_v2_parametric_seed.jsonl` |

此脚本先校验 parent/train、held-out overlap、文本重复和工具拓扑，再在快照
replay 中执行派生任务；输入 split 和 snapshot 缺失会直接退出。

### 7.3 两类语言扩增和混合

```bash
python scripts/augment_sft_data.py \
  --input data/sft/sft_v1_seed.jsonl \
  --output data/sft/sft_v1_linguistic.jsonl \
  --split-file data/splits/v2_62.json

python scripts/augment_parametric_sft_data.py \
  --input data/sft/sft_v2_parametric_seed.jsonl \
  --split-file data/splits/v2_62.json \
  --output data/sft/sft_v2_parametric_linguistic.jsonl

python scripts/build_sft_train_mix.py \
  --original data/sft/sft_v1_linguistic.jsonl \
  --parametric data/sft/sft_v2_parametric_linguistic.jsonl \
  --output data/sft/sft_v3_train_mix.jsonl \
  --seed 42
```

扩增器会写相邻的 `.manifest.json`；混合器默认要求原始来源为 1020 行、参数化
来源为 1908 行，并拒绝 hash、manifest kind、重复 messages 或输入覆盖。

## 8. 训练入口和硬件边界

下面只列源码中最常用、会改变实验语义的选项；完整参数以对应文件的
`argparse` 为准。

### 8.1 SFT / QLoRA

从仓库根目录或能导入 `AgenticArxiv` 的环境运行：

```bash
python -m AgenticArxiv.rl.train_sft --inspect_only --max_length 4096
python -m AgenticArxiv.rl.train_sft \
  --data data/sft/sft_v3_train_mix.jsonl \
  --data_manifest data/sft/sft_v3_train_mix.jsonl.manifest.json \
  --output_dir outputs/sft_qlora
```

| 参数 | 默认值 | 备注 |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | 模型名或本地目录 |
| `--data` | 脚本推导的训练文件 | 正式 QLoRA 应传混合 JSONL |
| `--data_manifest` | 与 data 同名后缀 | 默认强制验证 |
| `--output_dir` | `outputs/sft_qlora` | 输出模型目录 |
| `--epochs` | `3` | 训练轮数 |
| `--batch_size` | `1` | 每设备 batch |
| `--grad_accum` | `8` | 梯度累积 |
| `--lr` | `1e-4` | 学习率 |
| `--max_length` | `4096` | 超长样本会先被长度体检 |
| `--max_steps` | `-1` | `-1` 表示按 epoch；可用 30 做冒烟 |
| `--qlora` / `--no-qlora` | 开启 | 默认 4-bit QLoRA |
| `--inspect_only` | 关闭 | 只审计 manifest 和 token 长度，不加载模型 |
| `--skip_data_manifest_check` | 关闭 | 临时实验开关，结果不作为正式复现 |
| `--verify` / `--no-verify` | 开启 | 训练结束阶段验证 |
| `--report_to` | `none` | `none` / `auto` / `tensorboard` / `wandb` |

QLoRA 路径会检查 CUDA、BF16 和关键包版本；`--inspect_only` 是无 GPU 贡献者
可以执行的审计路径，但不等于训练已经运行。

### 8.2 DPO

入口：`python -m AgenticArxiv.rl.train_dpo`。

| 参数 | 默认值 |
|---|---|
| `--model` | `None`，SFT 模型路径 |
| `--data` | `None`，DPO 数据集 |
| `--output_dir` | `None` |
| `--verify` / `--no-verify` | 开启 |
| `--min_reward` | `-0.3` |
| `--report_to` | `none` |
| `--run_name` | `None` |

DPO 需要本地 SFT 模型和偏好数据；没有模型或数据时不要把命令示例写成已完成训练。

### 8.3 GRPO

入口：`python -m AgenticArxiv.rl.train_grpo`。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | `outputs/dpo/final` | 起始模型 |
| `--output_dir` | `outputs/grpo` | 输出目录 |
| `--max_steps` | `-1` | 优化步数 |
| `--beta` | `0.04` | KL 系数；`--dapo` 会改为 0 |
| `--reward_curriculum_steps` | `30` | SFT 起点可显式传 0 |
| `--num_generations` | `4` | 每个 prompt 的组采样数 |
| `--max_completion_length` | `256` | completion token 上限 |
| `--loss_type` | `None` | `grpo` / `dapo` / `bnpo` / `dr_grpo` / `cispo` / `sapo` / `luspo` |
| `--epsilon_high` | `None` | clip-higher 上界 |
| `--mask_truncated_completions` | `None` | 是否过滤超长轨迹 |
| `--dapo` | 关闭 | 一组 DAPO 预设；显式单项参数优先 |
| `--qlora` / `--no-qlora` | 开启 | 单卡默认路径 |
| `--snapshot` | `None` | 多轮 rollout 需要离线快照 |
| `--task_set` | `default` | `default` 或 `expanded` |
| `--split` | `None` | 例如 `data/splits/v3_81.json:rl_train` |
| `--allow_zero_variance` | 关闭 | 跳过组内零方差检查，通常不建议 |
| `--save_rollout_traces` | 关闭 | 写 rollout 审计 JSONL |
| `--rollout_trace_path` | `None` | 传入即自动启用审计 |
| `--rollout_trace_max_samples` | `0` | 0 表示全部 |

`--split` 需要 `--task_set expanded`；`rl_train` 由 split 文件中的 rates 动态
计算。零方差组不产生 GRPO 优势，默认检查就是为了避免训练静默空转。

### 8.4 OPD 和 PPO

OPD：`python -m AgenticArxiv.rl.train_opd`。

| 参数 | 默认值 |
|---|---|
| `--model` | `outputs/sft/final` |
| `--teacher` | `Qwen/Qwen2.5-7B-Instruct` |
| `--output_dir` | `outputs/opd` |
| `--max_steps` | `-1` |
| `--temperature` | `0.9` |
| `--lmbda` | `1.0`，当前实现只支持 1.0 |
| `--beta` | reverse-KL 默认值 |
| `--max_new_tokens` | `256` |
| `--max_turns` | `1` |
| `--max_observation_tokens` | `256` |
| `--snapshot` | `None` |
| `--task_set` | `default` |
| `--canary_steps` | `50`，0 禁用 |
| `--verify` / `--no-verify` | 开启 |

PPO：`python -m AgenticArxiv.rl.train_ppo`。当前入口的主要参数是
`--model outputs/grpo/final`、`--output_dir outputs/ppo`、`--epochs 1`、
`--batch_size 4`、`--mini_batch_size 2`、`--grad_accum 1`、`--lr 1e-6`、
`--init_kl_coef 0.05`、`--max_completion_length 256`、`--temperature 0.7`、
`--snapshot`、`--canary_steps 50`、`--min_canary_reward -1.0`、`--verify`、
`--report_to` 和 `--run_name`。PPO 是实验入口，不能根据存在 CLI 就声称它
已经完成可用性验证；当前历史设计也把它列为高风险路径。

## 9. 输出与排错速查

| 症状 | 首先检查 |
|---|---|
| `replay 模式下快照缺失` | 快照路径、工具参数 key、是否需要重新录制；不要把 miss 当成网络问题 |
| `切分中有任务不在当前任务池` | 是否用了 `--task-set expanded`；绑定快照的任务是否加了 `--offline` |
| `rl_train` 为空或没有 rates | split JSON 的 `rates` 和 train 覆盖范围；不要静默回退到全部 train |
| SFT 拒绝 `source_split` | 使用显式 `FILE:train`；检查是不是混入了 dev/iid/ood |
| manifest hash/rows 不一致 | 重新生成 manifest，不要手改 hash；确认输入没有被覆盖 |
| `blocked` 任务奖励异常 | 检查 FINISH Thought 是否包含声明的 terminal reason，而不只是“完成” |
| benchmark 输出数字不可比 | 固定 split 文件、snapshot、agent、seed、repeat、backend 和 task set |
| 图表分析返回空/越界 | 先检查 T4 `extract_paper_figures` 是否有目标图，再检查 `figure_no` 和快照 backend |

## 10. 命令核对原则

本参考只抄录对应入口的 `argparse`、默认常量和调用路径。新增参数时应同时
更新：

1. `docs/cli-reference.md` 和英文版。
2. 受影响的 `docs/data-formats.md` 或 `docs/offline-replay.md`。
3. 根 README 的短导航或 quick start（如果用户会从那里进入）。
4. PR 描述中的验证状态；没有运行的命令标为“按源码核对，未在本地执行”。
