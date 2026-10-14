# Benchmark：文本 ReAct 评测

活动评测只运行 ReAct，不再比较或选择其他 Agent 架构。以下命令均从仓库根目录执行；运行例子需要已有模型、快照和必要元数据，本次工程重构没有执行这些模型评测。

## 入口与正式切分

```bash
python -m AgenticArxiv.benchmark.run_benchmark --help

# 本地模型、86 任务定义中的开发集；这里只执行声明的 8 个 dev 任务
python -m AgenticArxiv.benchmark.run_benchmark \
  --backend transformers --model outputs/grpo/final \
  --task-set expanded --split data/splits/v7_86.json:dev \
  --offline --snapshot data/mock_arxiv_snapshot.json \
  --repeat 3 --seed 45 --save-traces \
  --output artifacts/eval_react_messages_dev

# 留出集评测：阶段对照必须保持模型以外的切分、协议与参数一致
python -m AgenticArxiv.benchmark.run_benchmark \
  --backend transformers --model outputs/grpo/final \
  --task-set expanded --split data/splits/v7_86.json:iid_test \
  --offline --snapshot data/mock_arxiv_snapshot.json \
  --repeat 3 --seed 45 --save-traces \
  --output artifacts/eval_react_messages_iid

# 显式 8 条冒烟任务，不代表扩展集的完整模型能力
python -m AgenticArxiv.benchmark.run_benchmark \
  --task-set default --offline --snapshot data/mock_arxiv_snapshot.json \
  --repeat 1 --output artifacts/eval_react_messages_smoke
```

最后一条默认使用 API 后端，须先配置 LLM API；`--offline` 只控制工具回放，不表示模型也离线。指定 `--backend transformers --model PATH` 才使用本地模型，模型缺失不能改走 API。快照缺失明确失败，不自动补请求。

`--task-set expanded` 只选择 86 条定义，正式对照仍要写明 `FILE:NAME`。不要使用裸 `--split iid_test`：它沿用历史默认 `v1.json`，不是 `v7_86.json`。

| 切分 | 条数 | 使用边界 |
| --- | ---: | --- |
| `v7_86.json:train` | 55 | 正式训练候选范围 |
| `v7_86.json:dev` | 8 | 开发与调试，不用于训练 |
| `v7_86.json:iid_test` | 19 | 同模板留出实例 |
| `v7_86.json:ood_test` | 4 | 未见模板/形态留出 |
| `v6_grpo_train.json:rl_train` | 33 | 历史信号子集，来源为旧训练切分，不是 v7 train |

任务初始化及依赖执行不计入策略步骤、奖励动作或训练标签。标准 steps、预期论文 ID 与评分字段只在评测侧；模型只能看到过滤后的必要初始状态。

## 参数

| 参数 | 含义 |
| --- | --- |
| `--backend api/transformers` | API 或指定本地 Hugging Face 模型；默认 API |
| `--model` | API 模型名称或本地模型目录 |
| `--task-set default/expanded` | 8 条冒烟或 86 条扩展定义 |
| `--split FILE:NAME` | 显式版本化成员选择 |
| `--task-ids ID ...`、`--tasks CATEGORY` | 小范围调试筛选，不代替正式切分 |
| `--offline`、`--snapshot PATH` | 显式快照工具回放 |
| `--repeat`、`--seed` | 重复次数和模型生成随机种子 |
| `--local-device`、`--local-dtype` | 本地后端设备和权重精度 |
| `--no-thinking` | 显式关闭支持此选项的模型思考模式 |
| `--save-traces [PATH]`、`--output` | 保存实际逐条轨迹及报告 |

## 评分与报告

- `task_completed`/`finish_rate` 只表示 FINISH，不等于业务成功。
- `strict_success_rate` 同时核对结束、工具路径、参数、引用与受阻任务语义。
- `false_finish` 记录未完成却声称完成；解析错误、执行错误和预算耗尽与 FINISH 分开。
- `arg_score` 检查参数取值，`ref_score` 检查实际论文对象，而不只比较 ref 写法。
- `pass^k` 在每任务重复结果上计算；重复数不足的任务单独标记，不混入平均。
- token、耗时及成功归一化成本与评分分别报告，不能把缺失历史字段推断成零成本。

`RewardCalculator`、benchmark、Canary 与 StageVerifier 消费实际多轮轨迹；奖励读取不重复执行策略工具，也不补造 FINISH。当前执行协议是 `react-runtime-v2` / `react-messages-v3`；报告分别保存执行 `protocol_metadata`、真实来源模型的 `training_protocol_metadata` 与 `protocol_relation`，不能用当前 context 冒充模型训练版本。已知不同为 cross_protocol，缺字段为 unknown，齐全相符才是 same_protocol；跨协议或未知关系不是同口径复现。源码接线与实际检查范围见迁移记录。

输出目录包含 `raw_data.csv`、`report.md`、`summary.json`；异常时还会有 `errors.csv`，启用 `--save-traces` 后保存 `traces.jsonl`。这不是三架构排行榜。

## 历史记录与坏例

既有报告/轨迹保留原件。历史协议和模板字段缺失保持未知，不补写成当前版本。读取历史记录是兼容性证据，不是重新执行模型或工具。

```bash
# 只读既有坏例并重评分，不调用模型或重执行工具
python eval/badcase_replay.py

# 以下重评分需要既有快照供标准步骤 oracle 使用；不执行原 Agent 轨迹
python -m AgenticArxiv.benchmark.rescore_traces \
  --traces artifacts/run/traces.jsonl \
  --summary artifacts/run/summary.json \
  --snapshot data/mock_arxiv_snapshot.json \
  --task-set expanded --split data/splits/v7_86.json:train \
  --output artifacts/run_rescore_react_messages
```

重评分输出应写新目录并声明输入和协议，不能覆盖原报告或伪装成原实验结果。`rescore_traces` 的标准步骤 oracle 在快照环境中执行，与坏例库仅重跑评分器不同。源码审查、已有测试和历史读取的范围见 [迁移记录](../../docs/mainline_migration.md)。

## 历史切分注记

以下成员数只描述旧切分，不是当前正式默认：`v2_62.json` 为 train 36、dev 8、iid_test 14、ood_test 4，旧 rates 选出 rl_train 6；`v1.json` 保存 59 条任务（train 42、iid 13、ood 4、rl_train 13）。这些原文件不回写。原三架构设计及示意统计见 [历史指标文档](../../docs/metric_stats.md)。

## 后续数据与模型对照

生成、扩增、混合、SFT、GRPO 及未来评测命令集中在 [主线指南](../../docs/rl_building.md)。留出集不得参与数据生成或信号子集选择。当前工程检查没有生成数据、运行训练、完整 86 任务评测或真实 tokenizer 集成；缺少对应资产不要求本次补齐，也不标记为已验证。
