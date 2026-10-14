# 文本 ReAct 评测与历史诊断

活动评测只运行文本 ReAct，不再比较或选择其他 Agent 架构。安装方式和源码对应关系见 [架构迁移说明](architecture_migration.md)、[主线指南](rl_building.md) 与实际 [包元数据](../pyproject.toml)。API 推理和历史评分使用基础安装；本地模型推理需要 `local`，训练才需要 `train`，课程事件分析需要 `logging`。

安装后可从任意目录执行 `agentic-arxiv`，`python -m agentic_arxiv` 使用同一参数入口，不需要 `PYTHONPATH`。全局 `--workspace` 默认当前工作目录；相对输入、模型、切分、快照与输出路径相对该工作区，绝对路径不改写。以下例子显式指定已有工作区：

```bash
WORKSPACE=/absolute/path/to/AgenticArXiv-RL
```

将占位路径改为自己的工作区。运行例子需要已有模型、快照和必要元数据；本次工程重构没有执行模型评测、真实 tokenizer、训练或完整任务验证。每次输出使用独立的新目录，不能复用历史报告目录。

## 入口与正式切分

```bash
agentic-arxiv evaluate run --help

# 本地模型、86 任务定义中的开发集；这里只执行声明的 8 个 dev 任务
agentic-arxiv --workspace "$WORKSPACE" evaluate run \
  --backend transformers --model outputs/grpo/final \
  --task-set expanded --split data/splits/v7_86.json:dev \
  --offline --snapshot data/mock_arxiv_snapshot.json \
  --repeat 3 --seed 45 --save-traces \
  --output artifacts/evaluation_new/dev

# 留出集评测：阶段对照必须保持模型以外的切分、协议与参数一致
agentic-arxiv --workspace "$WORKSPACE" evaluate run \
  --backend transformers --model outputs/grpo/final \
  --task-set expanded --split data/splits/v7_86.json:iid_test \
  --offline --snapshot data/mock_arxiv_snapshot.json \
  --repeat 3 --seed 45 --save-traces \
  --output artifacts/evaluation_new/iid

# 显式 8 条冒烟任务，不代表扩展集的完整模型能力
agentic-arxiv --workspace "$WORKSPACE" evaluate run \
  --task-set default --offline --snapshot data/mock_arxiv_snapshot.json \
  --repeat 1 --output artifacts/evaluation_new/smoke
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

`tasks` 将任务依赖和 setup 编译为初始化动作，`runtime` 只执行上层提供的计划，不查询任务目录或评分标准。初始化不计入策略步骤、奖励动作或训练标签；模型只能看到过滤后的必要初始状态。标准 steps、预期论文 ID 与评分字段不进入模型消息。

`tasks/reference` 的语义目标准备是上层显式请求的独立 replay episode，复用初始化计划，但不共享普通任务的可变状态。它不是评分器，也不是模型上下文的一部分，不能在历史重评分时隐式运行。

## 参数

| 参数 | 含义 |
| --- | --- |
| `--backend api/transformers` | API 或指定本地 Hugging Face 模型；默认 API |
| `--model` | API 模型名称或本地模型目录 |
| `--task-set default/expanded` | 8 条冒烟或 86 条扩展定义 |
| `--split FILE:NAME` | 显式版本化成员选择 |
| `--task-ids ID ...`、`--tasks CATEGORY` | 小范围调试筛选，不代替正式切分 |
| `--offline`、`--snapshot PATH` | 显式快照工具回放 |
| `--repeat`、`--seed` | 重复次数和模型生成随机种子，默认分别为 3 和 42 |
| `--local-device`、`--local-dtype` | 本地后端设备和权重精度 |
| `--no-thinking` | 显式关闭支持此选项的模型思考模式 |
| `--save-traces [PATH]`、`--output` | 保存实际逐条轨迹及报告；正式运行显式选择新输出目录 |

## Rollout 与基线诊断

```bash
# 查看参数声明，不执行模型或生成轨迹
agentic-arxiv evaluate rollout --help
agentic-arxiv evaluate baselines --help
```

`evaluate rollout` 按已有任务采集轨迹，使用 `--task-id ID` 或显式 `--all`，
`--task-set basic/expanded` 默认 basic，`--output-dir` 默认 `traces/train`。
它默认离线回放快照、温度 0.1、轮数预算 6；`--no-offline` 才显式允许在线工具。
该入口的 `--model` 是已有本地模型目录，需要 `local`；未指定时使用 API，离线工具不代表离线模型。普通 `run` 和
`evaluate run` 默认在线，不能因统一命令而将这些默认值混为一谈。

`evaluate baselines` 保留既有合成策略奖励诊断，支持 `--policies`、`--random-samples`、
`--min-reference-gap`、`--min-category-gap`、`--training-step` 与可选新 `--output`。
默认扩展任务、随机种子 42、随机样本 20、两个 gap 阈值 0.3、奖励课程步数 100。
这不是模型评测；执行会产生合成策略结果，本次迁移只检查帮助与已有回归，不运行基线生成。

## 评分与报告

- `task_completed`/`finish_rate` 只表示 FINISH，不等于业务成功。
- `strict_success_rate` 同时核对结束、工具路径、参数、引用与受阻任务语义。
- `false_finish` 记录未完成却声称完成；解析错误、执行错误和预算耗尽与 FINISH 分开。
- `arg_score` 检查参数取值，`ref_score` 检查实际论文对象，而不只比较 ref 写法。
- `pass^k` 在每任务重复结果上计算；重复数不足的任务单独标记，不混入平均。
- token、耗时及成功归一化成本与评分分别报告，不能把缺失历史字段推断成零成本。

`scoring` 的奖励和指标只消费已经提供的轨迹与参考数据，不加载模型、不执行工具、不读写文件，也不补造 FINISH。`evaluation` 负责读写报告并组合共同 episode 服务，训练侧 Canary/StageVerifier 可以调用该服务，评测不导入训练回调。

当前执行协议仍是 `react-runtime-v2` / `react-messages-v3`；报告分别保存执行 `protocol_metadata`、真实来源模型的 `training_protocol_metadata` 与 `protocol_relation`，不能用当前 context 冒充模型训练版本。已知不同为 cross_protocol，缺字段为 unknown，齐全相符才是 same_protocol；跨协议或未知关系不是同口径复现。源码接线与实际检查范围见 [架构迁移说明](architecture_migration.md)。

输出目录包含 `raw_data.csv`、`report.md`、`summary.json`；异常时还会有 `errors.csv`，启用 `--save-traces` 后保存 `traces.jsonl`。新轨迹同时冻结评分时实际使用的逐步 `expected_paper_ids`，不将它们注入策略 history；未知参考不补写。这不是三架构排行榜。

## 历史记录与坏例

既有报告/轨迹保留原件。历史协议和模板字段缺失保持未知，不补写成当前版本。读取历史记录是兼容性证据，不是重新执行模型或工具。

```bash
# 只读既有坏例并重评分，不调用模型或重执行工具
agentic-arxiv --workspace "$WORKSPACE" evaluate badcases replay

# 只读历史轨迹和原 summary，输出另存，不执行 reference 或工具
agentic-arxiv --workspace "$WORKSPACE" evaluate rescore \
  --traces artifacts/run/traces.jsonl \
  --summary artifacts/run/summary.json \
  --references artifacts/run/references.json \
  --task-set expanded --split data/splits/v7_86.json:train \
  --output artifacts/evaluation_new/rescore
```

重评分输出应写新目录并声明输入和协议，不能覆盖原报告或伪装成原实验结果。`evaluate rescore` 与坏例 replay 都只读历史评分，不调用模型、环境、工具或 `tasks/reference`；没有历史协议、模板或模型参数字段时保持未知，不根据当前源码回填。

`rescore` 使用轨迹中已冻结的 `expected_paper_ids`，或 `--references` 指定的已有 JSON 对象（`task_id` 映射到与标准工具步骤对齐的论文 ID 数组；检索步骤为 `null`）。两种来源同时存在时必须一致。需要论文身份参考的旧轨迹若未保存参考，也没有外部冻结文件或任务显式参考，将在写报告前报错，明确参数、指代和严格成功指标无法重算；不会退回 `ref` 字面比较，也不会根据策略实际返回值猜答案。只含检索或无工具的任务不需要论文身份参考。

示例中的 `references.json` 必须是原评测对应的已有参考，不由重评分命令生成；新轨迹已包含参考时可省略 `--references`。不再强制要求未被读取的默认快照；显式提供 `--snapshot` 只校验路径，不执行快照或参考步骤，也不能代替冻结参考。坏例 capture 与实际 run 的区别见 [坏例指南](../eval/readme.md)。

## 已有记录的只读诊断与资产核对

```bash
agentic-arxiv audit rollouts --help
agentic-arxiv audit curriculum --help
agentic-arxiv audit assets --help

# 输入必须是已有 GRPO 轨迹；诊断 JSON 另存新路径，不覆盖训练 summary
agentic-arxiv --workspace "$WORKSPACE" audit rollouts \
  outputs/grpo/rollout_traces.jsonl --show 10 \
  --summary-out artifacts/evaluation_new/rollout_audit.json

# 输入为已有单次训练的事件目录；只打印分析，不回写事件
agentic-arxiv --workspace "$WORKSPACE" audit curriculum \
  outputs/grpo/logs/EXISTING_RUN --curriculum-steps 30

# 默认为只读，对比完整规范化资产与原冻结基线
agentic-arxiv --workspace "$WORKSPACE" audit assets
```

`audit rollouts` 不采样、不训练；`audit curriculum` 延迟加载 TensorBoard，需要
`pip install 'agentic-arxiv[logging]'`，课程窗口须与原训练配置一致，不能用跨窗口总奖励
替代分量和权重分析。`EXISTING_RUN` 是已有事件目录的占位名称。

`audit assets` 对比全部 10 工具的 schema 与顺序、86 扩展/8 冒烟任务和工作区已有历史切分的内容及哈希，不只是数量。
`--dump` 只打印当前清单；`--baseline` 可显式指定基线；`--freeze` 是独立的新建能力，拒绝覆盖已有文件，不能用它更新 `artifacts/mainline_baseline/assets.json` 或掩盖差异。
上述检查不改研究资产；新诊断只写用户指定的新输出，不回填历史来源指纹。

## 历史切分注记

以下成员数只描述旧切分，不是当前正式默认：`v2_62.json` 为 train 36、dev 8、iid_test 14、ood_test 4，旧 rates 选出 rl_train 6；`v1.json` 保存 59 条任务（train 42、iid 13、ood 4、rl_train 13）。这些原文件不回写。原三架构设计及示意统计见 [历史指标文档](metric_stats.md)，旧工程记录 [mainline_migration.md](mainline_migration.md) 保持只读，不是当前命令指南。

## 后续数据与模型对照

生成、扩增、混合、SFT、GRPO 及未来评测命令集中在 [主线指南](rl_building.md)。留出集不得参与数据生成或信号子集选择。当前工程检查没有生成数据、运行训练、完整 86 任务评测或真实 tokenizer 集成；缺少对应资产不要求本次补齐，也不标记为已验证。
