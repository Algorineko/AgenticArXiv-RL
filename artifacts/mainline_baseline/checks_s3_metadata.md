# S3 元数据、信号选择与费用读取检查

检查日期：2026-10-14。范围为当前仓库源码、已有配置、切分、测试和历史记录；本记录是计算与检查证据，不是模型响应、轨迹、数据集或训练 manifest。

## 元数据契约与证据边界

- `rl/run_manifest.py` 记录请求配置、实际提供的解析配置和派生事实、源码基线、依赖版本、输入指纹、资源覆盖、请求/实际步数、停止原因和运行指纹。请求中的 `None` 单列为未设置，实际解析中的 `None` 单列为未知，不猜填训练库默认值。
- 本地模型指纹仅针对标准配置、tokenizer 文件、权重及标准 index 声明的分片流式计算，不递归扫描目录；没有可得 Hub revision 或权重时保留未知。当前仓库缺少本地模型权重，因此这部分仅源码审查，未执行模型加载或真实权重哈希集成。
- 记录工具递归排除凭据键，识别 snake/camel/大写键及 URL 凭据，不扫描环境变量，不序列化未知对象的 `repr`。只读复用 `configs/training/mainline.json` 确认全部主线 defaults/trainer_parameters 不被过滤；敏感字段排除以源码审查为依据，未建立含凭据的 fixture。
- 新数据与训练产物的目标 `training_protocol` 来自共同协议。普通执行不据当前 context 推断 checkpoint 的训练事实；`training_protocol_metadata` 必须来自真实模型 manifest 或明确声明，否则未知。
- `evaluation_protocol_relation` 比较真实模型 `training_protocol` 与实际评测 `context_version`，并核对可得运行时和模板字段。结果为 `same_protocol`、`cross_protocol` 或 `unknown`；未知模型不能默认同口径。
- 数据无实际 tokenizer 时 chat-template 哈希保持未知。训练兼容检查要求明确 runtime/context/training/prompt 字段；模型可要求实际 chat-template 哈希，已知值不等则拒绝。新 SFT 数据的协议检查不能通过跳过原 manifest 校验来绕过。

实际共同源码事实为 `react-runtime-v2` / `react-messages-v3`。`prompt_template_hash` 来自实际 `REACT_PROMPT_TEMPLATE` UTF-8 字节，检查时 SHA256 为 `80d5bd11f77feb3764041b05aea2e8e6dc1abcbb1e07ce0131fe0693ed4717cb`。chat-template 哈希代码读取实际选定模板正文，不把模板选择器名称当正文；未提供 tokenizer 的检查保持 `None`。本次未验证真实 tokenizer、模板前缀或真实模型协议集成。

## 历史记录只读检查

输入来自 `data/sft/*.manifest.json` 与 `artifacts/*/summary.json`，合计 63 个现有序列化文件。通过 `read_protocol_metadata` 核对缺失字段仍为未知，并用当前实际协议调用 `require_training_protocol` 确认未知输入不会被视为兼容。未补造新版元数据或写回原件。

其中 51 份历史评测 JSON 通过 `benchmark.report.read_report` 读取。原有 `summary_by_agent`、`summary_by_task`、`details`，以及原件已存在的费用字段保留原值；读取前后文件 SHA256 不变，缺失协议关系仍为未知。这只是历史文件读取，不代表重新执行工具、模型或完整任务，也不代表当前环境的模型效果。

## 信号选择保护

`scripts/build_grpo_signal_split.py` 保持原选择阈值与分桶逻辑，仅修复输入和输出保护：

- 候选来源只能是 `FILE:train`；拒绝重复 train 成员及与 dev/iid_test/ood_test 的交叉。
- 在读取 probe 或模型前拒绝既存输出路径和符号链接，最终使用独占创建防止竞争覆盖。
- 帮助例子明确 `--task_set expanded`，采用新协议模型/探测/选择路径，不指导覆盖历史 v6 文件；这些未来命令未执行。

使用原 `data/splits/v7_86.json` 检查：train 的原 55 个成员原样返回，dev/iid_test/ood_test 均拒绝。原 `v6_grpo_train.json:rl_train` 不能被当作新的信号选择候选来源；这不删除该历史子集，也不禁止 GRPO 显式使用它。以既存 v6/v7 文件作为输出运行拒绝分支，均在 probe/模型访问前失败，原文件哈希不变。没有生成选择文件、probe 或污染切分；交叉与重复的错误分支仅源码审查，没有新建坏切分输入。

## 未知 usage 与费用闭环源码审核

确认下列调用链保留未知而不补零：

1. `agents/agent_engine.py` 对每次请求逐字段读取实际 usage；缺报、非法计数和请求异常保持 `None`，累计时传播未知，明确报告的整数零保留为零。
2. `rl/trajectory.py` 的顶层 aggregate 可为 `None`；`create_trajectory` 只复制调用方实际 mapping。`rl/rollout.py` 传入实际 `result.token_usage`，不从预算或 steps 反推费用。
3. `benchmark.metrics.extract_metrics` 对非 mapping usage 保持未知，三个字段缺失默认 `None`；原 `TaskMetrics` 直接构造的默认零保留既有测试输入语义。共同 `project_result` 保留顶层 usage，不因评分投影丢弃。
4. 新费用聚合在任一 total 未知时，将 `avg_tokens` 与 `tokens_per_success` 标为未知；另列已知和、已知用量下界及逐字段覆盖率。Markdown 显示 `n/a` 和覆盖说明，不能将未知费用当成免费零费用；明确报告的零仍为零。
5. `rescore_traces.py` 根据 key 是否存在读取：trace 显式 `None`/非法 usage 不回落旧详情；仅缺少 trace key 时使用详情明确记录的 tokens，详情也缺失则未知。旧文件不写回。已有明确 token 值保留，原来补零的缺失值改为未知属于读取语义修正，不涉及奖励或成功判据变化。
6. 最后源码审核发现重评分报告未传入原 summary 的可得协议声明。主代理已将原 summary 和原 `training_protocol_metadata` 传给 `BenchmarkReport`，不使用当前协议 stamp。中央检查以后仅对这项小接线修正定向运行 CLI/费用回归，没有重新执行生产重评分或完整中央分组；实际新协议 summary 的动态文件集成没有现成输入，本次只核对源码传值。

费用断言复用原 `test_cost_metrics.py` 的指标实例，仅检查 usage 的未知/明确零语义，没有新增模型响应或行为 fixture。没有执行真实付费 API 请求，也没有测量真实模型费用。

## 实际检查命令与结果

以下运行有重叠，不能相加作为独立用例总数。

| 命令或检查 | 实际结果 |
| --- | --- |
| `PYTHONPATH=AgenticArxiv .venv/bin/python -m unittest AgenticArxiv.tests.test_grpo_signal_split AgenticArxiv.tests.test_splits AgenticArxiv.tests.test_cost_metrics AgenticArxiv.tests.test_reliability` | 102 项通过 |
| `PYTHONPATH=AgenticArxiv .venv/bin/python -m unittest AgenticArxiv.tests.test_trajectory AgenticArxiv.tests.test_cost_metrics AgenticArxiv.tests.test_reliability AgenticArxiv.tests.test_cli_entrypoints` | 47 项通过 |
| `/bin/sh /tmp/aa-mainline-check-o5cVGB/run_checks.sh -m pytest -q -ra --tb=short -p no:cacheprovider --basetemp=/tmp/aa-mainline-check-o5cVGB/pytest-s3-rescore-metadata AgenticArxiv/tests/test_cli_entrypoints.py AgenticArxiv/tests/test_cost_metrics.py` | 最后原 summary 协议保全小修后定向 14 项、6 subtests 通过，`git diff --check` 退出 0 |
| rollout audit / observability / selector / cost / reliability 原检查 | 68 项运行，其中 67 项通过、1 项既有跳过；原 mock 仅作接口回归 |
| `scripts/build_grpo_signal_split.py --help`、`scripts/audit_grpo_rollouts.py --help`、`python -m AgenticArxiv.benchmark.rescore_traces --help` | 通过，未进入生产流程 |
| 相关模块 `compileall`、`git diff --check` | 通过 |
| v7 原切分读取、留出候选拒绝、既存 v6/v7 输出拒绝 | 通过，文件哈希不变 |
| 63 份原元数据与 51 份原报告的只读检查 | 未补版本、未改原值、未写回 |

最初尝试 pytest 时当前 `.venv` 没有 pytest，随后使用已有 unittest 运行上述范围，没有安装依赖。真实权重/tokenizer、训练栈完整集成、SFT/GRPO 训练、数据生成、完整 86 任务执行及真实外部 API 未验证，且不是本次工程验收的新增待办。

## 冻结源码 SHA256

下面为本次证据记录时的字节哈希，供定位实现；不是历史实验版本或训练产物哈希。

| 文件 | SHA256 |
| --- | --- |
| `AgenticArxiv/rl/run_manifest.py` | `9181e8ac1ee93139ccd1d96199f2a2e0e62aff4c830cc39f235b2f809d839f89` |
| `AgenticArxiv/rl/rollout_audit.py` | `6711dba131a75279252dc85b90e5f91de5c734ddb3eb399ffdc68c2c5272d138` |
| `AgenticArxiv/rl/observability.py` | `50f140ff0b5028fa56e17ef0d1c3edb1a388579a43b04fd824190f9723f24682` |
| `AgenticArxiv/benchmark/report.py` | `460e1972f287fb9663be43aa4846febbe59d4e6c2fa6a013fe0ff5cc6ea316b4` |
| `scripts/audit_grpo_rollouts.py` | `90033e9f80ed4925eb5be6339292ee1f581c186cd96059f67167c7aaf97cbf1a` |
| `scripts/build_grpo_signal_split.py` | `9a2faa02a4db1b32efc0c4b46ca93b5eebb7fc844848d0328371d8eeff75e963` |
| `AgenticArxiv/benchmark/rescore_traces.py` | `07373ebd24ea12e391228ad73ea803c448482ad59c30614fdc5f156a218399cd` |
| `AgenticArxiv/agents/agent_engine.py` | `36b9e95f9e126565ec7f49516339adf054313c4dc14add5d92dd50f0a98b6e19` |
| `AgenticArxiv/rl/trajectory.py` | `3fd94c381a63c869a786d6930c62b9ef4851868a603a55981902c2caf8e9ee03` |
| `AgenticArxiv/rl/rollout.py` | `0856d04c5c80c8570d9cfdd3e88041f141f94595e560034a4e513ae21188499e` |
| `AgenticArxiv/benchmark/metrics.py` | `921a70fdc7e255bb94a904578561058effd790d5aafea1e5f42290c9e5fcfb92` |
