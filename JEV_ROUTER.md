# Jev 可选工具路由实验

## 1. 目的与范围

AgenticArXiv 原本由策略模型（当前实验权重为 Qwen2.5-1.5B-GRPO）同时完成两件事：

1. 从可用工具中选择下一步工具；
2. 生成该工具的参数和 ReAct thought。

本实验只把第一项交给 TypeSafe Jev。Jev 不生成任意字符串/整数参数、不执行工具，
也不参与 GRPO 梯度更新。启用优化后的 `guided` 模式时，用户已经明确给出的查询、
时间窗、数量、arXiv ID、论文序号和可选项由确定性解析器生成；含糊参数仍由 Qwen
生成；Python 环境负责最终执行。该功能仅接入普通 ReAct 推理路径，默认关闭，不改变
SFT、DPO、GRPO、PPO 或 OPD 训练主线。

```text
policy（baseline）:
task + state + all tools -> Qwen -> tool name + arguments -> environment

jev guided（实验组）:
task + state + all tools -> Jev -> selected tool
                                      |
                                      v
                    explicit args? --- yes ---> deterministic resolver
                         | no                         |
                         v                            |
                 Qwen sees selected schema only      |
                         |                            |
                         +-----------> schema/legality validator -> environment
```

## 2. 开关与 API 配置

配置模板位于 `jev_config.example.env`。将需要的条目复制到仓库根目录的
`.env.local`；该文件已被 `.gitignore` 排除，不会进入 commit。

默认行为：

```env
TOOL_ROUTER=policy
```

启用 Jev：

```env
TOOL_ROUTER=jev
TYPESAFE_API_KEY=your-key
JEV_MIN_CONFIDENCE=0.80
ROUTER_ARGUMENT_MODE=guided
```

`ROUTER_ARGUMENT_MODE=legacy` 保留最初的对照实现：Jev 只收窄 schema，所有参数仍交给
Qwen；`guided` 是当前推荐实验模式。

代码也支持在构造 `ReActAgent` 时显式注入 router。显式传入
`tool_router=None` 会强制关闭外部路由，适合测试或调用方按请求控制开关。

## 3. 运行时语义与回退

每个 ReAct turn 都重新路由，因为前一步 observation 会改变当前状态。

- `policy`：完全保持原行为，Qwen 看到全部工具并同时生成工具名和参数。
- Jev 高置信度选择工具：先确定性解析任务中可直接观察的显式参数；无法无歧义解析时，
  Qwen 只看到该工具的 schema 并生成参数。
- 所有 routed action 在执行前都经过工具名修复、schema 类型检查和通用合法性保护；
  例如 1-based 论文索引 `ref=0` 会解释原因并安全结束，不进入环境工具。
- Jev 高置信度选择 `FINISH`：直接结束本轮轨迹，不调用 Qwen。
- Jev 低于 `JEV_MIN_CONFIDENCE`：退回原始全工具 Qwen 路由。
- API 超时、TLS/连接失败、429/5xx、响应格式错误或未知工具：退回原始路由。
- Qwen 在受限 schema 下仍输出不同工具或提前 `FINISH`：再执行一次全工具 Qwen
  调用，以原策略结果为准；不会把不匹配的参数强行拼到 Jev 工具名上。

返回结果增加 `routing` 和 `timing.total_router_ms`，用于审计 Jev 是否真的被采用、
为何回退以及额外延迟。API key 不写入结果或日志。

Benchmark 会把原始 `routing.decisions` 连同 timing/token 一起写进 `traces.jsonl`，
并在 `summary.json`、`raw_data.csv` 和 `report.md` 中统计：请求数、接受率、实际采用率、
回退率及原因、首步路由准确率、平均路由延迟、输入/输出 token 和估算费用。这样网络
失败触发的 Qwen 回退不会被误记成“Jev 成功”。

## 4. 实验设计

### 4.1 研究问题

在不更换参数生成模型和工具环境的前提下，Jev 是否能提高下一步工具选择准确率？

### 4.2 对照组与实验组

| 组别 | 工具选择 | 参数生成 | 工具环境 |
| --- | --- | --- | --- |
| Baseline | Qwen2.5-1.5B-GRPO | 同一 Qwen | 同一 MockArxivEnv |
| Jev guided | Jev-latest | 显式参数确定性解析；其余同一 Qwen | 同一 MockArxivEnv |

实验仍使用同一 Qwen 和同一环境；guided 组额外测试一种明确的系统结构：让决策模型
负责离散工具选择，让代码负责可验证的显式槽位，让 Qwen 只处理剩余歧义。引入
DeepSeek 或更换执行模型会同时改变 router 和 executor，因此不属于本实验的主对照。

### 4.3 数据集

使用冻结的 `data/splits/v3_81.json` expanded benchmark，共 81 个任务：

| Split | 数量 |
| --- | ---: |
| train | 51 |
| dev | 8 |
| iid_test | 18 |
| ood_test | 4 |

路由输入只能包含任务、可用工具、历史 action/observation 和运行时会话状态；不得包含
`expected_tools`、`expected_tool_args` 或 reward。具有既有论文列表的任务应把真实候选
标题和 ID 提供给两个组，避免输入信息不对称。

### 4.4 指标

路由层主要指标：

- next-tool accuracy；
- `FINISH` / 非法请求识别率；
- 按 category 和 split 的准确率；
- confidence、低置信度 fallback 率；
- API error fallback 率；
- router latency、token 和估算费用。

完整 rollout 的最终指标：

- 完整工具序列准确率（项目现有 `tool_accuracy`）；
- argument accuracy；
- strict success rate；
- false FINISH rate；
- 平均总延迟、Qwen token 和 Jev 成本。

正式 A/B 应在相同 task、snapshot、Qwen checkpoint、seed 和 repeat 下运行。建议每题
至少重复 3 次，并对 `JEV_MIN_CONFIDENCE=0.70/0.80/0.85` 做阈值消融。

“选对工具”的判定来自冻结任务的 `TaskSpec.steps`：完整工具序列必须与
`expected_tools` 顺序完全一致、不能多调或漏调；没有期望工具的 infeasible 任务，
正确首步是 `FINISH`。参数和论文指代分别由 `expected_tool_args` 与冻结 snapshot 的
paper ID 判定，最终 `strict_success` 还要求正常结束、无解析/工具错误且终止语义正确。

## 5. 已完成的 routing-only pilot

报告：`artifacts/jev_route_expanded_all.json`。

该次 v1 实验使用 Jev-latest，对 81 条任务各调用一次，只评估首步路由：

| 指标 | 结果 |
| --- | ---: |
| next-tool accuracy | 70/81 = 86.42% |
| train | 43/51 = 84.31% |
| dev | 7/8 = 87.50% |
| iid_test | 16/18 = 88.89% |
| ood_test | 4/4 = 100% |
| 平均 confidence | 0.902 |
| 平均延迟（curl/Schannel） | 1.219 s |
| 估算总费用 | USD 0.00333 |

从已有 GRPO traces 单独抽取首步工具名后得到：

| Split | Jev（每题 1 次） | GRPO Qwen（每题 3 次） | GRPO Qwen 完整工具序列 |
| --- | ---: | ---: | ---: |
| dev | 87.50% | 62.50% | 62.50% |
| iid_test | 88.89% | 61.11% | 50.00% |
| ood_test | 100% | 100% | 58.33% |

这组数字表明 Jev 作为工具路由器值得继续验证，但不是严格的端到端因果结论：Jev
只有一次采样，且 Qwen traces 每题有三次采样。

### 5.1 错误分析

11 个错误全部集中在两个 category：

- `ref_form`：4/10。6 个标题文字指代任务被误选为关键词搜索。v1 输入只说明
  “会话有 5 篇候选论文”，没有提供真实标题，因此这部分同时暴露了实验状态构造缺陷。
- `infeasible`：0/5。越界 ref、不存在 ID、无会话指代和不支持操作没有选择
  `FINISH`，说明不可执行判定需要更明确的标准或独立 validator。

正确样本平均 confidence 为 0.933，错误样本为 0.702。在这一次样本上使用 0.80
阈值，可把 11 个错误中的 7 个交回 Qwen，同时将 70 个正确中的 5 个交回 Qwen；Jev
直接接受的 69 条决定准确率为 94.20%。这只是阈值选择依据，尚不是 cascade 的最终成功率。

### 5.2 结论边界

当前证据支持：Jev 能在该动作空间中进行有意义的工具分类，并值得作为默认关闭、
可回退的实验路由器。

当前证据不支持：Jev 已把完整任务成功率提升到 86.42%，或已经优于 Qwen 的完整
多轮 Agent。要做该结论仍需运行“Jev 选工具 + 同一 Qwen 生成参数 + 同一环境执行”
的完整 A/B rollout。

### 5.3 小型项目接线证明

`scripts/jev_project_integration_smoke.py` 不需要 GPU 或外部网络。它把 System One 的
真实响应结构确定性回放给正式 `JevToolRouter`，随后走完整项目链路：

```text
JevToolRouter -> ReActAgent -> MockArxivEnv -> extract_metrics -> BenchmarkReport
```

同一条 `search_AI_1d_3` 任务分别运行原 policy 和 Jev 路径，结果位于
`artifacts/jev_project_integration_smoke.json`：

| 检查 | 结果 |
| --- | ---: |
| policy strict success | PASS |
| Jev strict success | PASS |
| 两组完整工具序列相同 | PASS |
| policy 可见全部工具 schema | PASS |
| Jev 只向 Qwen 暴露所选工具 schema | PASS |
| observation 后重新路由 | PASS |
| Jev 两次决定均被正式采用 | PASS |
| 关闭 Jev 时外部决定数为 0 | PASS |

这项 smoke 证明接口和执行链可以互换；81 条在线 pilot 证明真实 Jev API 在项目动作空间
具有 86.42% 的首步分类准确率。两项证据结合，足以支持“默认关闭的实验路由功能可以
工作并已接入整体项目”。它仍不声称完整 Agent 成功率得到提升。

### 5.4 早期 legacy 端到端小样本 A/B

2026-10-04 使用公开 checkpoint
`Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO` 完成了 10 个预先固定任务的端到端
pilot。两组使用同一模型、离线 snapshot、seed=42，每题重复 3 次，因此每组各有 30 条
完整轨迹。原始 summary、CSV 和 trace 保存在交付证据包中；可提交的聚合结果位于
`artifacts/jev_e2e_pilot_10.json`。

| 指标 | policy baseline | Jev | 差异 |
| --- | ---: | ---: | ---: |
| 有效轨迹 / 异常 | 30 / 0 | 30 / 0 | 相同 |
| strict success | 40.00% | 40.00% | 0.00 pp |
| tool accuracy | 60.00% | 60.00% | 0.00 pp |
| argument accuracy | 50.00% | 50.00% | 0.00 pp |
| false FINISH | 40.00% | 40.00% | 0.00 pp |
| 平均 Qwen token | 3863.9 | 2723.2 | -29.52% |
| 平均总延迟 | 3374.9 ms | 7236.1 ms | +114.41% |

Jev 组共记录 57 次 router decision，其中 35 次达到接受条件、26 次实际控制动作；31 次
回退包含 22 次低 confidence 和 9 次 policy/router disagreement。首步路由准确率为 70%，
API 估算总费用为 USD 0.002718。这里的 Qwen token 不包含 Jev API 的 64,709 input 与
7,544 output token，二者必须分开报告。

这个小样本支持“真实 API 与真实 GRPO Qwen 的完整链路可互换，并在保持该 pilot 聚合
质量的同时减少本地 Qwen token”。它不支持“Jev 已提高完整 Agent 成功率”；本次质量
持平，且外部路由让平均延迟翻倍。若要评价总体质量，应扩大预先注册的代表性任务集，
而不是从本次 10 题中挑选有利案例。

### 5.5 guided 结构优化实验

legacy pilot 暴露了关键瓶颈：Jev 即使选对工具，1.5B Qwen 仍可能在受限 prompt 中输出
错误工具名或参数。因此增加 `routing/arguments.py`，把 TypeSafe 的 typed decision 与
项目中的 schema validator、显式参数解析和非法引用保护组合起来。

先用历史 trace 预选 6 条“Jev 高置信度选对、Qwen 多数失败”的 benefit slice，每题
重复 3 次。这个 slice 用来证明机制能在目标任务上产生效果，不代表总体分布：

| 指标 | policy | guided Jev |
| --- | ---: | ---: |
| 轨迹 | 18 | 18 |
| strict success | 0% | 100% |
| tool / argument / reference accuracy | 0% / 8.33% / 12.50% | 100% / 100% / 100% |
| 平均 Qwen token | 3348.7 | 0 |
| 平均总延迟 | 2852.1 ms | 3137.1 ms |

为避免只报告有利样本，又在原先固定的 mixed 10-task 集上，用相同 checkpoint、snapshot、
seed=42、repeat=3 重跑 guided 组，并与同任务的 policy baseline 比较：

| 指标 | policy baseline | guided Jev | 差异 |
| --- | ---: | ---: | ---: |
| 有效轨迹 / 异常 | 30 / 0 | 30 / 0 | 相同 |
| strict success | 40.00% | 60.00% | +20.00 pp |
| tool accuracy | 60.00% | 80.00% | +20.00 pp |
| argument accuracy | 50.00% | 70.00% | +20.00 pp |
| reference accuracy | 60.00% | 80.00% | +20.00 pp |
| false FINISH | 40.00% | 20.00% | -20.00 pp |
| 平均 Qwen token | 3864.7 | 1004.5 | -74.01% |
| 平均总延迟 | 3447.0 ms | 4720.9 ms | +36.96% |

guided 组共记录 60 次路由决定：45 次达到 Jev 置信门槛，42 次直接控制动作；15 次因
低置信度交回 Qwen，3 次高置信度错误工具被非法参数保护安全覆盖。首步路由准确率为
80%，Jev 估算费用 USD 0.003669。6 个任务达到 3/3 strict success，4 个仍是 0/3，
说明本结果是“小样本中有明确净改善”，而不是整个 81 题 benchmark 已全面提升。

## 6. 复现

仅测试 TypeSafe API 和一条路由：

```bash
python scripts/jev_route_smoke.py --transport curl --limit 1 --fresh \
  --output artifacts/jev_api_check.json
```

运行 expanded routing-only 实验：

```bash
python scripts/jev_route_smoke.py --transport curl
```

运行无需 GPU/网络的项目接线证明：

```bash
python scripts/jev_project_integration_smoke.py
```

脚本逐条 checkpoint；网络中断后重复同一命令即可继续。v2 脚本补充了真实 snapshot
候选论文上下文和更明确的 `FINISH` 判据，但受 API 网络超时影响，尚无可引用的完整
v2 结果，不能与 v1 结果混写。

运行纯离线测试：

```bash
python -m pytest \
  AgenticArxiv/tests/test_routed_arguments.py \
  AgenticArxiv/tests/test_tool_routing.py \
  AgenticArxiv/tests/test_jev_route_smoke.py \
  AgenticArxiv/tests/test_jev_project_integration_smoke.py \
  AgenticArxiv/tests/test_benchmark_routing_metrics.py -q
```

### 6.1 端到端 A/B

先在 8 条 dev 任务上各跑 1 次冒烟，再把 `--repeat` 改为 3。下面是 PowerShell 示例；
两组必须使用同一个模型目录、snapshot、split 和 seed：

```powershell
cd AgenticArxiv
$modelPath = "D:/models/Qwen2.5-1.5B-GRPO"  # 替换成实际目录
$env:TOOL_ROUTER = "policy"
python -m benchmark.run_benchmark --agents regex --task-set expanded `
  --split ../data/splits/v3_81.json:dev --repeat 1 --seed 42 `
  --offline --snapshot ../data/mock_arxiv_snapshot.json `
  --backend transformers --model $modelPath `
  --output ../artifacts/jev_ab/policy_dev --save-traces

$env:TOOL_ROUTER = "jev"
python -m benchmark.run_benchmark --agents regex --task-set expanded `
  --split ../data/splits/v3_81.json:dev --repeat 1 --seed 42 `
  --offline --snapshot ../data/mock_arxiv_snapshot.json `
  --backend transformers --model $modelPath `
  --output ../artifacts/jev_ab/jev_dev --save-traces
```

冒烟通过后，对 `dev`、`iid_test`、`ood_test` 分别跑 `repeat=3`。最终比较两组
`summary.json` 的 `strict_success_rate`、`tool_accuracy`、`arg_accuracy`、
`ref_accuracy` 和 `false_finish_rate`；Jev 组还必须报告 router use/fallback、延迟、
token 和费用。仓库里的 smoke snapshot 可用于小规模接线验证，但 IID 的正文、摘要和
图表任务需要完整 `data/mock_arxiv_snapshot.json`，不能用 smoke 结果冒充正式结论。

## 7. 代码位置

- `AgenticArxiv/routing/base.py`：provider-neutral 路由接口与决策结构；
- `AgenticArxiv/routing/jev.py`：TypeSafe Jev 适配、阈值和网络降级；
- `AgenticArxiv/routing/arguments.py`：显式参数解析、多步引用推进和非法引用保护；
- `AgenticArxiv/routing/factory.py`：环境变量开关；
- `AgenticArxiv/agents/base_agent.py`：逐 turn 路由、schema 收窄和原策略回退；
- `AgenticArxiv/agents/agent_engine.py`：只为普通 ReAct Agent 按配置启用；
- `AgenticArxiv/benchmark/metrics.py`、`report.py`：端到端路由指标与聚合报告；
- `AgenticArxiv/benchmark/run_benchmark.py`：把路由决策写入可重评分 trace；
- `scripts/jev_route_smoke.py`：routing-only pilot 与 checkpoint；
- `scripts/jev_project_integration_smoke.py`：policy/Jev 完整项目接口接线证明；
- `AgenticArxiv/tests/test_tool_routing.py`：不联网的集成行为测试；
- `AgenticArxiv/tests/test_jev_route_smoke.py`：实验驱动和断点恢复测试；
- `AgenticArxiv/tests/test_jev_project_integration_smoke.py`：端到端接口互换回归测试；
- `AgenticArxiv/tests/test_benchmark_routing_metrics.py`：trace/report 指标链测试。
