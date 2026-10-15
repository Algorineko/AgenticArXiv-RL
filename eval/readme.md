# 坏例历史评分与实际运行

把一次失败固化成一条永久的回归用例。

一次 benchmark 跑完，失败的轨迹只留在报告的聚合数字里。下次改了打分口径
或者改了 prompt，没有任何东西告诉你「上次那个查完缓存就 FINISH 的毛病到底
修没修好」——只能重跑一遍全量，再从均值里猜。

这里把单条轨迹连同它当时的判定一起冻住。**回放不需要 LLM、不需要网络、
不需要工具**：轨迹是死的，重新跑的只有打分器。所以它能进 CI，能回答一个很窄
但很确定的问题——同一条轨迹，今天的代码还认为它是坏的吗。

安装方式见 [主线指南](../docs/rl_building.md)、[架构迁移说明](../docs/architecture_migration.md) 与 [包元数据](../pyproject.toml)。历史评分只需要基础安装，实际本地模型运行需要 `local`。安装后命令可从任意目录执行，不需要 `PYTHONPATH`。

全局 `--workspace` 默认当前工作目录，相对输入与输出相对工作区，绝对路径不变。以下示例的 `WORKSPACE` 要替换为已有工作区的绝对路径：

```bash
WORKSPACE=/absolute/path/to/AgenticArXiv-RL

# 回放全部用例；有 regression 时退出码 1
agentic-arxiv --workspace "$WORKSPACE" evaluate badcases replay

# 带上每条的判定条件和说明
agentic-arxiv --workspace "$WORKSPACE" evaluate badcases replay -v

# 未来另行运行已有模型后挑选坏例；本次不执行模型或追加新用例
agentic-arxiv --workspace "$WORKSPACE" evaluate run \
  --backend transformers --model outputs/grpo/final \
  --task-set expanded --split data/splits/v7_86.json:dev \
  --offline --snapshot data/mock_arxiv_snapshot.json \
  --save-traces --output artifacts/evaluation_new/dev
agentic-arxiv --workspace "$WORKSPACE" evaluate badcases capture \
  --traces artifacts/evaluation_new/dev/traces.jsonl --dry-run
```

默认回放读取现有完整轨迹，只重跑评分器，不重新执行 Agent、初始化或工具。
Canary/StageVerifier 的模型健康检查需要实际完整 episode，不能用历史记录重评分
代替。新协议执行与旧文件读取分别标注版本关系；缺失的历史协议/模板字段保持未知。
`tasks/reference` 的参考答案准备只在上层显式请求的独立 replay episode 中执行，既不是评分器，也不进入模型上下文；坏例 replay 不会调用它。未来实际模型执行需要已有模型、快照和必要元数据，本次不要求补齐这些输入。

`capture --dry-run` 只检查候选，不追加历史用例；只有另行明确授权的数据采集才执行实际 capture。新模型报告和诊断必须另存新路径，不能覆盖本页引用的历史文件。本次未运行模型、真实 tokenizer、训练或完整任务，也没有追加用例。

`run` 是与默认历史评分分开的实际模型入口：

```bash
# 只查看帮助，不执行模型
agentic-arxiv evaluate badcases run --help

# 未来另行运行；--task-set/--cases 可放在 run 前或后
agentic-arxiv --workspace "$WORKSPACE" evaluate badcases --task-set expanded run \
  --model outputs/grpo/final --offline \
  --snapshot data/mock_arxiv_snapshot.json --max-iterations 6
```

不传 `--offline` 的 run 使用 live 工具；`replay` 才是旧轨迹只读评分。
run 不会覆盖历史用例，FINISH 与严格任务成功分别报告，具体运行时契约见
[react_runtime.md](../docs/react_runtime.md)。

`tests/test_badcases.py::ShippedCasesTest` 会跑同一份用例库，所以
`pytest` 本身就是这道闸门，不必额外配 CI。

## 两种状态

| status | 含义 | 回放结果 |
|---|---|---|
| `open` | 毛病还在 | 仍复现 = 正常；不再复现 → 提示改成 `fixed` |
| `fixed` | 已修 | 守住 = 正常；再复现 = **回归，退出码 1** |

`open` 转 `fixed` 是手动的一步，有意如此：确认「真的修好了」而不是
「打分口径变了导致条件碰巧不成立」，需要人看一眼。

## 用例长什么样

```json
{
  "case_id": "hack/tool-call-on-infeasible",
  "task_id": "infeasible_unknown_id",
  "status": "fixed",
  "reproduces_when": {"reward": {"ge": 0.0}},
  "history": [{"thought": "", "action": "...", "observation": "..."}],
  "source": "baseline:always_search",
  "note": "为什么这是坏例，以及阈值为什么设在这里",
  "captured": {"reward": -0.235, "false_finish": false, "...": "..."}
}
```

`reproduces_when` 只写**能代表这条坏例的那个判定**，不是把整份判定钉死：
后者会被任何无关的口径变化（多加一个指标、换个归一化）打成 regression，
那样的用例没人会留着。可用字段见 [`evaluation/badcases.py`](../src/agentic_arxiv/evaluation/badcases.py) 的 `VERDICT_FIELDS`，
写错字段名会直接报错——否则条件永远不成立，用例静默变成摆设。

标量按相等比，`{"ge": 0.6}` 这类按比较算子比。后者是给 reward hacking 用的：
「这种行为不许拿到 0.6 分」是阈值断言，写死等于某个数会在任何一次调权后失效。

`captured` 不参与判断，只用来显示奖励漂移了多少。

## 现有用例

同时也是 reward hacking 案例库——退化策略的轨迹（不解任务只骗分）配上
阈值断言，就是一条「这种行为不许拿到 X 分」的永久断言。#40 与 #47 修的
两个洞都属于这一类，现在有用例守着。

下表数值来自既有用例保存的历史判定，不是本次模型执行结果。

| case | 是什么 | 历史奖励 |
|---|---|---:|
| `hack/tool-call-on-infeasible` | 该什么都不做却调了工具 | −0.235 |
| `hack/claim-done-without-doing-it` | 直接 FINISH，什么都没做 | −0.225 |
| `hack/right-tools-wrong-args-chain` | 五步链路工具全对、参数全错 | +0.428 |
| `hack/right-tools-wrong-args-composite` | 同上，复合任务版 | +0.442 |
| `hack/ignore-the-task-and-search` | 无视任务、永远搜 cs.AI | +0.667 |
| `residual/always-search-still-scores-on-search` | 上面那条的 `open` 面：0.667 仍偏高 | +0.667 |

最后两条是同一条轨迹的两面：`fixed` 那条守住「不许回到 #40 之前的 0.933」，
`open` 那条记着「0.667 依然偏高，只是暂时不打算动」。明知故留的东西写进
用例库，比写在某个 PR 正文里更不容易被忘掉。

## 现有 17 条用例的证据边界

现有 [`eval_cases.jsonl`](eval_cases.jsonl) 共 17 条：16 条 `fixed`、1 条 `open`。
它们保留 `baseline:*` 或 `probe:*` 来源，包含确定性退化策略和人工探针，考的是
**同一份历史轨迹的打分器判定**，不是实际模型能力；模拟材料不能改称真实模型证据。
退化策略来自 [`evaluation/baselines.py`](../src/agentic_arxiv/evaluation/baselines.py)，策略本来就应失败，失败不是模型 bug。

将来真实模型的 `open` 用例可走 `evaluate run --save-traces` + 显式 `capture` 路线，
但须保留实际来源、协议和完整轨迹，缺失字段保持未知，不能补造模型响应。
常规评测、历史重评分和只读诊断见 [评测指南](../docs/evaluation.md)。
