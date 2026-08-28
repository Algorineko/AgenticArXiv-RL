# 坏例回放

把一次失败固化成一条永久的回归用例。

一次 benchmark 跑完，失败的轨迹只留在报告的聚合数字里。下次改了打分口径
或者改了 prompt，没有任何东西告诉你「上次那个查完缓存就 FINISH 的毛病到底
修没修好」——只能重跑一遍全量，再从均值里猜。

这里把单条轨迹连同它当时的判定一起冻住。**回放不需要 LLM、不需要网络、
不需要工具**：轨迹是死的，重新跑的只有打分器。所以它能进 CI，能回答一个很窄
但很确定的问题——同一条轨迹，今天的代码还认为它是坏的吗。

```bash
# 回放全部用例；有 regression 时退出码 1
python eval/badcase_replay.py

# 带上每条的判定条件和说明
python eval/badcase_replay.py -v

# 从一次 benchmark 跑的轨迹里挑坏例，追加进用例库
python -m benchmark.run_benchmark --task-set expanded --offline --save-traces
python eval/badcase_replay.py capture --traces data/traces.jsonl --dry-run
```

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
那样的用例没人会留着。可用字段见 `benchmark/badcases.py` 的 `VERDICT_FIELDS`，
写错字段名会直接报错——否则条件永远不成立，用例静默变成摆设。

标量按相等比，`{"ge": 0.6}` 这类按比较算子比。后者是给 reward hacking 用的：
「这种行为不许拿到 0.6 分」是阈值断言，写死等于某个数会在任何一次调权后失效。

`captured` 不参与判断，只用来显示奖励漂移了多少。

## 现有用例

同时也是 reward hacking 案例库——退化策略的轨迹（不解任务只骗分）配上
阈值断言，就是一条「这种行为不许拿到 X 分」的永久断言。#40 与 #47 修的
两个洞都属于这一类，现在有用例守着。

| case | 是什么 | 现在 |
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

## 目前只有 fixed 用例，为什么

`open` 用例要从真实模型跑出来的轨迹里捞，那需要 LLM。仓库里这批种子用例
全部来自确定性退化策略（`benchmark/baselines.py`），不需要模型也就跑得动，
但它们考的是**打分器**而不是模型——策略本来就该失败，失败不是 bug。

真实模型的 `open` 用例走 `--save-traces` + `capture` 那条路补进来。
