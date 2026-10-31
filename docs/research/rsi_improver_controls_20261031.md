# 三组改进器控制与 GEPA 接入（2026-10-31）

## 实际实现范围

`AgenticArxiv/rsi/improver.py` 实现 instruction-level 候选生成请求和合法性检查。只允许提交目标指令字符串，不执行模型代码，不修改工具、预算或外部检查器；尚未实现完整的调度/接受循环，也尚未运行递归效果对照。合法候选不等于有效或接受的改进。

| 组别 | 修改 solver 时使用谁 | 修改 improver 时使用谁 |
| --- | --- | --- |
| fixed_experience | 固定初始 M0，经验可以更新 | 禁止修改 M |
| fixed_outer | 当前 M_t | 固定 M0 作为外层修改策略 |
| recursive | 当前 M_t | 当前 M_t 自己产生下一版候选 |

同一 solver、目标 M_t、经验输入下，fixed_outer 与 recursive 的元更新仅 proposer policy 不同；输入经验摘要应一致。初始 M_t=M0 时两组首个元更新请求完全相同，所以一次改写本身不能区分递归与固定外层优化；至少需要后续有效代际/移植评估，并估计 API 噪声。

请求谱系记录 parent/proposer/seed/active/solver/experience/request 摘要。候选解析拒绝截断、空白和多个提交；no-op 保留并标记。不能以 API 调用成功、语法合法、元指令变长或自评更好作为接受规则。未来接受须依赖外部任务表现，计入 meta 生成、候选 solver 生成和验证费用。

模型参数仍固定；这不是权重 RL 或参数 TTT。固定外层也允许使用学到的 M_t 改 solver，不能把对照削弱成始终只用 M0 做所有事。

## GEPA 强固定基线

官方源码固定 `80a62c879ecdeb8741faac72a844dceb1c18691f`，位于 repo 外 cache。未修改 upstream core。`AgenticArxiv/rsi/gepa_adapter.py` 对接官方 EvaluationBatch，通过调用方提供的 bounded episode callback 使用同一外部评分器，并预先检查完整 batch 的显式 train/selection ID allowlist。只允许修改 solver_harness。

`BudgetedReflectionLM` 可将官方反思提示通过同一个 API 计量对象调用；solver 与 reflection 必须共享预算账本。API 客户端对无工具的文本反思省略 tools/tool_choice 字段，工具任务请求格式保留。不能仅以 GEPA metric-call 上限替代 API/token 总预算。

实际完成的是离线接入检查：官方 optimize 执行了 6 次合成 fixture episode、1 次脚本化反思；检验候选搜索、独立 SQLite 终局评分、metric-call 计数与 allowlist 拒绝。没有调用真实 LLM，train/val 在这个测试里刻意使用同一 toy fixture，只验证管道，绝不是 GEPA/RSI 的成绩。机器摘要：`artifacts/rsi_gepa_adapter_v1/offline_smoke.json`。

正式任务优化尚未启动，需等固定批量接口训练标定完成。GEPA 的官方反思模板/候选选择等正式配置仍需单独冻结；其强度与最近前作对照不能由一次手写提示替代。

## 验证

`python3 -m unittest discover -s AgenticArxiv/tests -p 'test_rsi_*.py'`：12 项通过，覆盖 batch 的状态/预算边界、三组 proposer 区别、同起点请求一致性、无效/no-op 候选，以及无工具请求格式与未知用量重试计量。无 GPU，无真实 API 请求。

官方来源：https://github.com/gepa-ai/gepa/tree/80a62c879ecdeb8741faac72a844dceb1c18691f ；https://gepa-ai.github.io/gepa/guides/adapters/
