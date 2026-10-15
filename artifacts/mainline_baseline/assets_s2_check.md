# S2 资产定义核对

检查日期：2026-10-14（Asia/Shanghai）。

本记录仅计算当前仓库定义与 S0 冻结清单的差异，不新增行为输入、不执行工具、模型、数据生成或完整任务回放。原 `assets.json` 未覆盖，未执行 `--freeze`。

## 输入与规范化

- 原基线：`artifacts/mainline_baseline/assets.json`，源码起点 `ad0c067e8c5584b32bd7d697cb38f299bdbddf2e`。
- 规范化：UTF-8 JSON、键排序、无额外空白、保留数组顺序、禁止 NaN。
- 工具清单仅剔除 `parameters.properties` 与 `required` 中的框架 `session_id`；其他业务 schema 原样核对。剔除依据是变更 `design.md` 的执行边界约定及原 `agents/base_agent.py` 仅向复制后的执行参数注入会话 ID 的实现。
- 定义来源：`tools/bootstrap.py` 与注册工具 schema、`benchmark/task_spec.py`、`benchmark/tasks.py`、`benchmark/tasks_expanded.py`、`data/splits/*.json`。
- 资产总规范化 SHA256 保持 `751b45441cb88cc59b92d2a26e2e6f4633a76b30e34339653169bc163a41cd12`。

## 实际命令与结果

| 命令 | 结果 |
| --- | --- |
| `.venv/bin/python scripts/freeze_mainline_assets.py` | 退出码 0；主线业务资产与冻结基线一致 |
| `.venv/bin/python -m unittest discover -s AgenticArxiv/tests -p test_mainline_assets.py -v` | 退出码 0；8 项通过 |
| `PYTHONPATH=AgenticArxiv .venv/bin/python -m unittest discover -s AgenticArxiv/tests -p test_splits.py` | 退出码 0；68 项通过 |
| `git diff -- AgenticArxiv/tools/bootstrap.py AgenticArxiv/tools/tool_registry.py AgenticArxiv/benchmark/task_spec.py AgenticArxiv/benchmark/tasks.py AgenticArxiv/benchmark/tasks_expanded.py data/splits` | 退出码 0；输出为空 |
| `git diff --check` | 退出码 0；输出为空 |

## 覆盖与结论边界

10 工具名称、业务 schema 和注册顺序，86 条完整 TaskSpec 与展开任务，8 条冒烟 TaskSpec 与展开任务，以及全部 9 份历史切分定义和原文件字节哈希均与 S0 一致。`v6_grpo_train.json` 的 33 条 `rl_train` 只来自 `v3_81.json:train`，与原留出集无交叉；`v7_86.json` 的 55/8/19/4 分区完整覆盖 86 条任务、成员唯一且无交叉污染，保留 v3 原分区成员。

这些结果证明定义和切分成员保全，不证明当前代码实际执行了全部 86 条任务，也不代表真实在线工具、tokenizer、训练或模型能力已验证。这些动态运行未执行。
