# 文档索引

本目录同时保存当前实现的操作手册和早期设计记录。阅读时先看“状态”列：
带有“现行实现”标记的文档应当与当前源码一起维护；“历史设计稿”保留
背景和决策过程，但其中的计划、伪代码和旧路径不能当作 CLI 合约。

## 从哪里开始

| 目标 | 推荐入口 | 状态 |
|---|---|---|
| 了解数据字段、切分和泄漏边界 | [`data-formats.md`](data-formats.md) / [English](data-formats.en.md) | 现行实现 |
| 查命令、工作目录、参数和输出 | [`cli-reference.md`](cli-reference.md) / [English](cli-reference.en.md) | 现行实现 |
| 跑 benchmark、理解 v1/v2/v3 | [`../AgenticArxiv/benchmark/readme.md`](../AgenticArxiv/benchmark/readme.md) | 现行实现 |
| 排查快照 miss、离线桩和 T4/T5 | [`offline-replay.md`](offline-replay.md) | 现行实现 |
| 理解奖励分量、课程和安全闸门 | [`multigranular_rl.md`](multigranular_rl.md) | 现行实现 |
| 贡献一个文档或 CPU 级改动 | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | 现行实现 |

## 当前实现文档

### 数据与评测

- [`data-formats.md`](data-formats.md)：TaskSpec、split JSON、Trajectory JSONL、SFT 样本和 manifest 的字段契约。
- [`data-formats.en.md`](data-formats.en.md)：与中文版共享 JSON 示例的英文对应版。
- [`../AgenticArxiv/benchmark/readme.md`](../AgenticArxiv/benchmark/readme.md)：81 条 expanded 任务、v3 切分、基线和报告文件。
- [`offline-replay.md`](offline-replay.md)：`MockArxivEnv` 的 `replay`、`record`、`auto` 模式以及常见失败。
- [`cli-reference.md`](cli-reference.md)：从快照构建到 SFT/GRPO/PPO 的入口参数参考。
- [`cli-reference.en.md`](cli-reference.en.md)：命令参考的英文版。

### 设计与训练

- [`multigranular_rl.md`](multigranular_rl.md)：`RewardCalculator` 当前五个加权分量、两个诊断分量和非补偿性安全闸门。
- [`rl_building.md`](rl_building.md)：改造过程的历史记录；页首的现行实现对照表优先于旧计划。
- [`metric_stats.md`](metric_stats.md)：早期指标统计方案；现行字段以 `benchmark/metrics.py` 和 `benchmark/report.py` 为准。

## 版本与工作目录约定

仓库根目录是 `AgenticArXiv-RL/`。除非命令明确写出 `cd AgenticArxiv`，
否则命令均从根目录执行。benchmark 模块的 Python 包名仍然是
`benchmark`，因此从 `AgenticArxiv/` 运行 `python -m benchmark...`；
训练数据生成脚本位于根目录的 `scripts/`，从根目录直接运行。

文档中的数字和选项来自当前源码快照。它们不是实验结果：例如 v3 的
`rates` 只覆盖已有的 36 个 train 任务，缺少 rate 的新任务不会因为出现在
`train` 中就自动进入 `rl_train`。如需报告新成功率，应重新运行相应实验并
保存模型、环境、切分和重复次数等元数据。

## 文档维护清单

新增任务、工具或训练入口时，请同步检查以下位置：

1. `AgenticArxiv/benchmark/task_spec.py` 是否仍能由 `steps` 派生两份标准答案。
2. `data/splits/*.json` 是否记录了版本、来源、rates 和泄漏政策。
3. `AgenticArxiv/benchmark/readme.md` 是否仍区分历史切分与当前切分。
4. `docs/cli-reference.md` 的工作目录、参数默认值和输出路径是否仍与 `argparse` 一致。
5. `docs/data-formats.md` 是否说明新增字段的来源与常见误用。
6. `README.md` 的导航是否能指向新的长期文档。
7. 如果改变奖励或终止语义，`docs/multigranular_rl.md`、badcase 和测试是否一并更新。

文档示例不能暗示“命令已经成功运行”或“模型已经达到某个指标”。
如果本地没有运行某条命令，应在 PR 描述中写明“按源码核对，未在本地执行”。
