# RSI 推进记录（2026-10-31，校准进行中）

`feat/alp` 远端分支已删除；本地不存在该分支，无工作树占用。独立远端查询确认不再存在。

## 已完成

- 原 ToolRegistry 外新增固定批量接口：最多16个独立子调用，共享128次基础尝试预算；禁止嵌套，超预算整批拒绝，子调用异常停止并保留已产生副作用。
- 五项 SQLite 状态/预算测试、AppWorld 三次观察串行/批量一致性与预算耗尽检查通过。
- instruction-level 三组改进器候选请求已实现：固定经验、固定外层、递归；首轮同起点元请求完全一致，后续 proposer 来源分开。没有把合法候选当成接受或有效改进，调度与外部接受循环尚待接入。
- 固定 upstream GEPA `80a62c879ecdeb8741faac72a844dceb1c18691f`，官方 optimize 离线管道检查通过，6次脚本化 fixture episode、1次脚本化反思、0真实LLM请求；不是方法成绩。
- GEPA 和自建候选均限制可修改文本长度12000字符，运行器支持显式模型名称。无工具反思请求省略 tools/tool_choice；未知usage重试保守计量。
- 12项单元测试通过。无GPU占用。

## 实时校准

按冻结顺序运行 pool-v2 前四个 meta_train/inner_train；模型 doubao-seed-2.0-mini。每题24个模型响应、128次基础尝试，单批最多16个显式调用；全套100次API尝试、1.5M input/0.1M output计量上限。模型不接收参考源码、检查器代码或解答，未读取meta_dev/锁定最终任务内容。

本次记录时已结束 2/4，独立终局成功 0/2；完整状态 False。未完成时不能报告四题最终成绩。不可变检查点见 `artifacts/rsi_batch_train_calibration_v1/checkpoint_20261031.json`。

已接入 `scripts/finalize_rsi_train_stage.py`：完整或终止后独立回放，验证观察/终局及源码摘要，然后生成实际结果文档：`docs/research/rsi_batch_train_calibration_findings_20261031.md`，并复制到原项目local_docs同名日期版本。运行中原始summary会继续更新，最终结果尚未宣称完成。

若完整4题仍0/4，按另一个严格上限进行中性 Qwen 接口检查：最多2次API尝试（含重试）、30k input/2048 output，总共不运行任务评测。候选ID来自已有目录 `qwen/qwen3.6-27b`；[官方模型卡](https://huggingface.co/Qwen/Qwen3.6-27B)描述27B模型，但目录/响应别名不能证明供应商实际后端权重。该连接检查费用与四题校准分开记录，也不冒称能力评测。

## 证据边界

尚无RSI方法增益，尚未运行真实GEPA优化或三组有效性对照。当前优先解决健康基线门槛，再比较递归是否超过经验复用与固定元优化。模型/harness/环境共同更新、有限权重RL和参数TTT仍是后续有条件阶段。
