# 最终资产边界检查

日期：2026-10-14，Asia/Shanghai。只读核对当前源码与 S0 原资产，不生成输入或运行任务。

## 实际检查

```sh
/bin/sh /tmp/aa-mainline-check-o5cVGB/run_checks.sh scripts/freeze_mainline_assets.py
git diff --exit-code -- AgenticArxiv/tools/bootstrap.py AgenticArxiv/tools/tool_registry.py \
  AgenticArxiv/benchmark/tasks.py AgenticArxiv/benchmark/tasks_expanded.py \
  AgenticArxiv/benchmark/task_spec.py data/splits
git diff --name-status -- data artifacts outputs
```

三项退出均为 0。第一项输出「主线业务资产与冻结基线一致」，后两项无输出。未使用 `--freeze`，没有覆盖基线。环境沿用 [`checks_s2.md`](checks_s2.md) 的隔离、离线和网络阻断 wrapper。

- 10 工具业务 schema 与注册顺序、86 条完整 TaskSpec、8 条冒烟任务和 9 份切分规范化一致。
- 总 SHA256 仍为 `751b45441cb88cc59b92d2a26e2e6f4633a76b30e34339653169bc163a41cd12`。
- `v6:rl_train` 33 条和 `v7` 的 55/8/19/4 成员与来源保持，原任务/切分定义无 diff。
- 现存 tracked `data/artifacts/outputs` 未修改或删除。另一次独立只读边界审核核对 299 个历史文件 SHA256 与 S0 相等，工具公开参数与注册默认值静态差异为空，唯一删除 fixture 为授权退出的 Jev 专属输入。
- 起始用户未提交的资产脚本、测试和旧审计材料保留；旧合成 `behavior.json` 和其测试未执行、扩充或作为验收证据。

## 结论限制

这是定义、成员与文件保全检查，不是完整 86 任务执行或真实工具能力验证。未新建论文、快照、响应、token、轨迹或数据集。已有 logger 覆盖写入风险及起始检查可能截断 ignored 日志的事实仍按迁移文档披露，不能用研究资产哈希替代日志保全结论。
