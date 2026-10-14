# Jev 路由退出说明

当前受支持的运行主线只有文本 ReAct。Jev 外部路由、MCP 和 Skill CLI 已退出活动入口，
对应实现、配置示例、smoke 脚本和专属测试已删除；普通 Agent 不再读取路由配置或发起路由请求。

普通命令和评测入口见 [主线迁移说明](docs/mainline_migration.md) 及
[Benchmark 说明](AgenticArxiv/benchmark/readme.md)。Benchmark 无需选择 Agent 架构。

已有 Jev 实验报告保留在 `artifacts/` 中，属于原版本的历史记录，
不表示当前代码重新执行或支持该路由。读取旧记录时保留其原标签和已有字段，
缺少的协议或模型来源字段保持未知，不补造新结果。历史复现应使用原源码版本和原产物。

