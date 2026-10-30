# RSI 推进记录（2026-10-30，本轮完成）

## 本轮结论

研究继续，但尚无 RSI 方法增益；不要用当前受限代理上的失败推断方向无效。原框架开始实际复用，而非仅保留目录。

| 项目 | 完成结果 |
| --- | --- |
| 原 ToolRegistry 集成 | 独立实例，不初始化旧业务工具；历史 39 次观察、4 个终局判定、schema 和非法调用返回均一致，全局注册表未改变 |
| 原 Trajectory / JSONL 集成 | 新增无损适配层；4 条原生工具轨迹保存/读取后内容一致，不生成虚构 thought，不引入旧 shaped reward |
| AppWorld 24 步诊断 | 同两个暴露场景 0/2，均 step_budget；实际工具调用 27、25；不是 RSI 运行 |
| 诊断审计 | 52 次观察回放一致，两个最终判定一致，冻结源码摘要匹配；48 次请求均成功，无缺失计量；input 420,829、output 12,729 token |
| 计时修复 | 发现 AppWorld 冻结普通 datetime/monotonic；API 改用系统原生时钟，环境内外均通过短等待验证。历史 AppWorld 耗时/UTC 不作真实时间证据 |
| 可行性更正 | 原宽预算参考请求量 71/145；改为最多 48 次原生请求后两条都在请求上限处失败，0/2。参考开销不是最短解下界 |
| 新训练池剖析 | 按冻结顺序取前四个 meta_train/inner_train；no-op 0/4、参考 4/4，请求量 31/29/40/70；没有按结果挑题 |
| 数据边界 | 仅检查 train；未读取 meta_dev/最终任务内容。ID-only 池审计确认 28 组/84实例，成员与旧清单一致 |

24 步运行使用的是计时修复前已加载的冻结代码；不可报告真实每请求耗时或从日志推导实际限速间隔。请求数量、token 数、观察回放与终局分数不受此问题影响。tool_errors=0 只指代理/执行层异常计数，不等于无任何业务层错误。

## 研究推进决策

停止在原两个暴露场景上反复堆提示。先为已确定的四个训练实例统一冻结接口和充分计量的 API/模型响应预算，得到有成功也有失败的固定基线；保持各方法表达能力相同，参考源码和检查器不进入经验库。之后再比较固定经验改进、固定元优化、递归改进三条路径。最终评测独立锁定，不随训练环境演化。

本轮还核查了 GEPA 官方适配器、学习失败诊断和元优化管线支持，见 `rsi_strong_baseline_audit_20261030.md`。只做自动诊断或优化改进器不足以建立新颖性；候选贡献仍须用等经验/等预算对照验证。

本轮不占 GPU，无需用户补资源。旧研究保留在归档标签，原 AgenticArxiv 源码未弃用。当前尚未进入权重 RL 或参数 TTT。

## 入口

- `docs/research/RSI_INDEX.md`
- `docs/research/rsi_framework_reuse_and_status_20261030.md`
- `docs/research/rsi_appworld_feasibility_erratum_20261030.md`
- `docs/research/rsi_host_clock_erratum_20261030.md`
- `artifacts/rsi_appworld_horizon_v1/summary.json` 与 `replay_analysis.json`
