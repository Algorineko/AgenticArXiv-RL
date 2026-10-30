# 强固定改进器与新颖性边界核查（2026-10-30）

本轮核查的是官方实现能力，不是本地对照实验结果。GEPA 当前 upstream HEAD 记录为 `80a62c879ecdeb8741faac72a844dceb1c18691f`；尚未安装或运行，不能声称已复现。

## 对照必须足够强

[官方适配器说明](https://gepa-ai.github.io/gepa/guides/adapters/)允许把自定义系统的执行评分与反思数据构造接入优化。现有 AppWorld 原生工具轨迹、外部评分器可以作为适配边界；仍需接入统一预算账本，不能只用 metric-call 次数代替 token/API 成本。

[官方 optimize API](https://gepa-ai.github.io/gepa/api/core/optimize/)具有候选选择、验证策略与停止控制。正式对照必须固定版本和搜索配置，给相同 solver 候选空间、经验及总预算，不以手写一次提示代替 GEPA。

[2026-10-06 官方失败诊断文章](https://gepa-ai.github.io/gepa/blog/2026/10/06/named-failure-modes/)已讨论从轨迹学习诊断分类并增强反思。因此“自动分析失败再改提示”不能单独作为新贡献。诊断构建和后续使用成本也应纳入对照。

[2026-07-22 官方元优化管线文章](https://gepa-ai.github.io/gepa/blog/2026/07/22/optimize-anything-omni/)直接涉及组合优化器；仅仅“用优化器优化优化器”不能建立新颖性。该资料属于官方工程说明，不等于独立同行评审证据。

## 保留的可证伪问题

以相同起点、共享经验、相同访问权限和总预算，对比：固定 improver＋更新经验；固定外层优化器修改 improver；真正由更新后的 improver 产生后续 improver。另做强固定 solver 优化基线。只有递归路径仍产生额外可迁移收益，且预算分配/成本分析给出可验证的新机制，才继续扩大方法论文主张。

这仍是候选问题，不是已经证实的文献空白。当前两个暴露场景上的手工提示对照只用于诊断执行瓶颈，不能替代上述机制对照。
