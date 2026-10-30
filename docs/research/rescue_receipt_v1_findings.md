# 契约/凭据基线结果：拒绝晋级

完整48条模型轨迹，无框架执行异常。协议在767e016推送后推理；未修改候选。

| 配置 | 严格成功 | 请求操作凭据＋FINISH |
|---|---:|---:|
| 原始模型baseline | 4/12 | 8/12 |
| 原始模型receipt_contract | 5/12 | 7/12 |
| 发布GRPO baseline | 3/12 | 3/12 |
| 发布GRPO receipt_contract | 0/12 | 0/12 |
| 仅用户文本的规则执行器 | 12/12 | 12/12 |

原始模型候选修复2题、退化3题；GRPO候选修复0题、退化3题。两者均未通过预先规定的正收益/无回归门槛，因此新promotion不向模型开放，final继续封存。规则上界12/12说明这是有限语法的工程探针，不足以证明算法创新或自主进化收益。

本轮还纠正了状态探针的解释：旧原始模型strict4/12，但有效操作凭据＋FINISH8/12；不应因额外有效调用就把它判为没完成请求。新口径在这轮推理前已声明，旧轨迹重算明确标为post-hoc。

静态提示连续失败，下一步检查可复现的训练/运行接口差异，而非继续挑提示。当前代码中的SFT样本生成器会真实执行setup，却没有在build_sft_samples_from_history中把setup可见状态加入prompt；BenchmarkRunner则会加入build_visible_setup_context。应先审计这一差异及其真实影响，再谈训练或RSI。由于历史训练JSONL缺失，不能直接断言发布权重训练时使用的完整样本就是当前代码生成版本。

原始轨迹与分析：artifacts/rescue_receipt_v1；复算scripts/analyze_rescue_contract.py。所有下载、翻译均为离线契约凭据，不证明真实PDF质量。两族六对依赖参数实例不支持显著性主张。
