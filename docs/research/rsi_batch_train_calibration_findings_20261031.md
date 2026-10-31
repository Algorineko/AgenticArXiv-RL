# 固定批量接口训练校准：实际结果（2026-10-31）

完成状态：True；已结束 4/4 题，独立终局成功 0/4，正常 final 且成功 0。
这是四个预定 meta_train/inner_train 的开发校准，不是最终测试或 RSI 方法实验。

| 训练实例摘要前缀 | 成功 | 停止原因 | 模型响应 | 顶层工具调用 | 基础调用尝试 |
| --- | --- | --- | --- | --- | --- |
| 3193bf4191ec | False | step_budget | 24 | 24 | 24 |
| 1fed9164bbc8 | False | assistant_final | 17 | 16 | 15 |
| 09652b8caabc | False | step_budget | 24 | 24 | 46 |
| 1d6db95bcc62 | False | generation_truncated | 13 | 12 | 19 |

API 请求：78；计量 input：662831，output：31898；缺失提供方用量：0 次。缺失时采用预算预留估计，不能冒称实际账单。
停止原因：{'step_budget': 2, 'assistant_final': 1, 'generation_truncated': 1}。实际分发 batch_tools 11 次，其已执行子调用 39 次；不是把批量免费计作一次基础操作。
独立回放 76 条顶层观察，观察与最终判定均一致。源码摘要核验通过。

未读取 meta_dev/锁定最终任务内容，未给模型参考源码或解答，没有使用 GPU。

推进决策：0/4，未通过健康基线门槛；不启动三组效果对照，继续诊断模型/工具使用与响应预算。不能据此否定 RSI。

独立中性 Qwen 连通性检查：status=api_failure，native_tool_call_valid=False，usage={'calls': 2, 'accounted_input_tokens': 9008, 'accounted_output_tokens': 2048, 'calls_without_provider_usage': 2}。此费用独立于上述四题校准，未运行 Qwen 任务评测；模型别名不证明实际后端权重身份。

结构诊断补充：其中3/4个开发实例至少一次在通用批处理中使用未注册的子工具名；第二题已确认把原生业务 API 名称填作子工具名；另有一题出现单批超16项。第4题停止于生成截断。该计数不包含所有业务层 API 错误，不证明修正接口后任务一定可解。详见 `artifacts/rsi_batch_train_calibration_v1/structural_diagnosis.json`。

Qwen 中性探针两次均收到 HTTP 502，缺失usage按保守上限计入；不能把目录项当作可用服务或推断永久下线。原始中性探针请求/错误日志已移到私有 cache，公开摘要只保留状态、计量与摘要。
