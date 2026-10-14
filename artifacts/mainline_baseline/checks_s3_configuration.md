# S3 源码与配置检查

检查日期：2026-10-14。源码 HEAD：`ad676bfb6dd51d0763b37e7fa081e663fc3af060`；实际未提交源文件哈希见同目录 `configuration_s3.json`，不能只用 HEAD 代替执行代码身份。

## 已执行

- `python3 scripts/audit_mainline_configuration.py`：退出码 0，161 项通过，0 失败，0 未能静态解析项。只读 AST/JSON，不导入训练器。
- SFT/GRPO 使用共同 `react-runtime-v2` / `react-messages-v3` 协议与加载保护；SFT 检查数据 manifest、逐行协议和消息形状。
- SFT 的 `messages[:-1]` 留在 prompt，`messages[-1:]` 是唯一目标 assistant；completion 必须恰好一条，`completion_only_loss=True` 不能关闭。
- 模板哈希代码使用实际 tokenizer 选择后的模板正文。当前模板、数据和已有项目训练产物声明必须核对；原始基座的未知训练历史不猜填。
- GRPO 默认算法值、五项 `None` 省略、G=4 及累计完成预算 256 保持；原采样 ID 不以候选重编码替换，追加前缀不兼容时明确失败。环境与模板边界 mask=0，策略 mask=1。
- 来源 `training_protocol_metadata` 与当前 `sampling_protocol` 分列；setup 在实际执行后编码，不进入策略轨迹，G 不重复扩增。

SFT 未指定 data 的 fallback 改为 `data/sft/sft_react_messages_v3_train_mix.jsonl`，这是新协议数据入口修正，不是算法默认变化。本次没有生成该文件、运行生成器或改写历史数据。

## 未验证的集成

未加载真实 tokenizer、模型或训练配置对象，未执行生产训练/数据生成/完整 86 任务。真实模板前缀、EOS、loss 标签、训练库解析有效值与梯度行为不由本静态报告证明；既有 token 测试未重复执行。实际 tokenizer 哈希和库解析默认值均未取得，不将 `None` 填成事实。

本记录只声明源码接线与配置保全检查通过，不宣称历史配方等价、真实训练集成、全量任务执行或模型能力。
