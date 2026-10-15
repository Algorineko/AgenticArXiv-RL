# SFT / DPO 受管恢复

## 恢复不是模型重载

原有训练配置不需要新增管理字段，裸训练的默认超参数和保存间隔不变。SFT / DPO 新增的 `--resume-from-checkpoint` 必须与 `--managed-context` 一起使用。最终推理模型、只有权重的目录、历史 checkpoint 和尚未提交完整标记的目录都不是完整受管恢复点；不会自动给旧目录补写元数据。

受管运行由管理入口生成身份、训练配置和新尝试目录。容器内部等价调用如下，仅用于说明参数接线，不应手写身份文件或把示例路径作为真实输入：

```bash
agentic-arxiv train sft --config /data/experiments/runs/RUN/attempts/ATTEMPT/training-config.json --managed-context /data/experiments/runs/RUN/attempts/ATTEMPT/control/managed-context.json --output-dir /data/experiments/runs/RUN/attempts/ATTEMPT/output --resume-from-checkpoint /data/experiments/runs/RUN/attempts/PREVIOUS/output/checkpoint-600
agentic-arxiv train dpo --config /data/experiments/runs/RUN/attempts/ATTEMPT/training-config.json --managed-context /data/experiments/runs/RUN/attempts/ATTEMPT/control/managed-context.json --output-dir /data/experiments/runs/RUN/attempts/ATTEMPT/output --resume-from-checkpoint /data/experiments/runs/RUN/attempts/PREVIOUS/output/checkpoint-600
```

`RUN`、`ATTEMPT`、`PREVIOUS` 仅为说明性占位符。调用前必须已有真实输入和管理记录；本地工程验证不会创建模型、偏好或研究轨迹。正式使用应由 `experiment resume` 明确选择恢复点，而非直接修改旧尝试。

## 状态连续性和只读来源

- 原始总预算、学习率配置、种子、batch、梯度累积、输入指纹、镜像、GPU UUID 及 rank 顺序必须一致。第 600 步恢复原定 1000 步的作业仍只消耗剩余预算，不能改成额外 1000 步。
- 同时支持显式优化步预算和原始 epoch 预算。epoch 模式始终保留 `max_steps=-1` 和原 `epochs`，不得把核定步数回填为正 `max_steps`，也不自行改变学习率排程。首次尝试的优化步总数在 Trainer 根据真实数据加载器、分布式和梯度累积配置计算前保持未知；各 rank 在 `on_train_begin`、首次优化前核对该总数，保存 `control/resolved-budget.json`。恢复从完整 checkpoint 读取原实际总步数，重新核对 Trainer 算得的值；不重新发放 epoch、改排程或跳过数据消费恢复。
- 标准 Trainer 接收明确的 `resume_from_checkpoint`，承接策略、优化器、调度器、进度、数据消费跳过位置和各 rank 随机状态。不启用 `ignore_data_skip`；不把 `--model checkpoint` 视为续训。
- 固定 Transformers 4.57.6 / Accelerate 1.11.0 的跳批加载器会丢失非零 epoch。受管 SFT/DPO 恢复在当前 Trainer 的 `train` 作用域内保留原训练循环代码，只隔离替换 `skip_first_batches` 依赖：仍调用 Accelerate 的原跳批函数，再将原加载器 epoch 传给新加载器。不会把第二个及以后 epoch 的剩余数据误按第一个 epoch 重排，也不改 seed、跳过数量、梯度累积或预算。正常与异常退出均还原实例方法，非受管训练、GRPO 和同进程其他 Trainer 不受影响；未知依赖版本或接口直接拒绝。
- 受管 PEFT 恢复在标准模型加载边界关闭 `load_adapter` 的自动精度转换，保留 TRL 已设置的 BF16 或 FP32 参数精度。优化器状态加载前核对参数集合、参数对象、精度、冻结状态、活动 adapter 和优化器参数引用；发现变化即拒绝，不把可加载权重等同于正确续训。非受管路径不安装该保护。
- DPO 的 `model` 和偏好包审计始终绑定最初 SFT。策略和独立完整冻结参考都先从原始 SFT 重建，再只恢复训练策略；参考指纹需要匹配原身份，不进入策略优化器，也不与策略共享可变参数存储。
- 新尝试独立写输出和日志，拒绝重用原输出或嵌套在恢复来源的输出目录内。原模型、数据、manifest、checkpoint 和异常记录不被改写；DPO 原有的输出覆盖拒绝仍保留。
- Trainer 原始 `training_loss` 不被解释为跨尝试累计均值或恢复等价证明；缺少可靠的本尝试累计量时保持未知。

## 停止、验证和失败诊断

安全停止由本尝试身份匹配的请求触发，在完整优化步保存并提交 checkpoint 后确认。用户停止和保护器提前停止都不导出成功 `final`，也不自动运行正常完成验证。实际训练故障和预算完成不能被迟到的用户请求掩盖。

受管 SFT 开启阶段验证时，配方必须声明已有 `inputs.snapshot`。预检检查其可读性，训练入口将验证环境绑定到冻结输入；显式 `--snapshot` 必须与该身份一致，不能改用容器工作目录中的隐含快照。裸 SFT 未指定快照时保留原工作区默认行为，`--no-verify` 不要求验证快照。

`control/training-result.json` 分别记录训练和验证事实，并保存 Trainer 的 `planned_steps`、核定的 `total_budget` 和原始 `epoch_budget`，退出码 0 不等于预算完成。epoch 预算尚未核定时总步数保持未知，不能按 0 或 `-1` 判断完成。训练异常记录实际步数及异常类型；异常处理不追加新的分布式通信。正常训练完成后的阶段验证可以在 episode 边界取消，已完成训练与未完成验证独立记录，重试验证不应重训。训练耗时和阶段验证耗时分开记录，未知成本不填零。

DPO 导出完成后，在加载阶段验证模型之前同步释放 Trainer、策略、参考、优化器所依赖的受管 checkpoint 句柄及停止回调，避免它们通过强引用保留训练模型。该资源生命周期检查不替代真实 GPU 显存验证。

常见拒绝原因包括：阶段错误、恢复协议版本未知、文件/输入指纹变化、rank 状态缺失、保存未提交、关键配置或原总预算变化、非新尝试输出目录、冻结参考身份变化、不支持的训练库接口和版本。拒绝不自动退回其他 checkpoint，也不重新开始训练。

## 验证边界

工程回归覆盖 CLI 参数、上下文与输入指纹检查、已有 Trainer 替身的完整状态恢复参数、原总预算/学习率接线、DPO 原始参考来源与独立冻结/优化器排除，以及停止不发布 `final` 的控制流。epoch 补充回归直接调用固定 Transformers 版本的真实 Trainer 预算计算接口和受管预算绑定接口，只提供具有长度的控制对象，不初始化模型或遍历训练样本；核对首次绑定、恢复预算不一致拒绝、epoch/排程/数据消费参数和逻辑进度不被改写。接线和控制元数据测试不构成真实权重恢复、数据/RNG 连续性或数值等价证据。

`tests/test_managed_adapter_resume.py` 使用 CPU 微型 PEFT 层及固定 SFT/DPO Trainer 的真实加载接口，覆盖 BF16/FP32、safetensors/bin、优化器与调度器状态恢复及参数契约拒绝路径；不运行前向、反向或训练。DPO 清理回归另通过弱引用确认受管句柄退出后 Trainer、策略、参考及优化器可以回收。这些检查不替代量化模型、GPU 显存或连续训练对照。

`tests/test_managed_dataloader_resume.py` 执行生产 SFT/DPO 配置表达式，使用真实 Trainer 采样器、预算计算和 Accelerate 加载器，仅遍历内存整数索引。覆盖单分片、四分片的每个 rank、步数/epoch 预算、首 epoch 中途、后续 epoch 中途和边界，以及恢复后继续跨 epoch 的剩余索引一致性；另执行生产 `train` 边界片段核对绑定、退出还原和版本/接口拒绝。四分片测试不启动进程组、不加载模型或执行训练，只证明固定依赖的数据加载器索引连续性，不是四卡分布式训练或端到端 RNG/权重等价证据。

真实 SFT / DPO 单卡与四卡连续运行、受管安全停止恢复、强制中断恢复对照均待独立授权和合格已有输入。尚未操作服务器或启动真实训练，不能宣称真实恢复或模型质量已验证。
