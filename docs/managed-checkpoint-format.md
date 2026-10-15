# 受管 checkpoint 格式与工程验证

本协议只用于显式受管的 SFT、DPO、GRPO。原训练配置不增加必填字段，裸入口不会自动接入新协议。模型可加载、final 存在或旧 Trainer checkpoint 存在，均不证明完整恢复能力。旧文件不升级、不补写。

## 身份文件

每次尝试由管理器写入 `control/managed-context.json`，`schema_version` 固定为整数 `1`：

- `stage`：`sft`、`dpo`、`grpo`。
- `run_id`、`attempt_id`：运行身份与本次尝试身份。
- `total_budget`：原计划的总优化步数，不是恢复后新发放的步数。按步数训练时为正整数；首次按 epoch 训练时暂为 `null`，由 Trainer 在训练开始、任何优化前核定为正整数。
- `epoch_budget`：原计划的正整数 epoch 数；按步数训练时为 `null`。它描述训练预算，不改变周期 checkpoint 仍按优化步保存的策略。
- `identity.inputs`：原始输入的路径、内容指纹与来源链。
- `identity.training_config`：有效训练参数与种子，不包含允许变化的尝试目录。
- `identity.runtime`：镜像及运行时身份。
- `identity.models`：`policy`、`reference`、`base` 身份；DPO/GRPO 的 `reference` 始终是最初输入，不能换成更新后的 checkpoint 策略。
- `identity.devices`：按全局 rank 顺序排列的 GPU UUID，保持集合、顺序与卡数。
- `output_dir`、`control_dir`、`log_dir`：本次尝试的操作路径，不参加研究身份比较。

恢复只允许 `attempt_id` 和上述操作路径改变，所有输入、训练参数、预算、设备和运行时身份保持不变。控制端的路径归属检查仍是必要条件，文件哈希不把任意外部 pickle 变成可信输入。

### Epoch 预算核定与恢复

按 epoch 的配方保留原 `max_steps=-1` 和原 `epochs`，管理器不依据原始数据行数猜算优化步数。Trainer 完成数据处理、分布式 batch 与梯度累积计算后，受管回调在 `on_train_begin` 核对所有 rank 的实际 `state.max_steps` 一致，绑定 adapter 的实际总步数，并写入当前尝试的 `control/resolved-budget.json`。原始 `max_steps`、`num_train_epochs` 和学习率排程参数均不被改写。

已提交 checkpoint 的 `total_budget` 必须是这次核定的正整数，并同时保存原 `epoch_budget`。恢复管理器从明确 checkpoint 取得实际总步数，交给新尝试；仍以相同 epoch 参数重建 Trainer。新 Trainer 再次算出的总步数必须等于原 checkpoint，不能通过换数据长度、改 epoch 或把剩余步数变成一份新预算继续训练。

尚未核定步数的初始 context 可以检查 epoch 身份匹配的 checkpoint，但只允许忽略这个尚未知的数字，不能忽略 epoch 或任何其他身份差异；训练适配器随后立即绑定 checkpoint 中已知的原实际总步数。缺失 epoch 身份或实际总步数的已提交清单没有恢复资格。

## 文件清单

提交后的目录为 `output/checkpoint-<global_step>/`：

| 文件 | 合同 |
| --- | --- |
| 策略权重或完整权重分片及索引 | 由标准 Trainer 保存，兼容全参和 LoRA |
| `optimizer.pt`、`scheduler.pt`、`training_args.bin` | 标准训练恢复状态 |
| `trainer_state.json` | `global_step`、原 `max_steps` 和数据消费的 `epoch` 位置 |
| `rng_state.pth` 或 `rng_state_<rank>.pth` | 标准 Trainer 保存的 Python、NumPy、CPU 及适用设备 RNG |
| `scaler.pt` | 启用 GradScaler 时必须存在；否则清单明确写 `not_applicable` |
| `managed_state_<rank>.pt` | 版本化附加状态；GRPO 包含生成复用位置、未消费缓存与保护/统计状态 |
| `rank-<rank>.complete.json` | 该 rank 的版本、rank、步数和 attempt 确认 |
| `managed_checkpoint.json` | 完整身份、实际总步数和原 epoch 预算、逻辑步数、卡数、附加状态版本、数据位置协议、精度与终止原因，以及全部内容文件的 SHA-256 和大小 |
| `managed_checkpoint.complete.json` | 最后写入的完成标记，绑定清单 SHA-256 |

数据位置协议为 `trainer_global_step_epoch_skip`，继续使用固定版本 Trainer 的原 epoch/已消费 batch 跳过逻辑，不重写数据采样算法。首版要求 `dataloader_num_workers=0`，不声称可恢复多进程预取队列。`ignore_data_skip`、仅保存模型、Hub 发布、分片训练与最佳模型回载不在本协议支持范围，训练前明确拒绝。

## 事务顺序与保留策略

1. 主 rank 创建同文件系统的 `.checkpoint-<step>.incomplete/checkpoint-<step>/`，遇到已有目标直接拒绝，不覆盖旧证据。
2. 所有 rank 继续调用原 Trainer `_save_checkpoint`。只临时改变保存根目录，并禁用其过早轮转，不改变损失、采样、优化器或标准恢复逻辑。
3. 各 rank 保存附加状态和 rank 完成确认；所有 rank 在共同保存边界汇总结果。可报告的本地磁盘失败传播为失败；已损坏的 CUDA/通信上下文不追加用于报告的 collective。
4. 主 rank 检查公共文件、所有 rank、相同步数和原预算，刷新文件，写清单，最后写完成标记并持久化。
5. 原子重命名至正式 `checkpoint-<step>`，刷新父目录，再允许当前尝试内的保留策略。隐藏未完成目录即使已有标记也没有恢复资格。
6. 仅轮转属于当前 run/attempt 且验证通过的正式目录；不删除恢复来源、其他尝试或最后有效恢复点。保存失败绝不先删旧 checkpoint。

原 `_save_checkpoint` 返回后，标准 `on_save` 才被调用，因此安全停止回调只能在提交成功后确认。强制中断、保存异常或缺失 rank 不能产生安全停止成功证据。正常提交后空事务父目录保留，不构成有效恢复点。

## 本地接口与诊断

控制端可只用标准库调用 `validate_checkpoint(path, expected=context)`。该函数重新检查标记、清单、所有内容、必要公共/rank 状态、版本、步数、预算和身份，返回清单；不加载模型或 pickle。`require_remaining_budget=False` 只用于检查历史完整性和轮转，不给预算耗尽或保护停止的 checkpoint 恢复资格。

训练端通过 `attach_managed_checkpoint(...)` 接入标准保存，返回的 adapter 提供 `last_checkpoint` / `latest_checkpoint`、`validate_resume()`、`restore_additional_state()` 和 `bind_budget(...)`。接入时自动注册 `budget_callback`，在训练开始时核对/核定原排程预算。附加状态在 `train(...)` 之前恢复一次；模型、优化器、调度器、精度和 RNG 继续交给标准 `train(resume_from_checkpoint=...)`。输出必须位于新尝试，原参考与输入只读保留。

固定适配依据为 [Transformers 4.57.6 Trainer 源码](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/trainer.py) 与 [TRL 0.29.1 GRPOTrainer 源码](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/grpo_trainer.py)。启动前检查实际版本、私有方法签名和不支持的执行模式，不能静默降级到“只恢复权重”。

## 验证边界

`tests/test_managed_checkpoint.py` 使用明确的控制元数据写入替身测试事务、损坏拒绝、轮转和身份差异，并在固定版本环境中调用真实 Trainer 的标准保存编排方法，但不构建模型或优化器、不训练。替身中的哨兵文件不是模型，不能用于推断真实恢复成功。

真实单卡/四卡的 SFT、DPO、GRPO 连续训练与中断恢复对照仍需单独授权，检查预算、学习率、数据消费、RNG、策略/参考和保护/缓存状态。本地工程测试通过不代表容器运行或上述 GPU 恢复门禁通过。
