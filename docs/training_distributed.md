# SFT / DPO / GRPO 单机 DDP

SFT、可选 DPO 和文本多轮 GRPO 均支持每卡一个进程的 DDP，包括默认 NF4/双量化/BF16 QLoRA。默认仍是 SFT → GRPO，DPO 必须显式选择；原 SFT/GRPO 单卡默认不变，四卡不会自动调整学习率、epoch、每卡 batch 或梯度累积。

本次工程回归使用合成张量和 CPU/Gloo 多进程，不读取研究数据、加载预训练模型或启动真实训练。CUDA/NCCL、bitsandbytes 四卡训练速度与实际保存/重载仍须在独立冒烟中验证。不要更新正在训练的安装环境；应先安装到新的隔离环境，再在指定 GPU 空闲时启动新实验。

## 启动入口

安装包含本次代码的 `agentic-arxiv[train]` 后，可从任意目录使用新启动器，不依赖源码路径注入或已有单卡后台脚本：

```bash
# 只检查命令，不初始化 GPU、不创建日志、不启动训练
python -m agentic_arxiv.training.launch \
  --num-processes 4 --gpus 0,1,2,3 --dry-run \
  sft --max-length 9216 --batch-size 1 --grad-accum 2

# 下列模型、数据、manifest 均须预先存在；所有路径按实际位置替换
WORKSPACE=/data/ding/AgenticArXiv-RL
BASE=/data/ding/models/Qwen2.5-1.5B-Instruct
DATA=/data/ding/research/sft-react-messages-v3-20261015/data/sft/train49/sft_react_messages_v3_train_mix.jsonl
RUN=/data/ding/runs/sft-ddp-new-run
python -m agentic_arxiv.training.launch \
  --num-processes 4 --gpus 0,1,2,3 --log-dir "$RUN/rank-logs" \
  sft --workspace "$WORKSPACE" --model "$BASE" --data "$DATA" \
  --max-length 9216 --batch-size 1 --grad-accum 2 \
  --output-dir "$RUN/output"
```

启动器只委托当前 Python 的 `torch.distributed.run --standalone --module agentic_arxiv train STAGE`。启动器选项放在 `sft/dpo/grpo` 前，训练参数放在其后，原样透传。`--gpus` 仅修改子进程的 `CUDA_VISIBLE_DEVICES`；省略时保留现有可见设备，不会自动突破限制。可见卡少于进程数、重复 GPU、嵌套启动均明确报错。`--log-dir` 保存各 rank 的 stdout/stderr，并在控制台添加 rank 前缀；不指定则沿用 torchrun 标准输出。

也可以使用 Accelerate，新增的 `configs/accelerate/ddp_4gpu.yaml` 是四卡 BF16 DDP 配置：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file configs/accelerate/ddp_4gpu.yaml --module agentic_arxiv \
  train sft --data /已有/train_mix.jsonl --model /已有/base \
  --max-length 9216 --batch-size 1 --grad-accum 2 \
  --output-dir /data/ding/runs/sft-ddp-new-run/output
```

## Batch 与 loss

有效 batch = 每卡 batch × 进程数 × 梯度累积。原单卡 `1×1×8=8` 对应四卡 `1×4×2=8`；4680 行、3 epoch 均为 1755 个优化步。单卡与 DDP 的归约顺序、随机采样可能不同，不宣称逐位复现。动态 padding 保持不变，不把所有样本补齐到 9216。

SFT 显式启用 `average_tokens_across_devices=True`，要求模型/Trainer 支持 `num_items_in_batch`；由 Trainer 汇总整个累积窗口和所有 rank 的非 `-100` labels，保持 completion-only 监督，不能按每卡 loss 均值再次归一化。prompt、历史 assistant、环境 Observation 和 padding 仍不参与当前 completion loss。

GRPO 的整除约束针对**全局 generation batch**，不是每卡 batch。原单卡 `batch=4, accumulation=1, G=4` 对应四卡 `batch=1, accumulation=1, G=4`。不会再将每卡 batch 1 静默扩大为 4；全局 batch 不可被 G 整除时直接报错。

```bash
# 仅说明后续 GRPO 多卡入口，本次未运行 GRPO；输入必须是兼容的 SFT 训练产物
python -m agentic_arxiv.training.launch \
  --num-processes 4 --gpus 0,1,2,3 \
  --log-dir /data/ding/runs/grpo-ddp-new-run/rank-logs \
  grpo --workspace /data/ding/AgenticArXiv-RL \
  --model /已有/sft/final --snapshot /已有/arxiv_snapshot.json \
  --task-set expanded --split data/splits/v7_86.json:train \
  --batch-size 1 --grad-accum 1 --num-generations 4 \
  --output-dir /data/ding/runs/grpo-ddp-new-run/output
```

## 可选 DPO 的跨 rank 语义

以下仅为未执行示例；它只打印命令，不加载模型或执行训练：

```bash
python -m agentic_arxiv.training.launch \
  --num-processes 2 --gpus 0,1 --dry-run \
  dpo --model outputs/sft_qlora/final \
  --data data/dpo/dpo_react_messages_v3_train.jsonl \
  --snapshot data/mock_arxiv_snapshot.json --output-dir outputs/dpo
```

DPO 每个 rank 独立持有同源 SFT policy 和冻结 reference，两者不共享可变参数；双模型增加显存，不保证任意卡型可训练。全局偏好对 batch = 每卡偏好对 batch × world size × grad_accum，默认单卡为 `1×1×8=8`。这不是 SFT token loss 归一化，也不受 GRPO generation 组整除规则约束。主进程写公共 manifest/报告，阶段验证只执行一次且结果/异常广播所有 rank；失败保留已保存模型及失败报告并非零退出，`--no-verify` 记录 skipped，不自动继续 GRPO。

DPO 当前只核对工程接线及既有检查；没有执行真实 tokenizer、优化后参考权重不变、CUDA/NCCL、GPU/DDP 训练、显存容量或资源释放验证。设计默认与 dry-run 都不是历史有效配方、真实训练通过或收益证据。

## GRPO 的跨 rank 语义

- 数据分片、prompt 的 G 次重复和梯度同步交给 Trainer/Accelerate/TRL，不另套采样器、不在 rollout 中再次扩增 G。
- 每个 rank 独立完成自己的多轮工具执行，可以在不同轮数结束；原始采样 ID、EOS、环境 mask 和隐藏 task_id 契约不变。
- reward function 仍只返回本 rank 的原始奖励。TRL 汇总完整 generation 组并计算 advantage，本项目不二次归一化、不把本地单样本当作零方差组。
- 奖励分量和逐任务统计使用全局证据；同一任务在同一批次出现多个 generation 组时，仍分别统计组内方差。
- rollout 审计汇总完整的跨 rank 组，仅全局 rank 0 打开和写入 JSONL，保存上限是全局上限。审计写盘异常会广播给所有 rank。
- Canary 只在 rank 0 执行，结果/异常广播到所有 rank；所有 rank 更新相同的 patience 和停止状态。最终阶段验证同样只执行一次，失败不能让其他 rank 永久等待。

## 设备、checkpoint 和来源记录

量化加载前按 `LOCAL_RANK` 设置 CUDA 当前设备，再将完整量化模型复制到该设备；不使用全局 `RANK` 当设备编号，也不使用 `device_map="auto"` 拆分 DDP 副本。DDP 设置 `find_unused_parameters=False`、`broadcast_buffers=False`，保留非重入 gradient checkpointing。

普通 checkpoint 和最终模型保存由所有 rank 调用 Trainer API，由训练器决定实际 writer；tokenizer、run/training manifest、奖励摘要和最终验证报告仅 rank 0 写入。manifest 记录 world size、全局有效 batch、GRPO generation batch 或 DPO 全局偏好对 batch，以及各 rank 的 peak allocated/reserved 显存。原 SFT/GRPO 配置子树不变，本次新增独立 DPO 节；world size 作为资源覆盖记录，不能伪装成原单卡配方。

QLoRA 的 FSDP/DeepSpeed 分片未在此支持；多轮 GRPO 也拒绝分片/vLLM 生成路径，因为不同 rank 的工具轮数不同，不能安全插入这些路径的逐轮 collective。已有非量化 SFT 分片入口不在本次 DDP 验证范围内。现有 `fsdp_2gpu.yaml` 不是 QLoRA 多卡启动方案。

GRPO 原有 checkpoint 恢复仍由 Trainer 执行；不要把变更 world size、batch 或累积后的恢复称为等价续训。本次没有新增 SFT 恢复参数，也没有实际验证真实多卡 checkpoint 恢复。先进行短冒烟，检查所有卡的真实利用率、完整 generation 组、有限 loss/梯度、保存和单卡重新加载，再决定正式多卡训练。
