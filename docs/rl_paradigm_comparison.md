# RL 范式对比：GRPO / GSPO / Dr.GRPO / DAPO

> 路线图 P1「新方法训练与对比」的实验记录。四个变体共用同一套任务、奖励与算力协议，只差范式旗标。

## 实验设置

- **起点**：`outputs/sft/final`（Qwen2.5-1.5B-Instruct 全参 SFT，与已发布 GRPO v2 同源）
- **任务**：`data/splits/v6_grpo_train.json:rl_train`（冻结探测选出的 33 条信号任务），离线快照回放
- **公共配方**（对齐已发布 GRPO v2 manifest）：`lr 5e-6`、`beta 0.1`、`batch_size 16`（每步 4 prompt × 4 生成）、`num_generations 4`、`max_turns 6`、temperature 1.0、无奖励课程（`reward_curriculum_steps 0`）、seed 42、60 步、`--no-qlora` 全参、`--allow_zero_variance`
- **硬件**：单卡 Hygon DCU BW1000_H（64 GB），每变体单进程训练；评测 seed 45 / repeat 3 / regex agent / transformers 后端，与已发布基线完全同协议

### 各变体差异（唯一变量）

| 变体 | 旗标 | 语义 | 峰值显存 |
|---|---|---|---|
| GRPO（基线） | — | token 级 IS + 组内 std 归一化 | 34.8 GiB |
| GSPO | `--importance_sampling_level sequence` | 重要性采样升至序列级 | 37.6 GiB |
| Dr.GRPO | `--loss_type dr_grpo --scale_rewards none` | 无偏目标：去 std 归一化与长度偏置 | 37.6 GiB |
| DAPO | `--dapo --beta 0 --no-dynamic_sampling` | clip-higher（ε_high 0.28）+ 截断掩码 + 无 KL | 见 manifest |

两个实现注记（踩过的坑）：

1. **DAPO 的 β 交互**：`resolve_dapo_options` 只在 β 等于默认值 0.04 时归零；若显式传基线的 `--beta 0.1` 会静默保留 KL，偏离预设。DAPO 运行必须显式 `--beta 0`。
2. **动态采样在 33 任务池上不可行**：完整 DAPO（含动态采样）在第 5 步即 `DynamicSamplingExhausted`——SFT 起点零方差组占比高（v2 日志 `frac_reward_zero_std` 常态 0.75–1.0），29/32 次重采样全被拒后提示池耗尽。故 DAPO 变体关闭动态采样（保留 clip-higher、截断掩码、β=0 三组件），发布产物如实标注 `grpo_dapo_release_nodynamic`。33 条任务的小池子是根因，扩池或增大队列是后续可行方向。

## 结果

### 严格成功率 pass³（离线，seed 45 / repeat 3）

| 模型 | rl_train（99） | dev（24） | iid_test（54） | ood_test（12） |
|---|---|---|---|---|
| SFT（起点） | 0.081 | 0.000 | 0.056 | 0.000 |
| GRPO（已发布基线） | 0.636 | 0.375 | 0.444 | 0.500 |
| GSPO | 0.636 | 0.375 | 0.444 | 0.500 |
| Dr.GRPO | 0.657 | 0.375 | 0.444 | 0.500 |
| DAPO（无动态采样） | **0.667** | **0.500** | **0.481** | 0.500 |

### 辅助指标（tool_accuracy / false_finish / 平均总 token）

| 模型 | rl_train | iid_test | ood_test |
|---|---|---|---|
| GRPO（基线） | 0.85 / 0.15 / 6461 | 0.50 / 0.50 / 5141 | 0.58 / 0.42 / 8314 |
| GSPO | 0.85 / 0.15 / 6461 | 0.50 / 0.50 / 5141 | 0.50 / 0.50 / 8110 |
| Dr.GRPO | 0.84 / 0.16 / 6497 | 0.56 / 0.44 / 5257 | **0.67 / 0.33** / 8516 |
| DAPO（无动态采样） | 0.85 / **0.12** / 6581 | **0.65 / 0.35** / 5587 | 0.50 / 0.50 / 8102 |

## 训练动态观察

- **GSPO 与基线权重不同但评测逐位同分**：final 权重哈希互异、训练 reward 轨迹后段分化（GSPO 第 58 步 0.452 vs Dr.GRPO 0.765），但四个切分的 pass³ 与辅助指标几乎完全一致。同种子 + 确定性回放下，两种 IS 粒度收敛到了相同的行为模式——在本任务分布与 60 步预算内，序列级 IS 未带来可测差异。
- **Dr.GRPO 全面持平或略优**：rl_train pass³ +0.021（样本量下属弱信号），ood 的工具准确率与虚假完成率占优（0.67/0.33）；去 std 缩放后有效更新更大，但 60 步内无失稳（阶段验证 mean_reward 0.295，阈值 −0.2）。
- **DAPO（无动态采样）是三变体中最强**：pass³ 在 rl_train / dev / iid 三个切分领先（0.667 / 0.500 / 0.481，基线 0.636 / 0.375 / 0.444），iid 的工具准确率/虚假完成率优势最大（0.65/0.35 vs 0.50/0.50）。clip-higher + 无 KL 让策略在去 std 归一化方向上更进一步；训练全程无失稳，token 用量与基线相当。注意 dev +0.125 在 24 样本下置信区间宽。
- **loss 跨 loss_type 不可比**：不同范式的 loss 尺度定义不同（IS 粒度、归一化方式），只比 reward 曲线与评测，勿比 loss 绝对值（GSPO 0.0178 / Dr.GRPO 0.0012 仅为各自口径）。

## 产物

- 权重：ModelScope `Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-{GSPO,DrGRPO,DAPO}`
- 训练 manifest：`outputs/grpo_{gspo,drgrpo,dapo}_release/final/training_manifest.json`（注意 manifest 未记录 `importance_sampling_level` / `scale_rewards`，以 run_name 与本文档为准）
- 评测产物：`artifacts/eval_grpo_{gspo,drgrpo,dapo}_{v6_rl_train,v381_dev,v381_iid,v381_ood}_r3_s45/`

## 局限

- 单种子单次运行，60 步预算；不排除更长训练下范式差异放大
- dev/ood 样本量小（24/12），置信区间宽
- DAPO 变体不含动态采样（见实现注记 2），非完整 DAPO
- 离线快照回放口径，不代表在线 arXiv 表现
