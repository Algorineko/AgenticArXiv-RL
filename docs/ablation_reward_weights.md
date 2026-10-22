# 多粒度奖励权重消融实验设计

> **作者：** pznger  
> **日期：** 2026-10  
> **状态：** 实验设计完成，待执行

---

## 1. 背景与动机

### 1.1 当前奖励函数设计

项目采用多粒度可验证奖励（commit f70215a），将单一稀疏成功信号分解为五个层次化分量：

```text
R(tau, s) = Σ w_i(s) r_i(tau) / Σ |w_i(s)|
```

每个分量归一化至 [-1, 1]，通过动态权重组合形成最终标量奖励：

| 分量 | 默认权重 | 信号 |
|------|---------|------|
| `r_format` | 1.0 | 动作格式正确性（终止 token 或严格 JSON） |
| `r_tool` | 3.0 | 工具序列 LCS F1（顺序感知） |
| `r_argument` | 2.0 | 参数值匹配覆盖率 |
| `r_process` | 1.0 | 有效步骤积分减去惩罚 |
| `r_outcome` | 3.0 | 最终任务完成质量 |

**课程学习调度：** 前 30 步 tool/argument/outcome 权重乘以 1/3，使策略先学习结构化行为（format + process）再优化语义正确性。

### 1.2 待验证的问题

当前权重配置基于先验设计，但缺乏系统性验证：

1. **差异化权重的必要性** - 均匀权重能否达到相似效果？
2. **稀疏 vs 密集** - 仅保留 outcome 与完整五分量的性能差异？
3. **各分量的重要性排序** - tool/argument/outcome 谁最关键？
4. **课程学习的增益** - 30 步课程窗口是否必要？
5. **权重敏感性** - 小幅调整（如 tool 从 3→5）的影响？

当前最佳模型（8B GRPO v3）在 81 任务 agent benchmark 上达到 **strict 72.8%**，为 27B base 的 2.9 倍。本实验旨在通过消融研究，定量回答上述问题，为后续优化提供实证基础。

---

## 2. 实验设计

### 2.1 实验组配置

采用单变量控制原则，每次仅改变一个维度：

| 实验组 | 权重配置 (format, tool, argument, process, outcome) | 课程学习 | 验证目标 |
|--------|---------------------------------------------------|---------|----------|
| **Baseline** | (1, 3, 2, 1, 3) | 30 steps | 当前默认配置（基线） |
| **Uniform** | (1, 1, 1, 1, 1) | 30 steps | 验证差异化权重的必要性 |
| **OutcomeOnly** | (0, 0, 0, 0, 3) | 30 steps | 对比稀疏奖励（单一最终目标） |
| **NoOutcome** | (1, 3, 2, 1, 0) | 30 steps | 验证 outcome 分量的重要性 |
| **StrongFormat** | (3, 3, 2, 1, 3) | 30 steps | 增强格式正确性的影响 |
| **ToolFocus** | (1, 5, 1, 1, 3) | 30 steps | 强化工具选择准确性 |
| **ArgumentFocus** | (1, 3, 4, 1, 3) | 30 steps | 强化参数传递准确性 |
| **ProcessRich** | (1, 3, 2, 3, 3) | 30 steps | 密集过程反馈的影响 |
| **NoCurriculum** | (1, 3, 2, 1, 3) | 0 steps | 验证课程学习的必要性 |

### 2.2 实验参数

- **基础模型：** outputs/sft/final（SFT 阶段输出）
- **训练步数：** 100 steps（小规模验证，成本可控）
- **每步生成数：** 4 generations（与 GRPO 标准配置一致）
- **任务集：** expanded + rl_train split（排除零方差任务）
- **评测指标：**
  - `strict_success_rate`: 严格成功率（完整任务完成）
  - `completion_rate`: 完成率（到达终止状态）
  - `tool_accuracy`: 工具序列准确率
  - `argument_accuracy`: 参数匹配准确率
  - `false_finish_rate`: 假结束率
  - `reward_mean/std`: 奖励均值与方差
  - `frac_reward_zero_std`: 零方差组占比（训练稳定性）

### 2.3 预期结果

#### 假设 H1：差异化权重优于均匀权重
- **预测：** Baseline > Uniform，至少在 tool/outcome 准确率上显著领先
- **依据：** 工具选择与最终目标是任务核心，应获得更高权重

#### 假设 H2：多粒度优于稀疏奖励
- **预测：** Baseline > OutcomeOnly，尤其在早期收敛速度与训练稳定性上
- **依据：** 密集反馈减少探索盲区，降低零方差组占比

#### 假设 H3：outcome 是最关键分量
- **预测：** NoOutcome 显著劣于 Baseline，strict 成功率下降 >20%
- **依据：** outcome 直接衡量任务完成，缺失它会导致目标漂移

#### 假设 H4：课程学习加速收敛
- **预测：** Baseline 在前 30 步 format/process 收敛更快，后续 tool/argument 提升更稳定
- **依据：** 早期抑制语义权重避免过早优化，符合 `docs/multigranular_rl.md` 设计意图

#### 假设 H5：权重小幅调整影响有限
- **预测：** ToolFocus/ArgumentFocus/ProcessRich 与 Baseline 差异 <5%
- **依据：** 当前配置已接近局部最优，微调不会带来质变

---

## 3. 实验执行

### 3.1 环境准备

```bash
# 确保基础模型存在
ls outputs/sft/final/

# 检查快照数据
python -m AgenticArxiv.rl.build_snapshot  # 如需更新
```

### 3.2 运行实验

```bash
# 运行所有 9 组实验
python scripts/ablation_reward_weights.py --base_model outputs/sft/final

# 快速测试（10 steps）
python scripts/ablation_reward_weights.py --quick_test

# 仅运行特定实验组
python scripts/ablation_reward_weights.py --experiments baseline uniform outcome_only

# Dry run 模式（仅验证配置）
python scripts/ablation_reward_weights.py --dry_run
```

### 3.3 输出结构

```
artifacts/ablation_reward_weights/
├── experiments_meta.json          # 实验元信息
├── ablation_summary.json          # 汇总结果
├── baseline/
│   ├── config.json                # 权重配置
│   ├── training.log               # 训练日志
│   ├── eval.log                   # 评测日志
│   ├── final/                     # 训练后的模型
│   └── eval/
│       ├── summary.json           # 评测指标
│       └── traces.jsonl           # 轨迹记录
├── uniform/
│   └── ...
└── ...
```

---

## 4. 结果分析

### 4.1 对比维度

#### 4.1.1 训练稳定性
- **零方差组占比** (`frac_reward_zero_std`)：<10% 为健康
- **奖励方差** (`reward_std`)：合理范围 [0.2, 0.8]
- **训练曲线平滑度**：通过 TensorBoard 可视化

#### 4.1.2 评测表现
- **主指标：** strict 成功率（与 Baseline 72.8% 对比）
- **辅助指标：** 完成率、tool/argument 准确率、假结束率
- **效率：** 平均对话轮数、token 消耗

#### 4.1.3 分量级分析
从 TensorBoard 日志提取各分量曲线：
- `reward_components/format`
- `reward_components/tool`
- `reward_components/argument`
- `reward_components/process`
- `reward_components/outcome`

对比各组在训练中期（step 30-60）与后期（step 80-100）的分量分布。

### 4.2 可视化

生成对比图表（参考 `scripts/analyze_curriculum.py`）：

```python
import matplotlib.pyplot as plt
import pandas as pd

# 读取汇总数据
with open("artifacts/ablation_reward_weights/ablation_summary.json") as f:
    summary = json.load(f)

# 绘制 strict 成功率对比
configs = [exp["name"] for exp in summary]
strict_rates = [exp["metrics"]["strict_success_rate"] * 100 for exp in summary]

plt.figure(figsize=(12, 6))
plt.bar(configs, strict_rates)
plt.axhline(y=72.8, color='r', linestyle='--', label='Baseline (current best)')
plt.ylabel("Strict Success Rate (%)")
plt.xlabel("Experiment Config")
plt.title("Reward Weight Ablation: Strict Success Rate")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/ablation_reward_weights/strict_success_comparison.png", dpi=150)
```

### 4.3 统计显著性检验

对关键指标进行 t-test（如 Baseline vs. Uniform）：

```python
from scipy import stats

baseline_rewards = load_rollout_rewards("artifacts/ablation_reward_weights/baseline/")
uniform_rewards = load_rollout_rewards("artifacts/ablation_reward_weights/uniform/")

t_stat, p_value = stats.ttest_ind(baseline_rewards, uniform_rewards)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
```

显著性水平 α=0.05，p < 0.05 视为显著差异。

---

## 5. 预期产出

### 5.1 实验报告

文档路径：`artifacts/ablation_reward_weights/report.md`

内容包括：
1. 实验背景与假设
2. 各组详细结果（表格 + 图表）
3. 假设验证结论
4. 权重优化建议
5. 局限性与后续工作

### 5.2 权重优化建议

基于实验结果，可能的优化方向：

- 如 **ToolFocus** 显著优于 Baseline → 建议将 tool 权重提升至 5
- 如 **NoCurriculum** 性能相近 → 可简化训练流程，移除课程调度
- 如 **OutcomeOnly** 完全失败 → 验证多粒度设计的必要性，作为论文素材

### 5.3 论文素材

实验数据可支撑以下论点：

- **Claim 1：** 多粒度奖励优于稀疏奖励（OutcomeOnly vs. Baseline）
- **Claim 2：** 差异化权重优于均匀权重（Uniform vs. Baseline）
- **Claim 3：** 课程学习加速收敛（NoCurriculum vs. Baseline 的前 30 步曲线）
- **Claim 4：** 权重配置的敏感性分析（ToolFocus/ArgumentFocus 的微调效果）

---

## 6. 局限性与后续工作

### 6.1 当前局限

1. **训练规模：** 100 steps 仅为快速验证，完整训练需 500+ steps
2. **模型规模：** 仅在 8B 模型验证，27B 可能有不同表现
3. **任务覆盖：** 仅在 arXiv 检索任务上验证，泛化性待验证
4. **超参敏感性：** 未测试学习率、batch size 等对结果的影响

### 6.2 后续工作

1. **完整规模训练：** 对表现最佳的 2-3 组配置，跑完整 500 steps 训练
2. **跨模型验证：** 在 1.5B / 3B / 27B 上重复实验
3. **自动化权重搜索：** 基于贝叶斯优化的超参搜索
4. **动态权重调度：** 探索训练中动态调整权重的策略
5. **OPD 集成：** 测试多粒度奖励在 OPD（Outcome-supervised Process-reward Distillation）中的表现

---

## 7. 引用与致谢

- 多粒度奖励设计参考：LLM-TIR（Tool Invocation Reward）
- 课程学习设计参考：`docs/multigranular_rl.md`
- 当前最佳模型：8B GRPO v3 (strict 72.8%)，见 `docs/edge-vlm-posttrain.md`
- 实验脚本参考：`scripts/analyze_curriculum.py`, `scripts/audit_grpo_rollouts.py`

---

## 8. 附录

### 8.1 完整命令记录

```bash
# 实验执行
cd AgenticArXiv-RL/
source .venv/bin/activate
python scripts/ablation_reward_weights.py --base_model outputs/sft/final --num_steps 100

# 结果分析
python scripts/visualize_ablation_results.py artifacts/ablation_reward_weights/

# TensorBoard 可视化
tensorboard --logdir artifacts/ablation_reward_weights/
```

### 8.2 预期运行时间

单组实验（100 steps, 4 generations）：
- GPU: A100 40GB ~30 分钟
- DCU: 海光 Z100 ~45 分钟

全部 9 组实验：
- 串行执行 ~4-6 小时
- 并行执行（3 组同时）~2 小时

### 8.3 资源需求

- **显存：** 单组实验 ~20GB（8B 模型 GRPO）
- **磁盘：** 每组 ~2GB（模型 checkpoint + 日志 + 评测结果）
- **总计：** ~20GB 磁盘空间

---

**实验状态：** 🟡 设计完成，待执行  
**负责人：** pznger  
**预计完成时间：** 实验执行 1 天 + 结果分析 1 天 = 2 天
