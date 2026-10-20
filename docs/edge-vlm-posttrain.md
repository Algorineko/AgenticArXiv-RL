# 端侧多模态论文助手：Qwen3.8-27B 级 VLM 后训练实验记录

> 分支 `feat/edge-vlm-posttrain` · 2026-10-16 ~ 10-20 · 状态：**实验完成，双模型已发布**
>
> 目标：在端侧可部署的 VLM 上完成领域后训练（Agentic 工具决策 + 论文总结 + 图表识别），
> 沿 SFT→DPO→GRPO 全链路，双闸门评测（clean-val 图表 ROUGE + 81 任务 agent benchmark）。

## 1. 选型调研结论

| 候选 | 开源 | 本机 DCU 可训 | 结论 |
|---|---|---|---|
| MiniCPM-V 4.7（35B-A3B/1B） | ✅ Apache 2.0 | ❌ 原生支持需 transformers≥5.15 → torch≥2.5，DTK vendor torch 2.4.1 钉死 | 参照系 |
| **Qwen3.8-27B**（原生多模态 dense） | ✅ Apache 2.0 | ✅ transformers 5.14.1（torch floor 2.4）原生支持 `qwen3_5` | **主选** |
| Qwen3-VL 8B / 4B | ✅ | ✅ tf 4.57.3 原生（零升级保底） | 快速验证车 |

关键探针（实测）：`arxiv`/`triton 3.0.0` 在位；transformers 5.15+ 要求 torch≥2.5（被 DTK 钉死）；
MiniCPM-V 4.7 两 repo 均无 remote code → 本机不可训，Qwen3.8-27B 落在 tf 5.14.1 支持线内。

## 2. 基础设施

- 4×64GB 海光 DCU（BW1000_H）；**宿主 cgroup 64GiB 是真墙**（模型装载 page cache 计入配额）
- vendor 环境之上建 **transformers 5.14.1 覆盖层**（`pip --target` + `PYTHONPATH` 前置，零污染可回滚），
  含 3 处 torch 2.4 的 `torch.distributed.tensor` 属性级 shim
- DeepSpeed ZeRO 不可用 → 单进程 bf16 + device_map；vllm 0.6.2 不可用 → `AA_RL_BLOCK_VLLM=1` 屏蔽
- 训练器关键件：**切片 lm_head 损失**（248k 词表全序列 logits ~10GB → 目标位置切片 ~0.2GB，
  双卡时延对齐 Trainer 记账设备）、adapter-only 保存 + page-cache 逐出 + flock settle

## 3. 数据演进

| 版本 | 规模 | 变化 |
|---|---|---|
| figureqa_seed | 630 行（413 图/71 篇） | 原始：describe/axes/trend 三题型 |
| v2 | 1,046 行 | 扩池（修 22 篇损坏 PDF）+ 3 新题型（figure_type/reading/caption） |
| v2-clean | 611 多模态行 | **剔除 clipart 污染**（98/413 图为 <300px 装饰图，caption 错配） |
| v3-canonical | 2,988 工具轨迹 | **工具手册规范化**为当前 registry 逐字节渲染 |
| v4-multiturn | 2,508 行 | 同任务决策链 + 快照 Observation（实验证伪，见 §4） |
| 总结 SFT | 474 行 | 95 篇 × 3 风格 × 3 预算，基座自蒸馏 + 硬过滤 |
| 总结 DPO | 854 对 | SFT 模型自采样 k=3 + 规则软评分 |

## 4. 实验轮次全记录

| 轮次 | 配方 | 8B 结果 | 27B 结果 | 判定 |
|---|---|---|---|---|
| v1 | 原始混合集 | ROUGE ❌ / strict 72.8% ✅ | ROUGE 0.1575 ✅ / strict 13.6% ❌ | 8B 双过；27B 崩 |
| v2 | +clipart 过滤 | ROUGE 0.1486 ✅ / 60.5% | 未跑 | 权衡 |
| **v3** | +工具手册规范化 | **ROUGE 0.1446 ✅ / 66.7% ✅** | ROUGE **0.1662** ✅ / 22.2%(greedy) ❌ | **8B 配方定案** |
| v4 | +多轮历史 | 双败 ❌（53.1%/0.1179） | 已终止 | 多轮为负收益，回滚 |
| cap | v3 + r128/α256/lr1e-4 | — | ROUGE **0.2164** ✅ / strict 21.0% ❌ | 图表侧容量假设证实；agent 侧不及 base |

## 4.1 终版评测矩阵（81 任务 strict · greedy 协议 / clean-val ROUGE-1）

| 模型 | benchmark strict | 完成 | tool | arg | false_fin | clean ROUGE-1 |
|---|---|---|---|---|---|---|
| 27B base | 24.7% | 80.2% | 32.1% | 47.7% | 22.2% | 0.1349 |
| 27B v3 SFT | 22.2% | 55.6% | 27.2% | 41.2% | 12.3% | 0.1662 |
| **27B cap SFT** | 21.0% | 66.7% | 25.9% | 51.9% | **4.9%** | **0.2164** |
| 8B SFT v3 | 66.7% | 95.1% | 72.8% | 75.8% | 8.6% | 0.1446 |
| **8B GRPO v3（主发布）** | **72.8%** | **97.5%** | **82.7%** | **83.2%** | **6.2%** | 0.1509 |

GRPO 终版双次运行 strict 均 71.6%/72.8%（任务翻转率 4.9%）。

## 4.2 发布（ModelScope，adapter-only，Apache 2.0）

- **[Algorineko/AgenticArXiv-RL-Qwen3-VL-8B-GRPO](https://modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-8B-GRPO)** —— 端侧论文助手主模型（353 MiB）
- **[Algorineko/AgenticArXiv-RL-Qwen3.8-27B-SFT](https://modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3.8-27B-SFT)** —— 图表理解特化研究版（2.4 GiB；工具决策未达标已如实披露）

## 5. 根因分析（三层递进，均有消融实验支撑）

1. **clipart 污染**：T4 位图抽取把 PDF 装饰图当论文图（98/413），caption 错配 → 图表监督噪声。
   修复：小图过滤 + 去重（v2-clean）。
2. **前缀/注册表漂移**：SFT 轨迹生成于旧工具注册表，27B 把 prompt 前缀背死（r32@0.58% 容量下
   loss 0.37），benchmark 前缀 53 行差异即整体回落 base 行为。修复：工具手册规范化（v3，diff 53→3 行）。
3. **采样协议放大**：27B 适配在 temp 0.1 下被 base 先验淹没（英文思维链+空 args），greedy 下完美
   （受控消融：同 prompt 同权重，仅温度不同）。修复：`AA_EVAL_GREEDY=1` 确定性评测口径。
4. **（负结果）多轮历史 conditioning**：v4 让模型学会复读历史模式，反伤首轮决策——单轮决策样本
   + 强规范化手册的组合仍是小数据 LoRA 的最优解。

## 6. 评测矩阵（81 任务 strict / clean-val ROUGE-1）

| 模型 | benchmark strict | clean ROUGE-1 | 备注 |
|---|---|---|---|
| 27B base | 30.9% | 0.1349 | 零样本（greedy 24.7%） |
| 27B v3 SFT | 终验中 | **0.1662**（+23%） | benchmark 贪心口径出分中 |
| 8B base | — | 0.1139 | — |
| 8B SFT v3 | 66.7% | 0.1446（+27%） | 双闸门通过 |
| 8B DPO v3 | 65.4% | 0.1435 | 完成率 97.5% |
| **8B GRPO v3** | **71.6%**（双次可复现） | **0.1509（+32%）** | 全指标最优 |

运行方差：贪心下同模型两次运行任务翻转率 4.9%（GRPO）~17.3%（SFT），结论均以配对 repeat 支撑。

## 7. 结论与发布

- **8B GRPO 全链路版 = 主发布模型**：同口径（greedy）strict 72.8%，为 27B base 的 2.9 倍；
  全指标（完成/tool/arg/false_fin/token）最优；端侧友好。已发布 ModelScope。
- **27B cap = 图表理解特化研究版**：clean ROUGE +60%/+87%（全项目最优），假结束率 4.9%；
  工具决策未及 base（21.0% vs 30.9%）——已发布并如实标注边界。
- **27B agent 化的边界**：小数据 LoRA（4k 行）在 27B 上受强先验压制，容量 ×4 + lr ×3.3
  只把 strict 从 13.6%→21.0%，仍未跨过 base。突破该边界需要数据规模/多样性的量级提升
  或全参数训练——超出本轮范围，留待后续。
- 方法论沉淀：数据质量（clipart/会话块）> 配方花样（多轮负收益）；容量占比决定 LoRA
  可迁移性；评测协议（采样温度）本身可以构成大模型评测的混淆变量，需纳入消融。

## 8. 复现指南

```bash
# 环境（覆盖层，不污染 vendor）
export HCU_ROOT=~/env/hcu_root
export PYTHONHOME=$HCU_ROOT/usr
set +u; source $HCU_ROOT[LOCAL_PATH_WITHHELD] >/dev/null 2>&1; set -u
export PYTHONPATH="$HOME/env/pip_overlay_tf514:$HCU_ROOT/usr/local/lib/python3.10/dist-packages"
export AA_RL_BLOCK_VLLM=1            # 屏蔽 vendor 旧 vllm（GRPOTrainer 导入需要）
export AA_EVAL_GREEDY=1              # 大模型本地评测确定性口径
export HIP_VISIBLE_DEVICES=2,3       # 选卡用 HIP

# 数据 → SFT → 评测
python scripts/build_vlm_policy_sft_mix.py --min-image-side 300 --canonicalize-tools \
  --output data/vlm/vlm_policy_sft_v3_canonical.jsonl
python -m AgenticArxiv.rl.train_vlm_sft --model <base> \
  --data data/vlm/vlm_policy_sft_v3_canonical.jsonl --device_map auto --lr 3e-5
VLM_ADAPTER_PATH=<output>/final python scripts/eval_figure_qa_vlm.py ...
VLM_ADAPTER_PATH=<output>/final python -m AgenticArxiv.benchmark.run_benchmark \
  --task-set expanded --offline --backend transformers --agents regex ...
```

产物索引：训练 manifest 与逐轮评测 JSON 见 `artifacts/`（`vlm_figureqa_eval/`、
`bench_*_q38_27b*/`、`bench_*_q3vl8b*/`）；模型权重发布于 ModelScope（adapter-only）。
