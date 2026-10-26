# RLCD 自训 next-tool 判别头：实验记录

> Roadmap P2「RLCD 自训 Jev」的落地记录。目标：用 RLCD（Reinforcement
> Learning from Contrastive Distillation，[arXiv:2307.12950](https://arxiv.org/abs/2307.12950)
> 精神——正/负对比信号 + 对比损失，免人工标注）在仓库内训练 next-tool
> 判别头，替代外部 TypeSafe Jev API 依赖。训练于 2026-10-26 完成，单张
> 海光 DCU（64GB）。

## 1. 方法

**判别形态**：Qwen2.5-1.5B（起点 `outputs/sft/final`）全参 bf16，对 11 个
候选（10 工具 + FINISH）做 verbalizer 打分——prompt 单次 forward + KV cache
批量展开 + 11 个候选后缀一次 batch forward，**z 只聚合候选名 token 段**
（共享前缀 `"Next tool: "` 对所有候选等值，进均值只会稀释信号；冒烟阶段
实测此前缀把零样本准确率压到 2.1%，剔除后 53.3%）。

**RLCD 损失**：`L = CE(z/τ, gold) + λ·Σ_neg[-log σ(β·(z_gold − z_neg))]`。
论文的 preference model + PPO 由 RLVR 确定性标签替代（`expected_tools`
按位重算，免人工标注）；负样本 = 失败轨迹首分歧步上策略**真实选错的
工具**（天然 hard negative）。

**双臂对照**：臂 A = λ0（纯 CE 的 router SFT 基线）vs 臂 B = λ0.5。同数据
行流、同 lr 2e-5、同 seed permutation（sha256 写入两侧 manifest）、同
eval 协议；早停 = 连续 3 个冻结评测点复合分（acc − 0.3×ECE）无改善。

## 2. 数据（`data/rlcd/rlcd_v1.jsonl`，manifest SHA256 冻结）

2143 个决策点 = gold 链回放 232（151 工具步 + 81 链尾 FINISH，上采样 ×2）
+ policy 成功步 1546 + 硬负样本（挂 470 行，error_pattern 标注）。来源
为 58 个 `artifacts/*/traces.jsonl` 的全部 2408 条轨迹；任务级 v3_81 切分
（train 1381 / dev 230 / iid_test 353 / ood_test 179），dev/iid/ood 决策点
训练不可见。标签全部由 TaskSpec 确定性重算。

## 3. 双臂结果（冻结评测 992 点，含 S1 首步 81 / S2 gold 全链 232 /
S3 留出轨迹步 509 / S4 错误模式切片 170）

| suite | 臂 A（CE） | 臂 B（+对比项） | Δ |
|---|---|---|---|
| S1 首步 81 | 6.2% | **53.1%** | +46.9pp |
| S2 gold 全链 232 | 34.9% | **76.7%** | +41.8pp |
| S3 留出轨迹步 509 | 61.1% | **91.0%** | +29.9pp |
| S4 错误模式 170 | 38.2% | **61.8%** | +23.5pp |
| overall 992 | 46.6% | **79.5%** | +33.0pp |

McNemar 精确检验（配对 n=992，discordant 13/340）：**p < 0.0001**；
bootstrap 95% CI of Δ(B−A) = **[+29.8, +36.2]pp**。

**结论：RLCD 对比项是本任务的主要增益来源。** 纯 CE 臂从 SFT 起点几乎
无法提升（准确率停滞在 53%——被 eval 池中占多数的链中步行为锚定），
显式压低硬负样本的对比项把四个 suite 全部大幅拉升。臂 B 训练曲线
53.3% → 68.0% → 82.4%（训练评测池），置信度校准良好（ECE 0.14）。

## 4. 与外部 Jev 的对照（诚实口径）

| 决策面 | 外部 Jev (jev-latest) | 本地判别头（臂 B） |
|---|---|---|
| 81 任务首步 | **86.42%**（`jev_route_expanded_all.json`） | 53.1%（v1 同口径） |
| 链中步（S3，留出） | 无实测（pilot 只测首步） | **91.0%** |
| 接受门控（conf ≥ 0.80）内 | — | 全量 87.8% 接受 × 高准确率（S3：97% 接受 × 92.3%） |

**首步是已知短板，且失败形态高度集中**：S1 的 42 个错误中 26 个是同一
形态——`analyze_*`/`chain_*` 复合链（「检索最近 7 天 cs.AI 论文 5 篇，
下载第 4 篇…」）的首步，任务文本里「下载/图表」词汇显著，模型把**最
显著工具**当成了**第一步工具**，且 z 呈精确 50/50 平票（两类未分离）。
download/translate 等直接指认类任务首步为 12/12、8/8 全对。

**门控使该短板在线上安全**：这 26 个平票案例 conf≈0.50 < 0.80，guided
模式下全部回退给策略模型，判别头不参与决策。S3 类链中步（运行时出现
频次最高的决策面）接受率 97%、接受内准确率 92.3%。

## 5. 训练侧工程注记

- verbalizer scoring 的 KV cache 批量展开做了三代 Cache 结构探测
  （`to_legacy_cache` → `key_cache/value_cache` → `layers[i].keys`），
  未知结构响亮报错并回退 `--scoring batch`；双打分路径在注意力替身
  单测中验证逐位一致。
- BPE 边界：`": get"` 类合并使前缀无精确 token 边界，取「第一个解码
  超长于前缀的下标」为名字段起点，跨界 token 含名字首段信息，纳入
  打分而非丢弃。
- 单卡峰值显存 21.3GiB（cache 路径）；`foreach=False` AdamW +
  装载后 page cache 逐出 + `guarded_save` 全套 DCU 纪律。
- 训练预算：单臂 ~30 分钟（382 optim 步 × 2 evals/epoch），远低于预估。

## 6. 局限与下一步

1. **首步复合链判别未解决**：候选改进——① 对 analyze/chain 首步定向
   上采样 + 把「download」挖成 recent-gold 行的负样本（当前负样本池里
   该形态覆盖不足）；② 更大基座（Qwen3-1.7B）重训两臂；③ 首步直接
   保留 policy/Jev 处理、本地判别头只接管链中步（当前门控已自然实现）。
2. **接线（Roadmap 阶段 W）未做**：`TOOL_ROUTER=local` backend 与
   routing pilot 端到端验证待离线指标达标后进行；本记录的门控分析
   表明接线的主要价值在链中步。
3. 外部 Jev 没有链中步实测，两者只在首步 81 上同口径可比；「替代」
   的完整论断需要 e2e guided benchmark。
4. 评测协议均为确定性 argmax（RLVR 兼容）；置信度校准（ECE 0.14）
   支持 0.80 门控语义，但阈值未做扫描。

## 7. 复现

```bash
# 数据（CPU，~42s）
python scripts/build_jev_rlcd_dataset.py
# 双臂（单卡；两臂串行 ~1h）
HIP_VISIBLE_DEVICES=2 python -m AgenticArxiv.rl.train_jev_rlcd \
  --model outputs/sft/final --data data/rlcd/rlcd_v1.jsonl \
  --output_dir outputs/jev_rlcd_armA --lambda 0
HIP_VISIBLE_DEVICES=2 python -m AgenticArxiv.rl.train_jev_rlcd ... \
  --output_dir outputs/jev_rlcd_armB --lambda 0.5
# 冻结评测 + 对照
HIP_VISIBLE_DEVICES=2 python scripts/eval_jev_local_router.py \
  --router-model outputs/jev_rlcd_armB/final --output artifacts/jev_local_router_armB.json
python scripts/eval_jev_local_router.py --compare artifacts/jev_local_router_armA.json \
  --output artifacts/jev_local_router_armB.json
```

产物：`outputs/jev_rlcd_arm{A,B}/`（权重 + training_manifest）、
`artifacts/jev_local_router_arm{A,B}.json`、`outputs/jev_rlcd_armA/train_log.jsonl`。
