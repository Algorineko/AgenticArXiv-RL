# FigureQA 后训练前后对比（T5 env 侧 VLM）

- 模型：`Qwen3-VL-4B-Instruct`（base） vs `outputs/vlm_figureqa/merged`（LoRA r=32 合并版）
- 数据：`data/vlm/figureqa_seed.jsonl`，按论文切分（train 62 篇 / val 11 篇，留出 94 图 × 问题）
- 协议：与快照录制完全同构（缩放到最长边 1024、同一问题提示词、贪心解码、96 token 预算）；指标为对 caption 证据的 ROUGE-1/ROUGE-L F1（**代理指标**，非人工评判）
- 原始数据：`base.json` / `trained.json`（逐条 target/answer/ROUGE）

## 汇总（留出 94 条）

| 问法 | n | base R1 | 训练后 R1 | base RL | 训练后 RL | 空/错误 |
|---|---|---|---|---|---|---|
| describe | 65 | 0.108 | **0.136** | 0.081 | **0.111** | 0/0 |
| axes | 24 | 0.081 | **0.114** | 0.075 | **0.087** | 0/0 |
| trend | 5 | 0.178 | 0.049 | 0.118 | 0.049 | 0/0 |
| **合计** | 94 | 0.105 | **0.126** | 0.081 | **0.102** | 0/0 |

训练后回答显著变短（describe 平均 47 → 18 token），风格向 caption 收敛——这正是监督目标的形状；G-Eval 类质量评判不在本表覆盖范围。

## 样例（Δ = 训练后 RL − base RL）

```
[+0.21] 2609.14827v1 fig2 describe
  target : Learning outcomes versus number of states for Bertrand Congestion Model with We = 1, N = 2, δ = 0.99.
  base   : This figure presents three plots analyzing a Bertrand competition model with congestion, showing
           how equilibrium quantity, price, and revenue change a...
  trained: Bertrand with Congestion: Equilibrium Quantity vs Kstate (We=1)

[-0.15] 2609.24537v1 fig5 describe
  target : Pixel-level redaction quality: standard mask IoU and person coverage (|Mpred ∩ Mgt|/|Mgt|).
  base   : This bar chart compares the performance of two metrics — “Mask IoU” (green bars) and “Person
           Coverage” (orange bars) — across diff...
  trained: | Comparison with existing methods on the three evaluation scenarios. Each bar shows the mean
           score across 10 random seeds.
```

## 诚实说明

- ROUGE 对 caption 的重叠只是**代理指标**：回答可以是对的但与 caption 措辞不重叠（base 的定位即如此），也可能照抄 caption 却答非所问。
- trend 留出仅 5 条且本次测量回退——样本量下这几乎是噪声，但结论按其字面报告，不做挑选。
- 监督目标来自 caption 的关键词匹配（与项目抽取式后端同一套线索词），本身带噪声（例如「Lower-level」误命中 trend 线索）。
- 3 条录制失败（`2608.14539v1` fig5 图片文件缺失）未计入评测。
