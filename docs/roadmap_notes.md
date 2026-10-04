# Roadmap 调研笔记

> 承接 README 外移的方向性调研：SAO 论文综述、Jev 判别式决策模型、RSI 自进化。条目状态见 README 的 Roadmap 章节；本文档存完整论据与引入路径。

## SAO：下一代异步 Agentic RL 算法

> **SAO（Single-Rollout Asynchronous Optimization，单 rollout 异步优化）** 由清华大学 KEG 实验室提出（2026-07），是 GRPO 在**异步 agentic 训练**场景下的演进方向。核心动机：长程 agent 任务的 rollout 是训练瓶颈，GRPO 的组式采样在异步下会 off-policy、不稳定（典型 <200 步即崩）。
>
> 五个关键技术点：
> 1. **单 rollout 采样**：每个 prompt 只生成一条轨迹、随到随训，替代组式对比；
> 2. **DIS 直接双边重要性采样**：用 rollout 时记录的 token logprob 计算 `r_t = π_θ / π_rollout`，越出信任区间 `[1−ε_l, 1+ε_h]` 的 token **直接掩码为 0**（非 PPO 式单侧 clip）；
> 3. **value model 解耦更新**：策略:value = 1:2 更新频率，value 训练时**冻结注意力层**（只训 MoE 投影层）；
> 4. **skip-observation GAE**：优势只在模型生成的 token 之间传播，跳过环境观察 token，滤除环境噪声。
>
> 效果：稳定训练 ~1000 步，AIME2025 达 **97.3%**（vs GRPO 84.2%），SWE-Bench Verified 29.8%，已用于 GLM-5.2（750B）训练。
>
> 引入路径：多轮 rollout 已落地 → 引入 skip-observation 掩码与 DIS 双边裁剪 → 迁移 verl `fully_async_policy`（`gen_batch_size=1` / `staleness_threshold` / token 级 TIS 裁剪，与 SAO 思路一致）或 AReaL v1.0 实现全异步 + value model。
>
> 📄 **论文**：[Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning（arXiv:2607.07508）](https://arxiv.org/abs/2607.07508)（清华 KEG，官方代码尚未开源）

## Jev 式判别式决策组件

**Jev**（TypeSafe AI，2026-09）是近期走红的判别式/非自回归决策模型：放弃逐 token 生成，用并行计算直接输出结构化决策（`choice` / `score` 等），成本约为同任务 LLM 的 1/400；定位是 LLM 的 System-1 补充——「选择题交给 Jev，简答题交给 LLM」。

对本项目的引入思路（三条候选落点，按价值排序）：

1. **Verifier / 结果质量判别**：`result_quality` 与坏例回放目前是规则判定；Jev 式判别头可以学到更丰富的「observation 是否真实成功」分类，仍可离线回放、保持可复现。
2. **Reward router**：五分量奖励的加权/门控用判别头做路由（哪些分量该压制、哪些该放大），替代固定课程表。
3. **工具选择的结构化解码**：动作空间是封闭枚举（8 工具 + FINISH），天然适合判别式输出——把「选哪个工具」从自回归生成改为判别头打分，生成式部分只产参数，端侧推理成本大幅下降。

关键约束：本项目的一切奖励必须**可复现**（RLVR），Jev 头要么确定性推理（argmax），要么作为诊断信号不进梯度——与 T5 VLM「不进奖励」的隔离原则一致。

## RSI 受限自进化（bounded RSI）

**RSI（Recursive Self-Improvement，递归自我改进）** 是 2026 年的工程化热点：上交/清华/字节发布四步循环路线图（[arXiv:2609.11873](https://arxiv.org/abs/2609.11873)：重建环境 → 发现能力差距并生成训练数据 → 训练任务模型 → 迭代环境产出更多任务）；Sakana AI 设 RSI Lab；ICLR 2026 设专题 Workshop；智谱以「下一代模型在上一代模型构建的环境中训练」为路线融资。学界对「真 RSI vs 受限自动化优化」仍有分歧——本项目目标明确取**受限**（bounded）一侧。

本项目已具备 bounded RSI 的全部前置件，缺的只是自动闭环：

```
留出评测（run_release_eval_matrix.sh 四切分）
   ↓ 暴露弱点（失败轨迹 → 坏例库 eval_cases.jsonl 已有 open/fixed 状态机）
自动生成针对性训练数据（generate_parametric_sft_data.py 参数化派生已有）
   ↓
再训练（train_sft / train_grpo 管线已有）
   ↓ 迭代：冻结新留出集、防止数据自噬
```

验收口径沿用量化闸门：每轮自进化后四切分 strict pass^3 不回退、坏例库只增不减、生成数据经 manifest SHA256 冻结审计。

### 实现状态：编排器（`rl/self_evolve.py`）

闭环的 1~4、6 步是对 JSON 文件的纯函数，不加载模型、不联网、不 import torch；第 5 步「再训练」只是一条不透明命令。全部复用现有机制，不改奖励、工具与任务模板：

| 步骤 | 实现 | 复用的现有组件 |
|---|---|---|
| 1 diagnose | 读取留出评测 `summary.json` 的 `details`，按模板族聚合 strict 成功率 / pass³ / 失败模式（`false_finish`、`wrong_tools`、`wrong_args`、`wrong_ref`、`parse_fail`、`tool_fail`…） | `benchmark/report.py` 的 details 口径、`benchmark/splits.py::template_key`、`is_strict_success` 同款判定 |
| 2 mine | 把同一轮评测的 `traces.jsonl` 里的沉默失败固化为 **本轮** 坏例文件 `artifacts/self_evolve/round_N/eval_cases.open.jsonl`（`open`） | `benchmark.badcases.capture`（挑选规则与 `reproduces_when` 完全同库）；并入 `eval/eval_cases.jsonl` 与 `open→fixed` 仍是人工步，闭环从不改写主库 |
| 3 select | 只在 **train 切分** 里挑弱族的父任务（最差族优先、族内按父任务自身 strict 率升序、确定性），且父任务必须已有参数化派生 | 参数化 seed manifest 的 `tasks` 血缘（derived → parent） |
| 4 synthesize | 子进程调用 `scripts/generate_parametric_sft_data.py --only-parents id,id`（新增参数，未知 id 直接报错，manifest 记录 `only_parents`） | 既有派生 + 离线环境执行 + manifest |
| 5 freeze | `artifacts/self_evolve/round_N/manifest.json`：每个输入/输出的 SHA256、git revision、选择结果；`lineage_sha256` 可审计 | 与 SFT manifest 同款哈希约定 |
| 6 gate | 新旧四切分 strict pass³ 不回退：iid / ood 永远严格（容差 0），只有 dev（n=8，噪声大）可显式给 `--dev-tolerance`（如 0.125 = 一题）且记入 `gate_result.json`；坏例数只增不减；留出切分文件与冻结哈希逐字节一致。任一不满足则本轮拒绝、不晋升 | `report.py` 的 pass^k 定义（组合数口径，试验数 < k 的任务跳过） |

```bash
# 诊断（可单独跑）
python -m AgenticArxiv.rl.self_evolve diagnose --eval dev=eval_results/<run>_dev8/summary.json   --eval iid_test=eval_results/<run>/summary.json --split data/splits/v3_81.json
# 一轮：诊断 → 挖坏例 → 选父任务 → 定向派生 → 冻结 →（可选）再训练
python -m AgenticArxiv.rl.self_evolve run --round 1 --eval dev=... --eval iid_test=... --traces traces.jsonl   --split data/splits/v3_81.json --snapshot data/mock_arxiv_snapshot.json   --parametric-manifest data/sft/sft_v2_parametric_seed.jsonl.manifest.json   --train-cmd "python -m AgenticArxiv.rl.train_sft --data artifacts/self_evolve/round_1/... --skip_data_manifest_check"
# 闸门：新策略评测完成后
python -m AgenticArxiv.rl.self_evolve gate --round-dir artifacts/self_evolve/round_1   --prev dev=... --prev iid_test=... --prev ood_test=... --new dev=... --new iid_test=... --new ood_test=...   --cases-before 14 --cases-after 17 --split data/splits/v3_81.json
```

刻意不做：LLM-as-judge、新奖励分量、新工具/模板、自动 `open→fixed`、训练循环改动。真实一轮（Qwen2.5-1.5B-SFT 起步 + 四切分对照）待跑，结果无论闸门通过与否都如实记录。

## 端侧多模态文档理解参照系

P0（策略侧多模态化）的外部参照：SmolDocling-256M（IBM+HF，端到端文档转换，DocTags 结构化输出）、PaddleOCR 3.0（轻量工程基座）、MiniCPM5-2B（128K 上下文端侧稠密模型）、InternVL/Qwen-VL 1B-2B 档。共同结论：端侧文档 VLM 的可行规模在 0.25B–4B 之间，本项目 Qwen3-VL-4B（env 侧已验证）与 Qwen2.5-VL 2B 档是策略侧候选。
