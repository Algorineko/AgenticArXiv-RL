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

## 端侧多模态文档理解参照系

P0（策略侧多模态化）的外部参照：SmolDocling-256M（IBM+HF，端到端文档转换，DocTags 结构化输出）、PaddleOCR 3.0（轻量工程基座）、MiniCPM5-2B（128K 上下文端侧稠密模型）、InternVL/Qwen-VL 1B-2B 档。共同结论：端侧文档 VLM 的可行规模在 0.25B–4B 之间，本项目 Qwen3-VL-4B（env 侧已验证）与 Qwen2.5-VL 2B 档是策略侧候选。
