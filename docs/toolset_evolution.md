# 工具集演进设计（增量路线图）

> 本文档承接 README 外移的「工具集演进设计」章节，记录 T1–T5 工具从设计到落地的完整决策链。
> 本项目的最终目标是**本地部署的轻量多模态模型独立完成 arXiv 论文的检索、下载、翻译、总结与图表分析**。T1–T5 已全部落地。

## 现状盘点与缺口

| 工具 | 能力 | 边界 |
|------|------|------|
| `get_recently_submitted_cs_papers` | 子领域 + 时间窗检索 | 只认 `cat:cs.*` + 提交日期，无翻页；摘要截断 200 字符 |
| `search_arxiv_papers` | 关键词/篇名/作者检索 | 无翻页；离线模式对快照外查询返回显式标记的确定性降级结果 |
| `download_arxiv_pdf` | 下载 PDF | — |
| `translate_arxiv_pdf` | pdf2zh 整篇翻译 | 产出是翻译后的 PDF 文件，翻译后的正文不进入模型上下文；依赖可选 extra |
| `get_paper_cache_status` | 查缓存 | — |
| `get_paper_content` | 读取摘要或 method/result/conclusion 章节 | 需要先下载 PDF；确定性抽取，不调用 LLM |
| `summarize_paper` | 按 style/budget 生成摘要 | 需要先下载 PDF；默认走确定性抽取式后端，`local_model` 后端为可选 |
| `extract_paper_figures` | 抽出内嵌图表图片 + caption，返回文件路径 | 需要先下载 PDF；纯向量图（无内嵌位图）会返回 `count: 0` |
| `analyze_figure`（T5，多模态） | env 侧调本地 VLM 读图回答 | VLM 只在 env 侧；策略仍是纯文本小模型 |

三个结论：

1. **「检索」半环已打通**：既支持时间窗浏览，也支持关键词、篇名和作者查找；翻页仍未实现。
2. **「分析解读」闭环已打通**：读内容 → 总结 → 抽图三步都是确定性工具，模型可以独立完成一篇论文的解读链。图表**语义分析**（T5）已落地：env 侧 VLM 由本项目后训练（FigureQA）并在构建快照时录制答案，策略学会「检索 → 下载 → 抽图 → 分析」四步链。
3. **动作空间不是越大越好**：策略是 1.5B 量级小模型，每加一个工具都放大工具选择与 JSON 格式的学习负担。新增工具的准入标准是「能开启一类新任务」，而不是「可能有用」。

## 工具演进记录（按依赖顺序）

| 优先级 | 工具 | 设计要点 | 为什么是 RLVR 友好的 |
|--------|------|----------|----------------------|
| **T1** ✅ | `search_arxiv_papers(query, max_results, days=None)` | 关键词检索，映射 arXiv API 的 `all:` / `ti:` / `au:` 字段；与现有工具并存（时间窗浏览 vs 精确查找是两类任务） | 期望工具/参数照常由 `task_spec.steps` 派生；`MockArxivEnv` 按查询串哈希离线回放，未收录的 query 走**确定性降级**（返回固定子集并在 observation 里显式标注），保证可复现、也防止模型把空结果当检索成功 |
| **T2** ✅ | `get_paper_content(ref, section=None)` | PDF → 纯文本（PyMuPDF），默认返回 title/abstract，可按节取（method / result / conclusion） | 确定性文本抽取，无 LLM 参与；快照预存抽取结果。**它是全部解读类任务的前置件** |
| **T3** ✅ | `summarize_paper(ref, style, max_words)` | 总结论文：**env 侧**生成摘要（输入来自 T2 的文本），返回摘要文本 | 可训练的是「何时调、对哪个 ref 调、style/长度参数对不对」——全部规则可判；摘要质量本身**不进奖励**（见下） |
| **T4** ✅ | `extract_paper_figures(ref)` | 图表可视化准备：抽出图表图片 + caption，返回文件路径列表 | 确定性；验证「ref 正确 + 数量 ≥ 1」 |
| **T5** ✅ | `analyze_figure(ref, figure_no, question=None)` | 图表分析：env 侧调本地 VLM（本项目后训练的 Qwen3-VL-4B FigureQA）读图回答 | 规则只判「调没调对、参数对不对」；VLM 回答质量不进奖励，避免把第三方模型的噪声写进策略梯度 |

**T3 的摘要后端**：落地时默认采用**确定性抽取式后端**（按 section 取整句、按词数预算裁剪，不采样、不调模型），原因有三：它让同一条轨迹在任意时刻回放都逐字节一致；它不需要在 `build_snapshot` 之外再引入一个需要权重的前置条件；而奖励只看工具调用决策、不看摘要文字，模型后端对训练信号没有贡献。需要更自然语言的摘要时可用 `SUMMARY_BACKEND=local_model SUMMARY_MODEL_PATH=<本地模型目录>` 切到模型后端（贪心解码，仍然确定性），代价是快照构建阶段要加载权重。

配套任务模板（沿用 `tasks_expanded.py` 的分类，全部由 `task_spec.steps` 声明式派生）：

- `search_kw_*`：关键词检索类（T1）
- `paper_reading`：检索 → 下载 → 读内容（T2），5 条
- `paper_summary`：检索 → 下载 → 总结（T3），5 条
- `long_chain` 的 `chain_ai5_read_then_summary`：读 → 总结串成 4 步链（T2+T3）
- `figure_extraction`：检索 → 下载 → 抽图（T4），4 条
- `analyze_figure`（T5）：检索 → 下载 → 抽图 → 图表分析，仅多模态环境启用

## 训练与评测侧的连带设计

1. **快照扩展**：`build_snapshot.py` 一次跑齐——除搜索结果外，对快照论文**预抽取全文文本与图表文件**；新工具全部离线回放，维持「build_snapshot 是唯一联网步骤」的约定。
2. **奖励零改动**：五分量方案原样复用；`expected_tools` / `expected_tool_args` 从 `steps` 派生，`reward.py` 与课程学习均不需动。
3. **防 reward hacking**：解读类工具天然多出「乱调工具刷 process 分」的面——沿用 `run_baselines.py` 逐类目闸门 + `eval/eval_cases.jsonl` 单例钉死（例如：ref 指向不存在论文时调 `summarize_paper` 必须扣分）。
4. **摘要质量的奖励问题（刻意不做）**：把「摘要写得好不好」变成奖励需要 LLM-as-judge 或 rubric 打分，会引入非确定性奖励与新的 hacking 面。设计上先把总结收敛为**工具调用决策问题**（何时调、对谁调），质量评估留到长期单独立项。
5. **多模态的边界（刻意隔离）**：T5 的 VLM 只活在 env 侧，策略仍是纯文本小模型——动作空间里只有「调不调、怎么问」，看图能力外包给环境。只有策略本身换成多模态模型时，才考虑把图片放进 observation（见 README Roadmap P0）。
6. **硬件门槛**：只有 T5 需要一个 env 侧模型（VLM 约 6GB 量级，量化后更低），它与训练显存互不影响（不进梯度）。T3 默认的抽取式后端不加载任何权重，所以解读闭环（T1–T4）在训练机之外不需要额外硬件。

## 落地顺序

```
T1 关键词检索 ──→ T2 读内容 ──→ T3 总结          （解读闭环，已完成）
                      └────→ T4 抽图 → T5 图分析 （全部完成）
```

每落一个工具：扩任务模板 → 重跑 `run_baselines.py` 重新卡各类目区分度门槛 → 重新生成 SFT/DPO 数据 → `eval/eval_cases.jsonl` 补对应用例。T3/T4/T5 落地时同步做了：

- 任务模板：新增 `paper_reading`（5 条）、`paper_summary`（5 条）、`figure_extraction`（4 条）与 4 步解读链 1 条，扩展集 62 → 81 条（十三类模板）
- 切分：新增 `data/splits/v3_81.json`（v1/v2 保持不变，历史实验的成功率仍然可比）
- 区分度：`run_baselines.py` 逐类目闸门覆盖新类目（`tests/test_reward_discrimination.py`）
- 坏例：`eval/eval_cases.jsonl` 新增 8 条 `hack/summary-*` / `hack/read-*` / `hack/figure-*` 用例，钉住「预算传错」「跳过下载直接总结/抽图」「多传 section」「抽错论文」等骗分形态（用例库现共 14 条）

## T5 落地记录（2026-09）

工具、环境集成、任务模板、单测与参数化 SFT 派生规则之外：

- **env 侧 VLM 后训练**：Qwen3-VL-4B + caption 监督 → [FigureQA](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA)（451 可用图 / 73 篇论文 / 630 行种子数据，按论文切分防泄漏；留出 94 图 ROUGE-1 0.105 → 0.126、ROUGE-L 0.081 → 0.102，trend 小样本回退如实记录）
- **快照录制**：`FIGURE_ANALYSIS_BACKEND=vlm VLM_MODEL_PATH=<FigureQA 目录>` 运行 `python -m AgenticArxiv.rl.backfill_figure_analysis --snapshot ... --force --paper-id <arXiv id>`（`--paper-id` 可限定范围、`--force` 覆盖既有抽取式条目）；默认 `extractive` 后端不变
- **策略侧 SFT-T5**：[SFT-T5 权重](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5)（`data/sft/sft_v3_t5_parametric_seed.jsonl`，86 派生任务 / 249 行 + 语言扩增）。评测（seed 45 / repeat 3 / 离线）：`analyze_cv5_ref3_trend` strict 3/3；两个 `describe` 任务与一个 `axes` 任务工具调用步骤正确但 `question` 枚举绑定错误（误传 `tldr`/`trend`）——枚举在未见「父任务问法 × 图」组合上的泛化缺口如实记录
- **附带修复**：SFT 阶段验证此前用伪聊天格式裸文本（`System:/User:/Assistant:`）算 parse_rate，对 chat 模板训练的模型系统性压低（已发布 SFT 与新模型同测均 1/16）；改为按 chat 模板渲染 ReAct prompt 后，同口径为 0.750 / 0.625
