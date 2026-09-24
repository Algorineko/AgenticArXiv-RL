# 离线回放与快照排错

> 本文按当前 `main` 的源码编写。快照是环境输入，不是模型权重，也不代表
> benchmark 结果；没有快照或快照不完整时，最安全的行为是让命令明确失败。

## 1. 先区分三件事

离线 RL 流水线里有三个容易混淆的边界：

1. **模型后端**：`run_benchmark --backend api|transformers` 决定 LLM 从哪里来。
2. **工具环境**：`MockArxivEnv` 的 `mode` 决定工具结果从 snapshot 读取、真实调用并记录，还是 miss 后自动真实调用。
3. **网络输入**：`build_snapshot.py` 是准备 snapshot 的联网步骤；后续 replay 不应悄悄联网。

因此“本地模型”不等于“离线”，“快照回放”也不等于“模型不运行”。一个
本地 transformers 模型仍然可以在不加 `--offline` 时调用真实 arXiv；一个 API
模型也可以配合 `--offline` 使用固定工具结果。

## 2. `MockArxivEnv` 三种模式

源码：`AgenticArxiv/rl/env.py`。构造函数的 `mode` 只接受 `replay`、`record`
和 `auto`。

| 模式 | 命中 snapshot | miss 时行为 | 适用场景 |
|---|---|---|---|
| `replay` | 返回记录结果 | 计数为 miss 后抛 `KeyError`，不联网 | 正式离线训练、可复现 benchmark、CI |
| `record` | 不直接复用快照中的同名记录 | 调真实工具并写入 snapshot | 构建或补充快照 |
| `auto` | 返回记录结果 | 调真实工具；部分搜索失败时可进入固定 fallback，并写入成功结果 | 开发探索、调试，不适合作为严格对比 |

快照 key 会剔除 `session_id`、`output_path`、`save_to_file` 和内部的
`_resolved_paper_id` 等易变字段；论文正文、摘要、图表和图表分析还会先把
`ref` 解析为 paper id，避免不同会话中的 `ref=1` 指向不同论文却错误命中
同一条结果。

### 2.1 replay miss 为什么应该失败

`replay` 的 miss 错误类似：

```text
[MockArxivEnv] replay 模式下快照缺失: tool=<tool> key=<key>。
请先运行 `python -m rl.build_snapshot` 生成快照。
```

这不是“网络暂时不可用”的普通警告，而是实验输入不完整。若 replay miss
自动访问网络，以下问题会被掩盖：

- 同一 prompt 在不同时间看到不同论文池，成功率失去可比性。
- `ref=2` 解析到另一篇论文，参数评分和指代评分同时被污染。
- 一次 API 回退改变会话状态，后续步骤看似成功但不属于原始任务。
- 训练时偶尔联网，导致同一 checkpoint 无法重建相同 observation。

所以排查 miss 的顺序应是“确认 key 和 snapshot 版本”，而不是先打开网络。

## 3. 工具处理策略

| 工具 | replay 中的来源 | record 中的行为 | 重点排错 |
|---|---|---|---|
| `get_recently_submitted_cs_papers` | 按 aspect 从快照池切片 | 真实 arXiv 搜索并记录 | aspect、days、max_results、锚点池 |
| `search_arxiv_papers` | 按 query hash + days 命中 | 真实关键词搜索并记录 | query 空白、规范化、未命中 fallback |
| `download_arxiv_pdf` | 默认走离线下载桩，不发 HTTP | `build_snapshot` 阶段关闭桩并真实下载 | session 论文状态、ref、占位 PDF |
| `get_paper_content` | 按已解析 paper id 和 section 命中 | 读取下载内容并记录 | 论文 id、section 是否存在 |
| `summarize_paper` | 按 paper/style/budget bucket 命中 | 运行配置的摘要后端并记录 | `tldr/structured/bullet`、60/120/250 bucket |
| `extract_paper_figures` | 按 paper id 命中 | 读取 PDF 抽取图表并记录 | `count=0` 是事实，不等同于工具异常 |
| `analyze_figure` | 按 paper/figure/question 命中 | 使用 extractive 或 VLM 后端并记录 | 图表必须先抽取，question 是枚举 |
| `translate_arxiv_pdf` | 由 `LocalSideEffectManager` 截获 | 真实翻译副作用不走普通 snapshot | session、ref、线程参数和 side effect 状态 |
| `get_paper_cache_status` | 纯本地执行 | 纯本地执行 | 是否先建立同一 session 的论文状态 |

`download_arxiv_pdf` 在 rollout 的离线桩中会写一个固定的最小 PDF 头、更新
MemoryStore 的 `PdfAsset(status="READY")`，并把时间固定到 2000-01-01，避免
文件残留或墙上时间改变下一步 observation。它模拟契约和状态转移，不提供
真实 PDF 内容；需要正文、摘要或图表时仍必须使用包含内容记录的 snapshot。

## 4. 正确的快照生成路径

入口：`AgenticArxiv/rl/build_snapshot.py`。从 `AgenticArxiv/` 运行：

```bash
python -m rl.build_snapshot \
  --snapshot ../data/mock_arxiv_snapshot.json \
  --aspects AI LG CL CV RO CR \
  --max_results 50 \
  --days 30
```

生成过程分为四层：

1. 记录每个 aspect 的检索池和配置的关键词查询。
2. 固定 benchmark 所需的 AI/CV 参考论文并校验前五条池是否包含锚点。
3. 并行预取 PDF；预取有整体预算，卡住的任务会交给后续串行阶段重试。
4. 对每篇唯一论文记录正文、常见 section、摘要预算网格、图表抽取和图表分析。

`--content-max-ref N` 只限制正文和图表预取的深度，不减少搜索返回数量；
需要让所有 `ref` 任务可复现时，不能只因为搜索池存在就假设深层正文也存在。

### 4.1 记录完成后的只读核对

按源码可以做下列静态核对（本文不声称已执行这些命令）：

```powershell
# 查看顶层工具分组和记录数量；只读取 JSON
$snapshot = Get-Content -Raw data/mock_arxiv_snapshot.json | ConvertFrom-Json
$snapshot.PSObject.Properties | Select-Object Name
```

需要核对：

- 搜索工具至少有任务所需的 aspect 或 query 记录。
- `extract_paper_figures` 的结果包含 `paper_id`、`figures` 和 `count`。
- `count` 与 `figures.length` 一致；`count=0` 代表“没有抽取到位图”，不是必然错误。
- 目标 `paper_id` 的正文/摘要/分析 key 与任务使用的 ref 对应。
- 记录的 `analyze_figure` 结果包含 `question`、`answer`、`backend` 和 `image_path`。

## 5. T4 图表抽取与 T5 图表分析

扩展任务中的 T5 工具链是：

```text
搜索 → 下载 → extract_paper_figures → analyze_figure
```

`analyze_figure` 要求论文先有抽取结果，并且 `figure_no` 从 1 开始；当前
question 枚举是 `describe`、`axes`、`trend`。默认后端是 `extractive`，它
从已经抽取的 caption 生成确定性回答；`vlm` 后端读取本地图像并使用
`VLM_MODEL_PATH` 指向的本地视觉语言模型，贪心生成最多 96 个 token。

### 5.1 extractive 与 VLM 的快照边界

两种后端不能混用同一条观察的语义：

- `extractive` 的回答可以从 T4 的 figure metadata/caption 确定性重建。
- `vlm` 的回答来自模型对图像的推理；即使 question、figure 和 paper 一样，
  也必须把当时的 VLM 输出录入 snapshot 才能 replay。
- 改变 `FIGURE_ANALYSIS_BACKEND` 或 `VLM_MODEL_PATH` 后，旧的 `analyze_figure`
  记录不能被当作新后端的结果。

在当前 `main` 中，`build_snapshot.py` 已经会在抽取图表后遍历每个图号和三种
question，直接记录配置后端的结果。VLM 快照的准备方式是：

```bash
cd AgenticArxiv
$env:FIGURE_ANALYSIS_BACKEND = "vlm"
$env:VLM_MODEL_PATH = "C:\\models\\your-vlm"
python -m rl.build_snapshot --snapshot ../data/mock_arxiv_snapshot_vlm.json
```

该命令示例需要本地 VLM、图片资产和 GPU/依赖；不要把 `extractive` 的回填
结果标成 VLM 结果。

### 5.2 关于 `backfill_figure_analysis.py`

T5 快照补录脚本属于此前单独提交的分支内容，不在本 PR 的 `main` 基线中；
因此当前仓库不能直接运行它。若在包含该脚本的 revision 上使用：

```bash
python -m AgenticArxiv.rl.backfill_figure_analysis \
  --snapshot data/mock_arxiv_snapshot.json
```

它的契约是**只适用于 `FIGURE_ANALYSIS_BACKEND=extractive`**：读取已有 T4
`extract_paper_figures` 记录，验证 paper id、图号、caption 和 count，再原子
写入可确定性重建的 T5 observation；不请求 arXiv、不读取 PDF/图片，也不做
VLM 推理。VLM 结果仍需用真实的 VLM 后端录制，不能用该补录脚本伪造。

如果当前 checkout 没有这个文件，应使用本节前面的 `build_snapshot.py` 正常
录制，而不是复制一个缺少对应分支依赖的脚本到新文档或 PR 中。

## 6. 常见症状 → 原因 → 定位 → 下一步

| 症状 | 常见原因 | 定位文件/字段 | 下一步 |
|---|---|---|---|
| replay 在搜索处 miss | snapshot 没有同 aspect/query 池，或使用了不同的 days | `env.py` 的 `_derive_search_result`、`_keyword_search_key` | 先比较 task 参数和快照记录；必要时 record 新快照 |
| keyword 返回 `offline_fallback` | query 未命中已记录的 query hash | `env.py::_keyword_search_fallback` | 把它当工具失败信号；不要让 fallback 论文进入 session 记忆 |
| 下载后正文读取失败 | rollout 的离线下载桩只有占位 PDF | `env.py::_offline_download`、`paper_content_tool.py` | 使用包含正文的 snapshot；不要把 stub 当真实文章 |
| `ref=1` 读到错误论文 | 用原始 ref 做了 snapshot key，或 session 搜索顺序不一致 | `env.py` 的 paper key 函数 | 用解析后的 paper id，确认搜索先建立 session 状态 |
| 摘要 key miss | style 不在 `tldr/structured/bullet`，或 budget 没被 bucket 化 | `paper_summary_tool.py`、`_paper_summary_key` | 修正参数；合法 bucket 是 60/120/250 |
| `extract_paper_figures` 返回 count 0 | 论文确实没有可抽取的 raster figure | `paper_figures_tool.py`、任务 note | 换有图的快照/任务；不要把空结果写成抽取异常 |
| `figure_no` 越界 | T4 图数比任务预期少，或图号不是 1-based | `figure_analysis_tool.py::validate_figure_no` | 先确认 T4 结果，再修正 figure_no 或重新录制 |
| VLM backend 缺模型 | 没有 `VLM_MODEL_PATH`，或缺 torch/transformers 支持 | `figure_analysis_tool.py::_vlm_answer` | 设置本地模型路径并重新 record；不能回退成未声明的 extractive |
| `translate_arxiv_pdf` 状态不一致 | 把 side effect 当普通 snapshot tool | `tools/pdf_translate_tool.py`、`LocalSideEffectManager` | 检查队列 handle 和 session，不要把 session_id 写进模型 Action |
| 只在第二次运行失败 | 磁盘残留被当作 runtime state，或 store 没 reset | `env.py::reset_runtime_state`、runner | 每次 trial 使用独立 MemoryStore 和环境状态 |

## 7. `record`、`auto` 的使用边界

`record` 会对快照工具强制真实调用，即使同一个 key 已存在；这是为了避免
生成过程只保留第一条“派生池”记录。它适合明确的输入冻结操作，不适合在
没有记录版本/网络时间的情况下做性能比较。

`auto` 在 miss 时会真实调用；搜索失败时还能构造带 `_mock_env.offline_fallback`
标记的固定回退子集。回退结果不会同步进 session 记忆，避免 observation 明说
失败而后续 `ref` 却解析到一篇无关论文。`auto` 适合开发，不适合正式的离线
可复现实验。

## 8. 排错后的验收

完成快照或修复环境后，提交前逐项检查：

- [ ] 正式 benchmark/训练路径明确使用 `mode="replay"` 或 `--offline`。
- [ ] snapshot 文件路径在命令和报告中可追溯。
- [ ] 搜索池、论文内容、T4/T5 记录来自同一个 snapshot 版本。
- [ ] replay miss 不会偷偷触发真实工具调用。
- [ ] offline download stub 没有被误称为真实 PDF。
- [ ] `extractive` 补录和 VLM 录制没有混用。
- [ ] 任务使用的 ref、figure_no、summary budget 与快照中的 key 相符。
- [ ] 失败原因记录为数据缺口、参数错误或工具执行错误中的一种，而不是泛化为“模型不好”。
- [ ] PR 描述写明快照是否存在、是否运行了命令以及哪些检查只是源码核对。

## 9. 源码地图

| 主题 | 文件 |
|---|---|
| 模式、key、miss、离线下载桩 | `AgenticArxiv/rl/env.py` |
| 快照构建与预取预算 | `AgenticArxiv/rl/build_snapshot.py` |
| 论文正文 | `AgenticArxiv/tools/paper_content_tool.py` |
| 摘要 backend、budget bucket | `AgenticArxiv/tools/paper_summary_tool.py` |
| 图表抽取 | `AgenticArxiv/tools/paper_figures_tool.py` |
| 图表分析 backend | `AgenticArxiv/tools/figure_analysis_tool.py` |
| 多轮环境封装 | `AgenticArxiv/rl/multiturn_env.py` |
| 快照/回放测试 | `AgenticArxiv/tests/test_paper_content_tool.py`、`test_paper_figures_tool.py`、`test_figure_analysis_tool.py` |
