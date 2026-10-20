"""E11 泛化探针集 v8：3 族 × 5 题 = 15 条，全部答案由冻结快照确定性判定。

设计文档：AgenticArXiv-Study/docs/e11-probe-design.md。目的：dev(8)/ood(4)
分辨率不足（E4：dev 单任务翻转=12.5pp），给「语义分量承载跨模板迁移」假设
以统计功效。**仅评测、绝不入训练池、冻结协议不变**；切分在
``data/splits/v8_probe.json``（单 probe 切分），不复用 v6/v7/v3 的任何一份。

三族：

  A 族 probe_long_chain      长轨迹链（max_iterations 统一口径 12）
  B 族 probe_image_entry     图像入口扰动（见下方「B 族降级说明」）
  C 族 probe_integration     多任务对比整合（新增 integration_complete 判定）

判定键：
  - 全部 15 条沿用现有 strict 七条件（``metrics.is_strict_success``），
    不改 metrics.py / report.py；
  - C 族在 strict 之外以 ``expected_integration_paper_ids`` 字段表达
    「终答需同时引用全部目标论文」，配套 ``evaluate_integration()``；
    report.py 的 +2 键扩展留到评测侧再做，本文件不碰冻结的打分路径；
  - B5 的「正确拒绝」直接复用现有 blocked 终止语义
    （terminal_mode/terminal_reason + ``classify_blocked_terminal_semantics``）。

## B 族降级说明（诚实优先）

设计文档里 B 族输入是「冻结快照内论文的图表截图」。现有 regex agent 框架
**无法把图片放进用户输入**，因此降级为「图像描述文本」：

  1. Agent 输入通道是 ``BaseAgent.run(task: str, ...)``，用户消息只有文本；
  2. 仓库里存在 ``[figure:relpath]`` 图片部件机制（``base_agent.py``
     ``_inject_vision_parts``），但它只在 ``vision_policy=True`` 且
     ``llm_client.is_vlm`` 时生效，而 ``BenchmarkRunner._create_agent``
     从不传 ``vision_policy`` —— 评测路径上图片永远注入不了；
  3. ``[figure:]`` 标记引用的 JPG 位于 AgenticArxiv-RL 仓的
     ``AgenticArxiv/output/pdf_figures/``，本 worktree 内不存在，
     强行内嵌标记会让未来的 VLM 臂在 ``Image.open`` 上直接崩。

所以 B1-B4 的用户输入是**从快照元数据（extract_paper_figures 的 caption、
analyze_figure 的确定性回答）预生成的图像描述文本**，判定不变：目标论文
身份（oracle 的 ``expected_paper_ids``）+ 图表工具参数。模型是否 VLM 由
评测时的模型决定；要做真正的图像入口臂，需要先把图件落进本仓、再给
BenchmarkRunner 接通 vision_policy —— 两项都不属于「仅构造任务与判定」。

## 与设计文档的三处偏差（均为可判定性硬约束所致）

  1. **A3 失败恢复**：设计是「gold 链先包含一次必然被快照拒绝的下载」。
     不可行：``semantic_oracle.attach_expected_paper_ids`` 对 gold 步骤
     fail-fast（报错步骤让 oracle 预检直接失败），且 strict 七条件要求
     ``tool_exec_failures == 0``，含必然失败调用的 gold 链永远无法 strict
     通过。改为：失败由用户转述进入任务（「刚才工具报错说没有这篇」），
     正确行为是改走检索完成，而**重试坏 ID 会确定性地报错并 strict 失败**，
     恢复语义仍然可测。
  2. **不用 get_translated_content**：冻结快照里该工具记录数为 0
     （build_snapshot 未录译文页），replay 必 KeyError。翻译后的可读产出
     用 summarize_paper 的确定性摘要替代。
  3. **C2 时序对比**：快照的检索池与时间窗无关（replay 按 aspect 派生，
     忽略 days），所以「两窗结果集差异」确定性地为空。判定落在两次带窗
     检索的参数上（days=1 vs days=30），integration 引用交集（=全池），
     这是快照能给出的唯一定义。

## 快照耦合说明

``requires_offline=True`` 的任务绑定 data/mock_arxiv_snapshot.json：
池内序号、标题子串、章节/图表/摘要覆盖都按当前快照挑选（换快照需重验，
跑 ``python -m benchmark.tasks_probe`` 自检即可，见文件末尾）。
关键词检索只有 3 条记录在案（query + days=30），其它 query 在 replay 下
走确定性回退池且被上报为工具失败，因此 A3/C4 只用记录在案的 query。
"""

from typing import Any, Dict, List, Optional, Sequence

from benchmark.task_spec import Step, TaskSpec, build


# ============================================================================
# 族 A：长轨迹链（max_iterations 统一口径 12 —— 设计文档「max_turns=12 仅 A 族」。
# 统一预算是为了让长链失败不被 FORCE_STOP 混淆：预算不够是「没给钱」，
# 不是「不会规划」，两者混在一起这一族的失败率就没法解读。）
# ============================================================================
_LONG_CHAIN_PROBES: List[TaskSpec] = [
    # A1 检索 → 下载 → 翻译 → 双风格摘要（5 次工具调用）
    TaskSpec(
        id='probe_a1_translate_dual_summary',
        task='检索最近30天计算机视觉(cs.CV)论文5篇，下载第1篇，为它创建中文翻译任务并保留双语版本，'
             '然后分别用 60 词以内的一句话风格(tldr)和 120 词以内的结构化风格(structured)总结这篇论文',
        steps=(
            Step('get_recently_submitted_cs_papers',
                 {'aspect': 'CV', 'days': 30, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 1}),
            Step('translate_arxiv_pdf', {'ref': 1, 'keep_dual': True}),
            Step('summarize_paper', {'ref': 1, 'style': 'tldr', 'max_words': 60}),
            Step('summarize_paper', {'ref': 1, 'style': 'structured', 'max_words': 120}),
        ),
        category='probe_long_chain',
        difficulty='hard',
        max_iterations=12,
        requires_offline=True,
        note='CV 池第 1 篇 2608.14546v1：快照里有 tldr60 与 structured120 两档摘要记录。'
             '同链混排异步翻译与两次解读调用，考察 5 步规划的步序保持',
    ),
    # A2 三池扫描链（9 次工具调用，本族最长）
    TaskSpec(
        id='probe_a2_three_aspect_scan',
        task='依次做三组检索并处理：先检索最近30天 cs.AI 论文3篇，下载第1篇并用 60 词以内的一句话风格(tldr)总结；'
             '再检索最近30天 cs.LG 论文3篇，同样下载第1篇并用 tldr 60 词总结；'
             '最后检索最近30天 cs.RO 论文3篇，下载第1篇并用 tldr 60 词总结',
        steps=(
            Step('get_recently_submitted_cs_papers',
                 {'aspect': 'AI', 'days': 30, 'max_results': 3}),
            Step('download_arxiv_pdf', {'ref': 1}),
            Step('summarize_paper', {'ref': 1, 'style': 'tldr', 'max_words': 60}),
            Step('get_recently_submitted_cs_papers',
                 {'aspect': 'LG', 'days': 30, 'max_results': 3}),
            Step('download_arxiv_pdf', {'ref': 1}),
            Step('summarize_paper', {'ref': 1, 'style': 'tldr', 'max_words': 60}),
            Step('get_recently_submitted_cs_papers',
                 {'aspect': 'RO', 'days': 30, 'max_results': 3}),
            Step('download_arxiv_pdf', {'ref': 1}),
            Step('summarize_paper', {'ref': 1, 'style': 'tldr', 'max_words': 60}),
        ),
        category='probe_long_chain',
        difficulty='hard',
        max_iterations=12,
        requires_offline=True,
        note='9 次调用；三组里 ref=1 的指代对象随池切换，考察「同名参数、不同所指」'
             '的长程一致性（AI pos1=2608.14539v1, LG pos1=2609.24985v1, RO pos1=2609.24996v1）',
    ),
    # A3 失败恢复（见模块 docstring 偏差 1：坏 ID 不得重试，改走检索）
    TaskSpec(
        id='probe_a3_bad_id_recovery',
        task='我想下载 arXiv:2609.16015v1，但刚才工具报错说快照里没有这篇论文。'
             '请不要再用这个 ID 重试：改用关键词检索 "all:large language model"（最近30天，最多5篇），'
             '在结果里找到标题以 "Social Behavior" 开头的那篇论文并下载它',
        steps=(
            Step('search_arxiv_papers',
                 {'query': 'all:large language model', 'days': 30, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 'Social Behavior'}),
        ),
        category='probe_long_chain',
        difficulty='hard',
        max_iterations=12,
        requires_offline=True,
        note='重试坏 ID 会确定性报错（快照拒绝 + 会话内解析不到）→ strict 失败；'
             '该 query 是快照仅有的 3 条记录之一，换任何其它 query 都会走确定性回退池'
             '并被上报为工具失败。目标 2609.16013v1 是该检索结果第 1 篇',
    ),
    # A4 缓存感知条件分支（缓存命中 → 免下载直接读）
    TaskSpec(
        id='probe_a4_cache_branch_method',
        task='查看第2篇论文的缓存状态；如果它已经缓存，就直接读它的方法(method)章节，'
             '如果还没缓存，就先下载再读',
        steps=(
            Step('get_paper_cache_status', {'ref': 2}),
            Step('get_paper_content', {'ref': 2, 'section': 'method'}),
        ),
        category='probe_long_chain',
        difficulty='hard',
        max_iterations=12,
        setup=(
            Step('get_recently_submitted_cs_papers',
                 {'aspect': 'AI', 'days': 30, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 2}),
        ),
        requires_offline=True,
        note='setup 已把第 2 篇（2608.14530v1，快照确有 method 章节）铺成已缓存，'
             '因此正确分支是「不下载」：缓存判断后直接读。多余下载会被严格序列判错',
    ),
    # A5 图表链（衔接 VLM 工具；文本臂在 analyze 上拿不到视觉内容 = 有意义的下界）
    TaskSpec(
        id='probe_a5_figure_chain_summary',
        task='检索最近30天密码学与安全(cs.CR)论文5篇，下载第3篇，抽出它的图表，'
             '描述第1张图画的是什么，再用 250 词以内的结构化风格(structured)总结这篇论文',
        steps=(
            Step('get_recently_submitted_cs_papers',
                 {'aspect': 'CR', 'days': 30, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 3}),
            Step('extract_paper_figures', {'ref': 3}),
            Step('analyze_figure', {'ref': 3, 'figure_no': 1, 'question': 'describe'}),
            Step('summarize_paper', {'ref': 3, 'style': 'structured', 'max_words': 250}),
        ),
        category='probe_long_chain',
        difficulty='hard',
        max_iterations=12,
        requires_offline=True,
        note='CR 池第 3 篇 2609.24826v1：快照确有图表与 analyze(describe) 记录。'
             '抽图→看图→摘要三段衔接，VLM 臂与文本臂在这里预期拉开差距',
    ),
]


# ============================================================================
# 族 B：图像入口扰动（降级为图像描述文本，见模块 docstring）
# ============================================================================
# B1-B3 的描述文本全部预生成自冻结快照：caption 取 extract_paper_figures 的
# 记录（PDF 原文换行连字符如 "struc- ture" 已在描述中拼回，note 里注明），
# 与 analyze_figure(describe/axes) 的确定性回答同源。判定不含任何「看懂了图
# 没有」的模糊成分：目标论文身份 + 图表工具调用参数。
_FIGURE_ENTRY_SETUP_CV = (
    Step('get_recently_submitted_cs_papers', {'aspect': 'CV', 'days': 30, 'max_results': 5}),
)
_FIGURE_ENTRY_SETUP_LG = (
    Step('get_recently_submitted_cs_papers', {'aspect': 'LG', 'days': 30, 'max_results': 5}),
)

_IMAGE_ENTRY_PROBES: List[TaskSpec] = [
    # B1 图注含方法名 → 以图注为检索键定位论文（fig 1 / describe）
    TaskSpec(
        id='probe_b1_caption_grounding_describe',
        task='用户上传了一张论文图表的截图（当前输入通道只传文字，图的内容以下面这段图注描述给出）：'
             '"Fig. 1: MagnifiQ restores real-world low-quality inputs into coherent high-resolution '
             'outputs up to 4096×4096, supporting up to 32× scaling while preserving global structure '
             'and recovering fine details."。这张图出自最近30天 cs.CV 候选列表中的某篇论文：'
             '找到它并下载，抽出它的图表，然后描述第1张图',
        steps=(
            Step('download_arxiv_pdf', {'ref': 'MagnifiQ'}),
            Step('extract_paper_figures', {'ref': None}),
            Step('analyze_figure', {'ref': None, 'figure_no': 1, 'question': 'describe'}),
        ),
        category='probe_image_entry',
        difficulty='hard',
        max_iterations=6,
        setup=_FIGURE_ENTRY_SETUP_CV,
        requires_offline=True,
        note='图注 = 快照 extract_paper_figures("2608.14543v1") fig1 原文（"struc- ture" 已拼回）。'
             '正确论文是 CV 池第 2 篇 2608.14543v1；图注里的方法名 MagnifiQ 就是可解析的标题子串',
    ),
    # B2 图序传递：图注明确是 Figure 2，考察 figure_no 不被默认成 1
    TaskSpec(
        id='probe_b2_figure_index_grounding',
        task='用户给了一张论文图（Figure 2）的截图（输入通道只传文字，内容按图注描述）：'
             '"Figure 2: Annotating an agent trajectory with onPanda. Top left: the user prompt; '
             'top right: the connected MCP servers and tools."。它出自最近30天 cs.LG 候选列表中的'
             '某篇论文：找到并下载该论文，抽出图表，然后描述第2张图',
        steps=(
            Step('download_arxiv_pdf', {'ref': 'onPanda'}),
            Step('extract_paper_figures', {'ref': None}),
            Step('analyze_figure', {'ref': None, 'figure_no': 2, 'question': 'describe'}),
        ),
        category='probe_image_entry',
        difficulty='hard',
        max_iterations=6,
        setup=_FIGURE_ENTRY_SETUP_LG,
        requires_offline=True,
        note='图注 = 快照 extract_paper_figures("2609.24983v1") fig2（"on- Panda"/"con- nected" '
             '的换行连字符已拼回）。正确论文是 LG 池第 2 篇 2609.24983v1，该论文只有 2 张图；'
             '把 figure_no 默认成 1 的话参数分直接丢失',
    ),
    # B3 面板/指标类问题：axes 问法 + 图 1 的右面板（快照里是 figure_no=2）
    TaskSpec(
        id='probe_b3_panel_axes_grounding',
        task='用户发来一张论文里的对比面板图（内容按图注描述）：右面板展示与 Arena Image Edit '
             'Leaderboard 对比的 Spearman 相关系数与平均绝对误差(MAE)，图注写道 '
             '"CPI-Bench outperforms other benchmarks, achieving the highest correlation and the '
             'lowest error margin."。这张图出自最近30天 cs.CV 候选列表中的某篇论文：'
             '找到并下载它，抽出图表，再说明第2张图用了哪些坐标轴/指标(axes)',
        steps=(
            Step('download_arxiv_pdf', {'ref': 'CPI-Bench'}),
            Step('extract_paper_figures', {'ref': None}),
            Step('analyze_figure', {'ref': None, 'figure_no': 2, 'question': 'axes'}),
        ),
        category='probe_image_entry',
        difficulty='hard',
        max_iterations=6,
        setup=_FIGURE_ENTRY_SETUP_CV,
        requires_offline=True,
        note='描述预生成自快照：caption 含 "Right: Spearman correlation coefficients and Mean '
             'Absolute Error (MAE) with Arena rank"（"Leader- board" 连字符已拼回）。'
             '正确论文是 CV 池第 1 篇 2608.14546v1；快照把 caption-Figure-1 的左右面板存成 '
             'figure_no=1/2 两个文件，axes 答案与描述同源',
    ),
    # B4 图内水印 arXiv ID → 直接按 ID 下载（ID 在池内第 6 位，序号猜测会下错论文）
    TaskSpec(
        id='probe_b4_watermark_id_download',
        task='用户给了一张论文图表的截图，右下角有水印文字 "arXiv:2609.24967v1"（输入通道只传文字，'
             '水印内容如前所述）。请下载这篇论文',
        steps=(
            Step('download_arxiv_pdf', {'ref': '2609.24967v1'}),
        ),
        category='probe_image_entry',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers',
                 {'aspect': 'CL', 'days': 30, 'max_results': 50}),
        ),
        requires_offline=True,
        note='2609.24967v1 是 CL 池第 6 篇，按序号瞎猜解析到的是别的论文，'
             'oracle 的 expected_paper_ids 会拆穿；正确动作是从描述里的 ID 直接取',
    ),
    # B5 无关图像 → 正确拒绝（复用现有 blocked 终止语义，见 classify_blocked_terminal_semantics）
    TaskSpec(
        id='probe_b5_non_paper_image_refusal',
        task='用户发来一张照片：厨房桌上一盘番茄炒蛋旁边的手写家常菜谱笔记，'
             '要求"找出这张图出自的那篇论文并下载它"',
        category='probe_image_entry',
        difficulty='hard',
        terminal_mode='blocked',
        terminal_reason='unsupported_capability',
        setup=(
            Step('get_recently_submitted_cs_papers',
                 {'aspect': '*', 'days': 30, 'max_results': 5}),
        ),
        note='非论文图像，且会话里有 5 篇候选也依然无法定位出处 —— 排除「会话为空」的'
             'missing_context 借口，正确行为是说明能力边界，而不是编造检索或乱下论文。'
             '文本臂与 VLM 臂在这条上应当同分（校准下界）',
    ),
]


# ============================================================================
# 族 C：多任务对比整合（strict 之外新增 integration_complete：
# 终答需同时引用全部目标论文，以 expected_integration_paper_ids 表达 +
# evaluate_integration() 判定；report.py 的 +2 键扩展在评测侧做）
# ============================================================================
_INTEGRATION_PROBES: List[TaskSpec] = [
    # C1 双篇方法对比
    TaskSpec(
        id='probe_c1_two_paper_method_contrast',
        task='先检索最近30天 cs.AI 论文3篇，下载第2篇并读它的方法(method)章节；'
             '再检索最近30天 cs.CR 论文3篇，下载第3篇并读它的方法(method)章节；'
             '最后在最终回答中对比两篇论文的方法差异，回答里需要同时提到这两篇论文',
        steps=(
            Step('get_recently_submitted_cs_papers',
                 {'aspect': 'AI', 'days': 30, 'max_results': 3}),
            Step('download_arxiv_pdf', {'ref': 2}),
            Step('get_paper_content', {'ref': 2, 'section': 'method'}),
            Step('get_recently_submitted_cs_papers',
                 {'aspect': 'CR', 'days': 30, 'max_results': 3}),
            Step('download_arxiv_pdf', {'ref': 3}),
            Step('get_paper_content', {'ref': 3, 'section': 'method'}),
        ),
        category='probe_integration',
        difficulty='hard',
        max_iterations=8,
        requires_offline=True,
        note='两篇都按快照挑过章节覆盖：AI pos2=2608.14530v1、CR pos3=2609.24826v1，均有 method',
    ),
    # C2 时序对比（见模块 docstring 偏差 3：快照池与时间窗无关，差异确定为空）
    TaskSpec(
        id='probe_c2_time_window_contrast',
        task='分别检索最近1天与最近30天的机器学习(cs.LG)论文各5篇，对比两个时间窗结果集的差异，'
             '并在最终回答中列出两个窗口都出现的论文（需要给出论文编号）',
        steps=(
            Step('get_recently_submitted_cs_papers',
                 {'aspect': 'LG', 'days': 1, 'max_results': 5}),
            Step('get_recently_submitted_cs_papers',
                 {'aspect': 'LG', 'days': 30, 'max_results': 5}),
        ),
        category='probe_integration',
        difficulty='medium',
        max_iterations=4,
        requires_offline=True,
        note='判定核心是 days=1 与 days=30 两个窗口参数都被真实传对；'
             'integration 引用两窗交集（快照里=LG 池前 5 篇，确定性地无差异）',
    ),
    # C3 翻译一致性（A1 的组件化对照：前缀由 setup 铺好，只考尾部 + 可选参数）
    TaskSpec(
        id='probe_c3_translate_variants_summarize',
        task='第1篇论文已经下载好了。请先用 google 服务把它翻译一次，'
             '再用保留双语版本(keep_dual)的方式翻译一次，'
             '最后用 60 词以内的一句话风格(tldr)总结这篇论文作为汇总',
        steps=(
            Step('translate_arxiv_pdf', {'ref': None, 'service': 'google'}),
            Step('translate_arxiv_pdf', {'ref': None, 'keep_dual': True}),
            Step('summarize_paper', {'ref': 1, 'style': 'tldr', 'max_words': 60}),
        ),
        category='probe_integration',
        difficulty='hard',
        max_iterations=5,
        setup=(
            Step('get_recently_submitted_cs_papers',
                 {'aspect': 'CL', 'days': 30, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 1}),
        ),
        requires_offline=True,
        note='同一篇论文的两种翻译产物（指定服务 / 双语）+ 一次汇总摘要，'
             'service 与 keep_dual 两个可选参数都要真的传对；ref=null 指向 setup 下载的那篇'
             '（CL pos1=2609.24985v1）',
    ),
    # C4 跨工具冗余对比：search vs get_recently 同目标，要求只走更精准的一条
    TaskSpec(
        id='probe_c4_tool_choice_keyword_over_list',
        task='目标：下载标题含 "Doc-REFRAG" 的论文。有两条路可选：'
             '按关键词检索 "all:retrieval augmented generation"（最近30天，最多5篇），'
             '或者拉取最近30天的 cs.CL 论文列表。关键词检索更精准——'
             '只走关键词检索这条路完成下载，不要两条都跑',
        steps=(
            Step('search_arxiv_papers',
                 {'query': 'all:retrieval augmented generation', 'days': 30, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 'Doc-REFRAG'}),
        ),
        category='probe_integration',
        difficulty='hard',
        max_iterations=4,
        requires_offline=True,
        note='测工具选择语义：两个检索工具都能拿到候选时，按任务约束只调一个；'
             '多调 get_recently 会被严格序列判错。目标 2608.30163v1 是该 query 结果第 1 篇',
    ),
    # C5 三篇聚合（C1 的三篇版：读摘要 + 汇总三篇编号）
    TaskSpec(
        id='probe_c5_three_paper_abstract_aggregate',
        task='依次处理三组检索：检索最近30天 cs.CV 论文3篇，下载第1篇并读出它的摘要(abstract)；'
             '检索最近30天 cs.CL 论文3篇，下载第1篇并读出摘要；'
             '检索最近30天 cs.CR 论文3篇，下载第1篇并读出摘要；'
             '最后在最终回答中汇总三篇论文，回答里需要同时给出三篇的论文编号',
        steps=(
            Step('get_recently_submitted_cs_papers',
                 {'aspect': 'CV', 'days': 30, 'max_results': 3}),
            Step('download_arxiv_pdf', {'ref': 1}),
            Step('get_paper_content', {'ref': 1, 'section': 'abstract'}),
            Step('get_recently_submitted_cs_papers',
                 {'aspect': 'CL', 'days': 30, 'max_results': 3}),
            Step('download_arxiv_pdf', {'ref': 1}),
            Step('get_paper_content', {'ref': 1, 'section': 'abstract'}),
            Step('get_recently_submitted_cs_papers',
                 {'aspect': 'CR', 'days': 30, 'max_results': 3}),
            Step('download_arxiv_pdf', {'ref': 1}),
            Step('get_paper_content', {'ref': 1, 'section': 'abstract'}),
        ),
        category='probe_integration',
        difficulty='hard',
        max_iterations=12,
        requires_offline=True,
        note='9 次调用；三组 ref=1 随池切换（CV pos1=2608.14546v1, CL pos1=2609.24985v1, '
             'CR pos1=2609.24994v1），摘要(abstract)是快照里每篇都有的章节',
    ),
]


# C 族的 integration 目标：终答必须同时引用的论文（与快照内池位一一对应，
# oracle 预检会用 gold 重放的解析结果逐一核对，见本文件 `python -m` 自检）。
PROBE_INTEGRATION: Dict[str, List[str]] = {
    'probe_c1_two_paper_method_contrast': ['2608.14530v1', '2609.24826v1'],
    'probe_c2_time_window_contrast': [
        '2609.24985v1', '2609.24983v1', '2609.24979v1', '2609.24972v1', '2609.24969v1',
    ],
    'probe_c3_translate_variants_summarize': ['2609.24985v1'],
    'probe_c4_tool_choice_keyword_over_list': ['2608.30163v1'],
    'probe_c5_three_paper_abstract_aggregate': [
        '2608.14546v1', '2609.24985v1', '2609.24994v1',
    ],
}


PROBE_SPECS: List[TaskSpec] = (
    _LONG_CHAIN_PROBES + _IMAGE_ENTRY_PROBES + _INTEGRATION_PROBES
)

# 与 tasks_expanded.EXPANDED_TASKS 同构：expected_tools / expected_tool_args
# 由同一份 steps 派生，C 族在此之上补 integration 目标字段。
PROBE_TASKS: List[Dict[str, Any]] = build(PROBE_SPECS)
for _task in PROBE_TASKS:
    _integration = PROBE_INTEGRATION.get(_task['id'])
    if _integration:
        _task['expected_integration_paper_ids'] = list(_integration)


def get_probe_tasks() -> List[Dict[str, Any]]:
    """run_benchmark --task-set probe 的任务池（15 条，全部为评测探针）。"""
    return list(PROBE_TASKS)


def offline_only_ids() -> List[str]:
    """绑定冻结快照的探针 id（联网跑判分无意义，需跳过）。"""
    return [t['id'] for t in PROBE_TASKS if t.get('requires_offline')]


def integration_complete(
    history: Sequence[Dict[str, Any]],
    expected_paper_ids: Sequence[str],
) -> bool:
    """integration_complete 判定：终答文本同时引用全部目标论文。

    只读最后一个 FINISH 动作的 thought（模型最终回答），比对方式与
    ``metrics._same_paper`` 同口径——忽略版本号后缀，因此终答写
    ``2608.14530`` 或 ``2608.14530v1`` 都算引用了该论文。

    保守性说明：这是「提到了编号」的必要条件判定，不验证语义正确性；
    语义正确性由 strict 七条件里的 arg/ref 分数承担，两者AND起来才是
    完整的 C 族判定。历史轨迹里没有 FINISH 时返回 False。
    """
    if not expected_paper_ids:
        return False
    final_text = ""
    for step in reversed(history or []):
        if str(step.get('action', '')).strip().upper() == 'FINISH':
            final_text = str(step.get('thought', ''))
            break
    if not final_text:
        return False
    # `2608.14530v1` -> 取 '2608.14530'；`2609.18317v2` 这类 v2 同理
    bases = []
    for pid in expected_paper_ids:
        text = str(pid)
        base = text
        marker = text.rfind('v')
        if marker > 0 and text[marker + 1:].isdigit():
            base = text[:marker]
        bases.append(base)
    return all(base in final_text for base in bases)


def evaluate_integration(task_def: Dict[str, Any], history: Sequence[Dict[str, Any]]) -> Optional[bool]:
    """任务级入口：未声明 integration 目标时返回 None（不适用）。"""
    expected = task_def.get('expected_integration_paper_ids')
    if not expected:
        return None
    return integration_complete(history, expected)


if __name__ == '__main__':
    # 快速自检：id 唯一、integration 目标只挂在 C 族、离线绑定齐全。
    ids = [t['id'] for t in PROBE_TASKS]
    assert len(ids) == len(set(ids)) == 15, ids
    for tid in PROBE_INTEGRATION:
        task = next(t for t in PROBE_TASKS if t['id'] == tid)
        assert task['category'] == 'probe_integration', tid
        assert task.get('expected_integration_paper_ids'), tid
    assert set(offline_only_ids()) == {
        t['id'] for t in PROBE_TASKS if t['id'] != 'probe_b5_non_paper_image_refusal'
    }
    print('tasks_probe 自检通过：15 条探针（A5 / B5 / C5）')
    for t in PROBE_TASKS:
        print(f"  {t['id']:44s} {t['category']:18s} "
              f"steps={len(t['expected_tools'])} "
              f"max_iter={t.get('max_iterations')}")
