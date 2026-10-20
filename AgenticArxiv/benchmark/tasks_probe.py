"""E11 泛化探针集 v8 / v8.1：15 个模板 × 3 参数组 = 45 条，全部答案由冻结快照确定性判定。

设计文档：AgenticArXiv-Study/docs/e11-probe-design.md；R1.5 扩容（审稿人统计
功效要求）：每个模板实例化 3 个参数组，切分 ``data/splits/v8_1_probe.json``
（45 条）；v8 的 15 条 pilot（``data/splits/v8_probe.json``）是本集的子集，
id 不变。**仅评测、绝不入训练池、冻结协议不变**。

三族（命名 ``probe_<family><n>_<variant>``）：

  A 族 probe_long_chain      长轨迹链（max_iterations 统一口径 12）
  B 族 probe_image_entry     图像入口扰动（见下方「B 族降级说明」）
  C 族 probe_integration     多任务对比整合（新增 integration_complete 判定）

## 参数维度（为什么是这些维度）

关键词检索在冻结快照里只有 3 条在案记录（query + days=30 的 sha256 键），
任何新 query 在 replay 下都走确定性回退池并被上报为工具失败，所以**参数
变化优先落在：论文池 aspect、池内序号、每池篇数、时间窗、图号、问法
(describe/axes/trend)、章节(abstract/method/result/conclusion)、摘要风格与
词预算、在案 query 的不同目标序号**。3 个参数组之间不允许只换一句话说法
（那是 iid 换皮），每一维都必须改变标准答案本身。

## 判定键

  - 全部 45 条沿用现有 strict 七条件（``metrics.is_strict_success``），
    不改 metrics.py / report.py；
  - C 族（15 条）在 strict 之外以 ``expected_integration_paper_ids`` 表达
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
B 族的 3 个参数组 = 3 篇不同论文 × 不同图号（描述文本逐篇预生成，不共享
模板句）。

## 与设计文档的偏差（均为可判定性硬约束所致，v8 起未变）

  1. **A3 失败恢复**：设计是「gold 链先包含一次必然被快照拒绝的下载」。
     不可行：``semantic_oracle.attach_expected_paper_ids`` 对 gold 步骤
     fail-fast（报错步骤让 oracle 预检直接失败），且 strict 七条件要求
     ``tool_exec_failures == 0``，含必然失败调用的 gold 链永远无法 strict
     通过。改为：失败由用户转述进入任务（「刚才工具报错说没有这篇」），
     正确行为是改走检索完成，而**重试坏 ID 会确定性地报错并 strict 失败**，
     恢复语义仍然可测。3 个参数组 = 在案 3 条 query 各配一个不同目标序号。
  2. **不用 get_translated_content**：冻结快照里该工具记录数为 0
     （build_snapshot 未录译文页），replay 必 KeyError。翻译后的可读产出
     用 summarize_paper 的确定性摘要替代。
  3. **C2 时序对比**：快照的检索池与时间窗无关（replay 按 aspect 派生，
     忽略 days），所以「两窗结果集差异」确定性地为空。判定落在两次带窗
     检索的参数上（days=lo vs days=30），integration 引用交集（=全池），
     这是快照能给出的唯一定义。

## 快照耦合说明

``requires_offline=True`` 的任务绑定 data/mock_arxiv_snapshot.json：
池内序号、标题子串、章节/图表/摘要覆盖都按当前快照挑选（换快照需重验，
跑 ``python -m benchmark.tasks_probe`` 自检 + oracle 预检即可，见文件末尾）。
B 族 caption 引用自 extract_paper_figures 的记录，PDF 换行连字符
（如 "struc- ture"）已在描述中拼回并在 note 注明；个别 caption 的
"Figure N" 编号与抽取文件序号不一致（快照抽取 quirk），同样在 note 注明。
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmark.task_spec import Step, TaskSpec, build


# ---------------------------------------------------------------------------
# 小工具：让 45 条的"参数组"只写一份维度，steps 由同一份维度派生。
# ---------------------------------------------------------------------------

_CN = {
    "AI": "人工智能", "LG": "机器学习", "CL": "自然语言处理",
    "CV": "计算机视觉", "RO": "机器人学", "CR": "密码学与安全",
    "*": "计算机科学",
}
_STYLE_CN = {
    "tldr": "一句话风格(tldr)", "structured": "结构化风格(structured)",
    "bullet": "要点风格(bullet)",
}


def _recent(aspect: str, max_results: int, days: int = 30) -> Step:
    """检索步。days 一律 30（与在案池精确对齐）；时间窗探针显式传别的值。"""
    return Step("get_recently_submitted_cs_papers",
                {"aspect": aspect, "days": days, "max_results": max_results})


def _summarize(ref: Any, style: str, max_words: int) -> Step:
    return Step("summarize_paper", {"ref": ref, "style": style, "max_words": max_words})


def _content(ref: Any, section: str) -> Step:
    return Step("get_paper_content", {"ref": ref, "section": section})


def _analyze(ref: Any, figure_no: int, question: str) -> Step:
    return Step("analyze_figure", {"ref": ref, "figure_no": figure_no, "question": question})


def _kw_search(query: str, max_results: int = 5) -> Step:
    return Step("search_arxiv_papers",
                {"query": query, "days": 30, "max_results": max_results})


# ============================================================================
# 族 A：长轨迹链（max_iterations 统一口径 12 —— 设计文档「max_turns=12 仅 A 族」。
# 统一预算是为了让长链失败不被 FORCE_STOP 混淆：预算不够是「没给钱」，
# 不是「不会规划」，两者混在一起这一族的失败率就没法解读。）
# ============================================================================

# ---- A1 检索 → 下载 → 翻译 → 双风格摘要（5 次工具调用）---------------------
# 参数维度：池 aspect × 池内序号 × 摘要风格对（快照里每篇论文都有 3 风格 × 3 预算）
_A1_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "probe_a1_translate_dual_summary",
        "aspect": "CV", "pos": 1, "pool_n": 5,
        "s1": ("tldr", 60), "s2": ("structured", 120),
        "text": "检索最近30天计算机视觉(cs.CV)论文5篇，下载第1篇，为它创建中文翻译任务并保留双语版本，"
                "然后分别用 60 词以内的一句话风格(tldr)和 120 词以内的结构化风格(structured)总结这篇论文",
        "note": "CV 池第 1 篇 2608.14546v1：快照里有 tldr60 与 structured120 两档摘要记录。"
                "同链混排异步翻译与两次解读调用，考察 5 步规划的步序保持",
    },
    {
        "id": "probe_a1_ai2_tldr_bullet",
        "aspect": "AI", "pos": 2, "pool_n": 5,
        "s1": ("tldr", 60), "s2": ("bullet", 250),
        "text": "检索最近30天人工智能(cs.AI)论文5篇，下载第2篇，为它创建中文翻译任务并保留双语版本，"
                "然后分别用 60 词以内的一句话风格(tldr)和 250 词以内的要点风格(bullet)总结这篇论文",
        "note": "AI 池第 2 篇 2608.14530v1；与 v1 换池、换序号、换风格对（bullet 250），"
                "考察同链在不同参数组上的步序保持",
    },
    {
        "id": "probe_a1_ro3_tldr_structured",
        "aspect": "RO", "pos": 3, "pool_n": 5,
        "s1": ("tldr", 120), "s2": ("structured", 250),
        "text": "检索最近30天机器人学(cs.RO)论文5篇，下载第3篇，为它创建中文翻译任务并保留双语版本，"
                "然后分别用 120 词以内的一句话风格(tldr)和 250 词以内的结构化风格(structured)总结这篇论文",
        "note": "RO 池第 3 篇 2609.24976v1；两段摘要预算同时抬高（120/250），"
                "考察预算维度是否随池与序号一起被真实传递",
    },
]


def _a1_build(v: Dict[str, Any]) -> TaskSpec:
    ref = v["pos"]
    return TaskSpec(
        id=v["id"],
        task=v["text"],
        steps=(
            _recent(v["aspect"], v["pool_n"]),
            Step("download_arxiv_pdf", {"ref": ref}),
            Step("translate_arxiv_pdf", {"ref": ref, "keep_dual": True}),
            _summarize(ref, *v["s1"]),
            _summarize(ref, *v["s2"]),
        ),
        category="probe_long_chain",
        difficulty="hard",
        max_iterations=12,
        requires_offline=True,
        note=v["note"],
    )


# ---- A2 三池扫描链（9 次工具调用，本族最长）--------------------------------
# 参数维度：三个 aspect 的组合 × 每池篇数 × 组内序号 × 摘要预算
_A2_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "probe_a2_three_aspect_scan",
        "groups": (("AI", 3, 1), ("LG", 3, 1), ("RO", 3, 1)),
        "style": ("tldr", 60),
        "text": "依次做三组检索并处理：先检索最近30天 cs.AI 论文3篇，下载第1篇并用 60 词以内的一句话风格(tldr)总结；"
                "再检索最近30天 cs.LG 论文3篇，同样下载第1篇并用 tldr 60 词总结；"
                "最后检索最近30天 cs.RO 论文3篇，下载第1篇并用 tldr 60 词总结",
        "note": "9 次调用；三组里 ref=1 的指代对象随池切换，考察「同名参数、不同所指」"
                "的长程一致性（AI pos1=2608.14539v1, LG pos1=2609.24985v1, RO pos1=2609.24996v1）",
    },
    {
        "id": "probe_a2_cv_cr_cl_pos2_scan",
        "groups": (("CV", 3, 2), ("CR", 3, 2), ("CL", 3, 2)),
        "style": ("tldr", 60),
        "text": "依次做三组检索并处理：先检索最近30天 cs.CV 论文3篇，下载第2篇并用 60 词以内的一句话风格(tldr)总结；"
                "再检索最近30天 cs.CR 论文3篇，同样下载第2篇并用 tldr 60 词总结；"
                "最后检索最近30天 cs.CL 论文3篇，下载第2篇并用 tldr 60 词总结",
        "note": "换组合（CV/CR/CL）且组内序号全部换成 2（CV pos2=2608.14543v1, "
                "CR pos2=2609.24980v1, CL pos2=2609.24983v1），同名 ref=2 指向三篇不同论文",
    },
    {
        "id": "probe_a2_lg_ro_ai_pos3_scan",
        "groups": (("LG", 5, 3), ("RO", 5, 3), ("AI", 5, 3)),
        "style": ("tldr", 120),
        "text": "依次做三组检索并处理：先检索最近30天 cs.LG 论文5篇，下载第3篇并用 120 词以内的一句话风格(tldr)总结；"
                "再检索最近30天 cs.RO 论文5篇，同样下载第3篇并用 tldr 120 词总结；"
                "最后检索最近30天 cs.AI 论文5篇，下载第3篇并用 tldr 120 词总结",
        "note": "每池篇数抬到 5、序号换 3、预算换 120（LG pos3=2609.24979v1, "
                "RO pos3=2609.24976v1, AI pos3=2608.14528v1），三维度同时变化",
    },
]


def _a2_build(v: Dict[str, Any]) -> TaskSpec:
    style, budget = v["style"]
    steps: List[Step] = []
    for aspect, pool_n, pos in v["groups"]:
        steps.append(_recent(aspect, pool_n))
        steps.append(Step("download_arxiv_pdf", {"ref": pos}))
        steps.append(_summarize(pos, style, budget))
    return TaskSpec(
        id=v["id"],
        task=v["text"],
        steps=tuple(steps),
        category="probe_long_chain",
        difficulty="hard",
        max_iterations=12,
        requires_offline=True,
        note=v["note"],
    )


# ---- A3 失败恢复（坏 ID 不得重试，改走在案 query 的检索）-------------------
# 参数维度：在案 3 条 query 之一 × 目标在检索结果里的序号 × 标题钩子
_A3_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "probe_a3_bad_id_recovery",
        "query": "all:large language model",
        "hook": "Social Behavior", "bad_id": "2609.16015v1",
        "text": "我想下载 arXiv:2609.16015v1，但刚才工具报错说快照里没有这篇论文。"
                "请不要再用这个 ID 重试：改用关键词检索 \"all:large language model\"（最近30天，最多5篇），"
                "在结果里找到标题以 \"Social Behavior\" 开头的那篇论文并下载它",
        "note": "重试坏 ID 会确定性报错（快照拒绝 + 会话内解析不到）→ strict 失败；"
                "该 query 是快照仅有的 3 条记录之一。目标是该检索结果第 1 篇 2609.16013v1",
    },
    {
        "id": "probe_a3_bad_id_recovery_arm",
        "query": "all:agentic reinforcement learning",
        "hook": "Markets with Congestion", "bad_id": "2608.77777v1",
        "text": "我想下载 arXiv:2608.77777v1，但刚才工具报错说快照里没有这篇论文。"
                "请不要再用这个 ID 重试：改用关键词检索 \"all:agentic reinforcement learning\"（最近30天，最多5篇），"
                "在结果里找到标题含 \"Markets with Congestion\" 的那篇论文并下载它",
        "note": "换在案 query 并把目标挪到第 2 篇（2609.14827v1），"
                "考察恢复路径在不同检索目标序号上是否稳定",
    },
    {
        "id": "probe_a3_bad_id_recovery_rag",
        "query": "all:retrieval augmented generation",
        "hook": "ISO-RAG", "bad_id": "2607.33333v1",
        "text": "我想下载 arXiv:2607.33333v1，但刚才工具报错说快照里没有这篇论文。"
                "请不要再用这个 ID 重试：改用关键词检索 \"all:retrieval augmented generation\"（最近30天，最多5篇），"
                "在结果里找到标题以 \"ISO-RAG\" 开头的那篇论文并下载它",
        "note": "第 3 条在案 query，目标是第 3 篇（2609.00513v1）；"
                "三条参数组合起来覆盖快照全部在案 query 与 1/2/3 三个目标序号",
    },
]


def _a3_build(v: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        id=v["id"],
        task=v["text"],
        steps=(
            _kw_search(v["query"]),
            Step("download_arxiv_pdf", {"ref": v["hook"]}),
        ),
        category="probe_long_chain",
        difficulty="hard",
        max_iterations=12,
        requires_offline=True,
        note=v["note"],
    )


# ---- A4 缓存感知条件分支（缓存命中 → 免下载直接读）-------------------------
# 参数维度：池 aspect × 池内序号 × 章节（setup 把目标铺成已缓存）
_A4_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "probe_a4_cache_branch_method",
        "aspect": "AI", "pos": 2, "section": "method",
        "text": "查看第2篇论文的缓存状态；如果它已经缓存，就直接读它的方法(method)章节，"
                "如果还没缓存，就先下载再读",
        "note": "setup 已把第 2 篇（2608.14530v1，快照确有 method 章节）铺成已缓存，"
                "因此正确分支是「不下载」：缓存判断后直接读。多余下载会被严格序列判错",
    },
    {
        "id": "probe_a4_cache_branch_result_cl",
        "aspect": "CL", "pos": 3, "section": "result",
        "text": "查看第3篇论文的缓存状态；如果它已经缓存，就直接读它的实验结果(result)章节，"
                "如果还没缓存，就先下载再读",
        "note": "CL 池第 3 篇 2609.24974v1（快照有 result 章节）；换池、换序号、换章节",
    },
    {
        "id": "probe_a4_cache_branch_conclusion_cr",
        "aspect": "CR", "pos": 4, "section": "conclusion",
        "text": "查看第4篇论文的缓存状态；如果它已经缓存，就直接读它的结论(conclusion)章节，"
                "如果还没缓存，就先下载再读",
        "note": "CR 池第 4 篇 2609.24801v1（快照有 conclusion 章节）；序号加深到 4，"
                "排除「永远答第 2 篇」的位置捷径",
    },
]


def _a4_build(v: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        id=v["id"],
        task=v["text"],
        steps=(
            Step("get_paper_cache_status", {"ref": v["pos"]}),
            _content(v["pos"], v["section"]),
        ),
        category="probe_long_chain",
        difficulty="hard",
        max_iterations=12,
        setup=(
            _recent(v["aspect"], 5),
            Step("download_arxiv_pdf", {"ref": v["pos"]}),
        ),
        requires_offline=True,
        note=v["note"],
    )


# ---- A5 图表链（抽图 → 看图 → 摘要）----------------------------------------
# 参数维度：池 aspect × 序号 × 图号 × 问法(describe/trend/axes) × 摘要风格/预算
_A5_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "probe_a5_figure_chain_summary",
        "aspect": "CR", "pos": 3, "figure_no": 1, "question": "describe",
        "style": ("structured", 250),
        "text": "检索最近30天密码学与安全(cs.CR)论文5篇，下载第3篇，抽出它的图表，"
                "描述第1张图画的是什么，再用 250 词以内的结构化风格(structured)总结这篇论文",
        "note": "CR 池第 3 篇 2609.24826v1：快照确有图表与 analyze(describe) 记录。"
                "抽图→看图→摘要三段衔接，VLM 臂与文本臂在这里预期拉开差距",
    },
    {
        "id": "probe_a5_figure_chain_trend_ro",
        "aspect": "RO", "pos": 1, "figure_no": 1, "question": "trend",
        "style": ("tldr", 120),
        "text": "检索最近30天机器人学(cs.RO)论文5篇，下载第1篇，抽出它的图表，"
                "总结第1张图展示的趋势(trend)，再用 120 词以内的一句话风格(tldr)总结这篇论文",
        "note": "RO 池第 1 篇 2609.24996v1；问法换 trend、摘要换 tldr 120，"
                "问法维度不再随 describe 走默认印象",
    },
    {
        "id": "probe_a5_figure_chain_axes_cv",
        "aspect": "CV", "pos": 2, "figure_no": 2, "question": "axes",
        "style": ("bullet", 60),
        "text": "检索最近30天计算机视觉(cs.CV)论文5篇，下载第2篇，抽出它的图表，"
                "说明第2张图用了哪些坐标轴/指标(axes)，再用 60 词以内的要点风格(bullet)总结这篇论文",
        "note": "CV 池第 2 篇 2608.14543v1；图号换 2、问法换 axes、摘要换 bullet 60，"
                "figure_no 与 question 两个易错参数同时偏离默认值",
    },
]


def _a5_build(v: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        id=v["id"],
        task=v["text"],
        steps=(
            _recent(v["aspect"], 5),
            Step("download_arxiv_pdf", {"ref": v["pos"]}),
            Step("extract_paper_figures", {"ref": v["pos"]}),
            _analyze(v["pos"], v["figure_no"], v["question"]),
            _summarize(v["pos"], *v["style"]),
        ),
        category="probe_long_chain",
        difficulty="hard",
        max_iterations=12,
        requires_offline=True,
        note=v["note"],
    )


# ============================================================================
# 族 B：图像入口扰动（降级为图像描述文本，见模块 docstring）
# ============================================================================
# B1-B3 的描述文本全部预生成自冻结快照：caption 取 extract_paper_figures 的
# 记录（PDF 原文换行连字符如 "struc- ture" 已在描述中拼回，note 里注明），
# 与 analyze_figure(describe/axes) 的确定性回答同源。3 个参数组 = 3 篇不同论文
# × 不同图号/问法；判定不含任何「看懂了图没有」的模糊成分：目标论文身份 +
# 图表工具调用参数。落地钩子是 caption 里出现、同时出现在论文标题里的
# 方法名（ref 标题子串），这是纯文本输入下唯一确定性的落地路径。

# ---- B1 图注含方法名 → 以图注为检索键定位论文（describe）--------------------
_B1_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "probe_b1_caption_grounding_describe",
        "setup": _recent("CV", 5),
        "ref": "MagnifiQ", "figure_no": 1,
        "caption": "Fig. 1: MagnifiQ restores real-world low-quality inputs into coherent "
                   "high-resolution outputs up to 4096×4096, supporting up to 32× scaling "
                   "while preserving global structure and recovering fine details.",
        "text": "用户上传了一张论文图表的截图（当前输入通道只传文字，图的内容以下面这段图注描述给出）："
                "\"Fig. 1: MagnifiQ restores real-world low-quality inputs into coherent high-resolution "
                "outputs up to 4096×4096, supporting up to 32× scaling while preserving global structure "
                "and recovering fine details.\"。这张图出自最近30天 cs.CV 候选列表中的某篇论文："
                "找到它并下载，抽出它的图表，然后描述第1张图",
        "note": "图注 = 快照 extract_paper_figures(\"2608.14543v1\") fig1 原文（\"struc- ture\" 已拼回）。"
                "正确论文是 CV 池第 2 篇 2608.14543v1；图注里的方法名 MagnifiQ 就是可解析的标题子串",
    },
    {
        "id": "probe_b1_caption_grounding_harness",
        "setup": _recent("CL", 5),
        "ref": "Harness-Zero", "figure_no": 1,
        "caption": "Figure 1: Overview of HARNESS-ZERO. (1) Evolve and adapt. We evolve a "
                   "student-side harness on training tasks, and adapt it into a private "
                   "reference harness.",
        "text": "用户给了一张论文架构图的截图（输入通道只传文字，内容按图注描述）："
                "\"Figure 1: Overview of HARNESS-ZERO. (1) Evolve and adapt. We evolve a student-side "
                "harness on training tasks, and adapt it into a private reference harness.\"。"
                "这张图出自最近30天 cs.CL 候选列表中的某篇论文：找到并下载该论文，抽出图表，描述第1张图",
        "note": "图注 = 快照 extract_paper_figures(\"2609.24974v1\") fig1（上标 ⋆ 等排版符号已化简）。"
                "正确论文是 CL 池第 3 篇 2609.24974v1，describe 回答与图注同源",
    },
    {
        "id": "probe_b1_caption_grounding_dualwam",
        "setup": _recent("RO", 10),
        "ref": "DualWAM", "figure_no": 1,
        "caption": "Figure 1 Overview. DualWAM couples low-frequency global world-action "
                   "planning with high-frequency wrist-centric refinement along a shared "
                   "denoising trajectory.",
        "text": "用户发来一张论文总览图的截图（输入通道只传文字，内容按图注描述）："
                "\"Figure 1 Overview. DualWAM couples low-frequency global world-action planning with "
                "high-frequency wrist-centric refinement along a shared denoising trajectory.\"。"
                "这张图出自最近30天 cs.RO 候选列表中的某篇论文：找到并下载它，抽出图表并描述第1张图",
        "note": "图注 = 快照 extract_paper_figures(\"2609.24868v1\") fig1 原文。正确论文是 "
                "RO 池第 7 篇 2609.24868v1，因此种子池给到 10 篇（序号 7 不在 5 篇池内）；"
                "hook=DualWAM 标题子串",
    },
]


def _b1_build(v: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        id=v["id"],
        task=v["text"],
        steps=(
            Step("download_arxiv_pdf", {"ref": v["ref"]}),
            Step("extract_paper_figures", {"ref": None}),
            _analyze(None, v["figure_no"], "describe"),
        ),
        category="probe_image_entry",
        difficulty="hard",
        max_iterations=6,
        setup=(v["setup"],),
        requires_offline=True,
        note=v["note"],
    )


# ---- B2 图序传递：图注明确是 Figure 2+，考察 figure_no 不被默认成 1 ----------
_B2_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "probe_b2_figure_index_grounding",
        "setup": _recent("LG", 5),
        "ref": "onPanda", "figure_no": 2,
        "caption": "Figure 2: Annotating an agent trajectory with onPanda. Top left: the "
                   "user prompt; top right: the connected MCP servers and tools.",
        "text": "用户给了一张论文图（Figure 2）的截图（输入通道只传文字，内容按图注描述）："
                "\"Figure 2: Annotating an agent trajectory with onPanda. Top left: the user prompt; "
                "top right: the connected MCP servers and tools.\"。它出自最近30天 cs.LG 候选列表中的"
                "某篇论文：找到并下载该论文，抽出图表，然后描述第2张图",
        "note": "图注 = 快照 extract_paper_figures(\"2609.24983v1\") fig2（\"on- Panda\"/\"con- nected\" "
                "的换行连字符已拼回）。正确论文是 LG 池第 2 篇 2609.24983v1，该论文只有 2 张图；"
                "把 figure_no 默认成 1 的话参数分直接丢失",
    },
    {
        "id": "probe_b2_figure_index_rare_event",
        "setup": _recent("LG", 5),
        "ref": "Rare Event Estimation", "figure_no": 2,
        "caption": "Figure 2. Importance sampling with Iterative Unalignment for rare-event "
                   "probability estimation. Naive Monte Carlo from the base model rarely "
                   "observes the rare target event.",
        "text": "用户给了一张示意图（Figure 2）的截图（输入通道只传文字，内容按图注描述）："
                "\"Figure 2. Importance sampling with Iterative Unalignment for rare-event probability "
                "estimation. Naive Monte Carlo from the base model rarely observes the rare target "
                "event.\"。它出自最近30天 cs.LG 候选列表中的某篇论文：找到并下载该论文，抽出图表，"
                "描述第2张图",
        "note": "图注 = 快照 extract_paper_figures(\"2609.24969v1\") fig2（数学记号化简）。"
                "正确论文是 LG 池第 5 篇 2609.24969v1；hook 同时出现在图注与标题"
                "（Iterative Unalignment），fig2 描述与 analyze 回答同源",
    },
    {
        "id": "probe_b2_figure_index_opbackdoor",
        "setup": _recent("CR", 5),
        "ref": "OPBackdoor", "figure_no": 2,
        "caption": "Figure 1: OPBACKDOOR coding instantiation across the four combinations "
                   "of opportunity and trigger presence.",
        "text": "用户给了一张论文标注示例图的截图（输入通道只传文字，内容按图注描述）："
                "\"Figure 1: OPBACKDOOR coding instantiation across the four combinations of opportunity "
                "and trigger presence.\"。它出自最近30天 cs.CR 候选列表中的某篇论文：找到并下载该论文，"
                "抽出图表，然后描述第2张图",
        "note": "快照 quirk：该论文抽取出的第 2 个图文件（figure_no=2）携带的 caption 文本以 "
                "\"Figure 1:\" 开头——描述按快照原文引用，判定仍以 figure_no=2 为准"
                "（正确论文 CR 池第 3 篇 2609.24826v1，analyze(2, describe) 与 caption 同源）",
    },
]


def _b2_build(v: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        id=v["id"],
        task=v["text"],
        steps=(
            Step("download_arxiv_pdf", {"ref": v["ref"]}),
            Step("extract_paper_figures", {"ref": None}),
            _analyze(None, v["figure_no"], "describe"),
        ),
        category="probe_image_entry",
        difficulty="hard",
        max_iterations=6,
        setup=(v["setup"],),
        requires_offline=True,
        note=v["note"],
    )


# ---- B3 面板/指标类问题：axes 问法 ------------------------------------------
_B3_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "probe_b3_panel_axes_grounding",
        "setup": _recent("CV", 5),
        "ref": "CPI-Bench", "figure_no": 2,
        "caption": "Right: Spearman correlation coefficients and Mean Absolute Error (MAE) "
                   "with Arena rank. CPI-Bench outperforms other benchmarks, achieving the "
                   "highest correlation and the lowest error margin.",
        "text": "用户发来一张论文里的对比面板图（内容按图注描述）：右面板展示与 Arena Image Edit "
                "Leaderboard 对比的 Spearman 相关系数与平均绝对误差(MAE)，图注写道 "
                "\"CPI-Bench outperforms other benchmarks, achieving the highest correlation and the "
                "lowest error margin.\"。这张图出自最近30天 cs.CV 候选列表中的某篇论文："
                "找到并下载它，抽出图表，再说明第2张图用了哪些坐标轴/指标(axes)",
        "note": "描述预生成自快照：caption 含 \"Right: Spearman correlation coefficients and Mean "
                "Absolute Error (MAE) with Arena rank\"（\"Leader- board\" 连字符已拼回）。"
                "正确论文是 CV 池第 1 篇 2608.14546v1；快照把 caption-Figure-1 的左右面板存成 "
                "figure_no=1/2 两个文件，axes 答案与描述同源",
    },
    {
        "id": "probe_b3_panel_axes_marionette",
        "setup": _recent("AI", 5),
        "ref": "Marionette", "figure_no": 3,
        "caption": "Figure 1 Marionette overview. Bottom: the pipeline. ActionGPT makes "
                   "discrete per-entity action decisions as a streaming token sequence; "
                   "PoseGPT turns them into a continuous articulated state.",
        "text": "用户给了一张论文流程总览图的截图（输入通道只传文字，内容按图注描述）："
                "\"Marionette overview. Bottom: the pipeline. ActionGPT makes discrete per-entity "
                "action decisions as a streaming token sequence; PoseGPT turns them into a continuous "
                "articulated state.\"。它出自最近30天 cs.AI 候选列表中的某篇论文：找到并下载该论文，"
                "抽出图表，说明第3张图用了哪些坐标轴/面板标注(axes)",
        "note": "图注 = 快照 extract_paper_figures(\"2608.14530v1\") fig3；正确论文 AI 池第 2 篇 "
                "2608.14530v1，figure_no=3 + axes 双参数偏离默认",
    },
    {
        "id": "probe_b3_panel_axes_gamehorizon",
        "setup": _recent("CV", 5),
        "ref": "GameHorizon", "figure_no": 3,
        "caption": "Figure 2. Overview of GameHorizon Suite. GameHorizon-Annotator "
                   "automatically produces a three-level pyramid of short-horizon tasks.",
        "text": "用户给了一张论文基准总览图的截图（输入通道只传文字，内容按图注描述）："
                "\"Overview of GameHorizon Suite. GameHorizon-Annotator automatically produces a "
                "three-level pyramid of short-horizon tasks.\"。它出自最近30天 cs.CV 候选列表中的"
                "某篇论文：找到并下载该论文，抽出图表，说明第3张图的坐标轴/结构标注(axes)",
        "note": "图注 = 快照 extract_paper_figures(\"2609.25001v1\") fig3；正确论文 CV 池第 4 篇 "
                "2609.25001v1（同池不同序号，与 B3 v1 形成池内对照）",
    },
]


def _b3_build(v: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        id=v["id"],
        task=v["text"],
        steps=(
            Step("download_arxiv_pdf", {"ref": v["ref"]}),
            Step("extract_paper_figures", {"ref": None}),
            _analyze(None, v["figure_no"], "axes"),
        ),
        category="probe_image_entry",
        difficulty="hard",
        max_iterations=6,
        setup=(v["setup"],),
        requires_offline=True,
        note=v["note"],
    )


# ---- B4 图内水印 arXiv ID → 直接按 ID 下载（ID 藏在池内第 5-7 位）------------
_B4_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "probe_b4_watermark_id_download",
        "setup": _recent("CL", 50),
        "paper_id": "2609.24967v1",
        "text": "用户给了一张论文图表的截图，右下角有水印文字 \"arXiv:2609.24967v1\"（输入通道只传文字，"
                "水印内容如前所述）。请下载这篇论文",
        "note": "2609.24967v1 是 CL 池第 6 篇，按序号瞎猜解析到的是别的论文，"
                "oracle 的 expected_paper_ids 会拆穿；正确动作是从描述里的 ID 直接取",
    },
    {
        "id": "probe_b4_watermark_id_ai",
        "setup": _recent("AI", 50),
        "paper_id": "2609.24984v1",
        "text": "用户给了一张论文图表的截图，右下角有水印文字 \"arXiv:2609.24984v1\"（输入通道只传文字，"
                "水印内容如前所述）。请下载这篇论文",
        "note": "AI 池第 5 篇 2609.24984v1；与 v1 换池、换位置，同一「读水印 ID」语义",
    },
    {
        "id": "probe_b4_watermark_id_cv",
        "setup": _recent("CV", 50),
        "paper_id": "2609.24997v1",
        "text": "用户给了一张论文图表的截图，右下角有水印文字 \"arXiv:2609.24997v1\"（输入通道只传文字，"
                "水印内容如前所述）。请下载这篇论文",
        "note": "CV 池第 5 篇 2609.24997v1；三条参数组合覆盖 3 个池、3 个 ≥5 的深位置",
    },
]


def _b4_build(v: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        id=v["id"],
        task=v["text"],
        steps=(
            Step("download_arxiv_pdf", {"ref": v["paper_id"]}),
        ),
        category="probe_image_entry",
        difficulty="medium",
        setup=(v["setup"],),
        requires_offline=True,
        note=v["note"],
    )


# ---- B5 无关图像 → 正确拒绝（复用现有 blocked 终止语义）----------------------
# 参数维度：无关图像类型 × 种子池（有候选池也必须拒绝，排除 missing_context 借口）
_B5_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "probe_b5_non_paper_image_refusal",
        "setup": _recent("*", 5),
        "text": "用户发来一张照片：厨房桌上一盘番茄炒蛋旁边的手写家常菜谱笔记，"
                "要求\"找出这张图出自的那篇论文并下载它\"",
        "note": "非论文图像，且会话里有 5 篇候选也依然无法定位出处 —— 排除「会话为空」的"
                "missing_context 借口，正确行为是说明能力边界，而不是编造检索或乱下论文。"
                "文本臂与 VLM 臂在这条上应当同分（校准下界）",
    },
    {
        "id": "probe_b5_non_paper_image_refusal_pet",
        "setup": _recent("CV", 3),
        "text": "用户发来一张照片：一只橘猫睡在笔记本电脑键盘上，屏幕上没有任何论文内容，"
                "要求\"找出这张图出自的那篇论文并下载它\"",
        "note": "换无关图像类型（宠物照）并换 CV 种子池 3 篇；同样必须拒绝而非乱检索",
    },
    {
        "id": "probe_b5_non_paper_image_refusal_landmark",
        "setup": _recent("LG", 5),
        "text": "用户发来一张照片：傍晚的城市天际线风景照，没有任何文字或图表，"
                "要求\"找出这张图出自的那篇论文并下载它\"",
        "note": "第三种无关图像（风景照）+ LG 种子池；三条参数组共同压住"
                "「换个池子就能找到」的幻觉方向",
    },
]


def _b5_build(v: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        id=v["id"],
        task=v["text"],
        category="probe_image_entry",
        difficulty="hard",
        terminal_mode="blocked",
        terminal_reason="unsupported_capability",
        setup=(v["setup"],),
        note=v["note"],
    )


# ============================================================================
# 族 C：多任务对比整合（strict 之外新增 integration_complete：
# 终答需同时引用全部目标论文，以 expected_integration_paper_ids 表达 +
# evaluate_integration() 判定；report.py 的 +2 键扩展在评测侧做）
# ============================================================================

# ---- C1 双篇对比（两组「池+序号」+ 章节）------------------------------------
_C1_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "probe_c1_two_paper_method_contrast",
        "left": ("AI", 2), "right": ("CR", 3), "section": "method", "pool_n": 3,
        "section_cn": "方法(method)", "thing": "方法差异",
        "text": "先检索最近30天 cs.AI 论文3篇，下载第2篇并读它的方法(method)章节；"
                "再检索最近30天 cs.CR 论文3篇，下载第3篇并读它的方法(method)章节；"
                "最后在最终回答中对比两篇论文的方法差异，回答里需要同时提到这两篇论文",
        "note": "两篇都按快照挑过章节覆盖：AI pos2=2608.14530v1、CR pos3=2609.24826v1，均有 method",
    },
    {
        "id": "probe_c1_two_paper_method_contrast_alt",
        "left": ("CR", 4), "right": ("AI", 5), "section": "method", "pool_n": 5,
        "section_cn": "方法(method)", "thing": "方法差异",
        "text": "先检索最近30天 cs.CR 论文5篇，下载第4篇并读它的方法(method)章节；"
                "再检索最近30天 cs.AI 论文5篇，下载第5篇并读它的方法(method)章节；"
                "最后在最终回答中对比两篇论文的方法差异，回答里需要同时提到这两篇论文",
        "note": "快照里共有 4 篇带 method 章节，这是不与 v1 重叠的另一对："
                "CR pos4=2609.24801v1、AI pos5=2609.24984v1；每池篇数同步抬到 5",
    },
    {
        "id": "probe_c1_two_paper_result_contrast",
        "left": ("LG", 1), "right": ("RO", 1), "section": "result", "pool_n": 3,
        "section_cn": "实验结果(result)", "thing": "实验结果差异",
        "text": "先检索最近30天 cs.LG 论文3篇，下载第1篇并读它的实验结果(result)章节；"
                "再检索最近30天 cs.RO 论文3篇，下载第1篇并读它的实验结果(result)章节；"
                "最后在最终回答中对比两篇论文的实验结果差异，回答里需要同时提到这两篇论文",
        "note": "章节维度换成 result（LG pos1=2609.24985v1、RO pos1=2609.24996v1 均有 result），"
                "模板形状不变、对比对象从方法换成结果",
    },
]


def _c1_build(v: Dict[str, Any]) -> TaskSpec:
    la, lp = v["left"]
    ra, rp = v["right"]
    return TaskSpec(
        id=v["id"],
        task=v["text"],
        steps=(
            _recent(la, v["pool_n"]),
            Step("download_arxiv_pdf", {"ref": lp}),
            _content(lp, v["section"]),
            _recent(ra, v["pool_n"]),
            Step("download_arxiv_pdf", {"ref": rp}),
            _content(rp, v["section"]),
        ),
        category="probe_integration",
        difficulty="hard",
        max_iterations=8,
        requires_offline=True,
        note=v["note"],
    )


# ---- C2 时序对比（快照池与时间窗无关，见模块 docstring 偏差 3）---------------
# 参数维度：aspect × 近端窗口 × 每池篇数
_C2_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "probe_c2_time_window_contrast",
        "aspect": "LG", "days_lo": 1, "pool_n": 5,
        "text": "分别检索最近1天与最近30天的机器学习(cs.LG)论文各5篇，对比两个时间窗结果集的差异，"
                "并在最终回答中列出两个窗口都出现的论文（需要给出论文编号）",
        "note": "判定核心是 days=1 与 days=30 两个窗口参数都被真实传对；"
                "integration 引用两窗交集（快照里=LG 池前 5 篇，确定性地无差异）",
    },
    {
        "id": "probe_c2_time_window_contrast_cv",
        "aspect": "CV", "days_lo": 3, "pool_n": 3,
        "text": "分别检索最近3天与最近30天的计算机视觉(cs.CV)论文各3篇，对比两个时间窗结果集的差异，"
                "并在最终回答中列出两个窗口都出现的论文（需要给出论文编号）",
        "note": "换 CV 池、近端窗口换 3 天、篇数换 3；交集=CV 池前 3 篇",
    },
    {
        "id": "probe_c2_time_window_contrast_ai",
        "aspect": "AI", "days_lo": 7, "pool_n": 5,
        "text": "分别检索最近7天与最近30天的人工智能(cs.AI)论文各5篇，对比两个时间窗结果集的差异，"
                "并在最终回答中列出两个窗口都出现的论文（需要给出论文编号）",
        "note": "近端窗口换 7 天（与既有 expanded 池的常用窗口对齐）、AI 池；交集=AI 池前 5 篇",
    },
]


def _c2_build(v: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        id=v["id"],
        task=v["text"],
        steps=(
            _recent(v["aspect"], v["pool_n"], days=v["days_lo"]),
            _recent(v["aspect"], v["pool_n"], days=30),
        ),
        category="probe_integration",
        difficulty="medium",
        max_iterations=4,
        requires_offline=True,
        note=v["note"],
    )


# ---- C3 翻译一致性（A1 的组件化对照：前缀由 setup 铺好）----------------------
# 参数维度：池 aspect × 序号 × 汇总摘要的风格/预算（两个翻译参数两 variant 固定）
_C3_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "probe_c3_translate_variants_summarize",
        "aspect": "CL", "pos": 1, "style": ("tldr", 60),
        "text": "第1篇论文已经下载好了。请先用 google 服务把它翻译一次，"
                "再用保留双语版本(keep_dual)的方式翻译一次，"
                "最后用 60 词以内的一句话风格(tldr)总结这篇论文作为汇总",
        "note": "同一篇论文的两种翻译产物（指定服务 / 双语）+ 一次汇总摘要，"
                "service 与 keep_dual 两个可选参数都要真的传对；ref=null 指向 setup 下载的那篇"
                "（CL pos1=2609.24985v1）",
    },
    {
        "id": "probe_c3_translate_variants_summarize_cv",
        "aspect": "CV", "pos": 1, "style": ("structured", 120),
        "text": "第1篇论文已经下载好了。请先用 google 服务把它翻译一次，"
                "再用保留双语版本(keep_dual)的方式翻译一次，"
                "最后用 120 词以内的结构化风格(structured)总结这篇论文作为汇总",
        "note": "CV 池第 1 篇 2608.14546v1；翻译对不变、汇总摘要换 structured 120",
    },
    {
        "id": "probe_c3_translate_variants_summarize_ai",
        "aspect": "AI", "pos": 2, "style": ("bullet", 250),
        "text": "第2篇论文已经下载好了。请先用 google 服务把它翻译一次，"
                "再用保留双语版本(keep_dual)的方式翻译一次，"
                "最后用 250 词以内的要点风格(bullet)总结这篇论文作为汇总",
        "note": "AI 池第 2 篇 2608.14530v1；序号、池、摘要风格三维度同时变化，"
                "setup 里 download 的 ref 与任务话术同步为第2篇",
    },
]


def _c3_build(v: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        id=v["id"],
        task=v["text"],
        steps=(
            Step("translate_arxiv_pdf", {"ref": None, "service": "google"}),
            Step("translate_arxiv_pdf", {"ref": None, "keep_dual": True}),
            _summarize(v["pos"], *v["style"]),
        ),
        category="probe_integration",
        difficulty="hard",
        max_iterations=5,
        setup=(
            _recent(v["aspect"], 5),
            Step("download_arxiv_pdf", {"ref": v["pos"]}),
        ),
        requires_offline=True,
        note=v["note"],
    )


# ---- C4 跨工具冗余对比：search vs get_recently，只走更精准的一条 -------------
# 参数维度：在案 query × 目标序号/标题钩子 × 被排除的池 aspect
_C4_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "probe_c4_tool_choice_keyword_over_list",
        "query": "all:retrieval augmented generation",
        "hook": "Doc-REFRAG", "distractor": "CL",
        "text": "目标：下载标题含 \"Doc-REFRAG\" 的论文。有两条路可选："
                "按关键词检索 \"all:retrieval augmented generation\"（最近30天，最多5篇），"
                "或者拉取最近30天的 cs.CL 论文列表。关键词检索更精准——"
                "只走关键词检索这条路完成下载，不要两条都跑",
        "note": "测工具选择语义：两个检索工具都能拿到候选时，按任务约束只调一个；"
                "多调 get_recently 会被严格序列判错。目标 2608.30163v1 是该 query 结果第 1 篇",
    },
    {
        "id": "probe_c4_tool_choice_keyword_over_list_arl",
        "query": "all:agentic reinforcement learning",
        "hook": "HARTS", "distractor": "RO",
        "text": "目标：下载标题含 \"HARTS\" 的论文。有两条路可选："
                "按关键词检索 \"all:agentic reinforcement learning\"（最近30天，最多5篇），"
                "或者拉取最近30天的 cs.RO 论文列表。关键词检索更精准——"
                "只走关键词检索这条路完成下载，不要两条都跑",
        "note": "换在案 query（agentic RL），目标仍是该结果第 1 篇 2608.28158v1，干扰池换 RO",
    },
    {
        "id": "probe_c4_tool_choice_keyword_over_list_llm",
        "query": "all:large language model",
        "hook": "SWRouter", "distractor": "CR",
        "text": "目标：下载标题含 \"SWRouter\" 的论文。有两条路可选："
                "按关键词检索 \"all:large language model\"（最近30天，最多5篇），"
                "或者拉取最近30天的 cs.CR 论文列表。关键词检索更精准——"
                "只走关键词检索这条路完成下载，不要两条都跑",
        "note": "第 3 条在案 query，目标挪到结果第 2 篇（2609.11414v1），干扰池换 CR；"
                "三条组合覆盖全部在案 query 与 1/1/2 的目标序号",
    },
]


def _c4_build(v: Dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        id=v["id"],
        task=v["text"],
        steps=(
            _kw_search(v["query"]),
            Step("download_arxiv_pdf", {"ref": v["hook"]}),
        ),
        category="probe_integration",
        difficulty="hard",
        max_iterations=4,
        requires_offline=True,
        note=v["note"],
    )


# ---- C5 三篇聚合（读摘要 + 汇总三篇编号）-------------------------------------
# 参数维度：三个 aspect 的组合 × 组内序号（abstract 每篇都有）
_C5_VARIANTS: List[Dict[str, Any]] = [
    {
        "id": "probe_c5_three_paper_abstract_aggregate",
        "groups": (("CV", 1), ("CL", 1), ("CR", 1)), "pool_n": 3,
        "text": "依次处理三组检索：检索最近30天 cs.CV 论文3篇，下载第1篇并读出它的摘要(abstract)；"
                "检索最近30天 cs.CL 论文3篇，下载第1篇并读出摘要；"
                "检索最近30天 cs.CR 论文3篇，下载第1篇并读出摘要；"
                "最后在最终回答中汇总三篇论文，回答里需要同时给出三篇的论文编号",
        "note": "9 次调用；三组 ref=1 随池切换（CV pos1=2608.14546v1, CL pos1=2609.24985v1, "
                "CR pos1=2609.24994v1），摘要(abstract)是快照里每篇都有的章节",
    },
    {
        "id": "probe_c5_three_paper_abstract_aggregate_pos2",
        "groups": (("AI", 2), ("LG", 2), ("RO", 2)), "pool_n": 3,
        "text": "依次处理三组检索：检索最近30天 cs.AI 论文3篇，下载第2篇并读出它的摘要(abstract)；"
                "检索最近30天 cs.LG 论文3篇，下载第2篇并读出摘要；"
                "检索最近30天 cs.RO 论文3篇，下载第2篇并读出摘要；"
                "最后在最终回答中汇总三篇论文，回答里需要同时给出三篇的论文编号",
        "note": "换组合（AI/LG/RO）且序号全部换 2（2608.14530v1 / 2609.24983v1 / 2609.24995v1），"
                "与 v1 无一篇重叠",
    },
    {
        "id": "probe_c5_three_paper_abstract_aggregate_pos3",
        "groups": (("CL", 3), ("CV", 3), ("RO", 3)), "pool_n": 3,
        "text": "依次处理三组检索：检索最近30天 cs.CL 论文3篇，下载第3篇并读出它的摘要(abstract)；"
                "检索最近30天 cs.CV 论文3篇，下载第3篇并读出摘要；"
                "检索最近30天 cs.RO 论文3篇，下载第3篇并读出摘要；"
                "最后在最终回答中汇总三篇论文，回答里需要同时给出三篇的论文编号",
        "note": "序号加深到 3（2609.24974v1 / 2608.14539v1 / 2609.24976v1），"
                "组合与 v1/v2 均不同，排除位置捷径",
    },
]


def _c5_build(v: Dict[str, Any]) -> TaskSpec:
    steps: List[Step] = []
    for aspect, pos in v["groups"]:
        steps.append(_recent(aspect, v["pool_n"]))
        steps.append(Step("download_arxiv_pdf", {"ref": pos}))
        steps.append(_content(pos, "abstract"))
    return TaskSpec(
        id=v["id"],
        task=v["text"],
        steps=tuple(steps),
        category="probe_integration",
        difficulty="hard",
        max_iterations=12,
        requires_offline=True,
        note=v["note"],
    )


# ---------------------------------------------------------------------------
# 组装：15 模板 × 3 参数组 = 45 条
# ---------------------------------------------------------------------------

def _expand(variants: List[Dict[str, Any]], builder) -> List[TaskSpec]:
    return [builder(v) for v in variants]


_LONG_CHAIN_PROBES: List[TaskSpec] = (
    _expand(_A1_VARIANTS, _a1_build)
    + _expand(_A2_VARIANTS, _a2_build)
    + _expand(_A3_VARIANTS, _a3_build)
    + _expand(_A4_VARIANTS, _a4_build)
    + _expand(_A5_VARIANTS, _a5_build)
)

_IMAGE_ENTRY_PROBES: List[TaskSpec] = (
    _expand(_B1_VARIANTS, _b1_build)
    + _expand(_B2_VARIANTS, _b2_build)
    + _expand(_B3_VARIANTS, _b3_build)
    + _expand(_B4_VARIANTS, _b4_build)
    + _expand(_B5_VARIANTS, _b5_build)
)

_INTEGRATION_PROBES: List[TaskSpec] = (
    _expand(_C1_VARIANTS, _c1_build)
    + _expand(_C2_VARIANTS, _c2_build)
    + _expand(_C3_VARIANTS, _c3_build)
    + _expand(_C4_VARIANTS, _c4_build)
    + _expand(_C5_VARIANTS, _c5_build)
)


# C 族的 integration 目标：终答必须同时引用的论文（与快照内池位一一对应，
# oracle 预检会用 gold 重放的解析结果逐一核对，见本文件 `python -m` 自检）。
PROBE_INTEGRATION: Dict[str, List[str]] = {
    # C1: 两侧 (aspect, pos) 的目标论文
    "probe_c1_two_paper_method_contrast": ["2608.14530v1", "2609.24826v1"],
    "probe_c1_two_paper_method_contrast_alt": ["2609.24801v1", "2609.24984v1"],
    "probe_c1_two_paper_result_contrast": ["2609.24985v1", "2609.24996v1"],
    # C2: 两窗交集 = 池前 N 篇（快照池与窗口无关，见 docstring 偏差 3）
    "probe_c2_time_window_contrast": [
        "2609.24985v1", "2609.24983v1", "2609.24979v1", "2609.24972v1", "2609.24969v1",
    ],
    "probe_c2_time_window_contrast_cv": ["2608.14546v1", "2608.14543v1", "2608.14539v1"],
    "probe_c2_time_window_contrast_ai": [
        "2608.14539v1", "2608.14530v1", "2608.14528v1", "2609.25001v1", "2609.24984v1",
    ],
    # C3: setup 下载的那篇
    "probe_c3_translate_variants_summarize": ["2609.24985v1"],
    "probe_c3_translate_variants_summarize_cv": ["2608.14546v1"],
    "probe_c3_translate_variants_summarize_ai": ["2608.14530v1"],
    # C4: 关键词检索结果里的目标论文
    "probe_c4_tool_choice_keyword_over_list": ["2608.30163v1"],
    "probe_c4_tool_choice_keyword_over_list_arl": ["2608.28158v1"],
    "probe_c4_tool_choice_keyword_over_list_llm": ["2609.11414v1"],
    # C5: 三组 (aspect, pos) 的目标论文
    "probe_c5_three_paper_abstract_aggregate": ["2608.14546v1", "2609.24985v1", "2609.24994v1"],
    "probe_c5_three_paper_abstract_aggregate_pos2": ["2608.14530v1", "2609.24983v1", "2609.24995v1"],
    "probe_c5_three_paper_abstract_aggregate_pos3": ["2609.24974v1", "2608.14539v1", "2609.24976v1"],
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
    """run_benchmark --task-set probe 的任务池（45 条，全部为评测探针）。"""
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
    # 快速自检：id 唯一、每模板 3 个参数组、integration 目标只挂在 C 族、
    # 离线绑定齐全（除 B5 的 3 条拒绝题）。
    ids = [t['id'] for t in PROBE_TASKS]
    assert len(ids) == len(set(ids)) == 45, len(ids)
    for prefix in ('probe_a1', 'probe_a2', 'probe_a3', 'probe_a4', 'probe_a5',
                   'probe_b1', 'probe_b2', 'probe_b3', 'probe_b4', 'probe_b5',
                   'probe_c1', 'probe_c2', 'probe_c3', 'probe_c4', 'probe_c5'):
        n = sum(1 for tid in ids if tid.startswith(prefix + '_'))
        assert n == 3, (prefix, n)
    for tid in PROBE_INTEGRATION:
        task = next(t for t in PROBE_TASKS if t['id'] == tid)
        assert task['category'] == 'probe_integration', tid
        assert task.get('expected_integration_paper_ids'), tid
    assert len(PROBE_INTEGRATION) == 15
    assert set(offline_only_ids()) == {
        t['id'] for t in PROBE_TASKS if not t['id'].startswith('probe_b5')
    }
    print('tasks_probe 自检通过：15 模板 × 3 参数组 = 45 条探针（A15 / B15 / C15）')
    for t in PROBE_TASKS:
        print(f"  {t['id']:52s} {t['category']:18s} "
              f"steps={len(t['expected_tools'])} "
              f"max_iter={t.get('max_iterations')}")
