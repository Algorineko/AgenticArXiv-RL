"""标准化测试集：由 task_spec 声明式定义展开。

相比 benchmark/tasks.py 的 8 条，补齐了几个此前完全没覆盖的维度：

  - **指代形态**（ref_form）：同一篇论文的序号 / arXiv ID / 标题子串 / null 四种写法
  - **可选参数**（optional）：force / service / threads / keep_dual
  - **跨步状态**（state）：「刚下载的那篇」这类必须靠会话状态才能解析的指代
  - **多跳链路**（composite / long_chain）：2~5 步的工具链
  - **不可行请求**（infeasible）：正确行为是**不调用任何工具**

`expected_tools` 与 `expected_tool_args` 都从 `steps` 派生（见
benchmark/task_spec.py），不存在两份标准答案漂移的可能。

category=ref_form 与 state 中标 requires_offline 的任务，ground truth 绑定
data/mock_arxiv_snapshot.json，需配合 run_benchmark.py --offline 运行。
其中 4 条 ref_stress_* 用的标题子串经过挑选：完整短语与截断后的首词会命中
**不同**的论文，因此能区分「参数是否被完整传递」。
"""

from typing import Any, Dict, List

from benchmark.task_spec import Step, TaskSpec, build, family

# arXiv CS 子类 -> 中文名，供任务描述渲染
_CN = {
    "AI": "人工智能", "LG": "机器学习", "CL": "自然语言处理",
    "CV": "计算机视觉", "RO": "机器人学", "CR": "密码学与安全",
}

# 四种自然语言说法轮换，避免模型靠句式模板而不是语义来抽参数
_SEARCH_PHRASINGS = (
    "检索最近{days}天内{cn}(cs.{a})方向的论文，最多{n}篇",
    "帮我找一下最近{days}天{cn}(cs.{a})的新论文，最多{n}篇",
    "获取最近{days}天{cn}方向(cs.{a})的最新论文，数量上限{n}篇",
    "搜索 cs.{a} 方向最近{days}天的论文，最多返回{n}篇",
)

_SEARCH_PARAMS = [
    {"a": "AI", "days": 1, "n": 3, "phrasing": 0},
    {"a": "AI", "days": 30, "n": 25, "phrasing": 1},
    {"a": "LG", "days": 3, "n": 10, "phrasing": 2},
    {"a": "LG", "days": 14, "n": 1, "phrasing": 3},
    {"a": "CL", "days": 7, "n": 5, "phrasing": 0},
    {"a": "CL", "days": 30, "n": 10, "phrasing": 1},
    {"a": "CV", "days": 1, "n": 1, "phrasing": 2},
    {"a": "CV", "days": 14, "n": 3, "phrasing": 3},
    {"a": "RO", "days": 3, "n": 8, "phrasing": 0},
    {"a": "RO", "days": 30, "n": 25, "phrasing": 1},
    {"a": "CR", "days": 7, "n": 5, "phrasing": 2},
    {"a": "CR", "days": 14, "n": 10, "phrasing": 3},
]

# 任务描述与标准答案由同一份参数渲染，因此「描述里写 7 天、答案里写 30 天」
# 这种不一致在结构上就不可能发生。
_SEARCH = family(
    task_id=lambda p: f"search_{p['a']}_{p['days']}d_{p['n']}",
    text=lambda p: _SEARCH_PHRASINGS[p["phrasing"]].format(
        days=p["days"], cn=_CN[p["a"]], a=p["a"], n=p["n"]),
    steps=lambda p: [Step("get_recently_submitted_cs_papers",
                          {"aspect": p["a"], "days": p["days"], "max_results": p["n"]})],
    params=_SEARCH_PARAMS,
    category="search",
    difficulty="easy",
)

_SEED_AI5 = (Step("get_recently_submitted_cs_papers",
                  {"aspect": "AI", "days": 7, "max_results": 5}),)

_OTHERS: List[TaskSpec] = [
    # ---------- optional ----------
    TaskSpec(
        id='opt_force_dl',
        task='强制重新下载第1篇论文',
        steps=(
            Step('download_arxiv_pdf', {'ref': 1, 'force': True}),
        ),
        category='optional',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
        ),
    ),
    TaskSpec(
        id='opt_keep_dual',
        task='翻译第1篇论文，保留双语版本',
        steps=(
            Step('translate_arxiv_pdf', {'ref': 1, 'keep_dual': True}),
        ),
        category='optional',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 1}),
        ),
    ),
    TaskSpec(
        id='opt_service',
        task='用 google 服务翻译第1篇论文',
        steps=(
            Step('translate_arxiv_pdf', {'ref': 1, 'service': 'google'}),
        ),
        category='optional',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 1}),
        ),
    ),
    TaskSpec(
        id='opt_threads',
        task='翻译第1篇论文，用8个线程',
        steps=(
            Step('translate_arxiv_pdf', {'ref': 1, 'threads': 8}),
        ),
        category='optional',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 1}),
        ),
    ),
    TaskSpec(
        id='opt_force_tr',
        task='强制重新翻译第1篇论文',
        steps=(
            Step('translate_arxiv_pdf', {'ref': 1, 'force': True}),
        ),
        category='optional',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 1}),
        ),
    ),
    # ---------- ref_form ----------
    TaskSpec(
        id='ref_stress_learning_state',
        task='下载标题里包含 "Learning State" 的那篇论文',
        steps=(
            Step('download_arxiv_pdf', {'ref': 'Learning State'}),
        ),
        category='ref_form',
        difficulty='hard',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
        ),
        requires_offline=True,
        note='截断成 Learning 会命中 2608.14539v1（错误）',
    ),
    TaskSpec(
        id='ref_stress_state_across',
        task='查看标题里含 "State Across" 那篇论文的缓存状态',
        steps=(
            Step('get_paper_cache_status', {'ref': 'State Across'}),
        ),
        category='ref_form',
        difficulty='hard',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
        ),
        requires_offline=True,
        note='截断成 State 会命中 2608.14530v1（错误）',
    ),
    TaskSpec(
        id='ref_stress_image_restoration',
        task='下载标题包含 "Image Restoration" 的论文',
        steps=(
            Step('download_arxiv_pdf', {'ref': 'Image Restoration'}),
        ),
        category='ref_form',
        difficulty='hard',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'CV', 'days': 7, 'max_results': 5}),
        ),
        requires_offline=True,
        note='截断成 Image 会命中 2608.14546v1（错误）',
    ),
    TaskSpec(
        id='ref_stress_uncertainty_aware',
        task='把标题里有 "An Uncertainty-Aware" 的那篇下载下来',
        steps=(
            Step('download_arxiv_pdf', {'ref': 'An Uncertainty-Aware'}),
        ),
        category='ref_form',
        difficulty='hard',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'CV', 'days': 7, 'max_results': 5}),
        ),
        requires_offline=True,
        note='截断成 An 会命中 2608.14546v1（错误）',
    ),
    TaskSpec(
        id='ref_ctrl_word_marionette',
        task='下载标题里包含 Marionette 的那篇论文',
        steps=(
            Step('download_arxiv_pdf', {'ref': 'Marionette'}),
        ),
        category='ref_form',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
        ),
        requires_offline=True,
        note='单词，无引号需求',
    ),
    TaskSpec(
        id='ref_ctrl_word_handover',
        task='下载标题包含 Handover 的论文',
        steps=(
            Step('download_arxiv_pdf', {'ref': 'Handover'}),
        ),
        category='ref_form',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
        ),
        requires_offline=True,
        note='单词，无引号需求',
    ),
    TaskSpec(
        id='ref_ctrl_id_download',
        task='下载 2608.14528v1 这篇论文',
        steps=(
            Step('download_arxiv_pdf', {'ref': '2608.14528v1'}),
        ),
        category='ref_form',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
        ),
        requires_offline=True,
        note='完整ID含字母，不会被类型推断改变',
    ),
    TaskSpec(
        id='ref_ctrl_id_cache',
        task='查一下 2608.14543v1 的缓存状态',
        steps=(
            Step('get_paper_cache_status', {'ref': '2608.14543v1'}),
        ),
        category='ref_form',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'CV', 'days': 7, 'max_results': 5}),
        ),
        requires_offline=True,
        note='同上',
    ),
    TaskSpec(
        id='ref_ctrl_null_translate',
        task='把刚才那篇论文翻译一下',
        steps=(
            Step('translate_arxiv_pdf', {'ref': None}),
        ),
        category='ref_form',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 2}),
        ),
        requires_offline=True,
        note='prompt 要求传 JSON null',
    ),
    TaskSpec(
        id='ref_ctrl_null_cache',
        task='看看刚操作的那篇论文缓存好了没',
        steps=(
            Step('get_paper_cache_status', {'ref': None}),
        ),
        category='ref_form',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 3}),
        ),
        requires_offline=True,
        note='同上',
    ),
    # ---------- composite ----------
    TaskSpec(
        id='multi_cv3_dl1',
        task='搜索最近7天计算机视觉(cs.CV)的论文(最多3篇)，然后下载第1篇',
        steps=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'CV', 'days': 7, 'max_results': 3}),
            Step('download_arxiv_pdf', {'ref': 1}),
        ),
        category='composite',
        difficulty='hard',
    ),
    TaskSpec(
        id='multi_cl5_dl2_cache',
        task='检索最近7天自然语言处理(cs.CL)论文5篇，下载第2篇，再查它的缓存状态',
        steps=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'CL', 'days': 7, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 2}),
            Step('get_paper_cache_status', {'ref': 2}),
        ),
        category='composite',
        difficulty='hard',
    ),
    TaskSpec(
        id='multi_ai5_dl1_tr',
        task='找最近7天人工智能(cs.AI)的论文5篇，下载第1篇并翻译它',
        steps=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 1}),
            Step('translate_arxiv_pdf', {'ref': 1}),
        ),
        category='composite',
        difficulty='hard',
    ),
    TaskSpec(
        id='multi_ro3_dl_two',
        task='检索最近3天机器人学(cs.RO)论文3篇，把前两篇都下载下来',
        steps=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'RO', 'days': 3, 'max_results': 3}),
            Step('download_arxiv_pdf', {'ref': 1}),
            Step('download_arxiv_pdf', {'ref': 2}),
        ),
        category='composite',
        difficulty='hard',
    ),
    TaskSpec(
        id='multi_cr5_cache1',
        task='搜索最近7天密码学与安全(cs.CR)论文5篇，查看第1篇的缓存状态',
        steps=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'CR', 'days': 7, 'max_results': 5}),
            Step('get_paper_cache_status', {'ref': 1}),
        ),
        category='composite',
        difficulty='hard',
    ),
    TaskSpec(
        id='multi_lg10_dl3',
        task='获取最近14天机器学习(cs.LG)论文10篇，下载第3篇',
        steps=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'LG', 'days': 14, 'max_results': 10}),
            Step('download_arxiv_pdf', {'ref': 3}),
        ),
        category='composite',
        difficulty='hard',
    ),
    TaskSpec(
        id='multi_cv5_dl1_cache_tr',
        task='检索计算机视觉(cs.CV)最近7天5篇论文，下载第1篇，确认缓存后翻译',
        steps=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'CV', 'days': 7, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 1}),
            Step('get_paper_cache_status', {'ref': 1}),
            Step('translate_arxiv_pdf', {'ref': 1}),
        ),
        category='composite',
        difficulty='hard',
    ),
    TaskSpec(
        id='multi_ai3_tr1',
        task='搜最近7天AI论文3篇，直接翻译第1篇',
        steps=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 3}),
            Step('translate_arxiv_pdf', {'ref': 1}),
        ),
        category='composite',
        difficulty='hard',
    ),
    # ---------- state ----------
    TaskSpec(
        id='state_ref_ordinal_cn',
        task='把第三篇论文下载下来',
        steps=(
            Step('download_arxiv_pdf', {'ref': 3}),
        ),
        category='state',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
        ),
    ),
    TaskSpec(
        id='state_ref_last_active',
        task='刚才下载的那篇，帮我翻译一下',
        steps=(
            Step('translate_arxiv_pdf', {'ref': None}),
        ),
        category='state',
        difficulty='hard',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 2}),
        ),
    ),
    TaskSpec(
        id='state_ref_cache_active',
        task='查一下我刚下载的那篇论文的缓存状态',
        steps=(
            Step('get_paper_cache_status', {'ref': None}),
        ),
        category='state',
        difficulty='hard',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 2}),
        ),
    ),
    TaskSpec(
        id='state_cache_two',
        task='第1篇和第2篇，分别查一下缓存状态',
        steps=(
            Step('get_paper_cache_status', {'ref': 1}),
            Step('get_paper_cache_status', {'ref': 2}),
        ),
        category='state',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
        ),
    ),
    TaskSpec(
        id='state_dl_then_cache',
        task='下载第1篇，然后确认它是否已经缓存',
        steps=(
            Step('download_arxiv_pdf', {'ref': 1}),
            Step('get_paper_cache_status', {'ref': 1}),
        ),
        category='state',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
        ),
    ),
    TaskSpec(
        id='state_cache_before_dl',
        task='先看看第2篇有没有下载过，没有的话下载它',
        steps=(
            Step('get_paper_cache_status', {'ref': 2}),
            Step('download_arxiv_pdf', {'ref': 2}),
        ),
        category='state',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
        ),
    ),
    TaskSpec(
        id='state_ref_last_of_10',
        task='把最后那篇（第10篇）下载下来',
        steps=(
            Step('download_arxiv_pdf', {'ref': 10}),
        ),
        category='state',
        difficulty='medium',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'CL', 'days': 7, 'max_results': 10}),
        ),
    ),
    TaskSpec(
        id='state_translate_active',
        task='翻译刚才那篇论文',
        steps=(
            Step('translate_arxiv_pdf', {'ref': None}),
        ),
        category='state',
        difficulty='hard',
        setup=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 1}),
        ),
    ),
]


# ============================================================================
# 不可行请求：正确行为是**不调用任何工具**
# ============================================================================
# 现有任务全都在奖励「做对了什么」，没有一条惩罚「做了不该做的事」。
# 但 Agent 最典型的失效模式恰恰是幻觉调用——会话里只有 5 篇，却硬要下载第 20 篇。
# 这类任务的 steps 为空，`_tool_score` 在 expected 为空时只有一条路：
# 不调工具得 1.0，调了就得 0.0。
#
# 它同时是 reward hacking 的探针：只靠「多调工具」刷分的策略在这里必然掉分。
_INFEASIBLE: List[TaskSpec] = [
    TaskSpec(
        id='infeasible_index_out_of_range',
        task='把第20篇论文下载下来',
        category='infeasible',
        difficulty='hard',
        setup=_SEED_AI5,
        note='会话里只有 5 篇；正确反应是说明做不到，而不是编一个 ref 去调用',
    ),
    TaskSpec(
        id='infeasible_unknown_id',
        task='下载 2999.99999v1 这篇论文',
        category='infeasible',
        difficulty='hard',
        setup=_SEED_AI5,
        requires_offline=True,
        note='该 arXiv ID 不在快照里；解析不到就不应该调用下载',
    ),
    TaskSpec(
        id='infeasible_no_session',
        task='把刚才那篇论文翻译一下',
        category='infeasible',
        difficulty='hard',
        note='没有 setup，会话是空的——「刚才那篇」无从指代。'
             '这是与 ref_ctrl_null_translate 的对照组：同样的话术，'
             '有会话时该调用，没会话时不该调用',
    ),
    TaskSpec(
        id='infeasible_unsupported_action',
        task='把第1篇论文的作者邮箱找出来发一封约稿邮件',
        category='infeasible',
        difficulty='hard',
        setup=_SEED_AI5,
        note='四个工具都做不到这件事；正确反应是说明能力边界',
    ),
]


# ============================================================================
# 长链：4~5 步
# ============================================================================
# 项目定位是长程工具调用，但原 43 条里 4 步的只有 1 条、3 步 3 条，74% 是单步。
# 这一族把链长顶到 Agent 的实际上限。
#
# 注意 max_iterations：Agent 默认 5 轮 = 最多 4 次工具调用 + 一次 FINISH。
# 5 步链必须显式抬高，否则会被判成 FORCE_STOP —— 那是「预算不够」而不是
# 「不会规划」，两者混在一起会让这一族的失败率完全没法解读。
_LONG_CHAIN: List[TaskSpec] = [
    TaskSpec(
        id='chain_ai5_dl2_tr2',
        task='找最近7天人工智能(cs.AI)的论文5篇，把第1篇和第2篇都下载下来，然后翻译第2篇',
        steps=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'AI', 'days': 7, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 1}),
            Step('download_arxiv_pdf', {'ref': 2}),
            Step('translate_arxiv_pdf', {'ref': 2}),
        ),
        category='long_chain',
        difficulty='hard',
    ),
    TaskSpec(
        id='chain_cv5_cache_dl_tr_cache',
        task='检索最近7天计算机视觉(cs.CV)论文5篇，先查第1篇缓存状态，下载它，翻译它，最后再查一次缓存',
        steps=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'CV', 'days': 7, 'max_results': 5}),
            Step('get_paper_cache_status', {'ref': 1}),
            Step('download_arxiv_pdf', {'ref': 1}),
            Step('translate_arxiv_pdf', {'ref': 1}),
            Step('get_paper_cache_status', {'ref': 1}),
        ),
        category='long_chain',
        difficulty='hard',
        max_iterations=7,
        note='5 次工具调用 + FINISH，默认 5 轮预算不够，显式给到 7',
    ),
    TaskSpec(
        id='chain_cl5_dl3_tr3',
        task='搜索最近7天自然语言处理(cs.CL)论文5篇，下载第3篇，确认缓存后翻译它',
        steps=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'CL', 'days': 7, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 3}),
            Step('get_paper_cache_status', {'ref': 3}),
            Step('translate_arxiv_pdf', {'ref': 3}),
        ),
        category='long_chain',
        difficulty='hard',
    ),
    TaskSpec(
        id='chain_ro5_dl_three',
        task='检索最近7天机器人学(cs.RO)论文5篇，把前三篇依次下载下来',
        steps=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'RO', 'days': 7, 'max_results': 5}),
            Step('download_arxiv_pdf', {'ref': 1}),
            Step('download_arxiv_pdf', {'ref': 2}),
            Step('download_arxiv_pdf', {'ref': 3}),
        ),
        category='long_chain',
        difficulty='hard',
        note='同一工具连调三次、只有 ref 递增，考察是否会漏调或重复调同一篇',
    ),
    TaskSpec(
        id='chain_lg10_dl_tr_last',
        task='获取最近14天机器学习(cs.LG)论文10篇，下载最后一篇，然后翻译它并保留双语',
        steps=(
            Step('get_recently_submitted_cs_papers', {'aspect': 'LG', 'days': 14, 'max_results': 10}),
            Step('download_arxiv_pdf', {'ref': 10}),
            Step('translate_arxiv_pdf', {'ref': 10, 'keep_dual': True}),
        ),
        category='long_chain',
        difficulty='hard',
        note='把「最后一篇」的序数推断和可选参数叠在同一条链上',
    ),
]


EXPANDED_TASKS: List[Dict[str, Any]] = build(_SEARCH + _OTHERS + _INFEASIBLE + _LONG_CHAIN)


def get_expanded_tasks() -> List[Dict[str, Any]]:
    return list(EXPANDED_TASKS)


def get_by_category(category: str) -> List[Dict[str, Any]]:
    return [t for t in EXPANDED_TASKS if t["category"] == category]


def offline_only_ids() -> List[str]:
    return [t["id"] for t in EXPANDED_TASKS if t.get("requires_offline")]
