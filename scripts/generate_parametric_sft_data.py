#!/usr/bin/env python3
"""从 v2 train 模板生成可执行、可审计的参数化 SFT 专家数据。

与 linguistic augmentation 的区别：本脚本真的改变工具参数和环境状态，但不改变
父任务的工具链拓扑。每个派生任务都要在离线环境中执行，并通过 Benchmark 的严格
成功判定后才会写出。这样可以增加参数覆盖度，同时不把 OOD 留出的三/四步
composite 链路偷渡进训练集。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark.task_spec import Step, TaskSpec  # noqa: E402
from benchmark.tasks_expanded import EXPANDED_SPECS  # noqa: E402


CN = {
    "AI": "人工智能", "LG": "机器学习", "CL": "自然语言处理",
    "CV": "计算机视觉", "RO": "机器人学", "CR": "密码学与安全",
}


@dataclass(frozen=True)
class DerivedTask:
    spec: TaskSpec
    parent_task_id: str
    parameters: Mapping[str, Any]


def step(tool: str, **args: Any) -> Step:
    return Step(tool, args)


def build_parametric_tasks(*, include_t5: bool = False) -> List[DerivedTask]:
    """声明仅用于训练的派生任务，并保持 v2 默认数据集不变。"""
    out: List[DerivedTask] = []

    def add(
        parent: str,
        suffix: str,
        task: str,
        steps: Sequence[Step],
        *,
        setup: Sequence[Step] = (),
        parameters: Mapping[str, Any],
    ) -> None:
        parent_spec = {item.id: item for item in EXPANDED_SPECS}[parent]
        out.append(DerivedTask(
            spec=TaskSpec(
                id=f"augp1_{parent}_{suffix}",
                task=task,
                steps=tuple(steps),
                category=parent_spec.category,
                difficulty=parent_spec.difficulty,
                setup=tuple(setup),
                template=parent_spec.template,
                requires_offline=True,
                terminal_mode=parent_spec.terminal_mode,
                terminal_reason=parent_spec.terminal_reason,
                max_iterations=parent_spec.max_iterations,
                note=f"parametric_v1 parent={parent}",
            ),
            parent_task_id=parent,
            parameters=dict(parameters),
        ))

    # 1) 单步检索：覆盖新的类别/天数/数量组合，不复制 heldout 的具体组合。
    search_grid = {
        "AI": [(2, 4), (10, 8)],
        "LG": [(5, 4), (21, 7)],
        "CL": [(3, 4), (14, 8)],
        "CV": [(3, 6), (21, 9)],
        "RO": [(5, 4), (14, 7)],
        "CR": [(3, 6), (21, 9)],
    }
    search_parent = {
        "AI": "search_AI_30d_25", "LG": "search_LG_3d_10",
        "CL": "search_CL_7d_5", "CV": "search_CV_14d_3",
        "RO": "search_RO_3d_8", "CR": "search_CR_7d_5",
    }
    for aspect, pairs in search_grid.items():
        for days, count in pairs:
            args = {"aspect": aspect, "days": days, "max_results": count}
            add(
                search_parent[aspect], f"{days}d_{count}",
                f"检索最近{days}天{CN[aspect]}(cs.{aspect})方向论文，最多返回{count}篇",
                [Step("get_recently_submitted_cs_papers", args)],
                parameters=args,
            )

    # 2) 关键词检索：快照只真实覆盖 train 中的 LLM query，因此仅改变数量。
    for count in (2, 8, 12):
        args = {
            "query": "all:large language model", "max_results": count, "days": 30,
        }
        add(
            "search_kw_llm", f"n{count}",
            f"按关键词 all:large language model 检索 arXiv，最多返回{count}篇",
            [Step("search_arxiv_papers", args)], parameters=args,
        )

    # 3) 翻译可选参数。setup 与目标 ref 同步变化，保证轨迹可执行。
    for threads in (2, 4, 12):
        setup = [
            step("get_recently_submitted_cs_papers", aspect="AI", days=7, max_results=5),
            step("download_arxiv_pdf", ref=1),
        ]
        add(
            "opt_threads", f"t{threads}", f"翻译第1篇论文，使用{threads}个线程",
            [step("translate_arxiv_pdf", ref=1, threads=threads)], setup=setup,
            parameters={"ref": 1, "threads": threads},
        )
    for ref in (2, 4):
        setup = [
            step("get_recently_submitted_cs_papers", aspect="AI", days=7, max_results=5),
            step("download_arxiv_pdf", ref=ref),
        ]
        add(
            "opt_keep_dual", f"ref{ref}", f"翻译第{ref}篇论文并保留双语版本",
            [step("translate_arxiv_pdf", ref=ref, keep_dual=True)], setup=setup,
            parameters={"ref": ref, "keep_dual": True},
        )
    for ref in (2, 3):
        setup = [
            step("get_recently_submitted_cs_papers", aspect="AI", days=7, max_results=5),
            step("download_arxiv_pdf", ref=ref),
        ]
        add(
            "opt_force_tr", f"ref{ref}", f"强制重新翻译第{ref}篇论文",
            [step("translate_arxiv_pdf", ref=ref, force=True)], setup=setup,
            parameters={"ref": ref, "force": True},
        )

    # 4) 会话状态和停止边界。
    seed5 = [step("get_recently_submitted_cs_papers", aspect="AI", days=7, max_results=5)]
    for ref in (1, 2, 4, 5):
        add(
            "state_ref_ordinal_cn", f"ref{ref}", f"把列表中的第{ref}篇论文下载下来",
            [step("download_arxiv_pdf", ref=ref)], setup=seed5,
            parameters={"ref": ref},
        )
    for count in (6, 8, 12):
        setup = [
            step("get_recently_submitted_cs_papers", aspect="CL", days=7, max_results=count)
        ]
        add(
            "state_ref_last_of_10", f"last{count}",
            f"把刚才列出的{count}篇自然语言处理论文中的最后一篇（第{count}篇）下载下来",
            [step("download_arxiv_pdf", ref=count)], setup=setup,
            parameters={"max_results": count, "ref": count},
        )
    for ref in (1, 3, 4):
        add(
            "state_cache_before_dl", f"ref{ref}",
            f"先查看第{ref}篇论文是否缓存，没有的话再下载它",
            [step("get_paper_cache_status", ref=ref), step("download_arxiv_pdf", ref=ref)],
            setup=seed5, parameters={"ref": ref},
        )

    # 5) 显式负向约束：工具序列不变，因此训练的是“完成后停止”。
    for ref in (1, 3, 4):
        add(
            "constraint_cache_only", f"ref{ref}",
            f"只检查第{ref}篇论文的缓存状态，不要下载或翻译",
            [step("get_paper_cache_status", ref=ref)], setup=seed5,
            parameters={"ref": ref},
        )
    for ref in (2, 3, 5):
        add(
            "constraint_download_only", f"ref{ref}",
            f"只下载第{ref}篇论文，不要翻译，也不要检查缓存",
            [step("download_arxiv_pdf", ref=ref)], setup=seed5,
            parameters={"ref": ref},
        )
    for first, second in ((1, 2), (2, 4), (4, 5)):
        add(
            "constraint_two_downloads_only", f"refs{first}_{second}",
            f"下载第{first}篇和第{second}篇论文，完成后停止，不要翻译或检查缓存",
            [step("download_arxiv_pdf", ref=first), step("download_arxiv_pdf", ref=second)],
            setup=seed5, parameters={"refs": [first, second]},
        )
    for aspect, days, count in (("AI", 4, 4), ("LG", 9, 6), ("CR", 18, 8)):
        args = {"aspect": aspect, "days": days, "max_results": count}
        add(
            "constraint_search_only", f"{aspect.lower()}_{days}_{count}",
            f"只检索最近{days}天{CN[aspect]}(cs.{aspect})论文，最多{count}篇；不要下载或翻译",
            [Step("get_recently_submitted_cs_papers", args)], parameters=args,
        )

    # 6) 不可行请求：仍然没有工具步骤，只允许 FINISH。
    for value, label in ((-1, "neg1"), (6, "six"), (50, "fifty")):
        wording = f"下载第{value}篇论文" if value != -1 else "下载第-1篇论文"
        add(
            "infeasible_zero_index", label, wording, [], setup=seed5,
            parameters={"invalid_ref": value},
        )
    for paper_id in ("2998.12345v1", "3999.00001v2"):
        add(
            "infeasible_unknown_id", paper_id.replace(".", "_"),
            f"请下载 arXiv 论文 {paper_id}", [], setup=seed5,
            parameters={"unknown_id": paper_id},
        )
    for suffix, wording in (
        ("translate", "继续翻译上一条提到的论文"),
        ("cache", "查看刚刚那篇论文的缓存状态"),
    ):
        add(
            "infeasible_no_session", suffix, wording, [],
            parameters={"session_state": "empty"},
        )

    # 7) 两步 composite。v2 的 OOD 是 composite 的三/四步链，本轮不生成它们。
    for aspect, count, ref in (("AI", 4, 2), ("RO", 6, 3), ("CV", 7, 4)):
        args = {"aspect": aspect, "days": 7, "max_results": count}
        add(
            "multi_cr5_cache1", f"{aspect.lower()}_{count}_ref{ref}",
            f"检索{CN[aspect]}(cs.{aspect})论文{count}篇，然后查看第{ref}篇的缓存状态",
            [Step("get_recently_submitted_cs_papers", args), step("get_paper_cache_status", ref=ref)],
            parameters={**args, "ref": ref},
        )
    for aspect, count, ref in (("CL", 6, 4), ("AI", 7, 5), ("CR", 8, 6)):
        args = {"aspect": aspect, "days": 10, "max_results": count}
        add(
            "multi_lg10_dl3", f"{aspect.lower()}_{count}_ref{ref}",
            f"检索最近10天{CN[aspect]}(cs.{aspect})论文{count}篇，然后下载第{ref}篇",
            [Step("get_recently_submitted_cs_papers", args), step("download_arxiv_pdf", ref=ref)],
            parameters={**args, "ref": ref},
        )

    # 8) 长链只改变参数，严格保留每个 train 父任务的工具序列。
    for aspect, count, first, second in (("CL", 6, 2, 4), ("CR", 7, 1, 5)):
        args = {"aspect": aspect, "days": 7, "max_results": count}
        add(
            "chain_ai5_dl2_tr2", f"{aspect.lower()}_{count}_{first}_{second}",
            f"检索{CN[aspect]}论文{count}篇，下载第{first}篇和第{second}篇，再翻译第{second}篇",
            [Step("get_recently_submitted_cs_papers", args),
             step("download_arxiv_pdf", ref=first), step("download_arxiv_pdf", ref=second),
             step("translate_arxiv_pdf", ref=second)],
            parameters={**args, "refs": [first, second], "translate_ref": second},
        )
    for aspect, count, ref in (("AI", 4, 2), ("LG", 6, 4)):
        args = {"aspect": aspect, "days": 7, "max_results": count}
        add(
            "chain_cv5_cache_dl_tr_cache", f"{aspect.lower()}_{count}_ref{ref}",
            f"检索{CN[aspect]}论文{count}篇，先查第{ref}篇缓存，再下载、翻译并复查缓存",
            [Step("get_recently_submitted_cs_papers", args),
             step("get_paper_cache_status", ref=ref), step("download_arxiv_pdf", ref=ref),
             step("translate_arxiv_pdf", ref=ref), step("get_paper_cache_status", ref=ref)],
            parameters={**args, "ref": ref},
        )
    for aspect, count, refs in (("CV", 6, (1, 3, 5)), ("CR", 7, (2, 4, 6))):
        args = {"aspect": aspect, "days": 7, "max_results": count}
        add(
            "chain_ro5_dl_three", f"{aspect.lower()}_{count}_{''.join(map(str, refs))}",
            f"检索{CN[aspect]}论文{count}篇，依次下载第{refs[0]}、第{refs[1]}和第{refs[2]}篇",
            [Step("get_recently_submitted_cs_papers", args)]
            + [step("download_arxiv_pdf", ref=ref) for ref in refs],
            parameters={**args, "refs": list(refs)},
        )
    for aspect, count in (("AI", 6), ("RO", 8)):
        args = {"aspect": aspect, "days": 14, "max_results": count}
        add(
            "chain_lg10_dl_tr_last", f"{aspect.lower()}_last{count}",
            f"获取最近14天{CN[aspect]}论文{count}篇，下载最后一篇并翻译，保留双语版本",
            [Step("get_recently_submitted_cs_papers", args),
             step("download_arxiv_pdf", ref=count),
             step("translate_arxiv_pdf", ref=count, keep_dual=True)],
            parameters={**args, "ref": count, "keep_dual": True},
        )

    # 9) 解读族（README T2/T3）。工具链拓扑与父任务一致：检索 → 下载 → 读/总结。
    #
    # 快照里 get_paper_content 的**默认**读取（section=None，标题+摘要）对每篇论文
    # 都有记录；命名小节是尽力而为的，同一篇论文未必有 method/result/conclusion。
    # 所以只参数化确定可执行的那部分：read 族固定在 section=None 上变化 aspect/ref，
    # 小节抽取由父任务自己覆盖；summarize 族的 3 风格 × 3 预算对每篇论文都有记录，
    # 可以放心一起变。
    for ref in (1, 3, 4):
        add(
            "read_ai5_default", f"ref{ref}",
            f"检索最近7天人工智能(cs.AI)论文5篇，下载第{ref}篇，然后读出它的标题与摘要",
            [
                step("get_recently_submitted_cs_papers", aspect="AI", days=7, max_results=5),
                step("download_arxiv_pdf", ref=ref),
                step("get_paper_content", ref=ref, section=None),
            ],
            parameters={"aspect": "AI", "ref": ref, "section": None},
        )
    for aspect, ref in (("CV", 2), ("CL", 4)):
        add(
            "read_ai5_default", f"{aspect.lower()}_ref{ref}",
            f"检索最近7天{CN[aspect]}论文5篇，下载第{ref}篇，然后读出它的标题与摘要",
            [
                step("get_recently_submitted_cs_papers", aspect=aspect, days=7, max_results=5),
                step("download_arxiv_pdf", ref=ref),
                step("get_paper_content", ref=ref, section=None),
            ],
            parameters={"aspect": aspect, "ref": ref, "section": None},
        )

    for style, budget in (("structured", 60), ("bullet", 120), ("tldr", 250)):
        args = {"aspect": "AI", "ref": 2, "style": style, "max_words": budget}
        add(
            "summary_ai5_tldr60", f"{style}{budget}",
            f"检索最近7天人工智能(cs.AI)论文5篇，下载第2篇，"
            f"用 {budget} 词以内的 {style} 风格总结它",
            [
                step("get_recently_submitted_cs_papers", aspect="AI", days=7, max_results=5),
                step("download_arxiv_pdf", ref=2),
                step("summarize_paper", ref=2, style=style, max_words=budget),
            ],
            parameters=args,
        )
    for aspect, ref, style, budget in (
        ("CV", 3, "tldr", 60), ("CL", 2, "structured", 120), ("RO", 4, "bullet", 250),
    ):
        args = {
            "aspect": aspect, "ref": ref, "style": style, "max_words": budget,
        }
        add(
            "summary_ai5_tldr60", f"{aspect.lower()}_ref{ref}_{style}{budget}",
            f"检索最近7天{CN[aspect]}论文5篇，下载第{ref}篇，"
            f"用 {budget} 词以内的 {style} 风格总结它",
            [
                step("get_recently_submitted_cs_papers", aspect=aspect, days=7, max_results=5),
                step("download_arxiv_pdf", ref=ref),
                step("summarize_paper", ref=ref, style=style, max_words=budget),
            ],
            parameters=args,
        )

    # 10) 图表抽取（README T4）。ref 同样只走快照里确实有内嵌位图的位置：
    # 抽图在「论文没有位图」时返回 count=0 而不是报错，所以选错 ref 不会响亮
    # 失败，只会安静地生成一条空结果轨迹。CV 池前 12 篇都有，RO/AI 取已核对的
    # 序号。
    for aspect, ref in (("CV", 2), ("CV", 4), ("RO", 3), ("AI", 1)):
        add(
            "figure_cv5_ref1", f"{aspect.lower()}_ref{ref}",
            f"检索最近7天{CN[aspect]}论文5篇，下载第{ref}篇，把它的图表抽出来",
            [
                step("get_recently_submitted_cs_papers", aspect=aspect, days=7, max_results=5),
                step("download_arxiv_pdf", ref=ref),
                step("extract_paper_figures", ref=ref),
            ],
            parameters={"aspect": aspect, "ref": ref},
        )

    if include_t5:
        # 已发布的 v2 种子早于 T5，因此图表分析须显式启用。
        # 保留父任务的论文和图号，只改变问题，避免引入未经验证的图片资源。
        question_phrases = {
            "describe": "描述第{figure_no}张图展示的内容",
            "axes": "说明第{figure_no}张图的坐标轴信息",
            "trend": "概括第{figure_no}张图呈现的趋势",
        }
        for parent in (
            "analyze_cv5_ref1_desc",
            "analyze_cv5_ref3_trend",
            "analyze_ro5_ref2_axes",
        ):
            parent_spec = {item.id: item for item in EXPANDED_SPECS}[parent]
            search_args = parent_spec.steps[0].args
            analysis_args = parent_spec.steps[-1].args
            aspect = search_args["aspect"]
            ref = analysis_args["ref"]
            figure_no = analysis_args["figure_no"]
            for question, phrase in question_phrases.items():
                if question == analysis_args["question"]:
                    continue
                derived_args = {**analysis_args, "question": question}
                add(
                    parent, f"q_{question}",
                    f"检索最近7天{CN[aspect]}(cs.{aspect})论文5篇，"
                    f"下载第{ref}篇并抽取图表，随后"
                    f"{phrase.format(figure_no=figure_no)}",
                    [*parent_spec.steps[:-1], Step("analyze_figure", derived_args)],
                    parameters={
                        "aspect": aspect, "ref": ref,
                        "figure_no": figure_no, "question": question,
                    },
                )

    return out


def tool_names(steps: Iterable[Step]) -> Tuple[str, ...]:
    return tuple(item.tool for item in steps)


def validate_derived_tasks(
    derived: Sequence[DerivedTask], split_payload: Dict[str, Any]
) -> None:
    """验证血缘、拓扑和测试集边界；任何不确定项都 fail-fast。"""
    if not derived:
        raise ValueError("参数化任务为空")
    by_id = {spec.id: spec for spec in EXPANDED_SPECS}
    train = set(split_payload["split"]["train"])
    heldout = set(
        split_payload["split"]["dev"]
        + split_payload["split"]["iid_test"]
        + split_payload["split"]["ood_test"]
    )
    benchmark_texts = {spec.task.strip() for spec in EXPANDED_SPECS}
    benchmark_ids = set(by_id)
    derived_ids = [item.spec.id for item in derived]
    derived_texts = [item.spec.task.strip() for item in derived]

    if len(derived_ids) != len(set(derived_ids)):
        raise ValueError("参数化任务 id 重复")
    if len(derived_texts) != len(set(derived_texts)):
        raise ValueError("参数化任务文本重复")
    if set(derived_ids) & benchmark_ids:
        raise ValueError("参数化任务 id 与 Benchmark 冲突")
    if set(derived_texts) & benchmark_texts:
        raise ValueError("参数化任务复制了 Benchmark 原文")

    for item in derived:
        if item.parent_task_id not in train:
            raise ValueError(f"父任务不属于 train: {item.parent_task_id}")
        if item.parent_task_id in heldout:
            raise ValueError(f"父任务属于 heldout: {item.parent_task_id}")
        parent = by_id[item.parent_task_id]
        if tool_names(item.spec.steps) != tool_names(parent.steps):
            raise ValueError(
                f"派生任务改变了工具链: {item.spec.id}: "
                f"{tool_names(parent.steps)} -> {tool_names(item.spec.steps)}"
            )
        if tool_names(item.spec.setup) != tool_names(parent.setup):
            raise ValueError(
                f"派生任务改变了 setup 拓扑: {item.spec.id}: "
                f"{tool_names(parent.setup)} -> {tool_names(item.spec.setup)}"
            )
        if not item.parameters:
            raise ValueError(f"派生任务缺少参数记录: {item.spec.id}")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    # 运行期依赖延迟加载：静态数据边界测试不应强制安装 torch/loguru 等完整环境。
    from agents.prompt_templates import format_tool_description
    from generate_sft_data import generate_deterministic_trajectories
    from rl.env import MockArxivEnv
    from tools.tool_registry import registry

    parser = argparse.ArgumentParser(description="生成 train-only 参数化 SFT 专家数据")
    parser.add_argument("--split-file", default="data/splits/v2_62.json")
    parser.add_argument("--snapshot", default="data/mock_arxiv_snapshot.json")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--include-t5", action="store_true",
        help="加入 T5 图表分析派生任务；需要 v3_81 训练切分",
    )
    args = parser.parse_args()

    split_path = Path(args.split_file)
    snapshot_path = Path(args.snapshot)
    default_output = (
        "data/sft/sft_v3_t5_parametric_seed.jsonl" if args.include_t5
        else "data/sft/sft_v2_parametric_seed.jsonl"
    )
    output_path = Path(args.output or default_output)
    if not split_path.exists():
        raise SystemExit(f"切分文件不存在: {split_path}")
    if not snapshot_path.exists():
        raise SystemExit(f"离线快照不存在: {snapshot_path}")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    if args.include_t5 and (
        split_payload.get("version") != 3
        or not {
            "analyze_cv5_ref1_desc", "analyze_cv5_ref3_trend",
            "analyze_ro5_ref2_axes",
        } <= set(split_payload.get("split", {}).get("train", []))
    ):
        raise SystemExit("--include-t5 requires data/splits/v3_81.json:train")

    derived = build_parametric_tasks(include_t5=args.include_t5)
    validate_derived_tasks(derived, split_payload)
    env = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")
    tools_desc = format_tool_description(registry.list_tools())
    specs = [item.spec for item in derived]
    source_split = (
        f"{split_path.name}:train:parametric_v1_t5" if args.include_t5
        else f"{split_path.name}:train:parametric_v1"
    )
    rows = generate_deterministic_trajectories(
        specs, env, tools_desc, source_split=source_split
    )

    lineage = {item.spec.id: item for item in derived}
    fingerprints = set()
    for row in rows:
        item = lineage[row["source_task_id"]]
        row["derived_task_id"] = item.spec.id
        row["parent_task_id"] = item.parent_task_id
        row["generation_parameters"] = dict(item.parameters)
        row["dataset_stage"] = "parametric_v1_expert_seed"
        row["sample_sha256"] = canonical_hash(row["messages"])
        if row["sample_sha256"] in fingerprints:
            raise RuntimeError(f"生成了重复 messages: {row['source_task_id']}")
        fingerprints.add(row["sample_sha256"])

    write_jsonl(output_path, rows)
    parent_counts = Counter(item.parent_task_id for item in derived)
    category_counts = Counter(item.spec.category for item in derived)
    chain_counts = Counter(len(item.spec.steps) for item in derived)
    task_manifest = [
        {
            "derived_task_id": item.spec.id,
            "parent_task_id": item.parent_task_id,
            "task": item.spec.task,
            "steps": [{"name": s.tool, "args": s.args} for s in item.spec.steps],
            "terminal_mode": item.spec.terminal_mode,
            "terminal_reason": item.spec.terminal_reason,
            "setup": [{"name": s.tool, "args": s.args} for s in item.spec.setup],
            "parameters": dict(item.parameters),
        }
        for item in derived
    ]
    manifest = {
        "version": 1,
        "kind": "train_only_parametric_expert_seed",
        "split_file": str(split_path),
        "split_sha256": sha256_file(split_path),
        "snapshot": str(snapshot_path),
        "snapshot_sha256": sha256_file(snapshot_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "derived_tasks": len(derived),
        "sample_rows": len(rows),
        "unique_sample_fingerprints": len(fingerprints),
        "parent_train_tasks": len(parent_counts),
        "parent_counts": dict(sorted(parent_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "tool_chain_length_counts": {
            str(k): v for k, v in sorted(chain_counts.items())
        },
        "heldout_parent_overlap": 0,
        "exact_benchmark_text_overlap": 0,
        "topology_policy": "Each derived task preserves its train parent's step/setup tool sequence.",
        "task_manifest_sha256": canonical_hash(task_manifest),
        "tasks": task_manifest,
    }
    if args.include_t5:
        manifest["include_t5"] = True
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("PARAMETRIC_SFT_OK")
    print(f"derived tasks:    {len(derived)}")
    print(f"sample rows:      {len(rows)}")
    print(f"parent tasks:     {len(parent_counts)}")
    print(f"unique samples:   {len(fingerprints)}")
    print("heldout overlap:  0")
    print(f"output:           {output_path}")
    print(f"manifest:         {manifest_path}")


if __name__ == "__main__":
    main()
