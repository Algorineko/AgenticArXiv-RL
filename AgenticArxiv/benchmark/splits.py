"""训练/留出集切分。

后训练要证明「学到了能力」而非「背下了任务」，就必须在模型没见过的任务上评测。
但随机按任务切会泄漏：search_AI_1d_3 与 search_AI_30d_25 是同一模板换参数，
一个进训练一个进测试，等于没测。

所以切分在**模板**层面进行，并区分两种泛化：

    iid   同模板、未见过的参数实例      ——「换个参数还会不会」
    ood   完全未见过的模板/链长          ——「换个形态还会不会」

两者都留出才说得清提升来自哪里：只有 iid 提升可能是记住了模板，
只有 ood 提升更可能是基础能力变强。
"""

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 成功率分档边界。GRPO 的梯度来自组内奖励方差，两端不产生梯度，
# 所以训练集应以中间带为主，两端更适合留作评测的上下限。
FLOOR_MAX = 0.2
CEILING_MIN = 0.8


def template_key(task: Dict[str, Any]) -> Tuple[str, int]:
    """模板键：同模板的不同参数实例必须落在同一侧。

    默认由 (category, 工具链长度) 构成。任务可用 `template` 字段覆盖第一项，
    以便在同一 category 下区分能力不同的子族（例如指代形态的对照组与压力组）。
    """
    label = task.get("template") or task.get("category") or "?"
    return (str(label), len(task.get("expected_tools") or []))


def difficulty_band(rate: float) -> str:
    if rate < FLOOR_MAX:
        return "floor"
    if rate > CEILING_MIN:
        return "ceiling"
    return "middle"


def make_split(
    tasks: Sequence[Dict[str, Any]],
    *,
    ood_keys: Sequence[Tuple[str, int]] = (),
    iid_ratio: float = 0.25,
    seed: int = 0,
    rates: Optional[Dict[str, float]] = None,
) -> Dict[str, List[str]]:
    """切成 train / iid_test / ood_test 三份，返回任务 id。

    ood_keys 指定整体留出的模板键；其余模板按 iid_ratio 留出部分实例。
    给定 rates（任务 id -> 成功率）时按难度分档分层抽样，
    使 iid_test 的难度构成与训练集接近，否则「测试集更简单」会伪装成提升。
    """
    if not 0.0 <= iid_ratio < 1.0:
        raise ValueError("iid_ratio 需在 [0, 1) 内")

    ood_set = {tuple(k) for k in ood_keys}
    by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_key[template_key(task)].append(task)

    unknown = ood_set - set(by_key)
    if unknown:
        raise ValueError(f"ood_keys 中存在任务集里没有的模板键: {sorted(unknown)}")

    rng = random.Random(seed)
    split: Dict[str, List[str]] = {"train": [], "iid_test": [], "ood_test": []}

    for key in sorted(by_key):
        group = sorted(by_key[key], key=lambda t: t["id"])
        if key in ood_set:
            split["ood_test"] += [t["id"] for t in group]
            continue

        # 按难度分档分层，再在档内抽样，避免留出集恰好全是简单任务
        strata: Dict[str, List[str]] = defaultdict(list)
        for task in group:
            rate = (rates or {}).get(task["id"])
            strata["_" if rate is None else difficulty_band(rate)].append(task["id"])

        for band in sorted(strata):
            ids = sorted(strata[band])
            rng.shuffle(ids)
            # 比例非零时至少留 1 条，但绝不掏空一个模板 —— 否则它就成了 ood。
            # iid_ratio 为 0 时一条都不留：那是「只切 ood」的合法用法。
            if len(ids) > 1 and iid_ratio > 0:
                n_hold = min(len(ids) - 1, max(1, round(len(ids) * iid_ratio)))
            else:
                n_hold = 0
            split["iid_test"] += ids[:n_hold]
            split["train"] += ids[n_hold:]

    return {k: sorted(v) for k, v in split.items()}


def summarize(
    split: Dict[str, List[str]],
    tasks: Sequence[Dict[str, Any]],
    rates: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    """每一份的规模与难度构成，用于核对切分是否失衡。"""
    by_id = {t["id"]: t for t in tasks}
    out: Dict[str, Dict[str, Any]] = {}
    for name, ids in split.items():
        bands: Dict[str, int] = defaultdict(int)
        for tid in ids:
            rate = (rates or {}).get(tid)
            bands[difficulty_band(rate) if rate is not None else "unknown"] += 1
        out[name] = {
            "count": len(ids),
            "templates": len({template_key(by_id[tid]) for tid in ids if tid in by_id}),
            "bands": dict(bands),
        }
    return out


def rl_train_ids(
    split: Dict[str, List[str]], rates: Dict[str, float]
) -> List[str]:
    """从训练集中挑出中间带任务作为 RL 训练集。

    成功率贴近 0 或 1 的任务，同一 prompt 采样出的轨迹奖励一致，
    组内方差为零、优势为零，不产生任何梯度 —— 放进 RL 训练集是空转。
    """
    return sorted(
        tid for tid in split["train"]
        if tid in rates and difficulty_band(rates[tid]) == "middle"
    )
