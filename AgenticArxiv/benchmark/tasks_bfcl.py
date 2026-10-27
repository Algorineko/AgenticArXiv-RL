"""BFCL-V3 任务池 (轨道 B, 第二环境外部效度): multi_turn_base 200 任务。

数据: data/bfcl/v3_multi_turn_base.parquet (fireworks-ai/bfcl_v3_multi_turn_base,
经 hf-mirror 获取)。每任务: 单用户消息 + 多步 ground-truth 调用链 + 工具 schema。
无用户模拟器依赖 (base 类别: 一步指令, 多调用 gold) —— 回放环境友好。

打分 (PREREG_BFCL 五分量类比):
  - format: 可解析调用语法比例
  - call-name / call-arg: 逐调用名/参匹配部分分 (与 gold 集合匹配, 顺序不敏感)
  - process: 链前缀正确比例
  - outcome: 全链精确匹配 (集合 + 前缀一致)
确定性: 纯字符串/结构匹配, 无随机。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_PARQUET = Path("/home/tione/notebook/home/arianliu/data/bfcl/v3_multi_turn_base.parquet")


def _parse_call(text: str) -> Optional[Dict[str, Any]]:
    """'cd(folder="document")' -> {name, args}; 失败返回 None。"""
    m = re.match(r"^\s*([A-Za-z_][\w.]*)\s*\((.*)\)\s*$", text, re.S)
    if not m:
        return None
    name, argstr = m.group(1), m.group(2).strip()
    args: Dict[str, Any] = {}
    if argstr:
        try:
            # BFCL gold 参数是合法 python 字面量键值对; 只放行 dict
            ns: Dict[str, Any] = {}
            exec(f"_r = dict({argstr})", {"__builtins__": {"dict": dict}}, ns)  # noqa: S102 — 只读数据文件
            args = ns["_r"]
        except Exception:
            return None
    return {"name": name, "args": args}


def _arg_match(pred: Dict[str, Any], gold: Dict[str, Any]) -> float:
    """参数软匹配: 键值对完全一致比例 (gold 键集为分母)。"""
    ga = gold.get("args", {})
    if not ga:
        return 1.0 if not pred.get("args") else 0.0
    pa = pred.get("args", {})
    hit = sum(1 for k, v in ga.items() if pa.get(k) == v)
    return hit / len(ga)


def score_chain(pred_calls: List[str], gold_calls: List[str]) -> Dict[str, float]:
    """预测调用列表 vs gold 调用列表 -> 分量分。"""
    preds = [c for c in (_parse_call(x) for x in pred_calls) if c]
    golds = [c for c in (_parse_call(x) for x in gold_calls) if c]
    n = max(len(preds), len(golds), 1)
    fmt = len(preds) / max(len(pred_calls), 1) if pred_calls else 0.0
    # 名匹配: 每个 gold 找一个未占用的同名 pred
    used = set()
    name_hits = 0
    arg_score = 0.0
    for g in golds:
        for i, p in enumerate(preds):
            if i not in used and p["name"] == g["name"]:
                used.add(i)
                name_hits += 1
                arg_score += _arg_match(p, g)
                break
    # 前缀 (process): 从头逐位比较名匹配
    prefix = 0
    for p, g in zip(preds, golds):
        if p["name"] == g["name"]:
            prefix += 1
        else:
            break
    exact = int(len(preds) == len(golds) and name_hits == len(golds)
                and all(_arg_match(p, g) == 1.0 for p, g in zip(preds, golds)))
    return {
        "format": fmt,
        "call_name": name_hits / len(golds) if golds else 0.0,
        "call_arg": arg_score / len(golds) if golds else 0.0,
        "process": prefix / max(len(golds), 1),
        "outcome": float(exact),
    }




_EXAMPLE = json.dumps({
    "name": "submit_calls",
    "args": {"calls": json.dumps(["cd(folder='docs')", "send_mail(to='a@b.com', subject='hi')"])},
})
_BFCL_PROTOCOL = (
    "[提交协议] 本环境只有一个可调用工具 submit_calls：把你的完整函数调用链"
    "一次性提交，不要逐个函数分多轮调用。每个调用写成文本形式 name(k=v, ...)，"
    "参数用字面量（字符串参数用单引号），全部调用按顺序放进一个 JSON 数组，"
    "再整体作为 calls 字符串参数提交：\n"
    "Action: " + _EXAMPLE + "\n"
    "即 calls 的值是一个 JSON 数组字符串：数组元素用转义的双引号包住，调用内的"
    "字符串参数用单引号。提交后用 Action: FINISH 结束。"
)


def _bfcl_signature(tool):
    """一个 BFCL 函数 -> 'name(req[, opt: type])' 紧凑签名 (schema 超集防御)。"""
    params = tool.get("parameters")
    params = params if isinstance(params, dict) else {}
    props = params.get("properties")
    props = props if isinstance(props, dict) else {}
    required = params.get("required")
    required = [str(n) for n in list(required)] if required is not None else []
    opt = []
    for k, spec in props.items():
        if isinstance(spec, dict) and k not in required:
            t = spec.get("type", "?")
            opt.append(f"{k}: {t}" if isinstance(t, str) else f"{k}")
    sig = ", ".join(required + [f"[{o}]" for o in opt])
    return sig


def build_bfcl_task_text(task):
    """BFCL 任务 -> ReAct 任务文本: 用户问题 + 逐任务函数 schema + 提交协议。"""
    functions = "\n".join(
        f"- {t.get('name', '')}({_bfcl_signature(t)}): {t.get('description', '')}"
        for t in task["tools"]
    )
    return f"{task['prompt']}\n\n本任务可用的函数及其参数（必选参数在前，方括号内为可选参数）：\n{functions}\n\n{_BFCL_PROTOCOL}"


def get_bfcl_tasks(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载 parquet -> 任务池 (id/prompt/tools/gold_calls/gold_raw)。"""
    import pandas as pd  # 延迟导入: 仅 BFCL 轨需要

    df = pd.read_parquet(_PARQUET)
    tasks: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        gold_raw = json.loads(row["ground_truth"]) if isinstance(row["ground_truth"], str) else list(row["ground_truth"])
        # base 类别: 单轮多调用; 展平为一条 gold 链
        flat: List[str] = []
        for turn in gold_raw:
            if isinstance(turn, list):
                flat.extend(str(x) for x in turn)
            else:
                flat.append(str(turn))
        tools = [
            {"name": t.get("name", ""), "description": t.get("description", ""),
             "parameters": t.get("parameters", {})}
            for t in row["tools"]
        ]
        user_msgs = [m["content"] for m in row["messages"] if m.get("role") == "user"]
        base = dict(
            id=f"bfcl_base_{i:03d}",
            prompt=user_msgs[0] if user_msgs else "",
            tools=tools,
            gold_calls=flat,
            gold_raw=gold_raw,
        )
        base["task"] = build_bfcl_task_text(base)  # runner/agent 直接消费的统一字段
        tasks.append(base)
        if limit and len(tasks) >= limit:
            break
    return tasks


if __name__ == "__main__":
    ts = get_bfcl_tasks(limit=5)
    t0 = ts[0]
    print(f"任务数样例: {len(ts)} | t0: {t0['id']} gold={t0['gold_calls'][:3]} tools={len(t0['tools'])}")
    s = score_chain(t0["gold_calls"], t0["gold_calls"])
    print("自评(应全1):", s)
    s2 = score_chain([t0["gold_calls"][0]], t0["gold_calls"])
    print("半对(应部分):", {k: round(v, 2) for k, v in s2.items()})
