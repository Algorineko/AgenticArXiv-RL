"""Next-tool 判别头的 torch 计算图层。

候选打分的计算图（``--scoring cache``，默认）：

1. prompt 单次 forward（走 ``model.model``，跳过 lm_head，避免物化
   151k 词表 × prompt 长度的 logits）拿 KV cache；
2. cache 沿 batch 维展开成 11 份（``expand_cache``）；
3. 11 个候选后缀（``"Next tool: {name}"``，短）一次 batch forward；
4. 只在各行的真实预测位置 gather hidden，过 lm_head——全 batch 合计
   ~60 个位置，logits 内存可忽略；
5. 每候选的 token logprob 聚合成 z（默认 mean，长度归一化）。

``--scoring batch`` 回退路径逐候选跑全序列 forward+bwd，正确性相同、
慢 5-8 倍，仅在 cache 展开被当前栈证伪时使用。

import 本模块不触发 GPU/模型加载；所有函数都是显式传入 model/tensor 的
纯计算单元，可被 CPU 单测覆盖。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def cache_layers(cache) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """按优先级探测三代 Cache 内部结构，取每层 (k, v)。

    transformers 4.36→5.x 之间「从 Cache 对象取每层 (k,v)」的途径换了三代：
    ``to_legacy_cache()`` → ``key_cache/value_cache`` 列表 → ``layers[i].keys``。
    ``DynamicCache.update`` 的签名则全程稳定，所以展开层用它落地。
    取不到任何已知结构时响亮 raise（调用方指引切 ``--scoring batch``），
    绝不静默降级。
    """
    if hasattr(cache, "to_legacy_cache"):
        return list(cache.to_legacy_cache())
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    raise RuntimeError(
        "无法从 DynamicCache 提取每层 (k, v)：当前 transformers 版本的 Cache "
        "内部结构不在已知三代之内。请用 --scoring batch（正确性不变，更慢）。"
    )


def expand_cache(cache, batch: int):
    """把 batch=1 的 KV cache 沿 batch 维展开。

    ``Tensor.expand`` 产生 stride-0 视图，``DynamicCache.update`` 内部的
    ``torch.cat`` 会把它具体化，不保留 alias。梯度可以穿过 ``cat`` 回流到
    原 cache 的 K/V（训练路径依赖这一点）。
    """
    from transformers.cache_utils import DynamicCache

    new = DynamicCache()
    for layer, (k, v) in enumerate(cache_layers(cache)):
        new.update(k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1), layer)
    return new


def tokenize_suffixes(tokenizer, suffix_texts: Sequence[str]) -> List[List[int]]:
    """预 tokenize 候选后缀并 round-trip 断言。

    断言防 BPE 边界意外（例如后缀与 chat 模板结尾合并成别的 token）：
    ``decode(ids) != text`` 说明 tokenizer 在边界处做了合并，必须换写法
    而不是带着错误的监督信号训下去。
    """
    out: List[List[int]] = []
    for text in suffix_texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(ids)
        if decoded != text:
            raise ValueError(
                f"候选后缀 tokenize round-trip 失败：{text!r} -> {decoded!r}；"
                "请调整后缀写法（BPE 边界合并会破坏打分口径）。"
            )
        out.append(list(ids))
    return out


def build_suffix_batch(
    suffix_ids: Sequence[Sequence[int]], device: torch.device
) -> Tuple[torch.Tensor, List[int]]:
    """右 padding 成 (11, s_max) 的 LongTensor + 每行真实长度。"""
    lengths = [len(ids) for ids in suffix_ids]
    s_max = max(lengths)
    batch = torch.zeros((len(suffix_ids), s_max), dtype=torch.long, device=device)
    for i, ids in enumerate(suffix_ids):
        batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return batch, lengths


def name_token_starts(
    tokenizer, suffix_texts: Sequence[str], prefix_text: str
) -> List[int]:
    """每行候选后缀里「候选名」token 的起始下标。

    共享前缀（如 ``"Next tool: "``）的 token 不进 z 聚合——它们对所有
    候选等值，只会稀释判别信号。BPE 常把前缀尾与名字首合并成单 token
    (如 ``":Ġget"``)，此时不存在精确边界；规则取**第一个解码超长于前缀
    的下标**，那个跨界 token 含名字首段信息，纳入打分而非丢弃。
    """
    starts: List[int] = []
    for text in suffix_texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        start = len(ids)  # 兜底：找不到边界就退化为全后缀
        for cut in range(1, len(ids) + 1):
            decoded = tokenizer.decode(ids[:cut])
            if decoded == prefix_text:
                start = cut
                break
            if len(decoded) > len(prefix_text):
                start = cut
                break
        starts.append(start)
    return starts


def _token_logprobs_from_hidden(
    lm_head, hidden: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    """在选定 hidden 位置上算目标 token 的 logprob（fp32 softmax 保数值稳定）。

    hidden: (N, H)；target_ids: (N,)。只对 N 个位置过 lm_head，
    绝不物化整段序列的 logits（1.5B 词表 × 长 prompt 会吃掉数 GB）。
    """
    logits = lm_head(hidden).float()
    logprobs = F.log_softmax(logits, dim=-1)
    return logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def score_candidates(
    model,
    prompt_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    suffix_lens: Sequence[int],
    *,
    agg: str = "mean",
    scoring: str = "cache",
    name_starts: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """对 11 个候选后缀打分，返回 z 向量（shape (11,)，带梯度）。

    第 i 行的预测位置链：prompt 最后一个 hidden 预测后缀 token 0，
    后缀第 j 个 hidden 预测 token j+1——即每行的 hidden 序列是
    ``[h_prompt_last, suf[i, 0], ..., suf[i, sl-2]]``。

    ``name_starts`` 给出每行名字 token 的起始下标时，z 只聚合名字段
    （verbalizer scoring：共享前缀对所有候选等值，进均值只会稀释信号）；
    None 则聚合整个后缀。
    """
    if agg not in {"mean", "sum"}:
        raise ValueError(f"未知 z 聚合方式: {agg!r}")
    device = prompt_ids.device
    prompt_len = int(prompt_ids.shape[0])
    transformer = getattr(model, "model", model)
    lm_head = model.get_output_embeddings()

    # 每行参与聚合的目标区间 [lo, hi)——先统一算好，前向与聚合共用，
    # 防止两处口径漂移（聚合偏移必须按 span 长度累进，不是整行长度）
    spans: List[Tuple[int, int]] = []
    for i, length in enumerate(suffix_lens):
        lo = int(name_starts[i]) if name_starts is not None else 0
        lo = min(lo, length)
        if lo >= length:
            lo = 0
        spans.append((lo, length))

    hidden_parts: List[torch.Tensor] = []
    target_parts: List[torch.Tensor] = []

    if scoring == "cache":
        prompt_out = transformer(
            input_ids=prompt_ids.unsqueeze(0), use_cache=True
        )
        hidden_last = prompt_out.last_hidden_state[0, -1]
        shared = expand_cache(prompt_out.past_key_values, len(suffix_lens))
        s_max = int(suffix_ids.shape[1])
        attention_mask = torch.zeros(
            (len(suffix_lens), prompt_len + s_max), dtype=torch.long, device=device
        )
        attention_mask[:, :prompt_len] = 1
        for i, length in enumerate(suffix_lens):
            attention_mask[i, prompt_len : prompt_len + length] = 1
        suffix_out = transformer(
            input_ids=suffix_ids,
            attention_mask=attention_mask,
            past_key_values=shared,
            use_cache=True,
        )
        suffix_hidden = suffix_out.last_hidden_state
        for i, (lo, hi) in enumerate(spans):
            length = suffix_lens[i]
            parts: List[torch.Tensor] = []
            if lo == 0:
                parts.append(hidden_last.unsqueeze(0))
            if hi > max(lo, 1):
                parts.append(suffix_hidden[i, max(lo - 1, 0) : hi - 1])
            hidden_parts.append(torch.cat(parts, dim=0))
            target_parts.append(suffix_ids[i, lo:hi])
    elif scoring == "batch":
        for i, (lo, hi) in enumerate(spans):
            length = suffix_lens[i]
            full_ids = torch.cat(
                [prompt_ids, suffix_ids[i, :length]], dim=0
            ).unsqueeze(0)
            out = transformer(input_ids=full_ids, use_cache=False)
            # 位置 L-1 预测后缀 token 0；位置 L+j-1 预测 token j
            hidden_parts.append(
                out.last_hidden_state[0, prompt_len - 1 + lo : prompt_len + hi - 1]
            )
            target_parts.append(suffix_ids[i, lo:hi])
    else:
        raise ValueError(f"未知 scoring 模式: {scoring!r}（可选 cache|batch）")

    hidden_cat = torch.cat(hidden_parts, dim=0)
    target_cat = torch.cat(target_parts, dim=0)
    token_lp = _token_logprobs_from_hidden(lm_head, hidden_cat, target_cat)

    z_rows = []
    offset = 0
    for lo, hi in spans:
        count = hi - lo
        row_lp = token_lp[offset : offset + count]
        z_rows.append(row_lp.mean() if agg == "mean" else row_lp.sum())
        offset += count
    return torch.stack(z_rows)


def router_ce_loss(z: torch.Tensor, tau: float, gold_idx: int) -> torch.Tensor:
    """11 类 CE：把 z/τ 当类 logits，与推理 softmax 完全同分布。

    这不是 token 级 LM loss，而是「长度归一化 log-linear 分类器」——
    全部负候选进入归一化分母，是判别头显式压低混淆类的监督来源。
    """
    return F.cross_entropy(z / tau, torch.tensor(gold_idx, device=z.device))


def router_contrastive_loss(
    z: torch.Tensor,
    beta: float,
    gold_idx: int,
    negative_idxs: Sequence[int],
    margin: float = 0.0,
) -> torch.Tensor:
    """RLCD 对比项：``-log σ(β·(z[gold] − z[neg] − margin))`` 的负样本均值。

    gold 涨、neg 压，方向由 σ 的单调性保证；β 是逆温度（RLCD/DPO 原始形式）。
    """
    if not negative_idxs:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    gold_z = z[gold_idx]
    neg_z = z[list(negative_idxs)]
    return F.soft_margin_loss(gold_z - neg_z - margin, torch.ones_like(neg_z))


def decision_from_z(
    z: torch.Tensor, names: Sequence[str], tau: float
) -> Tuple[str, float, dict]:
    """z → (selected_tool, confidence, probabilities)。确定性 argmax，RLVR 兼容。"""
    probabilities = F.softmax(z.detach().float() / tau, dim=-1)
    best = int(torch.argmax(probabilities).item())
    distribution = {
        name: round(float(probabilities[i].item()), 6)
        for i, name in enumerate(names)
    }
    return str(names[best]), float(probabilities[best].item()), distribution
