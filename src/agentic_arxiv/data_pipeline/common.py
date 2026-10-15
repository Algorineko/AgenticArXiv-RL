"""训练数据的内容指纹和显式协议审计，不承担数据生产入口。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable

from agentic_arxiv.provenance import require_training_protocol
from agentic_arxiv.runtime.react_contract import protocol_metadata


def canonical_hash(value: Any) -> str:
    """保留样本原始 JSON 指纹语义，不对训练内容作来源脱敏。"""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_data_protocol(
    rows: Iterable[Dict[str, Any]], *, expected_protocol: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """新训练数据不能把历史未知协议重标为当前协议。"""
    expected = protocol_metadata() if expected_protocol is None else expected_protocol
    declarations = [require_training_protocol(
        row, expected, source_label=f"SFT row {index}",
    ) for index, row in enumerate(rows)]
    if not declarations:
        raise ValueError("SFT 数据为空")
    known_hashes = {item["chat_template_hash"] for item in declarations
                    if item["chat_template_hash"] is not None}
    if len(known_hashes) > 1:
        raise ValueError("SFT 来源使用不兼容的 chat template")
    declared = dict(declarations[0])
    # 部分来源未知时，混合 manifest 不把已知的一部分伪装成全部已知。
    if any(item["chat_template_hash"] is None for item in declarations):
        declared["chat_template_hash"] = None
        declared["metadata_status"] = "partial"
    return declared
