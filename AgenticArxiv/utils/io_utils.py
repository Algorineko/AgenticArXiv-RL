# AgenticArxiv/utils/io_utils.py
"""
JSONL 文件读写工具，供各数据脚本共享

此前 augment_sft_data / generate_parametric_sft_data /
build_figure_qa_dataset 各自维护一份 write_jsonl。三份实现语义一致
（UTF-8、ensure_ascii=False、"w" 覆盖写、自动建父目录、每行一条以换行
结尾），唯一差异是 build_figure_qa_dataset 会先把行拷成 dict，这里取
并集：非 dict 的 Mapping 一律先 dict(row) 再序列化，dict 行为不变。
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    逐行读取 JSONL 文件，跳过空行

    Args:
        path: JSONL 文件路径

    Returns:
        解析出的字典列表
    """
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """
    覆盖写 JSONL 文件（UTF-8、ensure_ascii=False、每行一条、行尾换行）

    Args:
        path: 目标文件路径，父目录不存在时自动创建
        rows: 待写出的行；非 dict 的 Mapping 会先拷成 dict
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
