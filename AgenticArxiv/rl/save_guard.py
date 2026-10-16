"""ckpt 保存窗的宿主内存安全（hygon-dcu-skill「保存窗 OOM 六连杀」的落地版）。

本机 cgroup 宿主内存墙（64GiB）先于显存墙：保存写盘的 page cache 计入配额
且内核不主动驱逐，多作业并发保存会互相顶爆。这里提供三个原语：

- :func:`guarded_save`——跨进程 flock 串行 + 锁内 ``os.sync()`` + 保存后对
  每个写出文件 ``posix_fadvise(DONTNEED)`` 逐出 page cache + 锁内 settle；
- :func:`evict_page_cache`——单独对文件做 fadvise 逐出；
- :func:`cgroup_memory_gib`——读当前 cgroup 用量（监控/断言用）。

27B 级模型一律 adapter-only 保存（``--no-merge``），merge 需要在宿主重组
全量权重，直接顶穿 cgroup——不要在本仓库的大模型路径上开启 merge。
"""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable


def cgroup_memory_gib() -> float | None:
    for path in (
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/memory.current",
    ):
        try:
            return round(int(Path(path).read_text()) / 2**30, 2)
        except OSError:
            continue
    return None


def evict_page_cache(paths: Iterable[os.PathLike | str]) -> int:
    """对写出的文件做 ``POSIX_FADV_DONTNEED``，把 page cache 还给配额。"""
    evicted = 0
    for path in paths:
        path = Path(path)
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            evicted += 1
        except OSError:
            continue
    return evicted


def evict_model_dir_page_cache(model_dir: os.PathLike | str) -> int:
    """模型装载后逐出 safetensors 等文件的 page cache。

    from_pretrained 顺序读 50GB+ 权重会把等量 page cache 留在宿主
    cgroup 配额里（本机实测装载后 cgroup 直冲 63.9/64GiB）；这些页
    对已上卡的权重毫无用处，读完立逐。
    """
    model_dir = Path(model_dir)
    files = [
        p for p in model_dir.rglob("*")
        if p.is_file() and p.suffix in {".safetensors", ".json", ".jinja", ".txt"}
    ]
    return evict_page_cache(files)


@contextmanager
def save_lock(lock_path: Path):
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def guarded_save(
    save_fn: Callable[[], object],
    *,
    lock_path: os.PathLike | str,
    out_dir: os.PathLike | str | None = None,
    settle_seconds: int = 120,
    sync: bool = True,
) -> object:
    """在跨进程保存锁内执行 ``save_fn``，保存后逐出 page cache 并 settle。

    ``save_fn`` 返回它写出的文件路径列表（或 ``out_dir`` 给目录，自动收集）。
    """
    with save_lock(Path(lock_path)):
        if sync:
            os.sync()
        result = save_fn()
        if out_dir is not None:
            files = [p for p in Path(out_dir).rglob("*") if p.is_file()]
        elif isinstance(result, (list, tuple)):
            files = [Path(p) for p in result]
        else:
            files = []
        evicted = evict_page_cache(files)
        if settle_seconds > 0:
            time.sleep(settle_seconds)
        print(
            f"💾 guarded_save: {len(files)} files, evicted {evicted}, "
            f"cgroup {cgroup_memory_gib()} GiB, settle {settle_seconds}s"
        )
        return result
