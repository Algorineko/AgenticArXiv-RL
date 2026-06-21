# AgenticArxiv/utils/pdf_translator.py
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Dict, Any

from utils.logger import log


@dataclass(frozen=True)
class Pdf2ZhResult:
    mono_path: str
    dual_path: Optional[str]
    stdout_path: Optional[str] = None


def _guess_outputs(out_dir: str, stem: str) -> Tuple[str, str]:
    """
    不同版本可能是 -mono / -zh 命名，这里做兼容：
    mono_candidates: stem-mono.pdf or stem-zh.pdf
    dual: stem-dual.pdf
    """
    mono1 = os.path.join(out_dir, f"{stem}-mono.pdf")
    mono2 = os.path.join(out_dir, f"{stem}-zh.pdf")
    dual = os.path.join(out_dir, f"{stem}-dual.pdf")
    mono = mono1 if os.path.exists(mono1) else mono2
    return mono, dual


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_TQDM_PERCENT_RE = re.compile(r"(\d{1,3})%\|")  # tqdm 典型： 12%|████...
_PLAIN_PERCENT_RE = re.compile(r"(?<!\d)(\d{1,3})%(?!\d)")
_PAGE_RE = re.compile(r"(?i)\bpage(?:s)?\b.*?(\d+)\s*/\s*(\d+)")
# 排除日期格式 [04/04/26]：(?<!\[) 拒绝 [ 开头，(?!/) 拒绝后续还有 /
_FRACTION_RE = re.compile(r"(?<!\[)(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)(?!/)")


def _extract_progress(text: str) -> Optional[float]:
    """
    从一段输出里尽量提取 0~1 的进度。
    优先：
      1) tqdm 百分比： 12%|...
      2) 普通百分比： 12%
      3) page i/n
      4) i/n
    只处理 tqdm 风格行（以空白+数字/百分比开头），跳过 INFO/Namespace 等日志行。
    """
    if not text:
        return None
    text = _ANSI_RE.sub("", text)
    # 跳过非进度行：只处理 strip 后以数字开头的行（tqdm 输出格式）
    stripped = text.strip()
    if not stripped or not stripped[0].isdigit():
        return None

    m = _TQDM_PERCENT_RE.search(text)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 100:
            return v / 100.0

    m = _PLAIN_PERCENT_RE.search(text)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 100:
            return v / 100.0

    m = _PAGE_RE.search(text)
    if m:
        i = int(m.group(1))
        n = int(m.group(2))
        if n > 0:
            return max(0.0, min(1.0, i / n))

    m = _FRACTION_RE.search(text)
    if m:
        i = int(m.group(1))
        n = int(m.group(2))
        if n > 1 and i <= n:
            return max(0.0, min(1.0, i / n))

    return None


def _run_with_pty(
    cmd: list[str],
    on_text: Callable[[str], None],
) -> int:
    """
    用伪终端跑子进程，能捕获 tqdm 使用 \\r 刷新的进度。
    仅适用于类 Unix（Linux/macOS）。
    返回子进程 returncode。
    """
    import pty
    import select

    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=slave_fd,
            stderr=slave_fd,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
    finally:
        try:
            os.close(slave_fd)
        except Exception:
            pass

    buf = b""
    try:
        while True:
            # 读输出（支持 \r / \n 分割）
            r, _, _ = select.select([master_fd], [], [], 0.2)
            if master_fd in r:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    data = b""

                if not data:
                    # 可能 EOF
                    if proc.poll() is not None:
                        break
                else:
                    buf += data
                    while True:
                        # 找到最早的 \r 或 \n
                        idx_r = buf.find(b"\r")
                        idx_n = buf.find(b"\n")
                        idxs = [i for i in [idx_r, idx_n] if i != -1]
                        if not idxs:
                            break
                        idx = min(idxs)
                        chunk = buf[:idx]
                        buf = buf[idx + 1 :]
                        if chunk:
                            on_text(chunk.decode("utf-8", errors="ignore"))
                        else:
                            on_text("")  # 空刷新也触发一次（可用于心跳/阶段判断）

            # 若进程结束，尽量把残余读完
            if proc.poll() is not None:
                # 再尝试快速 drain 一下
                for _ in range(5):
                    r2, _, _ = select.select([master_fd], [], [], 0.05)
                    if master_fd in r2:
                        try:
                            data2 = os.read(master_fd, 4096)
                        except OSError:
                            data2 = b""
                        if not data2:
                            break
                        buf += data2
                    else:
                        break
                break

        if buf:
            on_text(buf.decode("utf-8", errors="ignore"))

        return int(proc.wait())
    finally:
        try:
            os.close(master_fd)
        except Exception:
            pass


def _run_with_pipe(
    cmd: list[str],
    on_text: Callable[[str], None],
) -> int:
    """
    Windows/无 pty 环境的 fallback：PIPE 行读取。
    注意：若 tqdm 只用 \\r 不输出 \\n，这里更新会少一些（但通常也会打印阶段行）。
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        on_text(line.rstrip("\n"))
    return int(proc.wait())


def run_pdf2zh_translate(
    pdf2zh_bin: str,
    input_pdf: str,
    out_dir: str,
    service: str = "bing",
    threads: int = 4,
    keep_dual: bool = False,
    log_path: Optional[str] = None,
    # 进度回调（p: 0~1），detail 里可带 stage/line 等
    progress_cb: Optional[Callable[[float, Optional[Dict[str, Any]]], None]] = None,
) -> Pdf2ZhResult:
    """
    调用 pdf2zh 翻译全文：
      pdf2zh input.pdf -s bing -o out_dir -t 4

    关键改造：
    - 用 Popen 实时读取 stdout（Linux 用 pty 捕获 tqdm 的 \\r 刷新）
    - 解析 stdout 中的百分比/页码，并通过 progress_cb 推送
    """
    if not os.path.exists(input_pdf):
        raise FileNotFoundError(f"input pdf not found: {input_pdf}")

    bin_path = shutil.which(pdf2zh_bin) or pdf2zh_bin
    if shutil.which(bin_path) is None and not os.path.exists(bin_path):
        raise RuntimeError(
            f"未找到 pdf2zh 可执行文件：{pdf2zh_bin}。"
            f"请先安装：pip install pdf2zh，或设置环境变量 PDF2ZH_BIN 指向可执行文件。"
        )

    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(input_pdf))[0]

    cmd = [
        bin_path,
        input_pdf,
        "-s",
        service,
        "-o",
        out_dir,
        "-t",
        str(int(threads)),
    ]

    log.info(f"Run pdf2zh: {' '.join(cmd)}")

    stdout_file = None
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        stdout_file = open(log_path, "w", encoding="utf-8")

    # 用于错误信息（保留末尾若干行）
    tail: list[str] = []
    tail_max = 80

    last_p: Optional[float] = None

    def on_text(text: str) -> None:
        nonlocal last_p, tail

        if stdout_file:
            # tqdm 的 \r 刷新也写成一行，方便复盘（不影响）
            stdout_file.write(text + "\n")
            stdout_file.flush()

        if text:
            tail.append(text)
            if len(tail) > tail_max:
                tail = tail[-tail_max:]

        p = _extract_progress(text)
        if p is None:
            return

        # 进度必须单调不减（避免 tqdm 反复刷同一个百分比）
        if last_p is None or p > last_p:
            last_p = p
            if progress_cb:
                try:
                    progress_cb(float(p), {"stage": "pdf2zh", "line": text})
                except Exception:
                    pass

    try:
        if progress_cb:
            try:
                progress_cb(0.0, {"stage": "pdf2zh", "line": "pdf2zh started"})
            except Exception:
                pass

        # Linux 优先用 pty
        if os.name != "nt":
            rc = _run_with_pty(cmd, on_text=on_text)
        else:
            rc = _run_with_pipe(cmd, on_text=on_text)

        if rc != 0:
            tail_text = "\n".join(tail[-40:]) if tail else "(no output captured)"
            raise RuntimeError(f"pdf2zh exited with code={rc}\n---- tail ----\n{tail_text}")

    finally:
        if stdout_file:
            stdout_file.close()

    mono, dual = _guess_outputs(out_dir, stem)

    if not os.path.exists(mono):
        raise RuntimeError(
            f"pdf2zh 执行成功但未找到 mono 输出文件，期望位置之一："
            f"{os.path.join(out_dir, stem + '-mono.pdf')} 或 {os.path.join(out_dir, stem + '-zh.pdf')}"
        )

    dual_exists = os.path.exists(dual)
    if dual_exists and not keep_dual:
        try:
            os.remove(dual)
        except Exception as e:
            log.warning(f"删除 dual 失败（不影响返回 mono）：{dual}, err={e}")

    # 结束时补一次 1.0（注意：外层任务会再设 SUCCEEDED=1.0）
    if progress_cb:
        try:
            progress_cb(1.0, {"stage": "pdf2zh", "line": "pdf2zh finished"})
        except Exception:
            pass

    return Pdf2ZhResult(
        mono_path=mono,
        dual_path=dual if (dual_exists and keep_dual) else None,
        stdout_path=log_path,
    )