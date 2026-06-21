# AgenticArxiv/services/translate_runner.py
from __future__ import annotations

import os
import threading
import time
from typing import Optional, Union, Dict, Any, Callable

from models.store import store
from models.schemas import TranslateTask, Paper
from config import settings
from utils.logger import log


def _fallback_pdf_url(paper_id: str) -> str:
    return f"https://arxiv.org/pdf/{paper_id}.pdf"


class TranslateRunner:
    """
    异步翻译任务执行器（MVP）
    - enqueue：创建任务 + 后台线程执行 translate_arxiv_pdf
    - 通过 event_bus 推送任务状态变化（SSE）
    """

    def __init__(self, event_bus) -> None:
        self.event_bus = event_bus
        self._threads_lock = threading.RLock()
        self._threads: Dict[str, threading.Thread] = {}

    def _resolve_inputs(
        self,
        session_id: str,
        ref: Union[str, int, None],
        paper_id: Optional[str],
        pdf_url: Optional[str],
        input_pdf_path: Optional[str],
    ) -> Dict[str, Any]:
        """
        解析 paper_id / pdf_url / input_pdf_path
        支持三种入口：
          1) input_pdf_path
          2) paper_id/pdf_url
          3) ref（含 ref=None -> last_active）
        """
        sid = session_id or "default"

        # 1) input_pdf_path 优先
        if input_pdf_path and os.path.exists(input_pdf_path):
            pid = paper_id or os.path.splitext(os.path.basename(input_pdf_path))[0]
            purl = pdf_url  # 可为空，工具内部会 fallback
            return {"paper_id": pid, "pdf_url": purl, "input_pdf_path": input_pdf_path}

        # 2) paper_id 优先
        if paper_id:
            pid = paper_id
            purl = pdf_url or None
            return {"paper_id": pid, "pdf_url": purl, "input_pdf_path": None}

        # 3) ref / last_active
        paper: Optional[Paper] = None
        if ref is None:
            last_id = store.get_last_active_paper_id(sid)
            if not last_id:
                raise ValueError(
                    "未找到指代对象：请先下载/翻译/查状态某篇论文，或明确提供 ref（序号/id/标题）"
                )
            pid = last_id
            paper = store.resolve_paper(sid, pid)  # 可能 None（last_papers 过期）
            purl = (paper.pdf_url if paper else None) or _fallback_pdf_url(pid)
            return {"paper_id": pid, "pdf_url": pdf_url or purl, "input_pdf_path": None}

        paper = store.resolve_paper(sid, ref)
        if paper is None:
            raise ValueError(
                "未找到论文：请先调用 /arxiv/recent 写入 session 记忆，或传 paper_id/pdf_url/input_pdf_path"
            )
        pid = paper.id
        purl = pdf_url or paper.pdf_url or _fallback_pdf_url(pid)
        return {"paper_id": pid, "pdf_url": purl, "input_pdf_path": None}

    def enqueue(
        self,
        session_id: str = "default",
        ref: Union[str, int, None] = None,
        force: bool = False,
        service: Optional[str] = None,
        threads: Optional[int] = None,
        keep_dual: bool = False,
        paper_id: Optional[str] = None,
        pdf_url: Optional[str] = None,
        input_pdf_path: Optional[str] = None,
    ) -> TranslateTask:
        """
        创建任务并启动后台执行；立即返回 task。
        """
        sid = session_id or "default"
        service = service or settings.pdf2zh_service
        threads = int(threads or settings.pdf2zh_threads)

        resolved = self._resolve_inputs(
            session_id=sid,
            ref=ref,
            paper_id=paper_id,
            pdf_url=pdf_url,
            input_pdf_path=input_pdf_path,
        )
        pid: str = resolved["paper_id"]
        purl: Optional[str] = resolved["pdf_url"]
        in_path: Optional[str] = resolved["input_pdf_path"]

        store.set_last_active_paper_id(sid, pid)

        # READY 且不 force：fast-path
        asset = store.get_translate_asset(pid)
        if (
            (not force)
            and asset
            and asset.status == "READY"
            and os.path.exists(asset.output_mono_path or "")
        ):
            t = store.create_translate_task(
                session_id=sid,
                paper_id=pid,
                input_pdf_url=purl,
                meta={
                    "force": str(force).lower(),
                    "service": service,
                    "threads": str(threads),
                    "keep_dual": str(keep_dual).lower(),
                    "fast_path": "true",
                },
            )
            store.update_task(
                t.task_id,
                status="SUCCEEDED",
                progress=1.0,
                input_pdf_path=asset.input_pdf_path,
                output_pdf_path=asset.output_mono_path,
                error=None,
            )

            task_obj = store.get_task(t.task_id)
            self.event_bus.publish(
                sid,
                {"type": "task_created", "kind": "translate", "task": task_obj.model_dump()},
            )
            self.event_bus.publish(
                sid,
                {"type": "task_succeeded", "kind": "translate", "task": task_obj.model_dump()},
            )
            return task_obj

        # 正常创建 PENDING
        t = store.create_translate_task(
            session_id=sid,
            paper_id=pid,
            input_pdf_url=purl,
            meta={
                "force": str(force).lower(),
                "service": service,
                "threads": str(threads),
                "keep_dual": str(keep_dual).lower(),
            },
        )

        task_obj = store.get_task(t.task_id)
        self.event_bus.publish(
            sid,
            {"type": "task_created", "kind": "translate", "task": task_obj.model_dump()},
        )

        th = threading.Thread(
            target=self._run_task_thread,
            kwargs=dict(
                task_id=t.task_id,
                session_id=sid,
                paper_id=pid,
                pdf_url=purl,
                input_pdf_path=in_path,
                force=force,
                service=service,
                threads=threads,
                keep_dual=keep_dual,
            ),
            daemon=True,
        )
        with self._threads_lock:
            self._threads[t.task_id] = th
        th.start()

        return task_obj

    def _run_task_thread(
        self,
        task_id: str,
        session_id: str,
        paper_id: str,
        pdf_url: Optional[str],
        input_pdf_path: Optional[str],
        force: bool,
        service: str,
        threads: int,
        keep_dual: bool,
    ) -> None:
        sid = session_id or "default"

        #  每个任务独立节流，避免 SSE 太密
        last_sent_p = 0.0
        last_sent_t = 0.0

        def publish_progress(p: float, detail: Optional[Dict[str, Any]] = None) -> None:
            """
            p: 0~1 的绝对进度（我们在 translate_arxiv_pdf 内部映射好阶段）
            """
            nonlocal last_sent_p, last_sent_t

            try:
                pp = float(p)
            except Exception:
                return
            if pp < 0:
                pp = 0.0
            if pp > 1:
                pp = 1.0

            now = time.time()
            # 至少提升 1% 或者间隔>=0.5s 才推送（两者满足其一）
            if (pp - last_sent_p) < 0.01 and (now - last_sent_t) < 0.5:
                return
            if pp <= last_sent_p and (now - last_sent_t) < 1.5:
                return

            last_sent_p = max(last_sent_p, pp)
            last_sent_t = now

            try:
                store.update_task(task_id, status="RUNNING", progress=last_sent_p, error=None)
                cur = store.get_task(task_id)
                self.event_bus.publish(
                    sid,
                    {
                        "type": "task_progress",
                        "kind": "translate",
                        "task": cur.model_dump(),
                        "detail": detail or {},
                    },
                )
            except Exception:
                return

        try:
            store.update_task(task_id, status="RUNNING", progress=0.01, error=None)
            started = store.get_task(task_id)
            self.event_bus.publish(
                sid,
                {"type": "task_started", "kind": "translate", "task": started.model_dump()},
            )

            #  关键：把 progress_cb 传给 translate_arxiv_pdf（由它从子进程输出解析进度）
            from tools.pdf_translate_tool import translate_arxiv_pdf

            res = translate_arxiv_pdf(
                session_id=sid,
                ref=None,
                force=force,
                service=service,
                threads=threads,
                keep_dual=keep_dual,
                paper_id=paper_id,
                pdf_url=pdf_url,
                input_pdf_path=input_pdf_path,
                progress_cb=publish_progress,
            )

            store.update_task(
                task_id,
                status="SUCCEEDED",
                progress=1.0,
                input_pdf_path=res.get("input_pdf_path"),
                output_pdf_path=res.get("output_pdf_path"),
                error=None,
            )
            done = store.get_task(task_id)
            self.event_bus.publish(
                sid,
                {"type": "task_succeeded", "kind": "translate", "task": done.model_dump()},
            )

        except Exception as e:
            log.error(f"Translate task failed: task_id={task_id}, err={e}")
            store.update_task(task_id, status="FAILED", progress=1.0, error=str(e))
            failed = store.get_task(task_id)
            self.event_bus.publish(
                sid,
                {"type": "task_failed", "kind": "translate", "task": failed.model_dump()},
            )

        finally:
            with self._threads_lock:
                self._threads.pop(task_id, None)