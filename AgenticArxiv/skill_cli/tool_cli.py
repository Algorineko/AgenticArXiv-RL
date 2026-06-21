#!/usr/bin/env python3
"""arXiv 论文管理 CLI 工具 (Python Fire)

用法:
    python tool_cli.py search_papers --max_results=20 --aspect=AI --days=7
    python tool_cli.py download_pdf --session_id=default --ref=1
    python tool_cli.py translate_pdf --session_id=default --ref=1 --service=bing
    python tool_cli.py cache_status --session_id=default --ref=1
"""
import json
import sys
import os

# 确保项目根目录在 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import fire


def _json_out(data):
    """统一 JSON 输出到 stdout"""
    print(json.dumps(data, ensure_ascii=False, default=str))


class ArxivToolCLI:
    """arXiv 论文搜索、下载、翻译、缓存状态查询 CLI"""

    def search_papers(self, max_results: int = 50, aspect: str = "*", days: int = 7,
                      output_path: str = None, save_to_file: bool = True):
        """搜索最近提交的 CS 领域论文

        Args:
            max_results: 最大返回结果数 (1-100)
            aspect: CS 子领域代码，如 AI, CL, CV, LG 等, * 表示所有
            days: 查询最近多少天的论文 (1-30)
            output_path: 保存路径 (可选)
            save_to_file: 是否保存到文件
        """
        from tools.arxiv_tool import get_recently_submitted_cs_papers
        result = get_recently_submitted_cs_papers(
            max_results=max_results, aspect=aspect, days=days,
            output_path=output_path, save_to_file=save_to_file,
        )
        _json_out(result)

    def download_pdf(self, session_id: str = "default", ref=None, force: bool = False):
        """下载论文 PDF

        Args:
            session_id: 会话 ID
            ref: 论文引用 (序号/ID/标题), null 表示最近操作的论文
            force: 是否强制重新下载
        """
        from tools.pdf_download_tool import download_arxiv_pdf
        result = download_arxiv_pdf(session_id=session_id, ref=ref, force=force)
        _json_out(result)

    def translate_pdf(self, session_id: str = "default", ref=None, force: bool = False,
                      service: str = "bing", threads: int = 4, keep_dual: bool = False,
                      paper_id: str = None, pdf_url: str = None, input_pdf_path: str = None):
        """翻译论文 PDF

        Args:
            session_id: 会话 ID
            ref: 论文引用 (序号/ID/标题), null 表示最近操作的论文
            force: 是否强制重新翻译
            service: 翻译服务 (bing/deepl/google)
            threads: 线程数 (1-32)
            keep_dual: 是否保留双语 PDF
            paper_id: 直接指定 paper_id
            pdf_url: 直接指定 PDF URL
            input_pdf_path: 直接指定本地 PDF 路径
        """
        from tools.pdf_translate_tool import translate_arxiv_pdf
        result = translate_arxiv_pdf(
            session_id=session_id, ref=ref, force=force,
            service=service, threads=threads, keep_dual=keep_dual,
            paper_id=paper_id, pdf_url=pdf_url, input_pdf_path=input_pdf_path,
        )
        _json_out(result)

    def cache_status(self, session_id: str = "default", ref=None, paper_id: str = None):
        """查询论文缓存状态

        Args:
            session_id: 会话 ID
            ref: 论文引用 (序号/ID/标题), null 表示最近操作的论文
            paper_id: 直接指定 paper_id
        """
        from tools.cache_status_tool import get_paper_cache_status
        result = get_paper_cache_status(session_id=session_id, ref=ref, paper_id=paper_id)
        _json_out(result)


if __name__ == "__main__":
    fire.Fire(ArxivToolCLI)
