# AgenticArxiv/utils/file_writer.py
"""
文件写入工具，用于保存论文信息等数据
"""

import os
from datetime import datetime
from typing import List, Dict
from utils.logger import log


def save_papers_to_file(papers: List[Dict], output_path: str) -> None:
    """
    将论文列表保存到指定文件

    Args:
        papers: 论文信息列表
        output_path: 输出文件路径
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            # 写入文件头
            f.write("=" * 80 + "\n")
            f.write("ArXiv 计算机科学论文列表\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"论文数量: {len(papers)}\n")
            f.write("=" * 80 + "\n\n")

            # 写入每篇论文的详细信息
            for i, paper in enumerate(papers, 1):
                f.write(f"论文 {i}: {paper.get('title', '无标题')}\n")
                f.write("-" * 60 + "\n")
                f.write(f"ID: {paper.get('id', 'N/A')}\n")
                f.write(f"作者: {', '.join(paper.get('authors', ['未知']))}\n")
                f.write(f"发表时间: {paper.get('published', 'N/A')}\n")
                f.write(f"更新时间: {paper.get('updated', 'N/A')}\n")
                f.write(f"主要分类: {paper.get('primary_category', 'N/A')}\n")
                f.write(f"所有分类: {', '.join(paper.get('categories', []))}\n")
                f.write(f"PDF链接: {paper.get('pdf_url', 'N/A')}\n")

                if paper.get("comment"):
                    f.write(f"备注: {paper.get('comment', '')}\n")

                f.write(f"摘要: {paper.get('summary', '无摘要')}\n")

                # 添加链接信息
                if paper.get("links"):
                    f.write("相关链接:\n")
                    for link in paper.get("links", []):
                        f.write(f"  - {link}\n")

                f.write("\n" + "=" * 80 + "\n\n")

        log.info(f"论文已保存到: {output_path}")

    except Exception as e:
        log.error(f"保存论文到文件时出错: {str(e)}")
        # 不抛出异常，避免影响主流程
