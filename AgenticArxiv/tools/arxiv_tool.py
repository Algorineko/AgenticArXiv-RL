# AgenticArxiv/tools/arxiv_tool.py
import arxiv  # type: ignore
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.tool_registry import registry
from utils.file_writer import save_papers_to_file

cs_categories = {
    "*": "All Computer Science",
    "AI": "Artificial Intelligence",
    "AR": "Hardware Architecture",
    "CC": "Computational Complexity",
    "CE": "Computational Engineering, Finance, and Science",
    "CG": "Computational Geometry",
    "CL": "Computation and Language",
    "CR": "Cryptography and Security",
    "CV": "Computer Vision and Pattern Recognition",
    "CY": "Computers and Society",
    "DB": "Databases",
    "DC": "Distributed, Parallel, and Cluster Computing",
    "DL": "Digital Libraries",
    "DM": "Discrete Mathematics",
    "DS": "Data Structures and Algorithms",
    "ET": "Emerging Technologies",
    "FL": "Formal Languages and Automata Theory",
    "GL": "General Literature",
    "GR": "Graphics",
    "GT": "Computer Science and Game Theory",
    "HC": "Human-Computer Interaction",
    "IR": "Information Retrieval",
    "IT": "Information Theory",
    "LG": "Machine Learning",
    "LO": "Logic in Computer Science",
    "MA": "Multiagent Systems",
    "MM": "Multimedia",
    "MS": "Mathematical Software",
    "NA": "Numerical Analysis",
    "NE": "Neural and Evolutionary Computing",
    "NI": "Networking and Internet Architecture",
    "OH": "Other Computer Science",
    "OS": "Operating Systems",
    "PF": "Performance",
    "PL": "Programming Languages",
    "RO": "Robotics",
    "SC": "Symbolic Computation",
    "SD": "Sound",
    "SE": "Software Engineering",
    "SI": "Social and Information Networks",
    "SY": "Systems and Control",
}


def _default_output_path() -> str:
    # 项目根目录 = tools/ 的上一级
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "output", "recent_cs_papers.txt")


def get_recently_submitted_cs_papers(
    max_results: int = 50,
    aspect: str = "*",
    days: int = 7,
    output_path: Optional[str] = None,
    save_to_file: bool = True,
) -> List[Dict]:
    """
    获取最近 days 天内提交的 cs.<aspect> 论文列表
    """
    now_utc = datetime.now(timezone.utc)
    start_date = (now_utc - timedelta(days=days)).strftime("%Y%m%d")
    end_date = now_utc.strftime("%Y%m%d")

    if aspect == "*":
        query = f"cat:cs.* AND submittedDate:[{start_date} TO {end_date}]"
    else:
        query = f"cat:cs.{aspect} AND submittedDate:[{start_date} TO {end_date}]"

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: List[Dict] = []
    for result in client.results(search):
        paper_info = {
            "id": result.get_short_id(),
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": (result.summary[:200] + "...")
            if len(result.summary) > 200
            else result.summary,
            "published": result.published.strftime("%Y-%m-%d %H:%M:%S"),
            "updated": result.updated.strftime("%Y-%m-%d %H:%M:%S")
            if result.updated
            else None,
            "pdf_url": result.pdf_url,
            "primary_category": result.primary_category,
            "categories": result.categories,
            "comment": result.comment,
            "links": [link.href for link in result.links],
        }
        papers.append(paper_info)

    if save_to_file:
        path = output_path or _default_output_path()
        save_papers_to_file(papers, path)

    return papers


ARXIV_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "max_results": {
            "type": "integer",
            "description": "最大返回结果数",
            "minimum": 1,
            "maximum": 100,
            "default": 50,
        },
        "aspect": {
            "type": "string",
            "description": "计算机科学子领域代码",
            "enum": list(cs_categories.keys()),
            "default": "*",
        },
        "days": {
            "type": "integer",
            "description": "查询最近多少天的论文",
            "minimum": 1,
            "maximum": 30,
            "default": 7,
        },
        "output_path": {
            "type": "string",
            "description": "可选：保存到指定文件路径；不传则用项目 output/recent_cs_papers.txt",
        },
        "save_to_file": {
            "type": "boolean",
            "description": "是否将论文列表写入文件",
            "default": True,
        },
    },
    "required": [],
}

registry.register_tool(
    name="get_recently_submitted_cs_papers",
    description="获取最近提交的计算机科学领域论文列表，支持按子领域筛选",
    parameter_schema=ARXIV_TOOL_SCHEMA,
    func=get_recently_submitted_cs_papers,
)


if __name__ == "__main__":
    ASPECT = "CL"
    papers = get_recently_submitted_cs_papers(max_results=20, aspect=ASPECT)
    print(f"获取到 {len(papers)} 篇论文")
