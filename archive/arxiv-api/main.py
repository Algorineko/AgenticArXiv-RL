import arxiv # type: ignore
from datetime import datetime, timezone, timedelta
from typing import List, Dict

cs_categories = {
    "*": "All Computer Science",  # 所有计算机科学领域
    "AI": "Artificial Intelligence",  # 人工智能
    "AR": "Hardware Architecture",  # 硬件架构
    "CC": "Computational Complexity",  # 计算复杂性
    "CE": "Computational Engineering, Finance, and Science",  # 计算工程、金融与科学
    "CG": "Computational Geometry",  # 计算几何
    "CL": "Computation and Language",  # 计算与语言
    "CR": "Cryptography and Security",  # 密码学与安全
    "CV": "Computer Vision and Pattern Recognition",  # 计算机视觉与模式识别
    "CY": "Computers and Society",  # 计算机与社会
    "DB": "Databases",  # 数据库
    "DC": "Distributed, Parallel, and Cluster Computing",  # 分布式、并行与集群计算
    "DL": "Digital Libraries",  # 数字图书馆
    "DM": "Discrete Mathematics",  # 离散数学
    "DS": "Data Structures and Algorithms",  # 数据结构与算法
    "ET": "Emerging Technologies",  # 新兴技术
    "FL": "Formal Languages and Automata Theory",  # 形式语言与自动机理论
    "GL": "General Literature",  # 一般文献
    "GR": "Graphics",  # 图形学
    "GT": "Computer Science and Game Theory",  # 计算机科学与博弈论
    "HC": "Human-Computer Interaction",  # 人机交互
    "IR": "Information Retrieval",  # 信息检索
    "IT": "Information Theory",  # 信息论
    "LG": "Machine Learning",  # 机器学习
    "LO": "Logic in Computer Science",  # 计算机科学中的逻辑
    "MA": "Multiagent Systems",  # 多智能体系统
    "MM": "Multimedia",  # 多媒体
    "MS": "Mathematical Software",  # 数学软件
    "NA": "Numerical Analysis",  # 数值分析
    "NE": "Neural and Evolutionary Computing",  # 神经与进化计算
    "NI": "Networking and Internet Architecture",  # 网络与互联网架构
    "OH": "Other Computer Science",  # 其他计算机科学
    "OS": "Operating Systems",  # 操作系统
    "PF": "Performance",  # 性能
    "PL": "Programming Languages",  # 编程语言
    "RO": "Robotics",  # 机器人学
    "SC": "Symbolic Computation",  # 符号计算
    "SD": "Sound",  # 音频处理
    "SE": "Software Engineering",  # 软件工程
    "SI": "Social and Information Networks",  # 社会与信息网络
    "SY": "Systems and Control",  # 系统与控制
}


def get_recently_submitted_cs_papers(
    max_results: int = 50, aspect: str = "*"
) -> List[Dict]:
    now_utc = datetime.now(timezone.utc)
    start_date = (now_utc - timedelta(days=7)).strftime("%Y%m%d")
    end_date = now_utc.strftime("%Y%m%d")
    query = f"cat:cs.{aspect} AND submittedDate:[{start_date} TO {end_date}]"

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for result in client.results(search):
        paper_info = {
            "id": result.get_short_id(),
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary[:200] + "..."
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

    return papers


if __name__ == "__main__":
    # "*"表示获取所有子领域的论文
    ASPECT = "LO"
    papers = get_recently_submitted_cs_papers(max_results=20, aspect=ASPECT)
    if not papers:
        print("未获取到最近提交的论文")
    else:
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper['title']}")
            print(f"   作者: {', '.join(paper['authors'][:3])}")
            print(f"   发布时间: {paper['published']}")
            print(f"   PDF链接: {paper['pdf_url']}")
            print(f"   注释: {paper['comment']}")
            print("-" * 80)