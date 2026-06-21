# AgenticArxiv/main.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入工具模块，确保工具被注册
import tools.arxiv_tool  # noqa: F401
import tools.pdf_download_tool  # noqa: F401
import tools.pdf_translate_tool  # noqa: F401
import tools.cache_status_tool  # noqa: F401

from utils.llm_client import get_env_llm_client
from agents.agent_engine import ReActAgent
from utils.logger import log
import time


def main():
    log.info("=" * 50)
    log.info("启动Agentic Arxiv系统")
    log.info("=" * 50)

    try:
        os.makedirs("./output", exist_ok=True)

        log.info("初始化LLM客户端...")
        llm_client = get_env_llm_client()

        log.info("创建ReAct Agent...")
        agent = ReActAgent(llm_client)

        task = "获取最近7天内机器学习领域（ML）的最新论文,最多获取15篇"
        log.info(f"执行任务: {task}")

        start_time = time.time()
        result = agent.run(task)
        end_time = time.time()

        log.info(f"任务执行耗时: {end_time - start_time:.2f}秒")
        log.info(f"执行步数: {len(result['history'])}")

        final_obs = result.get("final_observation", "")
        log.info(f"最终结果: {final_obs}")

        print("任务完成！详细日志请查看 log.txt")
        print("执行结果已保存到 result.json")
        print(f"论文数据已保存到 {os.path.abspath('./output/recent_cs_papers.txt')}")

    except Exception as e:
        log.error(f"系统运行失败: {str(e)}")
        print(f"错误: {str(e)}")
        print("请检查日志文件 log.txt 获取详细信息")


if __name__ == "__main__":
    main()
