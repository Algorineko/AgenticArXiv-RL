# AgenticArxiv/utils/logger.py
import sys
from loguru import logger


def setup_logger():
    """配置日志记录器"""
    # 移除默认的日志处理器
    logger.remove()

    # 添加文件日志处理器，记录到log.txt
    logger.add(
        "./output/log.txt",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",  # 日志文件达到10MB时轮转
        retention="30 days",  # 保留30天的日志
        encoding="utf-8",
        backtrace=True,  # 记录堆栈跟踪
        diagnose=True,  # 显示变量值
        mode="w"  # "w"为覆盖写入,"a"为追加写入
    )
    return logger


# 创建全局logger实例
log = setup_logger()
