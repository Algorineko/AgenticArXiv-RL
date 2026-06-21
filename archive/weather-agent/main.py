#!/usr/bin/env python3
"""
Weather Agent 主启动文件
"""
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.api.main import app
import uvicorn
from loguru import logger


def main():
    """主函数"""
    logger.info("启动 Weather Agent API 服务...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )


if __name__ == "__main__":
    main()
