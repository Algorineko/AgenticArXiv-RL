#!/bin/bash
set -e

# 切换到脚本所在目录，确保 docker-compose 能找到 docker-compose.yml
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

# 自动检测 docker compose 命令（兼容 v1: docker-compose 和 v2: docker compose）
COMPOSE_CMD=""

if command -v docker-compose >/dev/null 2>&1; then
	COMPOSE_CMD="docker-compose"
elif docker compose version >/dev/null 2>&1; then
	COMPOSE_CMD="docker compose"
else
	echo "错误: 系统中未找到 'docker-compose' 或 'docker compose'。请安装 Docker Compose 或使用 Docker v2。"
	exit 1
fi

echo "构建Docker镜像..."
$COMPOSE_CMD build

echo "启动Agent服务..."
$COMPOSE_CMD up -d

echo "查看容器状态..."
docker ps | grep weather-agent || true

echo "查看日志... (按 Ctrl+C 退出)"
docker logs -f weather-agent-demo || true
