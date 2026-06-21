FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY . .

# 暴露端口
EXPOSE 8080

# 设置环境变量
ENV PYTHONPATH=/app:/app/src:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# 运行应用
CMD ["python", "main.py"]
