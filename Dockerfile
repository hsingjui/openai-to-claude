# 构建阶段
FROM python:3.11-slim AS builder

# 设置工作目录
WORKDIR /app

# 复制 pyproject.toml 文件
COPY pyproject.toml ./

# 安装项目依赖
RUN pip install --no-cache-dir uv && uv sync

# 运行阶段
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装 uv 工具
RUN pip install --no-cache-dir uv

# 从构建阶段复制已安装的依赖
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 复制项目文件
COPY . .

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 创建日志和配置目录
RUN mkdir -p /app/logs /app/config

# 暴露端口
EXPOSE 8000

# 启动应用
CMD ["uv", "run", "main.py"]
