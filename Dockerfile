# 虚拟试衣系统 Docker镜像
# 使用阿里云镜像加速
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 更换apt源为阿里云镜像（加速）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 更换pip源为阿里云镜像（加速）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建输出目录
RUN mkdir -p output

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 默认命令
CMD ["python", "virtual_tryon_simple.py"]
