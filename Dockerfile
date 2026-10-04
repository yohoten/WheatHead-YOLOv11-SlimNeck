# 优云智算平台 - Docker 配置
# 基于 NVIDIA CUDA 12.1 + PyTorch
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（使用清华源加速）
RUN pip3 install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 单独安装 CUDA 版 PyTorch（从官方源）
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 复制项目代码
COPY . .

# 创建必要的目录
RUN mkdir -p runs/train data/GlobalWheat2020

# 设置入口点
CMD ["python3", "scripts/train_uyun.py"]
