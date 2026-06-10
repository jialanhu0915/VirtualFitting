# Docker使用指南

> 本目录包含 Docker 镜像与编排配置（`Dockerfile`、`docker-compose.yml`、运行脚本）。下文命令均假设在 `docker/` 目录下执行；从项目根目录调用 `docker/run_docker.sh` / `docker/run_docker.ps1` 即可自动进入该目录。

## 🐳 为什么使用Docker？

使用Docker可以：
- ✅ 避免环境配置问题
- ✅ 确保所有依赖正确安装
- ✅ 跨平台一致性（Windows/Linux/Mac）
- ✅ 一键运行，无需手动配置

## 📋 前置要求

### Windows用户：
1. 安装 **Docker Desktop for Windows**
   - 下载地址：https://www.docker.com/products/docker-desktop
   - 安装后启动Docker Desktop
   - 确保Docker Desktop正在运行（任务栏图标）

2. 检查Docker是否安装成功：
   ```powershell
   docker --version
   docker-compose --version
   ```

### Linux用户：
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose

# 启动Docker服务
sudo systemctl start docker
sudo systemctl enable docker

# 将当前用户添加到docker组（避免每次使用sudo）
sudo usermod -aG docker $USER
```

### Mac用户：
1. 安装 **Docker Desktop for Mac**
   - 下载地址：https://www.docker.com/products/docker-desktop

## 🚀 快速开始

### 方法1：使用自动化脚本（推荐）

#### Windows PowerShell：
```powershell
# 运行脚本
.\run_docker.ps1
```

#### Linux/Mac：
```bash
# 给脚本执行权限
chmod +x run_docker.sh

# 运行脚本
./run_docker.sh
```

### 方法2：手动运行

```bash
# 1. 构建Docker镜像
docker-compose build

# 2. 运行程序
docker-compose run --rm virtual-tryon

# 3. 查看结果
# 输出文件会自动保存到 output/ 目录
```

## 🔧 常用Docker命令

### 查看镜像状态：
```bash
docker images
```

### 查看运行中的容器：
```bash
docker ps
```

### 进入容器交互模式：
```bash
# 启动交互式容器
docker-compose run --rm virtual-tryon /bin/bash

# 或者使用exec进入正在运行的容器
docker-compose exec virtual-tryon /bin/bash
```

### 查看容器日志：
```bash
docker-compose logs
```

### 清理资源：
```bash
# 停止所有容器
docker-compose down

# 删除镜像
docker-compose down --rmi all

# 清理所有未使用的资源
docker system prune -a
```

## 📁 目录挂载说明

Docker容器会自动挂载以下目录：

```
主机目录              容器目录
./data_picture   ->   /app/data_picture   (输入数据)
./output         ->   /app/output         (输出结果)
./               ->   /app                (代码开发)
```

这意味着：
- ✅ 你可以在主机上修改代码，容器内会自动更新
- ✅ 容器内的输出会自动保存到主机的 `output/` 目录
- ✅ 输入图像放在 `data_picture/` 目录即可

## 🐛 常见问题

### 1. Docker镜像构建失败

**问题**：网络问题导致依赖下载失败

**解决**：
```bash
# 使用国内镜像源
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
```

### 2. 权限问题（Linux）

**问题**：`permission denied`

**解决**：
```bash
# 给脚本执行权限
chmod +x run_docker.sh

# 或者使用sudo
sudo docker-compose run --rm virtual-tryon
```

### 3. 内存不足

**问题**：MediaPipe需要较多内存

**解决**：
- 在Docker Desktop设置中增加内存限制（推荐至少4GB）

### 4. GPU支持

**说明**：本系统使用CPU运行，不需要GPU

如果未来需要GPU支持，修改 `docker-compose.yml`：
```yaml
services:
  virtual-tryon:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 📊 性能优化

### 减小镜像大小：
```bash
# 使用多阶段构建
# Dockerfile中已经优化
```

### 加速构建：
```bash
# 使用构建缓存
docker-compose build --no-cache  # 不使用缓存
docker-compose build             # 使用缓存
```

## 🔄 开发工作流

### 开发模式：
```bash
# 1. 首次构建
docker-compose build

# 2. 开发时挂载代码目录
# docker-compose.yml已配置自动挂载

# 3. 修改代码后直接运行
docker-compose run --rm virtual-tryon
```

### 生产模式：
```bash
# 1. 构建生产镜像
docker-compose -f docker-compose.prod.yml build

# 2. 运行
docker-compose -f docker-compose.prod.yml up -d
```

## 📝 自定义配置

### 修改Python版本：
编辑 `Dockerfile`：
```dockerfile
FROM python:3.10-slim  # 改为你需要的版本
```

### 添加新依赖：
编辑 `requirements.txt`，然后重新构建：
```bash
docker-compose build
```

### 修改运行参数：
编辑 `docker-compose.yml`：
```yaml
environment:
  - PYTHONUNBUFFERED=1
  - OMP_NUM_THREADS=4  # 限制线程数
```

## 🎯 最佳实践

1. **定期清理**：使用 `docker system prune` 清理未使用的资源
2. **版本控制**：将 `Dockerfile` 和 `docker-compose.yml` 纳入版本控制
3. **安全更新**：定期更新基础镜像以获取安全补丁
4. **资源限制**：在生产环境中设置内存和CPU限制

---

**作者**: CV课程大作业  
**更新时间**: 2026年5月28日
