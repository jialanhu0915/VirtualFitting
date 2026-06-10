#!/bin/bash

# Docker运行脚本 - Linux/Mac版本

echo "🚀 虚拟试衣系统 - Docker版本"
echo "================================"

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

echo "✅ Docker已安装"

# 选择运行模式
echo -e "\n请选择运行模式："
echo "1. 构建并运行（首次使用）"
echo "2. 直接运行（已构建过）"
echo "3. 进入容器交互模式"
echo "4. 清理Docker镜像"

read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo -e "\n🔨 正在构建Docker镜像..."
        docker-compose build
        
        echo -e "\n🏃 正在运行虚拟试衣系统..."
        docker-compose run --rm virtual-tryon
        ;;
    2)
        echo -e "\n🏃 正在运行虚拟试衣系统..."
        docker-compose run --rm virtual-tryon
        ;;
    3)
        echo -e "\n🐳 进入容器交互模式..."
        docker-compose run --rm virtual-tryon /bin/bash
        ;;
    4)
        echo -e "\n🧹 清理Docker镜像..."
        docker-compose down --rmi all
        docker system prune -f
        echo "✅ 清理完成"
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo -e "\n✅ 完成！请查看 output/ 目录中的结果"
