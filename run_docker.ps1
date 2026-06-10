# Docker运行脚本 - PowerShell版本

Write-Host "🚀 虚拟试衣系统 - Docker版本" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan

# 检查Docker是否安装
try {
    docker --version | Out-Null
    Write-Host "✅ Docker已安装" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker未安装，请先安装Docker Desktop" -ForegroundColor Red
    exit 1
}

# 选择运行模式
Write-Host "`n请选择运行模式：" -ForegroundColor Yellow
Write-Host "1. 构建并运行（首次使用）" -ForegroundColor White
Write-Host "2. 直接运行（已构建过）" -ForegroundColor White
Write-Host "3. 进入容器交互模式" -ForegroundColor White
Write-Host "4. 清理Docker镜像" -ForegroundColor White

$choice = Read-Host "请输入选项 (1-4)"

switch ($choice) {
    "1" {
        Write-Host "`n🔨 正在构建Docker镜像..." -ForegroundColor Cyan
        docker-compose build
        
        Write-Host "`n🏃 正在运行虚拟试衣系统..." -ForegroundColor Cyan
        docker-compose run --rm virtual-tryon
    }
    "2" {
        Write-Host "`n🏃 正在运行虚拟试衣系统..." -ForegroundColor Cyan
        docker-compose run --rm virtual-tryon
    }
    "3" {
        Write-Host "`n🐳 进入容器交互模式..." -ForegroundColor Cyan
        docker-compose run --rm virtual-tryon /bin/bash
    }
    "4" {
        Write-Host "`n🧹 清理Docker镜像..." -ForegroundColor Cyan
        docker-compose down --rmi all
        docker system prune -f
        Write-Host "✅ 清理完成" -ForegroundColor Green
    }
    default {
        Write-Host "❌ 无效选项" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n✅ 完成！请查看 output/ 目录中的结果" -ForegroundColor Green
