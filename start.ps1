# 虚拟试衣系统 - Windows启动脚本

Write-Host "================================" -ForegroundColor Cyan
Write-Host "👕 虚拟试衣系统" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# 检查Docker
$dockerInstalled = $false
try {
    docker --version | Out-Null
    $dockerInstalled = $true
    Write-Host "✅ Docker已安装" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Docker未安装" -ForegroundColor Yellow
}

# 检查Python
$pythonAvailable = $false
try {
    python --version | Out-Null
    $pythonAvailable = $true
    Write-Host "✅ Python已安装" -ForegroundColor Green
} catch {
    Write-Host "❌ Python未安装" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "请选择运行方式:" -ForegroundColor Yellow
Write-Host "  1. Docker运行（推荐）" -ForegroundColor White
Write-Host "  2. 本地Python运行" -ForegroundColor White
Write-Host "  3. 查看使用说明" -ForegroundColor White
Write-Host "  4. 退出" -ForegroundColor White

$choice = Read-Host "`n请选择 (1-4)"

switch ($choice) {
    "1" {
        if (-not $dockerInstalled) {
            Write-Host "`n❌ Docker未安装，请先安装Docker Desktop" -ForegroundColor Red
            Write-Host "下载地址: https://www.docker.com/products/docker-desktop" -ForegroundColor Cyan
            exit 1
        }
        
        Write-Host "`n🐳 使用Docker运行..." -ForegroundColor Cyan
        
        # 检查Docker Desktop是否运行
        $dockerRunning = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
        if (-not $dockerRunning) {
            Write-Host "⚠️  Docker Desktop未运行，正在启动..." -ForegroundColor Yellow
            Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
            Start-Sleep -Seconds 10
        }
        
        # 运行Docker
        & ".\run_docker.ps1"
    }
    "2" {
        Write-Host "`n🐍 使用本地Python运行..." -ForegroundColor Cyan
        
        # 检查conda环境
        $condaEnv = conda info --envs | Select-String "cv"
        if ($condaEnv) {
            Write-Host "✅ 检测到cv环境" -ForegroundColor Green
            Write-Host "激活cv环境并运行..." -ForegroundColor Cyan
            conda activate cv
            python main.py
        } else {
            Write-Host "⚠️  未检测到cv环境，使用当前Python环境" -ForegroundColor Yellow
            python main.py
        }
    }
    "3" {
        Write-Host "`n📖 打开使用说明..." -ForegroundColor Cyan
        if (Test-Path "README.md") {
            Start-Process "notepad.exe" "README.md"
        } else {
            Write-Host "❌ README.md 文件不存在" -ForegroundColor Red
        }
    }
    "4" {
        Write-Host "`n👋 再见！" -ForegroundColor Green
        exit 0
    }
    default {
        Write-Host "`n❌ 无效选项" -ForegroundColor Red
        exit 1
    }
}
