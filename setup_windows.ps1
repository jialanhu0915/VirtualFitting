# 虚拟试衣系统 - Windows安装脚本
# 自动检测并安装所有依赖

Write-Host "================================" -ForegroundColor Cyan
Write-Host "👕 虚拟试衣系统 - 安装向导" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# 检查Python版本
function Check-Python() {
    try {
        $version = python --version 2>&1
        Write-Host "✅ Python已安装: $version" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "❌ Python未安装" -ForegroundColor Red
        Write-Host "请从以下地址下载安装Python 3.11+" -ForegroundColor Yellow
        Write-Host "https://www.python.org/downloads/" -ForegroundColor Cyan
        return $false
    }
}

# 安装依赖包
function Install-Dependencies() {
    Write-Host "`n📦 正在安装依赖包..." -ForegroundColor Cyan
    
    # 升级pip
    python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    # 安装核心依赖
    $packages = @(
        "opencv-python",
        "opencv-contrib-python",
        "numpy",
        "Pillow",
        "scipy",
        "matplotlib",
        "scikit-image"
    )
    
    Write-Host "正在安装: $($packages -join ', ')" -ForegroundColor Yellow
    
    # 使用清华源加速
    pip install $packages -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    # 尝试安装MediaPipe（可选）
    Write-Host "`n尝试安装MediaPipe（可选）..." -ForegroundColor Yellow
    try {
        pip install mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple
        Write-Host "✅ MediaPipe安装成功" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  MediaPipe安装失败，将使用OpenCV作为替代方案" -ForegroundColor Yellow
    }
    
    Write-Host "`n✅ 依赖包安装完成！" -ForegroundColor Green
}

# 检查输入文件
function Check-InputFiles() {
    Write-Host "`n📁 检查输入文件..." -ForegroundColor Cyan
    
    $personImage = "data_picture\people\image.png"
    $clothingImage = "data_picture\clothes\image.png"
    
    $allExist = $true
    
    if (Test-Path $personImage) {
        Write-Host "✅ 人体图像: $personImage" -ForegroundColor Green
    } else {
        Write-Host "❌ 人体图像不存在: $personImage" -ForegroundColor Red
        $allExist = $false
    }
    
    if (Test-Path $clothingImage) {
        Write-Host "✅ 服装图像: $clothingImage" -ForegroundColor Green
    } else {
        Write-Host "❌ 服装图像不存在: $clothingImage" -ForegroundColor Red
        $allExist = $false
    }
    
    return $allExist
}

# 创建输出目录
function Create-OutputDir() {
    if (-not (Test-Path "output")) {
        New-Item -ItemType Directory -Path "output" | Out-Null
        Write-Host "✅ 创建输出目录: output\" -ForegroundColor Green
    }
}

# 主流程
Write-Host "开始安装流程..." -ForegroundColor Yellow
Write-Host ""

# 1. 检查Python
if (-not (Check-Python)) {
    Write-Host "`n❌ 安装失败：Python未安装" -ForegroundColor Red
    exit 1
}

# 2. 安装依赖
Install-Dependencies

# 3. 检查输入文件
Check-InputFiles | Out-Null

# 4. 创建输出目录
Create-OutputDir

Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "✅ 安装完成！" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "运行方法:" -ForegroundColor Yellow
Write-Host "  python virtual_tryon_simple.py" -ForegroundColor White
Write-Host "  或" -ForegroundColor Yellow
Write-Host "  python main.py" -ForegroundColor White
Write-Host ""
