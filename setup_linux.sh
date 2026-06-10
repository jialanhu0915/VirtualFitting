#!/bin/bash

# 虚拟试衣系统 - Linux/Mac安装脚本

echo "================================"
echo "👕 虚拟试衣系统 - 安装向导"
echo "================================"
echo ""

# 检查Python版本
check_python() {
    if command -v python3 &> /dev/null; then
        version=$(python3 --version 2>&1)
        echo "✅ Python已安装: $version"
        return 0
    elif command -v python &> /dev/null; then
        version=$(python --version 2>&1)
        echo "✅ Python已安装: $version"
        return 0
    else
        echo "❌ Python未安装"
        echo "请安装Python 3.11或更高版本"
        return 1
    fi
}

# 安装依赖包
install_dependencies() {
    echo ""
    echo "📦 正在安装依赖包..."
    
    # 确定python命令
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
        PIP_CMD=pip3
    else
        PYTHON_CMD=python
        PIP_CMD=pip
    fi
    
    # 升级pip
    $PYTHON_CMD -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    # 安装核心依赖
    packages=(
        "opencv-python"
        "opencv-contrib-python"
        "numpy"
        "Pillow"
        "scipy"
        "matplotlib"
        "scikit-image"
    )
    
    echo "正在安装: ${packages[*]}"
    
    # 使用清华源加速
    $PIP_CMD install "${packages[@]}" -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    # 尝试安装MediaPipe（可选）
    echo ""
    echo "尝试安装MediaPipe（可选）..."
    if $PIP_CMD install mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null; then
        echo "✅ MediaPipe安装成功"
    else
        echo "⚠️  MediaPipe安装失败，将使用OpenCV作为替代方案"
    fi
    
    echo ""
    echo "✅ 依赖包安装完成！"
}

# 检查输入文件
check_input_files() {
    echo ""
    echo "📁 检查输入文件..."
    
    person_image="data_picture/people/image.png"
    clothing_image="data_picture/clothes/image.png"
    
    all_exist=true
    
    if [ -f "$person_image" ]; then
        echo "✅ 人体图像: $person_image"
    else
        echo "❌ 人体图像不存在: $person_image"
        all_exist=false
    fi
    
    if [ -f "$clothing_image" ]; then
        echo "✅ 服装图像: $clothing_image"
    else
        echo "❌ 服装图像不存在: $clothing_image"
        all_exist=false
    fi
}

# 创建输出目录
create_output_dir() {
    if [ ! -d "output" ]; then
        mkdir -p output
        echo "✅ 创建输出目录: output/"
    fi
}

# 主流程
echo "开始安装流程..."
echo ""

# 1. 检查Python
if ! check_python; then
    echo ""
    echo "❌ 安装失败：Python未安装"
    exit 1
fi

# 2. 安装依赖
install_dependencies

# 3. 检查输入文件
check_input_files

# 4. 创建输出目录
create_output_dir

echo ""
echo "================================"
echo "✅ 安装完成！"
echo "================================"
echo ""
echo "运行方法:"
echo "  python3 virtual_tryon_simple.py"
echo "  或"
echo "  python3 main.py"
echo ""
