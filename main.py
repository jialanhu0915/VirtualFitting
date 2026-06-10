"""
虚拟试衣系统 - 主程序入口
支持多种运行方式和关键点检测方法
"""

import os
import sys


def check_dependencies():
    """检查依赖包"""
    print("🔍 检查依赖包...")
    
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'matplotlib': 'matplotlib'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    # MediaPipe是可选的
    try:
        import mediapipe
        print(f"  ✅ mediapipe (可选)")
    except:
        print(f"  ⚠️  mediapipe (可选，未安装)")
    
    if missing:
        print(f"\n❌ 缺少必需依赖: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False
    
    print("\n✅ 所有必需依赖已安装")
    return True


def main():
    """主函数"""
    print("="*70)
    print("👕 虚拟试衣系统 - 自动关键点检测版本")
    print("="*70)
    print()
    
    # 检查依赖
    if not check_dependencies():
        return
    
    print("\n" + "="*70)
    print("选择运行模式:")
    print("  1. 快速模式（简化版，OpenCV检测）")
    print("  2. 完整模式（MediaPipe + OpenCV）")
    print("  3. Docker模式（推荐，避免环境问题）")
    print("="*70)
    
    choice = input("\n请选择 (1/2/3) [默认: 1]: ").strip() or '1'
    
    if choice == '3':
        print("\n🐳 启动Docker模式...")
        if os.name == 'nt':  # Windows
            os.system("powershell -File docker/run_docker.ps1")
        else:  # Linux/Mac
            os.system("./docker/run_docker.sh")
        return
    
    # 检查输入文件
    person_image = "data_picture/people/image.png"
    clothing_image = "data_picture/clothes/image.png"
    
    if not os.path.exists(person_image):
        print(f"❌ 人体图像不存在: {person_image}")
        return
    
    if not os.path.exists(clothing_image):
        print(f"❌ 服装图像不存在: {clothing_image}")
        return
    
    print(f"\n✅ 找到输入文件:")
    print(f"  • 人体图像: {person_image}")
    print(f"  • 服装图像: {clothing_image}")
    
    try:
        if choice == '2':
            print("\n🚀 启动完整模式...")
            try:
                from virtual_tryon_system import VirtualTryOnSystem
                system = VirtualTryOnSystem()
                system.run(person_image, clothing_image)
            except Exception as e:
                print(f"⚠️  完整模式失败: {e}")
                print("🔄 切换到简化模式...")
                from virtual_tryon_simple import SimpleVirtualTryOn
                system = SimpleVirtualTryOn()
                system.run(person_image, clothing_image)
        else:
            print("\n🚀 启动快速模式...")
            from virtual_tryon_simple import SimpleVirtualTryOn
            system = SimpleVirtualTryOn()
            system.run(person_image, clothing_image)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
