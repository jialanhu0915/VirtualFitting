# 虚拟试衣系统 - 快速使用指南

## 🚀 快速开始（5分钟上手）

### 第1步：安装依赖（首次使用）

```bash
pip install opencv-python opencv-contrib-python numpy Pillow scipy matplotlib scikit-image
```

### 第2步：准备输入图像

将你的图像放到以下位置：
```
data_picture/
├── clothes/
│   └── image.png    # 服装图像（干净背景）
└── people/
    └── image.png    # 人体图像
```

**服装图像要求：**
- ✅ 白色或浅色背景
- ✅ 服装完整可见
- ✅ 光线均匀

**人体图像要求：**
- ✅ 正面站立
- ✅ 全身或上半身可见
- ✅ 光线充足

### 第3步：运行程序

```bash
python virtual_tryon_simple.py
```

### 第4步：查看结果

程序运行后，在 `output/` 目录查看结果：
- `human_keypoints.jpg` - 人体关键点可视化
- `clothing_keypoints.jpg` - 服装关键点可视化

---

## 📖 详细使用方法

### 方法1：直接运行Python（推荐）

#### Windows用户：
```powershell
# 1. 安装依赖
pip install opencv-python opencv-contrib-python numpy Pillow scipy matplotlib scikit-image

# 2. 运行程序
python virtual_tryon_simple.py
```

#### Linux/Mac用户：
```bash
# 1. 安装依赖
pip3 install opencv-python opencv-contrib-python numpy Pillow scipy matplotlib scikit-image

# 2. 运行程序
python3 virtual_tryon_simple.py
```

### 方法2：使用安装脚本

#### Windows：
```powershell
# 运行安装脚本
powershell -ExecutionPolicy Bypass -File setup_windows.ps1

# 然后运行程序
python virtual_tryon_simple.py
```

#### Linux/Mac：
```bash
# 给脚本执行权限
chmod +x setup_linux.sh

# 运行安装脚本
./setup_linux.sh

# 运行程序
python3 virtual_tryon_simple.py
```

### 方法3：使用主菜单程序

```bash
# 运行主菜单
python main.py

# 选择运行模式：
# 1. 快速模式（简化版）
# 2. 完整模式（需要MediaPipe）
```

---

## 🎯 使用不同图像

如果你想使用自己的图像：

### 方法1：替换默认图像

```bash
# 直接替换这些文件
data_picture/people/image.png    # 替换为你的人体图像
data_picture/clothes/image.png   # 替换为你的服装图像
```

### 方法2：修改代码中的路径

编辑 `virtual_tryon_simple.py` 文件：

```python
# 找到main函数中的这几行
def main():
    # 修改这里的路径
    person_image = "data_picture/people/your_image.png"
    clothing_image = "data_picture/clothes/your_clothing.png"
```

---

## 🔧 常见问题解决

### 问题1：`ModuleNotFoundError: No module named 'cv2'`

**解决方案：**
```bash
pip install opencv-python opencv-contrib-python
```

### 问题2：`Permission denied` 错误

**解决方案（Windows）：**
```powershell
# 以管理员身份运行PowerShell
pip install --user opencv-python opencv-contrib-python
```

**解决方案（Linux/Mac）：**
```bash
sudo pip install opencv-python opencv-contrib-python
```

### 问题3：无法读取图像

**解决方案：**
1. 确认图像路径正确
2. 确认图像文件存在
3. 确认图像格式正确（PNG, JPG, JPEG）

```bash
# 检查文件是否存在
ls data_picture/people/image.png
ls data_picture/clothes/image.png
```

### 问题4：未检测到人脸

**说明：**
- 系统会自动使用图像中心估算关键点
- 这不影响后续处理

---

## 📊 输出结果说明

### 人体关键点
| 关键点         | 说明     | 坐标示例   |
| -------------- | -------- | ---------- |
| face           | 脸部中心 | (448, 228) |
| left_shoulder  | 左肩     | (299, 342) |
| right_shoulder | 右肩     | (597, 342) |
| left_hip       | 左臀     | (336, 682) |
| right_hip      | 右臀     | (560, 682) |

### 服装关键点
| 关键点         | 说明     | 坐标示例   |
| -------------- | -------- | ---------- |
| top_center     | 领口中心 | (208, 0)   |
| bottom_center  | 衣服底部 | (208, 818) |
| left_shoulder  | 左肩     | (62, 163)  |
| right_shoulder | 右肩     | (353, 163) |
| left_armpit    | 左腋下   | (41, 286)  |
| right_armpit   | 右腋下   | (374, 286) |

---

## 🎨 高级用法

### 批量处理多张图像

创建自定义脚本：

```python
from virtual_tryon_simple import SimpleVirtualTryOn

system = SimpleVirtualTryOn()

# 处理多对图像
pairs = [
    ("person1.png", "cloth1.png"),
    ("person2.png", "cloth2.png"),
    ("person3.png", "cloth3.png")
]

for person, cloth in pairs:
    system.run(person, cloth)
```

### 调整关键点位置

如果自动检测的关键点不准确，可以手动调整：

```python
# 在获取关键点后，手动调整坐标
human_kpts['left_shoulder']['x'] += 20  # 左肩向右移动20像素
human_kpts['left_shoulder']['y'] += 10  # 左肩向下移动10像素
```

---

## 📈 性能优化建议

### 加快检测速度
1. 降低图像分辨率
2. 使用更快的级联分类器

```python
# 降低图像分辨率
img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
```

### 提高检测精度
1. 使用高质量输入图像
2. 确保光照均匀
3. 使用正面站立的人体图像

---

## 🔄 工作流程图

```
┌─────────────┐
│  准备图像    │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  安装依赖    │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  运行程序    │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  查看结果    │
└─────────────┘
```

---

## 📞 获取帮助

如果遇到问题：

1. **查看文档**：
   - `README.md` - 项目说明
   - `PROJECT_SUMMARY.md` - 项目总结
   - `DOCKER_GUIDE.md` - Docker使用指南

2. **检查日志**：
   - 程序运行时会输出详细的调试信息

3. **常见错误**：
   - 大部分问题都是依赖安装问题
   - 使用 `pip install --user` 可以解决权限问题

---

## ✅ 成功运行示例

```
🚀 初始化虚拟试衣系统...
⚠️  MediaPipe不可用
✅ OpenCV可用

============================================================
🚀 虚拟试衣系统启动
============================================================

[步骤1/2] 检测人体关键点
🔍 检测人体关键点: data_picture/people/image.png
✅ 检测到人体关键点（基于人脸检测）

[步骤2/2] 检测服装关键点
👕 检测服装关键点: data_picture/clothes/image.png
✅ 检测到服装关键点

============================================================
✅ 关键点检测完成！
============================================================

📋 检测到的人体关键点:
  • 脸部中心: (448, 228)
  • 左肩: (299, 342)
  • 右肩: (597, 342)
  • 左臀: (336, 682)
  • 右臀: (560, 682)

📋 检测到的服装关键点:
  • 领口中心: (208, 0)
  • 衣服底部: (208, 818)
  • 左肩: (62, 163)
  • 右肩: (353, 163)
  • 左腋下: (41, 286)
  • 右腋下: (374, 286)

📁 输出文件:
  • output/human_keypoints.jpg - 人体关键点可视化
  • output/clothing_keypoints.jpg - 服装关键点可视化

🎯 下一步：
  1. 实现TPS变形算法
  2. 实现图像融合算法
```

看到这个输出说明程序运行成功！

---

**祝你使用顺利！** 🎉
