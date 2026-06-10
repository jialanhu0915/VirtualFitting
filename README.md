# 虚拟试衣系统 - 自动关键点检测方案

## 🎯 系统特点

✅ **全自动关键点检测**
- 人体关键点：使用 MediaPipe Pose 自动检测33个关键点
- 服装关键点：基于图像分析自动提取8个关键点
- 无需手动标注

✅ **传统计算机视觉方法**
- 不使用深度学习虚拟试衣算法
- 符合作业要求

✅ **模块化设计**
- 关键点检测模块
- TPS变形模块（待实现）
- 图像融合模块（待实现）

## 📦 安装依赖

### 方法1：Docker运行（推荐⭐）

**优点**：避免环境配置问题，一键运行

#### Windows用户：
```powershell
# 1. 确保Docker Desktop已安装并运行
# 2. 运行启动脚本
.\docker\run_docker.ps1

# 或者手动运行
docker-compose -f docker/docker-compose.yml build
docker-compose -f docker/docker-compose.yml run --rm virtual-tryon
```

#### Linux/Mac用户：
```bash
# 1. 给脚本执行权限
chmod +x docker/run_docker.sh

# 2. 运行启动脚本
./docker/run_docker.sh

# 或者手动运行
docker-compose -f docker/docker-compose.yml build
docker-compose -f docker/docker-compose.yml run --rm virtual-tryon
```

#### Docker运行选项：
- **选项1**：首次构建并运行
- **选项2**：已构建过直接运行
- **选项3**：进入容器交互模式（用于调试）
- **选项4**：清理Docker镜像

> 详细 Docker 使用请参见 [`docker/DOCKER_GUIDE.md`](docker/DOCKER_GUIDE.md)。

### 方法2：Conda环境运行

```bash
# 激活conda环境
conda activate cv

# 安装依赖
pip install -r requirements.txt
```

### 方法3：直接安装

```bash
pip install opencv-python mediapipe numpy Pillow scipy matplotlib scikit-image
```

## 🚀 使用方法

### Docker方式（推荐）：
```bash
# 自动运行
./docker/run_docker.ps1  # Windows
./docker/run_docker.sh   # Linux/Mac
```

### 直接运行：
```python
# 运行主程序
python virtual_tryon_system.py
```

## 📊 关键点说明

### 人体关键点（MediaPipe自动检测）

MediaPipe提供33个身体关键点，我们主要使用：

| 关键点 | 索引 | 用途         |
| ------ | ---- | ------------ |
| 左肩   | 11   | 服装对齐     |
| 右肩   | 12   | 服装对齐     |
| 左肘   | 13   | 袖子变形     |
| 右肘   | 14   | 袖子变形     |
| 左臀   | 23   | 服装底部对齐 |
| 右臀   | 24   | 服装底部对齐 |

### 服装关键点（自动提取）

系统自动提取8个关键点：

1. **top_center** - 领口中心
2. **bottom_center** - 衣服底部中心
3. **left_shoulder** - 左肩
4. **right_shoulder** - 右肩
5. **left_armpit** - 左腋下
6. **right_armpit** - 右腋下
7. **left_bottom** - 左下摆
8. **right_bottom** - 右下摆

## 🔧 技术架构

```
输入图像
    ↓
┌─────────────────────────────────────┐
│   人体关键点检测（MediaPipe Pose）    │
│   - 自动检测33个关键点                │
│   - 返回肩膀、肘部、臀部等关键位置      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   服装关键点检测（图像分析）          │
│   - 服装分割（颜色/边缘）             │
│   - 轮廓提取                         │
│   - 自动定义8个关键点                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   TPS变形（下一步实现）               │
│   - 将服装变形适配人体姿态            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   图像融合（下一步实现）              │
│   - 颜色调整                         │
│   - 边缘融合                         │
│   - 最终合成                         │
└─────────────────────────────────────┘
    ↓
输出试衣结果
```

## 📁 项目结构

```
大作业/
├── data_picture/
│   ├── clothes/
│   │   └── image.png          # 服装图像
│   └── people/
│       └── image.png          # 人体图像
├── output/
│   ├── human_keypoints_visualization.jpg    # 人体关键点可视化
│   └── clothing_keypoints_visualization.jpg # 服装关键点可视化
├── virtual_tryon_system.py    # 主程序
├── requirements.txt           # 依赖列表
└── README.md                  # 说明文档
```

## 🎨 可视化输出

运行程序后，会在 `output/` 文件夹生成：

1. **human_keypoints_visualization.jpg** - 显示检测到的人体骨架和关键点
2. **clothing_keypoints_visualization.jpg** - 显示提取的服装关键点

## ⚠️ 注意事项

### 服装图像要求

为了获得最佳效果，服装图像应该：
- ✅ 背景干净（白色或浅色背景最佳）
- ✅ 服装完整可见
- ✅ 光线均匀
- ❌ 避免复杂背景
- ❌ 避免服装遮挡

### 人体图像要求

- ✅ 人体正面站立
- ✅ 全身或上半身可见
- ✅ 光线充足
- ❌ 避免严重遮挡

## 🔬 算法原理

### 1. 人体关键点检测

**MediaPipe Pose** 使用深度学习模型实时估计人体姿态：

```
输入图像 → CNN特征提取 → 关键点回归 → 33个关键点
```

**优点**：
- 精度高（在标准数据集上准确率>90%）
- 速度快（CPU实时运行）
- 鲁棒性强（适应不同姿态）

### 2. 服装关键点自动提取

基于几何形状分析：

```
步骤1: 图像分割
  - 灰度化 → Otsu阈值分割
  - 形态学处理（开运算+闭运算）

步骤2: 轮廓提取
  - 查找外轮廓
  - 选择最大轮廓

步骤3: 关键点定义
  - 找到上下左右边界点
  - 根据比例关系定义其他关键点
```

## 📈 下一步开发

### 待实现功能

1. **TPS变形模块**
   ```python
   # 薄板样条插值变形
   def tps_warp(clothing_img, src_points, dst_points):
       # 计算TPS变换
       # 应用变形
       # 返回变形后的服装
   ```

2. **图像融合模块**
   ```python
   # 多尺度融合
   def blend_images(warped_clothing, person_img, mask):
       # 颜色校正
       # 泊松融合
       # 边缘平滑
   ```

3. **完整流程**
   ```python
   human_kpts → clothing_kpts → TPS变形 → 图像融合 → 输出结果
   ```

## 🆚 与深度学习方法对比

| 维度         | 本系统（传统方法） | VITON（深度学习） |
| ------------ | ------------------ | ----------------- |
| **训练数据** | ❌ 不需要           | ✅ 需要大量数据    |
| **速度**     | ⚡ 快速             | 🐌 较慢            |
| **精度**     | ⚠️ 中等             | ✅ 高              |
| **泛化能力** | ⚠️ 有限             | ✅ 强              |
| **可解释性** | ✅ 强               | ❌ 黑盒            |
| **资源需求** | ✅ CPU即可          | ⚠️ 需要GPU         |

## 📚 参考文献

1. MediaPipe Pose: [https://google.github.io/mediapipe/solutions/pose.html](https://google.github.io/mediapipe/solutions/pose.html)
2. TPS插值原理: Bookstein, F. L. (1989). "Principal warps: Thin-plate splines and the decomposition of deformations"

## 🤝 贡献

欢迎提出改进建议！

---

**作者**: CV课程大作业  
**日期**: 2026年5月28日  
**版本**: v1.0
