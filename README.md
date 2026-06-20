# 虚拟试衣系统 — 自动关键点检测方案

一个**不依赖深度学习虚拟试衣模型**的服装试穿工具：从平铺服装图 + 人体图出发，
通过**关键点检测 + 几何 warp + 图像融合**三步把衣服贴到人身上。
用于 CV 课程大作业演示，也作为后续工作的基线。

---

## 🎯 系统特点

- ✅ **人体关键点自动检测**：MediaPipe Pose (Tasks API, full 模型) + Haar 级联 + 启发式
  估算三级降级链，任何输入都能拿到结果。
- ✅ **服装关键点自动提取**：纯传统 CV（CLAHE + 多通道 Canny + 轮廓分析），
  8 个语义点（领口 / 肩 / 腋下 / 下摆）+ 30 点轮廓采样。
- ✅ **流水线 warp**：Stage A 仿射粗定位 + Stage B 按 silhouette 逐行 fit，
  躯干 / 左袖 / 右袖三区域独立处理。
- ✅ **人体关键点缓存**：写入 person 图同目录，`mtime` 失效，
  同一张人图反复试衣时不用重跑模型。
- ✅ **CLI 子命令**：分阶段运行（detect / detect-human / detect-clothing / run）。
- ✅ **诊断脚本齐全**：`scripts/` 下保留 13 个 warp / scale / 对称性诊断工具。

---

## 📦 安装依赖

### 方法 1：uv（推荐 ⭐，与 `pyproject.toml` 锁文件一致）

```bash
# 安装 uv（如果还没有）
pip install uv

# 在仓库根目录
uv sync                      # 创建 .venv 并按 uv.lock 安装
.venv/Scripts/python.exe main.py --help    # Windows
# 或：  uv run main.py --help
```

`.python-version` 锁定 **Python 3.13**。

### 方法 2：pip + venv

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 方法 3：Docker（避免环境配置）

```bash
# Windows
.\docker\run_docker.ps1
# Linux / Mac
chmod +x docker/run_docker.sh
./docker/run_docker.sh
```

详细 Docker 使用见 [`docker/DOCKER_GUIDE.md`](docker/DOCKER_GUIDE.md)。

> **首次运行会自动下载** MediaPipe Pose 模型到 `models/pose_landmarker_full.task`（约 12MB），
> 以及 rembg U²-Net 模型到 `models/u2net/`（约 170MB）。

---

## 🚀 使用方法

所有命令都以仓库根目录运行。

### 1. 检测关键点（不 warp）

```bash
# 同时检测人体 + 服装关键点
.venv/Scripts/python.exe main.py detect \
    --person data_picture/people/image.png \
    --clothing data_picture/clothes/image.png \
    --output output/detect

# 只检测人体 / 只检测服装
.venv/Scripts/python.exe main.py detect-human \
    --person data_picture/people/image.png \
    --output output/detect_human
```

### 2. 完整试衣流水线

```bash
.venv/Scripts/python.exe main.py run \
    --person data_picture/people/image.png \
    --clothing data_picture/clothes/image.png \
    --output output/run \
    --n-points 30 \
    --warp-method flow
```

- `--warp-method flow`（默认）：Stage A 仿射 + Stage B 流水式逐行 fit
- `--warp-method affine`：仅 Stage A 仿射（不 fit，调试用）
- `--n-points`：轮廓采样点数（默认 30）
- `--rebuild-human-cache`：忽略人体关键点缓存，强制重跑模型

### 3. 人体关键点缓存

```bash
# 显式预热 / 重建缓存（写入 person 图同目录 *.keypoints.json / *.keypoints.jpg）
.venv/Scripts/python.exe main.py cache-human \
    --person data_picture/people/image.png
```

同一张人图后续 `detect` / `run` 会直接命中缓存，跳过 MediaPipe 推理。

### 输出文件

| 文件 | 说明 |
|---|---|
| `human_keypoints.jpg` | 人体关键点可视化 |
| `clothing_keypoints.jpg` | 服装关键点可视化 |
| `warped_clothing.png` | 流水式 warp 后的服装（**PNG 无损**，避免 JPEG 压缩伪影） |
| `warped_mask.png` | warp 后的二值 mask |
| `result.jpg` | 最终试衣结果 |
| `debug_body_pts.jpg` | 身体轮廓多边形 overlay |
| `debug_clothing_pts.jpg` | 服装轮廓 + 领口锚点 overlay |
| `debug_warped_mask_overlay.jpg` | warped_mask 在人体图上的半透明叠加 |

---

## 🔧 技术架构

```
输入：person.png + clothing.png
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1  关键点检测                                      │
├─────────────────────────────────────────────────────────┤
│ 人体：RobustHumanDetector（三级降级）                      │
│   ① MediaPipe Pose (Tasks API, full 模型)               │
│   ② Haar Cascade 人脸检测 + 比例推算                      │
│   ③ 启发式估算（图像中心 + 身高比例）                       │
│   → 缓存到 person 图同目录，mtime 失效                     │
│                                                         │
│ 服装：ClothingDetector（纯传统 CV）                       │
│   ① 若带 alpha 且含透明信息 → 直接用 alpha 抠图           │
│   ② 否则 CLAHE 拉伸灰度 → 多通道 Canny → 闭运算封口         │
│      → 最大外轮廓 → 取领口凹点 / 肩 / 腋下 / 下摆 8 关键点   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2  warp（流水式）                                   │
├─────────────────────────────────────────────────────────┤
│ Stage A  仿射粗定位                                       │
│   - bbox-based isotropic scale（cap=1.10 防紧身衣放大过度） │
│   - 肩线中点 anchor：衣服 → 人体 neck（双肩中点）           │
│                                                         │
│ Stage B  按 silhouette 逐行 fit                           │
│   - 30 点身体轮廓（弧长均匀采样）+ 30 点服装轮廓            │
│   - 对每行 y：按 body_pts 左/右边 → cloth_pts 缩放         │
│   - 袖子：按 x/y 极值启发式划分躯干/左袖/右袖三区域独立 fit   │
│   - 强制对称（围绕 body_center）消除 qipao 类斜襟凸块       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 3  融合                                             │
│   mask 抠图 + alpha blend 写入人体图                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
output/<dir>/result.jpg
```

> TPS / Umeyama similarity / 8 点语义对应等旧路径已废弃并删除；
> 流水线 warp 是当前唯一路径。历史背景见 `docs/notes/` 与 `scripts/README.md`。

---

## 📁 项目结构

```
VirtualFitting/
├── main.py                       # CLI 入口（detect / detect-human / detect-clothing / cache-human / run）
├── src/
│   └── virtual_tryon/            # 核心包（src 布局，pyproject/uv 已配置）
│       ├── __init__.py
│       ├── keypoints.py          # Keypoint dataclass + MediaPipe 索引 + 8 点表 + 对应表
│       ├── human_detector.py     # MediaPipe + Haar + 启发式三级降级链
│       ├── human_cache.py        # 磁盘缓存（mtime 失效）
│       ├── clothing_detector.py  # CLAHE + Canny + alpha 抠图 + 8 关键点
│       ├── warp.py               # Stage A 仿射 + Stage B 流水式逐行 fit + 三区域
│       ├── io.py                 # 图像读写
│       └── visualize.py          # 关键点绘制
├── models/                       # MediaPipe Pose / rembg U²-Net 模型权重（运行时按需下载）
├── data_picture/
│   ├── clothes/                  # 服装样例
│   └── people/                   # 人体样例
├── scripts/                      # 诊断与回归脚本（详见 scripts/README.md）
│   ├── README.md
│   ├── grid_3x4.py               # 3 人 × 4 衣 全组合回归 contact sheet
│   ├── detect_all_people.py      # 批量人体关键点缓存
│   ├── diagnose_*.py             # 对称性 / scale / body_pts 调参
│   ├── qipao_bump_verify.py      # qipao 斜襟凸块复现
│   └── test_smoothing.py         # Stage B 改动必跑的回归
├── docs/
│   ├── notes/                    # 过程笔记（按 CLAUDE.md 规范分类）
│   └── 虚拟试衣报告.docx          # 课程报告
├── docker/                       # Docker（Dockerfile / docker-compose / 启动脚本 / DOCKER_GUIDE）
├── pyproject.toml                # uv 项目配置（依赖、Python 3.13）
├── uv.lock                       # uv 锁文件
├── requirements.txt              # pip 依赖（备用）
├── .python-version               # 3.13
└── LICENSE                       # MIT
```

`output/` 与 `models/` 通过 `.gitignore` 排除（部分模型除外）。

---

## 🔑 关键点说明

### 人体关键点（MediaPipe 自动检测）

MediaPipe Pose 提供 33 个关键点。试衣流水线实际使用：

| 关键点 | MediaPipe 索引 | 派生方式 | 用途 |
|---|---|---|---|
| `neck` | — | 由双肩中点派生 | Stage A anchor |
| `left_shoulder` / `right_shoulder` | 11 / 12 | 直接 | 对齐 + 肩宽 scale |
| `left_elbow` / `right_elbow` | 13 / 14 | 直接 | （历史对应表保留） |
| `left_hip` / `right_hip` | 23 / 24 | 直接 | 身体轮廓下界 |

其余 27 个关键点由 MediaPipe 输出但试衣流程不消费。

### 服装关键点（自动提取）

8 个语义关键点（`CLOTHING_KEYPOINTS`）：

| 关键点 | 几何来源 |
|---|---|
| `top_center` | 轮廓顶部凹点（领口） |
| `bottom_center` | 轮廓底部中点 |
| `left_shoulder` / `right_shoulder` | 轮廓上 1/4 处左右极值 |
| `left_armpit` / `right_armpit` | 轮廓中 1/3 处左右极值 |
| `left_bottom` / `right_bottom` | 轮廓下 1/4 处左右极值（向内收缩） |

---

## ⚠️ 输入要求

### 人体图像

- ✅ 正面或近似正面站立
- ✅ 全身或上半身可见
- ✅ 光线充足
- ❌ 避免严重遮挡 / 多人同框 / 极端侧身

### 服装图像

- ✅ 背景干净（白色或浅色背景最佳）
- ✅ 服装完整可见、无衣架
- ✅ 优先提供 PNG 透明背景（alpha 信息能简化抠图）
- ❌ 避免复杂背景 / 折叠 / 服装被遮挡

---

## 🧪 回归测试

任何改动 `src/virtual_tryon/warp.py` 后都应该重跑 `scripts/grid_3x4.py`：
3 人 × 4 衣 = 12 组合全跑一遍 + 生成 3×4 拼接 contact sheet，肉眼对照。

```bash
.venv/Scripts/python.exe scripts/grid_3x4.py
```

`scripts/README.md` 列出了每个 `diagnose_*` / `test_*` 脚本对应的问题和历史 commit。

---

## 🆚 与深度学习方法对比

| 维度 | 本系统（传统 CV） | VITON / CP-VTON 等（深度学习） |
|---|---|---|
| 训练数据 | ❌ 不需要 | ✅ 需要大量数据 |
| 速度 | ⚡ 快速（CPU） | 🐌 较慢 |
| 精度 | ⚠️ 中等（标准上衣 > 紧身 / 长裙 <） | ✅ 高 |
| 泛化能力 | ⚠️ 限于上衣 / 标准姿态 | ✅ 强 |
| 可解释性 | ✅ 每一步可视化 | ❌ 黑盒 |
| 资源需求 | ✅ CPU 即可 | ⚠️ GPU |

---

## ❓ 常见问题

**Q: `ModuleNotFoundError: No module named 'cv2'`**
A: `pip install opencv-python opencv-contrib-python`，或 `uv sync`。

**Q: 找不到 `data_picture/.../image.png`**
A: 检查路径与格式（PNG/JPG/JPEG 都可）。`main.py` 不限定文件名，
   通过 `--person` / `--clothing` 显式传入。

**Q: MediaPipe 模型下载慢**
A: 默认从 Google 公共存储下载；若网络受限可手动下载
   `pose_landmarker_full.task` 放到 `models/` 下，代码会自动跳过下载。

**Q: 人体检测不到任何关键点**
A: `RobustHumanDetector` 会自动降级到 Haar / 启发式，保证拿到结果；
   若全部失败会抛异常 — 此时检查图像质量。

**Q: Docker 构建慢或失败**
A: `Dockerfile` 已配置阿里云 apt + pip 镜像；若仍失败参见
   [`docker/DOCKER_GUIDE.md`](docker/DOCKER_GUIDE.md)。

**Q: `virtual_tryon_simple.py` / `virtual_tryon_system.py` 在哪？**
A: 已废弃删除。当前入口是 `main.py`（CLI 子命令）。
   `Dockerfile` 与 `docker-compose.yml` 里仍残留旧入口文件名 — 见已知问题。

---

## 📚 参考文献

1. MediaPipe Pose: <https://google.github.io/mediapipe/solutions/pose.html>

---

## 🤝 贡献

欢迎提出改进建议！过程笔记见 `docs/notes/`。

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

**作者**：CV 课程大作业
**日期**：2026-06-20
**版本**：v2.0