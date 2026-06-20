# scripts/ — 诊断与回归脚本

这一目录下的脚本是开发过程中用来验证 warp / 检测 / scale 行为的**一次性诊断工具**。它们不是产品代码，但保留下来作为以后类似问题复现 / 回归测试的参考。

所有脚本都用同一种方式跑：

```bash
.venv/Scripts/python.exe scripts/<script_name>.py
```

输出默认落到 `output/<dir>/`，可用 `--help` 看参数（如果脚本支持）。

---

## 单人 / 多人检测诊断

| 脚本 | 用途 |
|---|---|
| `detect_all_people.py` | 批量跑 `data_picture/people/` 下所有人图的 MediaPipe 检测，落 jpg/json 缓存（缓存规则见 `human_cache.py`） |
| `diagnose_all_people.py` | 对已缓存的 keypoints 跑 `body_region_contour`，量化 4 个对称性指标（肩 y 差、肩 x 对称、脖子中点偏置、轮廓质心偏置） |
| `diagnose_grayscale.py` | 跑灰度图 + 直方图，验证 MediaPipe 对彩色 / 灰度的输入要求 |

## Body polygon 调参

| 脚本 | 用途 |
|---|---|
| `diagnose_body_pts.py` | 对比 `expand_ratio=0.0 / 0.05 / 0.15` 三种参数下 `body_region_contour` 的多边形左右差异 |
| `diagnose_body_pts_mirror.py` | 镜像翻转输入 keypoints（左/右 swap + 水平 flip x），用来判断左右不对称是 MediaPipe 偏置还是算法偏置 |

## Scale 策略对比

| 脚本 | 用途 |
|---|---|
| `diagnose_scales.py` | 量化 12 组合（3 人 × 4 衣）的 Stage A `scale = max(dst_w/src_w, dst_h/src_h) * 1.05`，看 scale 跨人 / 跨衣的方差 |
| `diagnose_scales_v2.py` | 对比 bbox-based vs shoulder-width-based 两套 scale 矩阵，输出表格 |

## Qipao 凸块专项（commit 08f0d09 的来源）

| 脚本 | 用途 |
|---|---|
| `diagnose_qipao_bump.py` | 量化 qipao mask 自身对称性 + 3 人的 Stage B `s(y)` 序列；最先怀疑是 mask 本身或 s 跳变 |
| `qipao_bump_verify.py` | 用 `det.sample_contour` 返回的真 mask（不是 PNG alpha）画 silhouette + 30 点 polygon；证实 mask 本身没凸块，凸块是 Stage B 制造的 |
| `test_smoothing.py` | 跑 `qipao × 3 人` 完整 pipeline + 生成 contact sheet，每次改 `_warp_flow` 都要重跑一次对照 |

## Cross-test 网格

| 脚本 | 用途 |
|---|---|
| `grid_3x4.py` | 3 人 × 4 衣 = 12 组合全跑一遍 + 生成 3×4 拼接 contact sheet。改 warp 后必跑 |

---

## 历史背景

- `commit 86a8813` 引入 `cap=1.10` 防止紧身衣过度放大 → `diagnose_scales_v2.py` 是它的 scale 验证
- `commit 08f0d09` 引入 silhouette 对 body_center 强制对称消除 qipao 斜襟凸块 → `qipao_bump_verify.py` + `test_smoothing.py` 是它的复现脚本
- `grid_3x4.py` 是任何 warp 改动后的回归 baseline
