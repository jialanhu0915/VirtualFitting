"""验证：qipao 真实 mask（不是 alpha）和 sample 出的 30 点 polygon 是否对称。

分三步：
  1. 画 sample_contour 返回的 mask（不是 PNG alpha）—— 真实分割结果
  2. 画 30 点 polygon 叠加在 mask 上 —— 看采样是否引入噪声
  3. Stage A 单独跑（不开 Stage B），看仿射后 mask 是否已经有凸块
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from virtual_tryon import ClothingDetector
from virtual_tryon.human_detector import body_region_contour
from virtual_tryon.io import load_image

CLOTHES = ROOT / "data_picture" / "clothes"
PEOPLE = ROOT / "data_picture" / "people"
OUT = ROOT / "output" / "diag_qipao"
OUT.mkdir(parents=True, exist_ok=True)


# --- Step 1: 真实 mask ---
qipao_path = CLOTHES / "image.png"
img = load_image(qipao_path, with_alpha=True)
print(f"qipao 通道数: {img.shape[2]}")
print(f"qipao alpha 唯一值: {np.unique(img[:, :, 3])[:5]} ... ({len(np.unique(img[:, :, 3]))} 个)")

det = ClothingDetector()
pts, mask, _, kps = det.sample_contour(img, n_points=30)

# 看 mask 每行的 [left, right, w, center]
ys = np.where(mask.any(axis=1))[0]
print(f"\nmask y 范围: {ys[0]} - {ys[-1]}  (height={ys[-1] - ys[0] + 1})")
print(f"\n{'y':>5} {'left':>5} {'right':>6} {'w':>5} {'cx':>6} "
      f"{'cx-bbox_cx':>10}")
bbox_cx = (pts[:, 0].max() + pts[:, 0].min()) / 2
sample_ys = list(range(ys[0], ys[-1] + 1, 30))
for y in sample_ys:
    row = mask[y] > 0
    if not row.any():
        continue
    xs = np.where(row)[0]
    L, R, W = int(xs[0]), int(xs[-1]), int(xs[-1] - xs[0] + 1)
    CX = (L + R) / 2
    print(f"{y:>5} {L:>5} {R:>6} {W:>5} {CX:>6.1f} {CX - bbox_cx:>+10.1f}")

# 保存：mask 单独可视化（不画 polygon）
mask_only = np.zeros_like(img[:, :, :3])
mask_only[mask > 0] = (0, 255, 0)
mask_only = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if False else mask_only
cv2.imwrite(str(OUT / "01_mask_only.png"), mask_only)

# --- Step 2: 30 点 polygon 叠加在 mask 上 ---
polygon_vis = mask_only.copy()
pts_i = pts.astype(np.int32).reshape((-1, 1, 2))
cv2.polylines(polygon_vis, [pts_i], isClosed=True, color=(0, 0, 255), thickness=2)
# 标顶点
for i, (x, y) in enumerate(pts):
    cv2.circle(polygon_vis, (int(x), int(y)), 4, (255, 0, 255), -1)
    cv2.putText(polygon_vis, str(i), (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
cv2.imwrite(str(OUT / "02_mask_plus_polygon.png"), polygon_vis)

# --- Step 3: Stage A 单独跑（不开 Stage B），仿射后看 mask ---
print("\n=== Step 3: Stage A 单独跑（仿射后 mask） ===")
import json

# 用 image（人）作为 dst
person_path = PEOPLE / "image.png"
person_img = load_image(person_path)
H, W = person_img.shape[:2]
data = json.loads((person_path.parent / f"{person_path.stem}.keypoints.json").read_text())
from virtual_tryon.keypoints import Keypoint
kpts = {n: Keypoint(int(d["x"]), int(d["y"]), float(d.get("confidence", 1)), n)
        for n, d in data["keypoints"].items()}

# shoulder anchor
cls_x = float(kps["left_shoulder"].x)
crs_x = float(kps["right_shoulder"].x)
cls_y = float(kps["left_shoulder"].y)
crs_y = float(kps["right_shoulder"].y)
cloth_anchor = ((cls_x + crs_x) / 2, (cls_y + crs_y) / 2)

bls_x = float(kpts["left_shoulder"].x)
brs_x = float(kpts["right_shoulder"].x)
bls_y = float(kpts["left_shoulder"].y)
brs_y = float(kpts["right_shoulder"].y)
body_anchor = ((bls_x + brs_x) / 2, (bls_y + brs_y) / 2)

body_pts = body_region_contour(kpts, n_points=30, expand_ratio=0.05)
bw = body_pts[:, 0].max() - body_pts[:, 0].min()
bh = body_pts[:, 1].max() - body_pts[:, 1].min()
src_w = max(pts[:, 0].max() - pts[:, 0].min(), 1)
src_h = max(pts[:, 1].max() - pts[:, 1].min(), 1)
scale = min(max(bw / src_w, bh / src_h) * 1.05, 1.10)
print(f"scale={scale:.3f}  body=({bw:.0f}x{bh:.0f})  src=({src_w:.0f}x{src_h:.0f})")
print(f"cloth_anchor=({cloth_anchor[0]:.0f},{cloth_anchor[1]:.0f})  "
      f"body_anchor=({body_anchor[0]:.0f},{body_anchor[1]:.0f})")

cx, cy = cloth_anchor
bx, by = body_anchor
tx = bx - cx * scale
ty = by - cy * scale
M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
print(f"M = scale={scale:.3f}  tx={tx:.1f}  ty={ty:.1f}")

stage_a_mask = cv2.warpAffine(mask, M, (W, H),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)

# 量化 Stage A 仿射后 mask 的对称性
sa_ys = np.where(stage_a_mask.any(axis=1))[0]
print(f"\nStage A 后 mask y 范围: {sa_ys[0]} - {sa_ys[-1]}")
print(f"\n{'y':>5} {'left':>5} {'right':>6} {'w':>5} {'cx':>6} "
      f"{'shift_to_body_cx':>17}")

# body 中心
body_cx = (body_pts[:, 0].max() + body_pts[:, 0].min()) / 2
for y in range(int(sa_ys[0]), int(sa_ys[-1]) + 1, 30):
    row = stage_a_mask[y] > 0
    if not row.any():
        continue
    xs = np.where(row)[0]
    L, R, W = int(xs[0]), int(xs[-1]), int(xs[-1] - xs[0] + 1)
    CX = (L + R) / 2
    print(f"{y:>5} {L:>5} {R:>6} {W:>5} {CX:>6.1f} {CX - body_cx:>+17.1f}")

# 保存 Stage A 仿射后的 mask
stage_a_vis = person_img.copy()
stage_a_vis[stage_a_mask > 0] = (
    stage_a_vis[stage_a_mask > 0] * 0.5
    + np.array([0, 255, 255], dtype=np.float32) * 0.5
).astype(np.uint8)
cv2.imwrite(str(OUT / "03_stage_a_only.png"), stage_a_vis)
print(f"\n输出到 {OUT}/")
