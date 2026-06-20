"""分析 qipao 左侧凸块：量化 cloth mask 本身的不对称性 + 三个人的 s(y) 序列。

输出：
  - qipao mask 在每个 y 行的 [左, 右, 宽, 中心] — 看源图本身是否对称
  - 三个人 (image, image2, image3) 在 torso 区域 y 的 Stage B s(y) 序列
    — 看 fit 是否把不对称放大
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from virtual_tryon import ClothingDetector
from virtual_tryon.human_detector import body_region_contour
from virtual_tryon.io import load_image
from virtual_tryon.keypoints import Keypoint

PEOPLE = ROOT / "data_picture" / "people"
CLOTHES = ROOT / "data_picture" / "clothes"


def load_kpts(p: Path) -> dict[str, Keypoint]:
    data = json.loads((PEOPLE / f"{p.stem}.keypoints.json").read_text())
    return {n: Keypoint(int(d["x"]), int(d["y"]), float(d.get("confidence", 1)), n)
            for n, d in data["keypoints"].items()}


# --- Part 1: qipao mask 本身的对称性 ---
print("=" * 70)
print("Part 1: qipao mask 自身的不对称性")
print("=" * 70)

qipao_path = CLOTHES / "image.png"
img = load_image(qipao_path, with_alpha=True)
det = ClothingDetector()
pts, mask, _, kps = det.sample_contour(img, n_points=30)

# 提取 alpha 作为 mask（如果带 alpha）；否则用 mask
if img.shape[2] == 4:
    alpha = img[:, :, 3]
    src_mask = (alpha > 0).astype(np.uint8) * 255
else:
    src_mask = mask

ys_with_cloth = np.where(src_mask.any(axis=1))[0]
print(f"qipao 像素 y 范围: {ys_with_cloth[0]} – {ys_with_cloth[-1]}  "
      f"(height={ys_with_cloth[-1] - ys_with_cloth[0] + 1} px)")
print(f"qipao 整体 bbox: w={pts[:, 0].max()-pts[:, 0].min():.0f}  "
      f"h={pts[:, 1].max()-pts[:, 1].min():.0f}")
print(f"qipao 中心 x: {(pts[:, 0].max() + pts[:, 0].min()) / 2:.0f}")

print("\n按 y 行扫描 mask 宽度 + 中心（每隔 50px 采样）：")
print(f"  {'y':>5} {'left':>5} {'right':>6} {'w':>5} {'cx':>5} "
      f"{'cx_to_bbox_cx':>13}")
bbox_cx = (pts[:, 0].max() + pts[:, 0].min()) / 2
for y in range(ys_with_cloth[0], ys_with_cloth[-1] + 1, 50):
    row = src_mask[y] > 0
    if not row.any():
        continue
    xs = np.where(row)[0]
    L, R, W, CX = int(xs[0]), int(xs[-1]), int(xs[-1] - xs[0] + 1), (xs[0] + xs[-1]) / 2
    print(f"  {y:>5} {L:>5} {R:>6} {W:>5} {CX:>5.1f} {CX - bbox_cx:>+13.1f}")

# 量化"左侧凸出"：在 y ∈ [mid_height - 100, mid_height + 100] 范围内
# 计算左半和右半到 bbox_cx 的最大距离差
mid_y = (ys_with_cloth[0] + ys_with_cloth[-1]) // 2
print(f"\n中段 (y ∈ [{mid_y - 100}, {mid_y + 100}]) 单边最远距离：")
max_left_dist = 0.0
max_right_dist = 0.0
max_left_y = max_right_y = mid_y
for y in range(mid_y - 100, mid_y + 101):
    row = src_mask[y] > 0
    if not row.any():
        continue
    xs = np.where(row)[0]
    l_dist = bbox_cx - xs[0]
    r_dist = xs[-1] - bbox_cx
    if l_dist > max_left_dist:
        max_left_dist = l_dist
        max_left_y = y
    if r_dist > max_right_dist:
        max_right_dist = r_dist
        max_right_y = y
print(f"  左侧最远: {max_left_dist:.1f}px @ y={max_left_y}")
print(f"  右侧最远: {max_right_dist:.1f}px @ y={max_right_y}")
print(f"  不对称: 左侧 - 右侧 = {max_left_dist - max_right_dist:+.1f}px  "
      f"(正值 = 左侧凸出更多)")

# --- Part 2: 三个人 + qipao 的 Stage B s(y) 序列 ---
print("\n" + "=" * 70)
print("Part 2: qipao 在 3 个人身上的 Stage B s(y)（torso 区域）")
print("=" * 70)

people = sorted(p for p in PEOPLE.glob("image*")
                if ".keypoints" not in p.name and p.suffix.lower() in (".png", ".jpg"))

# 复用 main.py 的 run 流程：调 warp_clothing 看 log
# 但 log 是用 logging 输出的；更简单的办法是手动模拟 Stage B 的 s(y) 序列。
# 重新计算：取 Stage A 仿射后的 mask 在每行 cloth_width，按 body_w/cloth_w 求 s

for p in people:
    kpts = load_kpts(p)
    body_pts = body_region_contour(kpts, n_points=30, expand_ratio=0.05)
    bw = body_pts[:, 0].max() - body_pts[:, 0].min()
    bh = body_pts[:, 1].max() - body_pts[:, 1].min()

    # 重新跑 Stage A 仿射
    H, W = load_image(p).shape[:2]
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

    src_w = max(pts[:, 0].max() - pts[:, 0].min(), 1)
    src_h = max(pts[:, 1].max() - pts[:, 1].min(), 1)
    dst_w = bw
    dst_h = bh
    scale = min(max(dst_w / src_w, dst_h / src_h) * 1.05, 1.10)
    cx, cy = cloth_anchor
    bx, by = body_anchor
    tx = bx - cx * scale
    ty = by - cy * scale
    M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)

    # 抠出 clothing_fg + mask
    clothing_img = img
    if clothing_img.shape[2] == 4:
        clothing_rgb = clothing_img[:, :, :3]
    else:
        clothing_rgb = clothing_img
    mask_3ch = mask[:, :, np.newaxis] / 255.0
    clothing_fg = (clothing_rgb.astype(np.float32) * mask_3ch).astype(np.uint8)

    stage_a_rgb = cv2.warpAffine(clothing_fg, M, (W, H),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
    stage_a_mask = cv2.warpAffine(mask, M, (W, H),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)

    # 算 torso 区域 s(y)
    print(f"\n--- {p.stem} (body bbox w={bw:.0f} h={bh:.0f}, scale={scale:.3f}) ---")

    # 用 _body_silhouette_per_row 算 body left/right
    from virtual_tryon.warp import _body_silhouette_per_row

    # 采样 y：从 cloth 顶到 body 底
    rows_with_cloth = np.where(stage_a_mask.any(axis=1))[0]
    if len(rows_with_cloth) == 0:
        print("  (no cloth pixels)")
        continue
    top_y = float(rows_with_cloth[0])
    body_bot = float(body_pts[:, 1].max())
    n_samples = 30
    ys_q = np.linspace(top_y, max(body_bot, top_y + 1), n_samples)

    bl, br = _body_silhouette_per_row(body_pts, ys_q)
    # cloth silhouette（只 torso 区域）
    torso_x_lo = body_pts[:, 0].min() + 0.20 * (body_pts[:, 0].max() - body_pts[:, 0].min())
    torso_x_hi = body_pts[:, 0].min() + 0.80 * (body_pts[:, 0].max() - body_pts[:, 0].min())
    cl, cr = np.full(n_samples, -1.0), np.full(n_samples, -1.0)
    for i, y in enumerate(ys_q):
        yi = int(round(y))
        if yi < 0 or yi >= H:
            continue
        row = (stage_a_mask[yi] > 0)
        xs = np.where(row & (np.arange(W) >= torso_x_lo) & (np.arange(W) <= torso_x_hi))[0]
        if len(xs) > 0:
            cl[i] = float(xs[0])
            cr[i] = float(xs[-1])

    s_y = np.ones(n_samples)
    valid = np.isfinite(bl) & np.isfinite(br) & (cl >= 0) & (cr > cl) & (br > bl)
    bw_arr = br - bl
    cw_arr = cr - cl
    s_y[valid] = bw_arr[valid] / cw_arr[valid]
    s_y[valid] = np.clip(s_y[valid], 0.7, 1.5)

    # 算 body center & cloth center per row
    bcx = (bl + br) / 2
    ccx = (cl + cr) / 2
    center_shift = np.where(valid, ccx - bcx, 0.0)

    print(f"  {'y_query':>8} {'body_w':>7} {'cloth_w':>8} {'s':>5} "
          f"{'body_cx':>8} {'cloth_cx':>9} {'shift':>7}")
    for i, y in enumerate(ys_q):
        if not valid[i]:
            print(f"  {y:>8.1f}    (invalid)")
            continue
        print(f"  {y:>8.1f} {bw_arr[i]:>7.1f} {cw_arr[i]:>8.1f} {s_y[i]:>5.2f} "
              f"{bcx[i]:>8.1f} {ccx[i]:>9.1f} {center_shift[i]:>+7.1f}")
