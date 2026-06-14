"""可视化 TPS 对应关系——衣服轮廓点 vs 身体轮廓点在输出空间的位置。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from virtual_tryon import ClothingDetector, RobustHumanDetector
from virtual_tryon.human_detector import body_region_contour
from virtual_tryon.io import ensure_dir, load_image
from virtual_tryon.tps_warp import _align_contours

OUT = ensure_dir(Path(__file__).resolve().parent.parent / "output" / "viz_corresp")
PERSON = "data_picture/people/image.png"

for clothing_name in ["image.png", "image2.jpg"]:
    clothing_path = f"data_picture/clothes/{clothing_name}"
    name = Path(clothing_name).stem

    person_img = load_image(PERSON)
    clothing_img = load_image(clothing_path, with_alpha=True)
    if clothing_img.shape[2] == 4:
        clothing_rgb = clothing_img[:, :, :3]
    else:
        clothing_rgb = clothing_img

    # 人体
    human_det = RobustHumanDetector()
    human_kpts = human_det.detect(person_img)
    body_pts = body_region_contour(human_kpts, n_points=30)

    # 衣服
    clothing_det = ClothingDetector()
    c_pts, c_mask, c_anchor = clothing_det.sample_contour(clothing_img, n_points=30)

    # ---- 仿射矩阵（与 warp_clothing 一致）----
    src_pts = c_pts.astype(np.float32)
    dst_pts = body_pts.astype(np.float32)
    sx_min, sy_min = src_pts[:, 0].min(), src_pts[:, 1].min()
    sx_max, sy_max = src_pts[:, 0].max(), src_pts[:, 1].max()
    dx_min, dy_min = dst_pts[:, 0].min(), dst_pts[:, 1].min()
    dx_max, dy_max = dst_pts[:, 0].max(), dst_pts[:, 1].max()
    src_w = max(sx_max - sx_min, 1)
    src_h = max(sy_max - sy_min, 1)
    dst_w = max(dx_max - dx_min, 1)
    dst_h = max(dy_max - dy_min, 1)
    scale = max(dst_w / src_w, dst_h / src_h) * 1.05
    cx, cy = c_anchor
    nose_y = float(human_kpts["nose"].y)
    neck_x = float(human_kpts["neck"].x)
    neck_y = float(human_kpts["neck"].y)
    bx, by = neck_x, neck_y - max(0.0, (neck_y - nose_y) * 0.30)
    tx = bx - cx * scale
    ty = by - cy * scale
    M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)

    # ---- 仿射后的衣服轮廓点 ----
    src_aff = (M[:, :2] @ src_pts.T + M[:, 2:3]).T

    # ---- 对齐后的对应 ----
    s_aligned, d_aligned = _align_contours(src_aff.astype(np.float64), dst_pts.astype(np.float64))
    s_aligned = s_aligned.astype(np.int32)
    d_aligned = d_aligned.astype(np.int32)
    n = min(len(s_aligned), len(d_aligned))

    # ---- 图1：人体图 + 身体轮廓点(绿编号) + 仿射后衣服轮廓点(红编号) ----
    vis = person_img.copy()
    for i in range(n):
        dx, dy = d_aligned[i]
        cv2.circle(vis, (dx, dy), 5, (0, 255, 0), -1)
        cv2.putText(vis, str(i), (dx + 6, dy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1)
        sx, sy = s_aligned[i]
        cv2.circle(vis, (sx, sy), 5, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (sx + 6, sy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)
    cv2.imwrite(str(OUT / f"{name}_both_contours.jpg"), vis)

    # ---- 图2：对应连线 ----
    vis2 = person_img.copy()
    colors = [(0, 165, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
              (255, 0, 255), (0, 255, 255)]
    for i in range(0, n, 2):  # 每 2 个点画一条线
        sx, sy = s_aligned[i]
        dx, dy = d_aligned[i]
        color = colors[(i // 2) % len(colors)]
        cv2.line(vis2, (sx, sy), (dx, dy), color, 1)
        cv2.circle(vis2, (sx, sy), 4, (0, 0, 255), -1)
        cv2.circle(vis2, (dx, dy), 4, (0, 255, 0), -1)
    cv2.imwrite(str(OUT / f"{name}_correspondence.jpg"), vis2)

    # ---- 打印数据 ----
    print(f"\n=== {name} ===")
    print(f"衣服点范围(仿射后): x=[{src_aff[:, 0].min():.0f}, {src_aff[:, 0].max():.0f}], "
          f"y=[{src_aff[:, 1].min():.0f}, {src_aff[:, 1].max():.0f}]")
    print(f"身体点范围: x=[{dst_pts[:, 0].min():.0f}, {dst_pts[:, 0].max():.0f}], "
          f"y=[{dst_pts[:, 1].min():.0f}, {dst_pts[:, 1].max():.0f}]")
    print("对齐后前10对对应:")
    for i in range(min(10, n)):
        print(f"  [{i:2d}] 衣=({s_aligned[i, 0]:4d},{s_aligned[i, 1]:4d})  "
              f"→ 体=({d_aligned[i, 0]:4d},{d_aligned[i, 1]:4d})  "
              f"Δ=({d_aligned[i, 0]-s_aligned[i, 0]:4d},{d_aligned[i, 1]-s_aligned[i, 1]:4d})")
