"""可视化衣服轮廓点和身体轮廓点分布，用于调试 TPS 对应关系。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from virtual_tryon import ClothingDetector, RobustHumanDetector
from virtual_tryon.human_detector import body_region_contour
from virtual_tryon.io import ensure_dir, load_image

OUT = ensure_dir(Path(__file__).resolve().parent.parent / "output" / "viz_points")
PERSON = "data_picture/people/image.png"

for clothing_name in ["image.png", "image2.jpg"]:
    clothing_path = f"data_picture/clothes/{clothing_name}"
    name = Path(clothing_name).stem

    # 加载图像
    person_img = load_image(PERSON)
    clothing_img = load_image(clothing_path, with_alpha=True)

    # 人体轮廓点
    human_det = RobustHumanDetector()
    human_kpts = human_det.detect(person_img)
    body_pts = body_region_contour(human_kpts, n_points=30)

    # 衣服轮廓点
    clothing_det = ClothingDetector()
    c_pts, c_mask, c_anchor, c_kpts = clothing_det.sample_contour(clothing_img, n_points=30)

    # ---- 图1: 人体图 + 身体轮廓点 ----
    vis_h = person_img.copy()
    for i, (x, y) in enumerate(body_pts):
        cv2.circle(vis_h, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.putText(vis_h, str(i), (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.imwrite(str(OUT / f"{name}_body_points.jpg"), vis_h)

    # ---- 图2: 衣服图 + 轮廓采样点 ----
    if clothing_img.shape[2] == 4:
        vis_c = cv2.cvtColor(clothing_img, cv2.COLOR_BGRA2BGR)
    else:
        vis_c = clothing_img.copy()
    for i, (x, y) in enumerate(c_pts):
        cv2.circle(vis_c, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(vis_c, str(i), (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    # 画领口锚点
    cv2.circle(vis_c, (int(c_anchor[0]), int(c_anchor[1])), 12, (255, 255, 0), 2)
    cv2.imwrite(str(OUT / f"{name}_cloth_points.jpg"), vis_c)

    # ---- 图3: 人体图 + 轮廓多边形连线（看清楚顺序）----
    vis_poly = person_img.copy()
    pts = body_pts.astype(np.int32)
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        cv2.line(vis_poly, tuple(pts[i]), tuple(pts[j]), (0, 255, 255), 2)
        cv2.circle(vis_poly, tuple(pts[i]), 5, (0, 255, 0), -1)
    cv2.imwrite(str(OUT / f"{name}_body_polygon.jpg"), vis_poly)

    # ---- 图4: 衣服轮廓多边形 ----
    if clothing_img.shape[2] == 4:
        vis_cpoly = cv2.cvtColor(clothing_img, cv2.COLOR_BGRA2BGR)
    else:
        vis_cpoly = clothing_img.copy()
    cpi = c_pts.astype(np.int32)
    for i in range(len(cpi)):
        j = (i + 1) % len(cpi)
        cv2.line(vis_cpoly, tuple(cpi[i]), tuple(cpi[j]), (0, 255, 255), 2)
    # 标第0点（领口附近）
    cv2.circle(vis_cpoly, tuple(cpi[0]), 10, (255, 0, 0), -1)
    cv2.putText(vis_cpoly, "0", (cpi[0, 0] + 8, cpi[0, 1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite(str(OUT / f"{name}_cloth_polygon.jpg"), vis_cpoly)

    print(f"{name}:")
    print(f"  衣服点范围: x=[{c_pts[:, 0].min()}, {c_pts[:, 0].max()}], "
          f"y=[{c_pts[:, 1].min()}, {c_pts[:, 1].max()}]")
    print(f"  身体点范围: x=[{body_pts[:, 0].min()}, {body_pts[:, 0].max()}], "
          f"y=[{body_pts[:, 1].min()}, {body_pts[:, 1].max()}]")
    print(f"  领口锚点: ({c_anchor[0]:.0f}, {c_anchor[1]:.0f})")
    print(f"  身体脖子: ({human_kpts['neck'].x}, {human_kpts['neck'].y})")
    print()
