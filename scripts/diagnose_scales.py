"""量化 12 张组合的 Stage A scale 差异。

每张图：
  - 衣服 bbox (src_w, src_h)
  - 人体 body_pts bbox (dst_w, dst_h)
  - scale = max(dst_w/src_w, dst_h/src_h) * 1.05

输出：表格化对比 4 件衣服在 3 个人身上的 scale，看是否一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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


people = sorted(p for p in PEOPLE.glob("image*")
                if ".keypoints" not in p.name and p.suffix.lower() in (".png", ".jpg"))
clothes = sorted(c for c in CLOTHES.glob("image*")
                 if c.suffix.lower() in (".png", ".jpg"))

print(f"people: {[p.stem for p in people]}")
print(f"clothes: {[c.stem for c in clothes]}")

# 衣服 bbox
print("\n=== 衣服 bbox (src) ===")
clothing_bboxes = {}
det = ClothingDetector()
for c in clothes:
    img = load_image(c, with_alpha=True)
    pts, mask, _, _ = det.sample_contour(img, n_points=30)
    w = float(pts[:, 0].max() - pts[:, 0].min())
    h = float(pts[:, 1].max() - pts[:, 1].min())
    clothing_bboxes[c.stem] = (w, h)
    print(f"  {c.stem:>10}  w={w:>6.1f}  h={h:>6.1f}  ratio={h/w:.2f}")

# 人体 body_pts bbox
print("\n=== 人体 body_pts bbox (dst) ===")
body_bboxes = {}
for p in people:
    kpts = load_kpts(p)
    bpts = body_region_contour(kpts, n_points=30, expand_ratio=0.05)
    w = float(bpts[:, 0].max() - bpts[:, 0].min())
    h = float(bpts[:, 1].max() - bpts[:, 1].min())
    body_bboxes[p.stem] = (w, h)
    print(f"  {p.stem:>10}  w={w:>6.1f}  h={h:>6.1f}  ratio={h/w:.2f}")

# 12 张 scale 表
print("\n=== Stage A scale (max(dst_w/src_w, dst_h/src_h) * 1.05) ===")
print(f"  {'':>10} | " + " | ".join(f"{c.stem:>8}" for c in clothes))
print("  " + "-" * 70)
for p in people:
    dw, dh = body_bboxes[p.stem]
    row = f"  {p.stem:>10} | "
    scales = []
    for c in clothes:
        sw, sh = clothing_bboxes[c.stem]
        s = max(dw / sw, dh / sh) * 1.05
        scales.append(s)
        row += f"{s:>8.3f} | "
    print(row)
    print(f"  {'':>10}   (variance: {max(scales) - min(scales):.3f}, "
          f"max/min: {max(scales) / min(scales):.2f}x)")

# 4 件衣服的「期望物理尺寸」分析
print("\n=== 假设衣服物理尺寸固定：scale 应只取决于衣服本身，跨人不变 ===")
print("但当前实现 scale 同时受衣服 bbox 和 body_pts bbox 影响：")
print("  - 同件衣服在更宽/高的人身上会被放得更大")
print("  - 同件衣服在更窄/矮的人身上会被缩得更小")
print("\n这是为什么你看 contact_sheet 时：")
print("  - 同一件 qipao (col 1) 在 3 个人身上长度差异明显")
print("  - 同一件粉色 T (col 2) 在 image3 人身上显得过短")
