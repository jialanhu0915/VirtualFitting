"""对比新旧 scale 策略：bbox-based vs shoulder-width-based。

复算两套 scale 矩阵：
  - 旧: max(dst_w/src_w, dst_h/src_h) * 1.05
  - 新: (body_shoulder_w / cloth_shoulder_w) * 1.05

输入仍是同一组 keypoints。
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

# 衣服 bbox + shoulder 宽
print("\n=== 衣服信息 ===")
clothing_info = {}
det = ClothingDetector()
for c in clothes:
    img = load_image(c, with_alpha=True)
    pts, mask, _, kps = det.sample_contour(img, n_points=30)
    w = float(pts[:, 0].max() - pts[:, 0].min())
    h = float(pts[:, 1].max() - pts[:, 1].min())
    sw = abs(float(kps["left_shoulder"].x) - float(kps["right_shoulder"].x))
    clothing_info[c.stem] = {"w": w, "h": h, "shoulder_w": sw}
    print(f"  {c.stem:>10}  bbox={w:>6.0f}x{h:>6.0f}  shoulder_w={sw:>5.0f}")

# 人体 bbox + shoulder 宽
print("\n=== 人体信息 ===")
body_info = {}
for p in people:
    kpts = load_kpts(p)
    bpts = body_region_contour(kpts, n_points=30, expand_ratio=0.05)
    w = float(bpts[:, 0].max() - bpts[:, 0].min())
    h = float(bpts[:, 1].max() - bpts[:, 1].min())
    sw = abs(float(kpts["left_shoulder"].x) - float(kpts["right_shoulder"].x))
    body_info[p.stem] = {"w": w, "h": h, "shoulder_w": sw}
    print(f"  {p.stem:>10}  bbox={w:>6.0f}x{h:>6.0f}  shoulder_w={sw:>5.0f}")

# 旧 scale (bbox-based)
print("\n=== 旧 scale: max(dst_w/src_w, dst_h/src_h) * 1.05 ===")
print(f"  {'':>10} | " + " | ".join(f"{c.stem:>8}" for c in clothes))
for p in people:
    dw, dh = body_info[p.stem]["w"], body_info[p.stem]["h"]
    row = f"  {p.stem:>10} | "
    scales = []
    for c in clothes:
        sw, sh = clothing_info[c.stem]["w"], clothing_info[c.stem]["h"]
        s = max(dw / sw, dh / sh) * 1.05
        scales.append(s)
        row += f"{s:>8.3f} | "
    print(row)
    print(f"  {'':>10}   variance={max(scales) - min(scales):.3f}, "
          f"max/min={max(scales) / min(scales):.2f}x")

# 新 scale (shoulder-based)
print("\n=== 新 scale: body_shoulder / cloth_shoulder * 1.05 ===")
print(f"  {'':>10} | " + " | ".join(f"{c.stem:>8}" for c in clothes))
for p in people:
    bsw = body_info[p.stem]["shoulder_w"]
    row = f"  {p.stem:>10} | "
    scales = []
    for c in clothes:
        csw = clothing_info[c.stem]["shoulder_w"]
        s = (bsw / csw) * 1.05
        scales.append(s)
        row += f"{s:>8.3f} | "
    print(row)
    print(f"  {'':>10}   variance={max(scales) - min(scales):.3f}, "
          f"max/min={max(scales) / min(scales):.2f}x")
