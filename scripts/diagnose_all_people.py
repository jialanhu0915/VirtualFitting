"""三方对比 3 张人物图的 body_pts 对称性。

每张图读自己的 cache，跑 body_region_contour，量化 4 个对称性指标：
  1. 肩膀 y 差       (lsh.y - rsh.y)
  2. 肩膀 x 对称     (cx - rsh.x) - (lsh.x - cx)
  3. 肘 / 腕 x 对称   同上
  4. 髋 x 对称       同上
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from virtual_tryon.human_detector import body_region_contour
from virtual_tryon.io import load_image, save_image
from virtual_tryon.keypoints import Keypoint


PEOPLE = ROOT / "data_picture" / "people"
OUT = ROOT / "output" / "diagnose"
OUT.mkdir(parents=True, exist_ok=True)


def load_kpts(json_path: Path) -> dict[str, Keypoint]:
    data = json.loads(json_path.read_text())
    return {n: Keypoint(int(d["x"]), int(d["y"]), float(d.get("confidence", 1.0)), n)
            for n, d in data["keypoints"].items()}


def report(label: str, kpts: dict[str, Keypoint], img: np.ndarray) -> None:
    h, w = img.shape[:2]
    ls = kpts["left_shoulder"]; rs = kpts["right_shoulder"]
    le = kpts["left_elbow"];    re = kpts["right_elbow"]
    lw = kpts["left_wrist"];    rw = kpts["right_wrist"]
    lh = kpts["left_hip"];      rh = kpts["right_hip"]
    cx = (ls.x + rs.x) / 2
    print(f"\n{'=' * 70}")
    print(f"  {label}  ({w}x{h}, body_cx={cx:.0f})")
    print(f"{'=' * 70}")
    print(f"  {'关键点':<10} {'L (image右)':>16} {'R (image左)':>16} "
          f"{'x 距 cx':>20} {'y_Δ':>6}")
    for name, lk, rk in [
        ("shoulder", ls, rs), ("elbow", le, re),
        ("wrist", lw, rw), ("hip", lh, rh),
    ]:
        ldx = cx - lk.x     # image 右顶点到 cx 的距离（正值 = 在 cx 右侧）
        rdx = rk.x - cx     # image 左顶点到 cx 的距离（正值 = 在 cx 左侧）
        dyx = abs(rdx - ldx)  # x 距离差
        dyy = rk.y - lk.y
        print(f"  {name:<10} ({lk.x:>4},{lk.y:>4})  ({rk.x:>4},{rk.y:>4})  "
              f"L:{ldx:>+6.1f} R:{rdx:>+6.1f} Δx:{dyx:>5.1f}  {dyy:>+5.1f}")

    # 跑 30 点 resample，对比 4 个 y 层的左右宽度
    sampled = body_region_contour(kpts, n_points=30, expand_ratio=0.05)
    print(f"\n  polygon 在关键 y 层的左右宽度 (expand=0.05):")
    for target_y in [440, 530, 620, 740]:  # 肩/腋/肘/胯
        # 找左右最近点
        left_pts = [(i, p) for i, p in enumerate(sampled) if p[0] < cx and abs(p[1] - target_y) < 30]
        right_pts = [(i, p) for i, p in enumerate(sampled) if p[0] > cx and abs(p[1] - target_y) < 30]
        if not left_pts or not right_pts:
            continue
        # 取最接近 target_y 的左右各一
        li, lp = min(left_pts, key=lambda t: abs(t[1][1] - target_y))
        ri, rp = min(right_pts, key=lambda t: abs(t[1][1] - target_y))
        ldx = cx - lp[0]
        rdx = rp[0] - cx
        print(f"    y≈{target_y:>3}  L:{lp[0]:>5.1f} (idx {li:>2})  R:{rp[0]:>5.1f} (idx {ri:>2})  "
              f"Lx距:{ldx:>6.1f}  Rx距:{rdx:>6.1f}  Δ:{rdx - ldx:>+5.1f}")

    # overlay
    vis = img.copy()
    cv2.line(vis, (int(cx), 0), (int(cx), h), (255, 0, 0), 1)
    pts = sampled.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    for kp in kpts.values():
        cv2.circle(vis, (kp.x, kp.y), 6, (0, 0, 255), -1)
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
    save_image(OUT / f"all3_{label}.jpg", vis)


def main() -> None:
    images = []
    for p in sorted(PEOPLE.glob("image*")):
        if ".keypoints" in p.name or p.suffix.lower() not in (".png", ".jpg"):
            continue
        cache = PEOPLE / f"{p.stem}.keypoints.json"
        if not cache.exists():
            continue
        images.append((p.stem, p, cache))

    for stem, img_path, cache in images:
        kpts = load_kpts(cache)
        img = load_image(img_path)
        report(stem, kpts, img)

    print(f"\n[done] 详见 {OUT}")


if __name__ == "__main__":
    main()
