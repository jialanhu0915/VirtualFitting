"""诊断 body_region_contour 输出的左右对称性。

不修改主代码。直接调用 body_region_contour，对比 expand_ratio=0.05 和 0.15
两种参数下的多边形左右差异。

输出：
  - 终端打印 10 顶点 / 30 采样点的「到 cx 的距离」对比
  - 落盘两张 overlay：verts_only.jpg、verts_expand.jpg
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
from virtual_tryon.keypoints import Keypoint
from virtual_tryon.io import load_image, save_image


PEOPLE = ROOT / "data_picture" / "people" / "image.png"
JSON = ROOT / "data_picture" / "people" / "image.keypoints.json"
OUT = ROOT / "output" / "diagnose"
OUT.mkdir(parents=True, exist_ok=True)


def load_kpts(path: Path) -> dict[str, Keypoint]:
    """读缓存的 MediaPipe keypoints JSON。"""
    data = json.loads(path.read_text())
    out: dict[str, Keypoint] = {}
    for name, d in data["keypoints"].items():
        out[name] = Keypoint(
            x=int(d["x"]), y=int(d["y"]),
            confidence=float(d.get("confidence", 1.0)),
            name=name,
        )
    return out


def report_polygon(label: str, sampled: np.ndarray, body_cx: float) -> None:
    """把 30 个采样点按 y 排序后，逐个打印「到 body_cx 的距离」。"""
    print(f"\n=== {label} (body_cx={body_cx:.1f}) ===")
    print(f"{'idx':>4} {'x':>5} {'y':>5} {'dist_from_cx':>13} {'side':>5}")
    for i, (x, y) in enumerate(sampled):
        d = x - body_cx
        side = "L<" if d < 0 else ">R"  # image 左: x 小, dist<0
        print(f"{i:>4} {int(x):>5} {int(y):>5} {d:>+13.1f} {side:>5}")


def asymmetry_summary(label: str, sampled: np.ndarray, body_cx: float) -> None:
    """把 30 个采样点按 y 排序，对比左右侧最远点的距离。"""
    # 找每个 y 大致位置的左右端点
    by_y = sorted(sampled.tolist(), key=lambda p: p[1])

    # 取首末各 ~3 个点（最上和最下）算肩和胯的总宽度
    top_pts = sorted(by_y[:5], key=lambda p: p[0])  # 肩部附近
    bot_pts = sorted(by_y[-5:], key=lambda p: p[0])  # 胯部附近

    top_left = top_pts[0][0]
    top_right = top_pts[-1][0]
    bot_left = bot_pts[0][0]
    bot_right = bot_pts[-1][0]

    print(f"\n[{label}] 总宽")
    print(f"  肩部  left_x={top_left:.0f}  right_x={top_right:.0f}  "
          f"width={top_right - top_left:.1f}  "
          f"距 cx: 左={body_cx - top_left:.1f}  右={top_right - body_cx:.1f}  "
          f"Δ={top_right - body_cx - (body_cx - top_left):+.1f}  "
          f"({(top_right - body_cx) / max(body_cx - top_left, 1e-3) * 100 - 100:+.1f}%)")
    print(f"  胯部  left_x={bot_left:.0f}  right_x={bot_right:.0f}  "
          f"width={bot_right - bot_left:.1f}  "
          f"距 cx: 左={body_cx - bot_left:.1f}  右={bot_right - body_cx:.1f}  "
          f"Δ={bot_right - body_cx - (body_cx - bot_left):+.1f}  "
          f"({(bot_right - body_cx) / max(body_cx - bot_left, 1e-3) * 100 - 100:+.1f}%)")


def overlay_polygon(img: np.ndarray, sampled: np.ndarray, cx: float, label: str
                    ) -> np.ndarray:
    """把 30 采样点画在原图上，body_cx 用蓝竖线标。"""
    vis = img.copy()

    # body_cx 蓝竖线
    cv2.line(vis, (int(cx), 0), (int(cx), vis.shape[0]), (255, 0, 0), 1)

    # 30 采样点（青色多边形）
    pts = sampled.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    # 标注每个采样点的 index
    for i, (x, y) in enumerate(sampled):
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.putText(vis, str(i), (int(x) + 4, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # title
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
    return vis


def main() -> None:
    kpts = load_kpts(JSON)
    person = load_image(PEOPLE)
    print(f"image shape: {person.shape}")

    body_cx = (kpts["left_shoulder"].x + kpts["right_shoulder"].x) / 2
    print(f"body_cx (from shoulders) = {body_cx}")

    # Run 1: default expand_ratio=0.05
    sampled05 = body_region_contour(kpts, n_points=30, expand_ratio=0.05)
    report_polygon("expand_ratio=0.05 (default)", sampled05, body_cx)
    asymmetry_summary("expand=0.05", sampled05, body_cx)

    # Run 2: expand_ratio=0.15
    sampled15 = body_region_contour(kpts, n_points=30, expand_ratio=0.15)
    report_polygon("expand_ratio=0.15 (3x)", sampled15, body_cx)
    asymmetry_summary("expand=0.15", sampled15, body_cx)

    # Run 3: expand_ratio=0.0 (no expand, see raw shape)
    sampled00 = body_region_contour(kpts, n_points=30, expand_ratio=0.0)
    report_polygon("expand_ratio=0.0 (no expand)", sampled00, body_cx)
    asymmetry_summary("expand=0.0", sampled00, body_cx)

    # Save overlays
    save_image(OUT / "polygon_no_expand.jpg",
               overlay_polygon(person, sampled00, body_cx, "no expand"))
    save_image(OUT / "polygon_default.jpg",
               overlay_polygon(person, sampled05, body_cx, "expand=0.05"))
    save_image(OUT / "polygon_3x.jpg",
               overlay_polygon(person, sampled15, body_cx, "expand=0.15"))

    # Compare: per-side wrap distance from cx, at corresponding sample points
    print("\n=== 对比 expand=0.05 vs 0.15 ===")
    print(f"{'idx':>4} {'y':>5} {'cx-0.05':>10} {'cx-0.15':>10} {'delta':>10}")
    for i, (p05, p15) in enumerate(zip(sampled05, sampled15)):
        d05_l = body_cx - p05[0]
        d15_l = body_cx - p15[0]
        # 取绝对值
        print(f"{i:>4} {int(p05[1]):>5} {abs(d05_l):>10.1f} {abs(d15_l):>10.1f} "
              f"{abs(d15_l) - abs(d05_l):>+10.1f}")


if __name__ == "__main__":
    main()
