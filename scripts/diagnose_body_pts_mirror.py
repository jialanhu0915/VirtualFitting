"""对比原图 vs 水平镜像翻转后的 body_pts。

不重新跑 MediaPipe。直接把 keypoints 做「swap left/right + 水平镜像 x」，
模拟镜像翻转输入。这能区分：
  1. MediaPipe 对 subject 某一侧的偏置（如果原图与镜像的 Δy 一致且符号相同）
  2. 这张图姿势本身的真实不对称（如果原图与镜像的 Δy 符号相反 / 大小相近）
  3. polygon 构造的算法偏置（如果原图与镜像都产生相同的 Δy）
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


PEOPLE = ROOT / "data_picture" / "people" / "image.png"
JSON = ROOT / "data_picture" / "people" / "image.keypoints.json"
OUT = ROOT / "output" / "diagnose"
OUT.mkdir(parents=True, exist_ok=True)


def load_kpts(path: Path) -> dict[str, Keypoint]:
    data = json.loads(path.read_text())
    out: dict[str, Keypoint] = {}
    for name, d in data["keypoints"].items():
        out[name] = Keypoint(
            x=int(d["x"]), y=int(d["y"]),
            confidence=float(d.get("confidence", 1.0)),
            name=name,
        )
    return out


def mirror_kpts(kpts: dict[str, Keypoint], image_w: int) -> dict[str, Keypoint]:
    """swap left_/right_ + x 镜像，模拟水平翻转输入。"""
    out: dict[str, Keypoint] = {}
    for name, kp in kpts.items():
        if name.startswith("left_"):
            new_name = "right_" + name[len("left_"):]
        elif name.startswith("right_"):
            new_name = "left_" + name[len("right_"):]
        else:
            new_name = name
        out[new_name] = Keypoint(
            x=int(image_w - kp.x), y=int(kp.y),
            confidence=kp.confidence, name=new_name,
        )
    return out


def summarize_polygon(label: str, kpts: dict[str, Keypoint], body_cx: float,
                       expand_ratio: float = 0.05) -> dict:
    """重建 10 顶点 polygon（通过再调 body_region_contour 之前先得到 verts）。"""
    # body_region_contour 不暴露中间 verts，但 expand 后再 resample 30 点会让
    # 顶点位置丢失。所以这里直接重新实现 polygon 构造，提取 10 顶点。
    ls_x, ls_y = kpts["left_shoulder"].x, kpts["left_shoulder"].y
    rs_x, rs_y = kpts["right_shoulder"].x, kpts["right_shoulder"].y
    le_y = kpts["left_elbow"].y
    re_y = kpts["right_elbow"].y
    lh_x, lh_y = kpts["left_hip"].x, kpts["left_hip"].y
    rh_x, rh_y = kpts["right_hip"].x, kpts["right_hip"].y
    le_x = kpts["left_elbow"].x
    re_x = kpts["right_elbow"].x
    lw_x = kpts["left_wrist"].x; lw_y = kpts["left_wrist"].y
    rw_x = kpts["right_wrist"].x; rw_y = kpts["right_wrist"].y
    nose_y = kpts["nose"].y

    shoulder_y = (ls_y + rs_y) / 2
    neck_top_y = shoulder_y - (shoulder_y - nose_y) * 0.30
    armpit_y_l = ls_y + (le_y - ls_y) * 0.50
    armpit_y_r = rs_y + (re_y - rs_y) * 0.50

    shoulder_pad = 8; hip_pad = 30; elbow_pad = 5; wrist_pad = 3
    lsh_x = ls_x + shoulder_pad; rsh_x = rs_x - shoulder_pad
    larmpit_x = lsh_x; rarmpit_x = rsh_x
    lhip_x = lh_x + hip_pad; rhip_x = rh_x - hip_pad
    lelbow_x = le_x + elbow_pad; relbow_x = re_x - elbow_pad
    lwrist_x = lw_x + wrist_pad; rwrist_x = rw_x - wrist_pad

    cx_v = (ls_x + rs_x) / 2
    bottom_x = (lhip_x + rhip_x) / 2
    bottom_y = max(lh_y, rh_y)

    verts = np.array([
        (cx_v, neck_top_y),       # 0  neck_top
        (lsh_x, ls_y),            # 1  左肩 (image 右)
        (larmpit_x, armpit_y_l),  # 2  左腋
        (lelbow_x, le_y),         # 3  左肘
        (lwrist_x, lw_y),         # 4  左腕
        (bottom_x, bottom_y),     # 5  底
        (rwrist_x, rw_y),         # 6  右腕
        (relbow_x, re_y),         # 7  右肘
        (rarmpit_x, armpit_y_r),  # 8  右腋
        (rsh_x, rs_y),            # 9  右肩 (image 左)
    ], dtype=np.float32)

    if expand_ratio > 0:
        cxv = verts[:, 0].mean()
        cyv = verts[:, 1].mean()
        for i in range(1, len(verts)):
            dx, dy = verts[i][0] - cxv, verts[i][1] - cyv
            dist = np.sqrt(dx * dx + dy * dy) + 1e-6
            verts[i][0] += dx / dist * expand_ratio * (abs(dx) + abs(dy))
            verts[i][1] += dy / dist * expand_ratio * (abs(dx) + abs(dy))

    print(f"\n=== {label} (body_cx={body_cx:.1f}, expand={expand_ratio}) ===")
    print(f"{'vert':>6} {'role':>12} {'x':>6} {'y':>6} {'dist_from_cx':>14}")
    roles = ["neck_top", "lsh", "larmpit", "lelbow", "lwrist",
             "bottom", "rwrist", "relbow", "rarmpit", "rsh"]
    for i, (x, y) in enumerate(verts):
        d = x - body_cx
        print(f"{i:>6} {roles[i]:>12} {x:>6.1f} {y:>6.1f} {d:>+14.1f}")

    # 镜像对的不对称统计（顶点 1↔9, 2↔8, 3↔7, 4↔6）
    pairs = [(1, 9, "shoulder"), (2, 8, "armpit"), (3, 7, "elbow"),
             (4, 6, "wrist")]
    print(f"\n  镜像对 Δy (右-左)  → 期望接近 0:")
    for li, ri, name in pairs:
        dl = body_cx - verts[li][0]  # image 左侧距 (lsh 在 image 右)
        dr = verts[ri][0] - body_cx  # image 右侧距 (rsh 在 image 左)
        dy = verts[ri][1] - verts[li][1]  # 右侧 y - 左侧 y (image 坐标 y 增向下)
        print(f"    {name:>8}  L(at image-right)={dl:>6.1f}  "
              f"R(at image-left)={dr:>6.1f}  dist_Δ={dr-dl:>+6.1f}  "
              f"y_Δ(R-L)={dy:>+6.1f}")
    return {"verts": verts, "kpts": kpts}


def main() -> None:
    kpts_orig = load_kpts(JSON)
    person = load_image(PEOPLE)
    h, w = person.shape[:2]
    print(f"image: {w}x{h}")

    body_cx_orig = (kpts_orig["left_shoulder"].x + kpts_orig["right_shoulder"].x) / 2
    print(f"原图 body_cx = {body_cx_orig}")

    # 镜像版 keypoints
    kpts_mirror = mirror_kpts(kpts_orig, w)
    body_cx_mirror = (kpts_mirror["left_shoulder"].x +
                      kpts_mirror["right_shoulder"].x) / 2
    print(f"镜像 body_cx = {body_cx_mirror}")

    # 用同一 expand 跑两种输入
    summarize_polygon("原图 (subject 左在 image 右)", kpts_orig, body_cx_orig)
    summarize_polygon("镜像 (subject 左在 image 左)", kpts_mirror, body_cx_mirror)

    # 跑真实的 30 点 resample，对比 warp 输入
    print("\n=== 真实 30 点 resample 对比 (expand=0.05) ===")
    sampled_orig = body_region_contour(kpts_orig, n_points=30, expand_ratio=0.05)
    sampled_mirror = body_region_contour(kpts_mirror, n_points=30, expand_ratio=0.05)

    print(f"{'idx':>4} {'y_orig':>7} {'x_orig':>7} {'dist_orig':>10} | "
          f"{'x_mirror':>10} {'dist_mirror':>12}")
    for i in range(30):
        # 镜像后坐标系整体翻向，所以镜像的 x 等于 w - 原图 x
        # 关注 polygon 形状对称性：x 距 cx 在两种输入下应近似
        do = sampled_orig[i][0] - body_cx_orig
        dm = sampled_mirror[i][0] - body_cx_mirror
        print(f"{i:>4} {int(sampled_orig[i][1]):>7} {int(sampled_orig[i][0]):>7} "
              f"{do:>+10.1f} | {int(sampled_mirror[i][0]):>10} {dm:>+12.1f}")

    # 保存镜像原图供目视
    mirrored_img = cv2.flip(person, 1)
    save_image(OUT / "person_mirrored.jpg", mirrored_img)
    print(f"\n镜像图已保存: {OUT / 'person_mirrored.jpg'}")


if __name__ == "__main__":
    main()
