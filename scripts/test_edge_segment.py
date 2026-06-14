"""边缘检测抠图测试：对比 4 种策略在 3 张衣服图上的分割效果。

策略：
  A. 灰度图直接 Canny
  B. CLAHE 拉伸后 Canny
  C. 多通道 Canny（R/G/B 各自 Canny 后 OR 合并）
  D. 多通道 Canny + 闭运算 + 最大轮廓填充（完整抠图流水线）

输出：
  output/edge_test/<name>_panel.png    4 列对比图
  控制台打印每张图每种策略的边缘像素占比、最大填充轮廓占比
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from virtual_tryon.io import ensure_dir, load_image

OUT = ensure_dir(Path(__file__).resolve().parent.parent / "output" / "edge_test")
CLOTHES = [
    "data_picture/clothes/image.png",
    "data_picture/clothes/image2.jpg",
    "data_picture/clothes/image3.png",
]


def to_bgr(img: np.ndarray) -> np.ndarray:
    """统一成 3 通道 BGR。"""
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def canny_single(gray: np.ndarray) -> np.ndarray:
    """A. 灰度直接 Canny。"""
    return cv2.Canny(gray, 50, 150)


def canny_clahe(gray: np.ndarray) -> np.ndarray:
    """B. CLAHE 拉伸后 Canny。"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    return cv2.Canny(eq, 50, 150)


def canny_multichannel(bgr: np.ndarray) -> np.ndarray:
    """C. 多通道 Canny：R/G/B/Lab-L 分别 Canny 后 OR。"""
    edges = np.zeros(bgr.shape[:2], dtype=np.uint8)
    for ch in range(3):  # B, G, R
        edges = cv2.bitwise_or(edges, cv2.Canny(bgr[:, :, ch], 50, 150))
    # Lab 的 L 通道也试一下（亮度，与灰度略不同）
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    edges = cv2.bitwise_or(edges, cv2.Canny(lab[:, :, 0], 50, 150))
    return edges


def fill_largest(edges: np.ndarray) -> np.ndarray:
    """D. 闭运算闭合 + 最大轮廓填充，得到抠图 mask。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # 二次膨胀把缝隙彻底封死
    closed = cv2.dilate(closed, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(edges)
    # 找面积最大的 1 个轮廓并填充
    main = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [main], 255)
    return mask


def stats(name: str, edges: np.ndarray, mask: np.ndarray | None) -> str:
    total = edges.size
    edge_pct = float((edges > 0).sum()) / total * 100
    if mask is None:
        return f"  {name:<14} 边缘占比={edge_pct:5.2f}%"
    mask_pct = float((mask > 0).sum()) / total * 100
    return f"  {name:<14} 边缘占比={edge_pct:5.2f}%  填充占比={mask_pct:5.2f}%"


def make_panel(bgr: np.ndarray, gray: np.ndarray, edges: dict[str, np.ndarray],
               mask: np.ndarray) -> np.ndarray:
    """把原图 / 灰度 / 各 Canny / 抠图 mask 拼成 1x5 面板。"""
    h, w = bgr.shape[:2]
    # 等比缩放到统一高度 400
    target_h = 400
    scale = target_h / h
    new_w = int(w * scale)
    bgr_s = cv2.resize(bgr, (new_w, target_h))
    gray_s = cv2.resize(gray, (new_w, target_h))
    gray_3 = cv2.cvtColor(gray_s, cv2.COLOR_GRAY2BGR)

    panels = [bgr_s, gray_3]
    titles = ["原图", "灰度"]
    for name in ("A:Canny", "B:CLAHE+Canny", "C:多通道Canny"):
        e = cv2.resize(edges[name], (new_w, target_h))
        e_3 = cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)
        panels.append(e_3)
        titles.append(name)
    m = cv2.resize(mask, (new_w, target_h))
    m_3 = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    panels.append(m_3)
    titles.append("D:填充mask")

    # 拼成一行
    gap = 8
    out = np.full((target_h + 30, (new_w + gap) * len(panels), 3), 240, dtype=np.uint8)
    for i, (p, t) in enumerate(zip(panels, titles)):
        x = i * (new_w + gap)
        out[30:30 + target_h, x:x + new_w] = p
        cv2.putText(out, t, (x + 4, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    print(f"{'image':<12} {'策略':<18} {'结果'}")
    print("-" * 60)
    for path_str in CLOTHES:
        path = Path(path_str)
        name = path.stem
        img = load_image(path_str, with_alpha=True)
        bgr = to_bgr(img)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        edges = {
            "A:Canny": canny_single(gray),
            "B:CLAHE+Canny": canny_clahe(gray),
            "C:多通道Canny": canny_multichannel(bgr),
        }
        mask = fill_largest(edges["C:多通道Canny"])

        print(f"{name}")
        for k, v in edges.items():
            print(stats(k, v, None))
        print(stats("D:填充mask", edges["C:多通道Canny"], mask))
        print()

        panel = make_panel(bgr, gray, edges, mask)
        cv2.imwrite(str(OUT / f"{name}_panel.png"), panel)

    print(f"已写入 {OUT}")


if __name__ == "__main__":
    main()
