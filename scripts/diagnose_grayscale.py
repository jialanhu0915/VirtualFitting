"""诊断 3 张衣服图的灰度分布：判断边缘检测是否可行。

输出：
  output/diag_gray/<name>_gray.png    灰度图
  output/diag_gray/<name>_hist.png    灰度直方图
  控制台打印每张图的灰度统计（min/max/mean/std/低对比度像素占比）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from virtual_tryon.io import ensure_dir, load_image

OUT = ensure_dir(Path(__file__).resolve().parent.parent / "output" / "diag_gray")
CLOTHES = [
    "data_picture/clothes/image.png",
    "data_picture/clothes/image2.jpg",
    "data_picture/clothes/image3.png",
]


def to_gray(img: np.ndarray) -> np.ndarray:
    """统一转灰度：4 通道先取 BGR 再 cvtColor。"""
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        bgr = img
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def draw_hist(gray: np.ndarray, name: str) -> np.ndarray:
    """把 256-bin 直方图画成 400x300 的彩色图。"""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    # 归一化到画布高度
    h, w = 300, 400
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    hist_n = (hist / max(hist.max(), 1) * (h - 30)).astype(np.int32)
    for x in range(256):
        y_top = h - 20 - hist_n[x]
        cv2.line(canvas, (x, h - 20), (x, y_top), (60, 60, 60), 1)
    # 标注关键阈值
    cv2.line(canvas, (30, 0), (30, h - 20), (0, 0, 255), 1)
    cv2.putText(canvas, "30", (32, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.line(canvas, (220, 0), (220, h - 20), (0, 0, 255), 1)
    cv2.putText(canvas, "220", (218, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.putText(canvas, name, (4, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return canvas


def main() -> None:
    print(f"{'name':<12} {'shape':<12} {'min':>4} {'max':>4} {'mean':>6} {'std':>6} {'%<30':>6} {'%>220':>6}")
    print("-" * 64)
    for path_str in CLOTHES:
        path = Path(path_str)
        name = path.stem
        img = load_image(path_str, with_alpha=True)
        gray = to_gray(img)
        cv2.imwrite(str(OUT / f"{name}_gray.png"), gray)
        cv2.imwrite(str(OUT / f"{name}_hist.png"), draw_hist(gray, name))

        total = gray.size
        pct_low = float((gray < 30).sum()) / total * 100   # 接近黑的占比
        pct_high = float((gray > 220).sum()) / total * 100  # 接近白的占比
        print(f"{name:<12} {str(gray.shape):<12} {gray.min():>4} {gray.max():>4} "
              f"{gray.mean():>6.1f} {gray.std():>6.1f} {pct_low:>5.1f}% {pct_high:>5.1f}%")
    print()
    print(f"已写入 {OUT}")


if __name__ == "__main__":
    main()
