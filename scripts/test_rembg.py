"""一次性验证脚本：在3张衣服图上跑rembg抠图，输出到 output/test_rembg/。

不修改任何项目代码，只是：
  1. 加载 rembg（U2NET_HOME 已被 clothing_detector 模块重定向到项目内）
  2. 对每张图生成：原图副本、alpha mask、白底合成图
  3. 在 stdout 打印前景像素数

退出后即可在 output/test_rembg/ 下查看效果。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # 脚本在 scripts/ 下，向上到项目根
sys.path.insert(0, str(ROOT / "src"))

# 这一步会触发 clothing_detector 模块加载，U2NET_HOME 在那时被设好。
from virtual_tryon.clothing_detector import _REMBG_CACHE_ROOT  # noqa: F401

from rembg import new_session, remove

CLOTHES_DIR = ROOT / "data_picture" / "clothes"
OUT_DIR = ROOT / "output" / "test_rembg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAMES = ["image.png", "image2.jpg", "image3.png"]


def main() -> int:
    print("首次调用 new_session 会按需下载模型到：", _REMBG_CACHE_ROOT)
    session = new_session("u2net")

    for name in NAMES:
        src = CLOTHES_DIR / name
        bgr = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            print(f"[SKIP] {name}: 读取失败")
            continue
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        if bgr.shape[2] == 4:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # rembg 输出：ndarray uint8, RGBA
        rgba = remove(rgb, session=session)
        mask = rgba[:, :, 3]

        # 白底合成（前景贴白底，便于肉眼判断"衣服是否被完整抠出"）
        white = np.full_like(rgb, 255)
        a = mask[:, :, np.newaxis].astype(np.float32) / 255.0
        composed = (rgb.astype(np.float32) * a + white.astype(np.float32) * (1.0 - a)).astype(np.uint8)

        stem = Path(name).stem
        cv2.imwrite(str(OUT_DIR / f"{stem}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT_DIR / f"{stem}_mask.png"), mask)
        cv2.imwrite(str(OUT_DIR / f"{stem}_on_white.png"), cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))

        fg_px = int((mask > 10).sum())
        total = mask.size
        pct = 100.0 * fg_px / total
        print(f"[OK] {name}: 前景像素 {fg_px}/{total} ({pct:.1f}%)")
        print(f"     输出 -> {OUT_DIR / f'{stem}_rgb.png'}")
        print(f"            {OUT_DIR / f'{stem}_mask.png'}")
        print(f"            {OUT_DIR / f'{stem}_on_white.png'}")

    print("\n验证完成。请查看 output/test_rembg/ 下的9张图。")
    return 0


if __name__ == "__main__":
    sys.exit(main())