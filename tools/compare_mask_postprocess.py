"""3 张测试图 × 2 关键点方法 × 4 mask 配置 = 24 个组合的关键点对比。

跑完后打印每个组合的 8 个关键点坐标，便于人工对照 V1/V3 在不同 mask
处理路径下的差异。mask 配置:
  raw                : mask_postprocess=False（仅 CLAHE+Canny 原始 mask）
  post               : clean_mask() 基础后处理（闭运算+最大 CC+填洞+腐蚀）
  post+trim_solidity : clean_mask() + 行向实心度裁剪（裁掉底部阴影行）
  trim_only          : clean_mask() 关掉，只跑实心度裁剪（验证实心度裁剪
                       单独的效果）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from virtual_tryon.clothing_detector import ClothingDetector, set_debug_dir
from virtual_tryon.io import load_image

TEST_IMAGES = {
    "qipao":      "data_picture/clothes/image.png",
    "tshirt":     "data_picture/clothes/image2.jpg",
    "whiteshirt": "data_picture/clothes/image3.png",
}

METHODS = ["geometric", "width"]

# (post, trim, label)
MASK_CONFIGS = [
    (False, False, "raw"),
    (True,  False, "post"),
    (True,  True,  "post+trim"),
    (False, True,  "trim_only"),  # 实心度裁剪不依赖 clean_mask，单独验证
]


def run_one(image_path: Path, method: str, post: bool, trim: bool) -> dict:
    img = load_image(image_path, with_alpha=True)
    config_label = next(lbl for p, t, lbl in MASK_CONFIGS if p == post and t == trim)
    out_dir = ROOT / "output" / f"{method}_{config_label}_{image_path.stem}"
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    set_debug_dir(debug_dir)
    try:
        det = ClothingDetector(
            keypoint_method=method, mask_postprocess=post, mask_trim_solidity=trim,
        )
        kpts = det.detect(img)
    finally:
        set_debug_dir(None)
    return kpts


def main() -> int:
    rows = []
    for img_name, rel in TEST_IMAGES.items():
        img_path = ROOT / rel
        for method in METHODS:
            for post, trim, label in MASK_CONFIGS:
                tag = f"{method}_{label}_{img_name}"
                try:
                    kpts = run_one(img_path, method, post, trim)
                except Exception as e:
                    print(f"[{tag:40s}] FAILED: {e}")
                    continue
                sh = kpts["left_shoulder"]
                ar = kpts["left_armpit"]
                top = kpts["top_center"]
                bot = kpts["bottom_center"]
                print(
                    f"[{tag:40s}] top=({top.x:4d},{top.y:4d})  "
                    f"Lsh=({sh.x:4d},{sh.y:4d})  "
                    f"Lar=({ar.x:4d},{ar.y:4d})  "
                    f"bot=({bot.x:4d},{bot.y:4d})"
                )
                rows.append((tag, sh.y, ar.y, top.y, bot.y))
    return 0


if __name__ == "__main__":
    sys.exit(main())
