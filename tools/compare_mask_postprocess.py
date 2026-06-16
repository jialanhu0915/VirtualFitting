"""3 张测试图 × 2 关键点方法 × 2 mask 后处理 = 12 个组合的关键点对比。

跑完后打印每个组合的 8 个关键点坐标，便于人工对照 V1/V3 + raw/postprocessed
在 T 恤/白衬衫 mask 含阴影场景下的差异。
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
POSTPROCESS = [True, False]


def run_one(image_path: Path, method: str, post: bool) -> dict:
    img = load_image(image_path, with_alpha=True)
    out_dir = ROOT / "output" / f"{method}_{'post' if post else 'raw'}_{image_path.stem}"
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    set_debug_dir(debug_dir)
    try:
        det = ClothingDetector(keypoint_method=method, mask_postprocess=post)
        kpts = det.detect(img)
    finally:
        set_debug_dir(None)
    return kpts


def main() -> int:
    rows = []
    for img_name, rel in TEST_IMAGES.items():
        img_path = ROOT / rel
        for method in METHODS:
            for post in POSTPROCESS:
                tag = f"{method}_{'post' if post else 'raw'}_{img_name}"
                try:
                    kpts = run_one(img_path, method, post)
                except Exception as e:
                    print(f"[{tag}] FAILED: {e}")
                    continue
                sh = kpts["left_shoulder"]
                ar = kpts["left_armpit"]
                top = kpts["top_center"]
                bot = kpts["bottom_center"]
                print(
                    f"[{tag:35s}] top=({top.x:4d},{top.y:4d})  "
                    f"Lsh=({sh.x:4d},{sh.y:4d})  "
                    f"Lar=({ar.x:4d},{ar.y:4d})  "
                    f"bot=({bot.x:4d},{bot.y:4d})"
                )
                rows.append((tag, sh.y, ar.y, top.y, bot.y))
    return 0


if __name__ == "__main__":
    sys.exit(main())