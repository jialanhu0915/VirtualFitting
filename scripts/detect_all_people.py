"""为 data_picture/people/ 下所有人像跑 MediaPipe 检测（写缓存）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from virtual_tryon import RobustHumanDetector
from virtual_tryon.human_cache import cache_paths_for, load_human_cache, save_human_cache
from virtual_tryon.io import load_image
from virtual_tryon.visualize import draw_keypoints

PEOPLE = ROOT / "data_picture" / "people"
det = RobustHumanDetector()

for img_path in sorted(PEOPLE.glob("image*.png")) + sorted(PEOPLE.glob("image*.jpg")):
    # skip cache files
    if ".keypoints" in img_path.name:
        continue
    cached = load_human_cache(img_path)
    if cached is not None:
        print(f"[skip] {img_path.name} 已缓存")
        continue
    print(f"[run ] {img_path.name}")
    img = load_image(img_path)
    kpts = det.detect(img)
    json_path, jpg_path = cache_paths_for(img_path)
    save_human_cache(img_path, kpts, draw_keypoints(img, kpts))
    print(f"       → {json_path.name}")
