"""虚拟试衣流水线命令行入口。

Stage 1 子命令：
    detect           - 同时检测人体和服装关键点（完整流程）。
    detect-clothing  - 只检测服装关键点。
    detect-human     - 只检测人体关键点。

后续 Stage 计划加入：
    run      - 完整流水线（关键点 -> 变形 -> 融合）。
    ablation - 在同一组输入上跑多种 warper/blender 组合用于对比。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 把 src/ 加入 sys.path，让仓库内的包可以直接被 import，
# 避免必须先 `pip install -e .` 才能跑。
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from virtual_tryon import ClothingDetector, Keypoint, RobustHumanDetector
from virtual_tryon.human_detector import set_debug_dir as set_human_debug
from virtual_tryon.clothing_detector import set_debug_dir as set_clothing_debug
from virtual_tryon.io import ensure_dir, load_image, save_image
from virtual_tryon.visualize import draw_keypoints

logger = logging.getLogger(__name__)


def _dump_keypoints(label: str, kpts: dict[str, Keypoint]) -> None:
    """以表格形式打印关键点坐标和置信度。"""
    logger.info("%s 关键点（共 %d 个）：", label, len(kpts))
    for k, v in kpts.items():
        logger.info("  %-18s (%4d, %4d)  conf=%.2f", k, v.x, v.y, v.confidence)


def _run_human(person_path: Path, out_dir: Path) -> dict[str, Keypoint]:
    """加载人体图、设置 debug 目录、跑 RobustHumanDetector、保存可视化。"""
    person_img = load_image(person_path)
    debug_dir = ensure_dir(out_dir / "debug")
    set_human_debug(debug_dir)
    try:
        human_det = RobustHumanDetector()
        logger.info("正在检测人体关键点：%s", person_path)
        kpts = human_det.detect(person_img)
    finally:
        set_human_debug(None)
    save_image(
        out_dir / "human_keypoints.jpg",
        draw_keypoints(person_img, kpts),
    )
    _dump_keypoints("人体", kpts)
    return kpts


def _run_clothing(clothing_path: Path, out_dir: Path) -> dict[str, Keypoint]:
    """加载服装图、设置 debug 目录、跑 ClothingDetector、保存可视化。"""
    # 服装图可能带 alpha 通道（PNG 透明背景），用 with_alpha 保留。
    clothing_img = load_image(clothing_path, with_alpha=True)
    debug_dir = ensure_dir(out_dir / "debug")
    set_clothing_debug(debug_dir)
    try:
        clothing_det = ClothingDetector()
        logger.info("正在检测服装关键点：%s", clothing_path)
        kpts = clothing_det.detect(clothing_img)
    finally:
        set_clothing_debug(None)
    save_image(
        out_dir / "clothing_keypoints.jpg",
        draw_keypoints(clothing_img, kpts),
    )
    _dump_keypoints("服装", kpts)
    return kpts


def cmd_detect_human(args: argparse.Namespace) -> int:
    """`detect-human` 子命令：只检测人体关键点并保存可视化。"""
    out_dir = ensure_dir(args.output)
    _run_human(args.person, out_dir)
    logger.info("输出已写入 %s", out_dir)
    return 0


def cmd_detect_clothing(args: argparse.Namespace) -> int:
    """`detect-clothing` 子命令：只检测服装关键点并保存可视化。"""
    out_dir = ensure_dir(args.output)
    _run_clothing(args.clothing, out_dir)
    logger.info("输出已写入 %s", out_dir)
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    """`detect` 子命令：同时检测人体和服装关键点并保存可视化。"""
    out_dir = ensure_dir(args.output)
    _run_human(args.person, out_dir)
    _run_clothing(args.clothing, out_dir)
    logger.info("输出已写入 %s", out_dir)
    return 0


def main() -> int:
    """CLI 入口：解析子命令并分发。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="虚拟试衣流水线（Stage 1：关键点检测）"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_det = sub.add_parser(
        "detect", help="同时检测人体和服装关键点并保存可视化结果"
    )
    p_det.add_argument("--person", required=True, help="人体图像路径")
    p_det.add_argument("--clothing", required=True, help="服装图像路径")
    p_det.add_argument("--output", default="output", help="输出目录")
    p_det.set_defaults(func=cmd_detect)

    p_det_c = sub.add_parser(
        "detect-clothing", help="只检测服装关键点并保存可视化"
    )
    p_det_c.add_argument("--clothing", required=True, help="服装图像路径")
    p_det_c.add_argument("--output", default="output", help="输出目录")
    p_det_c.set_defaults(func=cmd_detect_clothing)

    p_det_h = sub.add_parser(
        "detect-human", help="只检测人体关键点并保存可视化"
    )
    p_det_h.add_argument("--person", required=True, help="人体图像路径")
    p_det_h.add_argument("--output", default="output", help="输出目录")
    p_det_h.set_defaults(func=cmd_detect_human)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
