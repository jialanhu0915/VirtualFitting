"""虚拟试衣流水线命令行入口。

Stage 1 子命令：
    detect           - 同时检测人体和服装关键点（完整流程）。
    detect-clothing  - 只检测服装关键点。
    detect-human     - 只检测人体关键点。

Stage 2 子命令：
    run              - 完整流水线（轮廓采样 → TPS 变形 → 融合）。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# 把 src/ 加入 sys.path，让仓库内的包可以直接被 import，
# 避免必须先 `pip install -e .` 才能跑。
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from virtual_tryon import ClothingDetector, Keypoint, RobustHumanDetector
from virtual_tryon.human_cache import (
    cache_paths_for,
    load_human_cache,
    save_human_cache,
)
from virtual_tryon.human_detector import body_region_contour
from virtual_tryon.clothing_detector import set_debug_dir as set_clothing_debug
from virtual_tryon.io import ensure_dir, load_image, save_image
from virtual_tryon.keypoints import CORRESPONDENCE
from virtual_tryon.tps_warp import blend, warp_clothing
from virtual_tryon.visualize import draw_keypoints

logger = logging.getLogger(__name__)


def _dump_keypoints(label: str, kpts: dict[str, Keypoint]) -> None:
    """以表格形式打印关键点坐标和置信度。"""
    logger.info("%s 关键点（共 %d 个）：", label, len(kpts))
    for k, v in kpts.items():
        logger.info("  %-18s (%4d, %4d)  conf=%.2f", k, v.x, v.y, v.confidence)


def _load_or_detect_human(
    person_path: str | Path, force_rebuild: bool = False,
) -> dict[str, Keypoint]:
    """优先读本地缓存的人体关键点；命中失败或强制重建时跑模型并写缓存。

    缓存文件位于 person 图像同目录（`*.keypoints.json` + `*.keypoints.jpg`），
    失效条件由 human_cache.load_human_cache 判断（mtime 不一致等）。
    """
    person_path = Path(person_path)
    if not force_rebuild:
        cached = load_human_cache(person_path)
        if cached is not None:
            logger.info("命中人体关键点缓存：%s", person_path)
            return cached

    logger.info("运行人体检测器：%s", person_path)
    person_img = load_image(person_path)
    kpts = RobustHumanDetector().detect(person_img)
    save_human_cache(person_path, kpts, draw_keypoints(person_img, kpts))
    return kpts


def _run_human(
    person_path: str | Path, out_dir: Path, force_rebuild: bool = False,
) -> dict[str, Keypoint]:
    """加载人体图（优先用缓存）、保存可视化到 out_dir。"""
    person_path = Path(person_path)
    kpts = _load_or_detect_human(person_path, force_rebuild=force_rebuild)
    person_img = load_image(person_path)
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
    _run_human(args.person, out_dir, force_rebuild=args.rebuild_human_cache)
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
    _run_human(args.person, out_dir, force_rebuild=args.rebuild_human_cache)
    _run_clothing(args.clothing, out_dir)
    logger.info("输出已写入 %s", out_dir)
    return 0


def cmd_cache_human(args: argparse.Namespace) -> int:
    """`cache-human` 子命令：显式预热/重建人体关键点缓存。"""
    _load_or_detect_human(args.person, force_rebuild=True)
    json_path, jpg_path = cache_paths_for(Path(args.person))
    logger.info("人体关键点缓存已就绪：%s / %s", json_path, jpg_path)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """`run` 子命令：完整虚拟试衣流水线。"""
    out_dir = ensure_dir(args.output)

    # 1. 加载人体关键点（命中缓存则跳过模型）
    human_kpts = _load_or_detect_human(
        args.person, force_rebuild=args.rebuild_human_cache,
    )
    person_img = load_image(args.person)
    body_pts = body_region_contour(human_kpts, n_points=args.n_points)

    # 2. 加载衣服
    clothing_img = load_image(args.clothing, with_alpha=True)
    if clothing_img.shape[2] == 4:
        clothing_rgb = clothing_img[:, :, :3]
    else:
        clothing_rgb = clothing_img

    # 3. 衣服轮廓采样（同时得到 mask、领口锚点、8 个语义关键点）
    clothing_det = ClothingDetector()
    clothing_pts, clothing_mask, cloth_anchor, cloth_8kpts = (
        clothing_det.sample_contour(clothing_img, n_points=args.n_points)
    )

    # 用 mask 抠掉衣服背景
    mask_3ch = clothing_mask[:, :, np.newaxis] / 255.0
    clothing_fg = (clothing_rgb.astype(np.float32) * mask_3ch).astype(np.uint8)

    # 人体脖子锚点：MediaPipe neck=双肩中点（肩线高度），
    # 衣服领口应略高于肩线。用人脸 nose 到 neck 距离的 30% 作向上偏移。
    nose_y = float(human_kpts["nose"].y)
    neck_x = float(human_kpts["neck"].x)
    neck_y = float(human_kpts["neck"].y)
    neck_offset = max(0.0, (neck_y - nose_y) * 0.30)
    body_anchor = (neck_x, neck_y - neck_offset)

    # 4. 变形（领口 → 脖子锚定；method=tps 时用 TPS 路径，anchor 被忽略）
    logger.info("正在进行 %s 变形（领口→脖子锚定）...", args.warp_method)

    # 8 对语义对应：按 CORRESPONDENCE 把衣服点 → 人体点。
    # CORRESPONDENCE 没列 bottom_center，用左右髋中点补齐，使配对数和
    # 衣服关键点数一致。
    semantic_pairs: np.ndarray | None = None
    if args.warp_method == "tps":
        pairs: list[list[list[float]]] = []
        for cloth_name, human_name in CORRESPONDENCE.items():
            ck = cloth_8kpts[cloth_name]
            hk = human_kpts[human_name]
            pairs.append([[float(ck.x), float(ck.y)], [float(hk.x), float(hk.y)]])
        # bottom_center -> (left_hip, right_hip) 中点
        bc = cloth_8kpts["bottom_center"]
        lh, rh = human_kpts["left_hip"], human_kpts["right_hip"]
        pairs.append([
            [float(bc.x), float(bc.y)],
            [(lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0],
        ])
        semantic_pairs = np.array(pairs, dtype=np.float32)
        logger.info("TPS 语义对应点数：%d", len(pairs))

    warped_rgb, warped_mask = warp_clothing(
        clothing_fg, clothing_mask, clothing_pts, body_pts,
        out_shape=person_img.shape[:2],
        clothing_anchor=cloth_anchor,
        body_anchor=body_anchor,
        method=args.warp_method,
        semantic_pairs=semantic_pairs,
    )

    # Debug overlay 1: body_pts 叠加在 person 图
    body_vis = person_img.copy()
    pts_i = body_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(body_vis, [pts_i], isClosed=True, color=(0, 255, 255), thickness=2)
    save_image(out_dir / "debug_body_pts.jpg", body_vis)

    # Debug overlay 2: clothing_pts + 领口锚点叠加在 clothing 图
    if clothing_img.shape[2] == 4:
        cloth_vis = np.ascontiguousarray(clothing_img[:, :, :3])
    else:
        cloth_vis = clothing_img.copy()
    cpts_i = clothing_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(cloth_vis, [cpts_i], isClosed=True, color=(255, 0, 255), thickness=2)
    cx_anchor, cy_anchor = int(cloth_anchor[0]), int(cloth_anchor[1])
    cv2.drawMarker(
        cloth_vis, (cx_anchor, cy_anchor), (0, 255, 255),
        markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3,
    )
    save_image(out_dir / "debug_clothing_pts.jpg", cloth_vis)

    # Debug overlay 3: warped_mask 染色叠加在 person 图（半透明青色）
    overlay = person_img.copy()
    overlay[warped_mask > 0] = (
        overlay[warped_mask > 0] * 0.5 + np.array([0, 255, 255], dtype=np.float32) * 0.5
    ).astype(np.uint8)
    save_image(out_dir / "debug_warped_mask_overlay.jpg", overlay)

    # 5. 融合
    result = blend(person_img, warped_rgb, warped_mask)
    save_image(out_dir / "result.jpg", result)

    # 中间产物
    save_image(out_dir / "warped_clothing.jpg", warped_rgb)
    save_image(out_dir / "warped_mask.jpg", warped_mask)

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
    p_det.add_argument(
        "--rebuild-human-cache", action="store_true",
        help="忽略人体关键点缓存，强制重新跑模型",
    )
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
    p_det_h.add_argument(
        "--rebuild-human-cache", action="store_true",
        help="忽略人体关键点缓存，强制重新跑模型",
    )
    p_det_h.set_defaults(func=cmd_detect_human)

    p_cache_h = sub.add_parser(
        "cache-human",
        help="预热/重建人体关键点缓存（写入 person 图同目录）",
    )
    p_cache_h.add_argument("--person", required=True, help="人体图像路径")
    p_cache_h.set_defaults(func=cmd_cache_human)

    p_run = sub.add_parser(
        "run", help="完整虚拟试衣流水线（轮廓采样 → TPS 变形 → 融合）"
    )
    p_run.add_argument("--person", required=True, help="人体图像路径")
    p_run.add_argument("--clothing", required=True, help="服装图像路径")
    p_run.add_argument("--output", default="output/run", help="输出目录")
    p_run.add_argument("--n-points", type=int, default=30,
                       help="轮廓采样点数（默认 30）")
    p_run.add_argument(
        "--rebuild-human-cache", action="store_true",
        help="忽略人体关键点缓存，强制重新跑模型",
    )
    p_run.add_argument(
        "--warp-method", choices=["affine", "tps"], default="affine",
        help="warp 算法：affine=等比缩放+对齐（默认），tps=薄板样条非线性形变",
    )
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
