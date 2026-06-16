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


# 进入 TPS 时排除下摆 → 髋的强对应：衣服下摆通常远低于髋（旗袍到脚踝），
# 强对应会把下半身压缩到髋高度。剔除后下摆按 TPS 的远距离仿射行为自然下垂。
_TPS_EXCLUDED_CORRESPONDENCE: set[str] = {"left_bottom", "right_bottom"}


def _dump_keypoints(label: str, kpts: dict[str, Keypoint]) -> None:
    """以表格形式打印关键点坐标和置信度。"""
    logger.info("%s 关键点（共 %d 个）：", label, len(kpts))
    for k, v in kpts.items():
        logger.info("  %-18s (%4d, %4d)  conf=%.2f", k, v.x, v.y, v.confidence)


def _cover_outpush(
    body_pt: np.ndarray, cloth_pt: np.ndarray,
    neck: np.ndarray, hip_mid: np.ndarray,
    cloth_anchor: np.ndarray, scale: float,
) -> np.ndarray:
    """覆盖性外推：消除"人体关节中心 vs 衣服轮廓边缘"的几何错配。

    人体关键点是关节中心、衣服关键点是轮廓边缘，直接做 TPS 控制点会把衣服
    收缩到骨架尺寸（旗袍尤其明显）。这里把 dst（人体侧）沿身体中轴的法向
    外推到至少和 affine 预测一样外，保证衣服 warp 后能覆盖人体；
    切向（沿中轴方向）保留人体姿态偏差，让 TPS 仍能拟合一肩高一肩低、
    抬手等非线性形变——这是 TPS 相对 affine 的核心优势，不能丢。

    Args:
        body_pt: 人体侧关键点（关节中心，会被外推的 dst）
        cloth_pt: 衣服侧关键点（轮廓边缘，TPS 的 src）
        neck: 人体颈中点（中轴起点 + affine 平移锚点）
        hip_mid: 人体髋中点（中轴终点）
        cloth_anchor: 衣服领中点（affine 平移锚点，对应 neck）
        scale: affine 的均匀缩放比

    Returns:
        外推后的 dst (2,) float32
    """
    # affine 预测：衣服点按 (scale, anchor=top_center→neck) 变换后的位置
    p_aff = neck + scale * (cloth_pt - cloth_anchor)

    # 身体中轴（neck → hip_mid）的单位向量和法向
    axis = hip_mid - neck
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        # 退化（neck 和 hip_mid 重合）：放弃外推，原样返回。
        return body_pt.astype(np.float32)
    axis_u = axis / axis_norm
    perp_u = np.array([-axis_u[1], axis_u[0]], dtype=np.float32)

    # 把 body_pt 和 p_aff 都分解到中轴坐标系（以 neck 为原点）
    body_rel = body_pt - neck
    aff_rel = p_aff - neck
    body_axial = float(body_rel @ axis_u)
    body_perp = float(body_rel @ perp_u)
    aff_perp = float(aff_rel @ perp_u)

    # 切向：完全保留 body（保留姿态偏差，TPS 的非线性能力在此体现）
    target_axial = body_axial
    # 法向：同侧时取更外的，异侧时保守用 body（衣服比人体窄的罕见 case）
    if (body_perp >= 0) == (aff_perp >= 0):
        target_perp = body_perp if abs(body_perp) > abs(aff_perp) else aff_perp
    else:
        target_perp = body_perp

    return (neck + target_axial * axis_u + target_perp * perp_u).astype(np.float32)


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

    # TPS 控制点构造：
    # - 排除 *_bottom（下摆 → 髋强对应会把长款衣服压到髋高度）
    # - top_center → neck 作为中线锚点，**不外推**（领中点和 neck 都在中线）
    # - 其余 4 对（左右肩、左右腋下）的 dst 用 _cover_outpush 做覆盖性外推，
    #   消除"关节中心 vs 轮廓边缘"的几何错配（详见函数 docstring）
    semantic_pairs: np.ndarray | None = None
    if args.warp_method == "tps":
        # affine 参考变换：领口 → neck 为锚点，scale 取衣服 5 个上半身语义点
        # 和对应人体点外接矩形的 max(W/w, H/h)，保证衣服 cover 上半身。
        cloth_anchor_pt = np.array(
            [cloth_8kpts["top_center"].x, cloth_8kpts["top_center"].y],
            dtype=np.float32,
        )
        neck_pt = np.array(
            [human_kpts["neck"].x, human_kpts["neck"].y], dtype=np.float32,
        )
        hip_mid_pt = np.array([
            (human_kpts["left_hip"].x + human_kpts["right_hip"].x) / 2.0,
            (human_kpts["left_hip"].y + human_kpts["right_hip"].y) / 2.0,
        ], dtype=np.float32)

        # scale：复用 affine 路径同款公式（轮廓 bbox 的 max(W/w, H/h)），
        # 而不是用 5 个稀疏语义点。语义点里 armpit↔elbow 不是"几何锚定"
        # 而是"穿上后大致到肘"的近似，纳入 bbox 会把 scale 拉偏（实测
        # 旗袍上算出 scale<1 导致外推全 no-op）。轮廓 bbox 反映整件衣服
        # vs 整个躯干区域的尺寸比，和 affine 路径里的 scale 含义一致，
        # 在"实测能 cover"这点上已被验证。
        cw = max(float(np.ptp(clothing_pts[:, 0])), 1.0)
        ch = max(float(np.ptp(clothing_pts[:, 1])), 1.0)
        bw = max(float(np.ptp(body_pts[:, 0])), 1.0)
        bh = max(float(np.ptp(body_pts[:, 1])), 1.0)
        affine_scale = max(bw / cw, bh / ch) * 1.05

        pairs: list[list[list[float]]] = []
        for cloth_name, human_name in CORRESPONDENCE.items():
            if cloth_name in _TPS_EXCLUDED_CORRESPONDENCE:
                continue
            ck = cloth_8kpts[cloth_name]
            hk = human_kpts[human_name]
            cloth_pt = np.array([ck.x, ck.y], dtype=np.float32)
            body_pt = np.array([hk.x, hk.y], dtype=np.float32)

            if cloth_name == "top_center":
                # 领 → neck 作为中线锚点，不外推
                dst_pt = body_pt
            else:
                dst_pt = _cover_outpush(
                    body_pt, cloth_pt, neck_pt, hip_mid_pt,
                    cloth_anchor_pt, affine_scale,
                )
            pairs.append([
                [float(cloth_pt[0]), float(cloth_pt[1])],
                [float(dst_pt[0]), float(dst_pt[1])],
            ])
            if cloth_name != "top_center":
                logger.info(
                    "  外推 %-14s body=(%.0f,%.0f) → dst=(%.0f,%.0f)",
                    cloth_name, body_pt[0], body_pt[1], dst_pt[0], dst_pt[1],
                )

        semantic_pairs = np.array(pairs, dtype=np.float32)
        logger.info(
            "TPS 语义对应点数：%d（affine 参考 scale=%.3f）",
            len(pairs), affine_scale,
        )

    warped_rgb, warped_mask = warp_clothing(
        clothing_fg, clothing_mask, clothing_pts, body_pts,
        out_shape=person_img.shape[:2],
        clothing_anchor=cloth_anchor,
        body_anchor=body_anchor,
        method=args.warp_method,
        semantic_pairs=semantic_pairs,
    )

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
