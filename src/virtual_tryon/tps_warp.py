"""TPS 变形 + 融合：将平铺服装 warp 到人体躯干区域。"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _align_contours(
    src: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """对齐两个轮廓的起点和方向。

    衣服轮廓"最高点"(y 最小) ≈ 领口 → 对齐到人体轮廓"最上点"。
    两者按相同方向（逆时针）排列后返回。
    """
    # 衣服：最高点的索引
    src_top = int(np.argmin(src[:, 1]))
    src = np.roll(src, -src_top, axis=0)
    # 人体：最上点的索引
    dst_top = int(np.argmin(dst[:, 1]))
    dst = np.roll(dst, -dst_top, axis=0)
    # 检查方向一致性：若 x 方向相反则翻转衣服
    if (src[1, 0] - src[0, 0]) * (dst[1, 0] - dst[0, 0]) < 0:
        src = src[::-1]
    return src, dst


def _tps_coefficients(
    src: np.ndarray, dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算 TPS 系数。

    Returns:
        w_x, w_y: 非线性部分系数 (n,)
        a_x, a_y: 仿射部分系数 (3,)
    """
    n = len(src)
    # K 矩阵: K_ij = r^2 * log(r), r = ||src_i - src_j||
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dx = src[i, 0] - src[:, 0]
        dy = src[i, 1] - src[:, 1]
        r2 = dx * dx + dy * dy
        # U(r) = r^2 * log(r), r>0; U(0)=0
        with np.errstate(divide="ignore", invalid="ignore"):
            K[i] = np.where(r2 > 1e-12, 0.5 * r2 * np.log(r2 + 1e-12), 0.0)

    # P 矩阵: [1, x, y]
    P = np.column_stack([np.ones(n), src[:, 0], src[:, 1]])
    # L = [K  P; P^T  0]
    L = np.zeros((n + 3, n + 3), dtype=np.float64)
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T

    # 右端: 目标坐标
    B = np.zeros((n + 3, 2), dtype=np.float64)
    B[:n, 0] = dst[:, 0]
    B[:n, 1] = dst[:, 1]

    coeffs = np.linalg.solve(L, B)
    w = coeffs[:n]     # (n, 2)
    a = coeffs[n:]     # (3, 2)
    return w[:, 0], w[:, 1], a[:, 0], a[:, 1]


def _tps_warp_points(
    src_ctl: np.ndarray,
    w_x: np.ndarray, w_y: np.ndarray,
    a_x: np.ndarray, a_y: np.ndarray,
    grid_pts: np.ndarray,
) -> np.ndarray:
    """用 TPS 系数将 grid_pts 从 src 空间映射到 dst 空间。

    Args:
        src_ctl: TPS 控制点 (n, 2)
        w_x, w_y: 非线性权重 (n,)
        a_x, a_y: 仿射系数 (3,)
        grid_pts: 要映射的点 (M, 2)

    Returns:
        warped (M, 2)
    """
    M = len(grid_pts)
    mapped = np.zeros((M, 2), dtype=np.float32)
    for i, (gx, gy) in enumerate(grid_pts):
        # 仿射部分
        mapped[i, 0] = a_x[0] + a_x[1] * gx + a_x[2] * gy
        mapped[i, 1] = a_y[0] + a_y[1] * gx + a_y[2] * gy
        # 非线性部分
        dx = gx - src_ctl[:, 0]
        dy = gy - src_ctl[:, 1]
        r2 = dx * dx + dy * dy
        with np.errstate(divide="ignore", invalid="ignore"):
            U = np.where(r2 > 1e-12, 0.5 * r2 * np.log(r2 + 1e-12), 0.0)
        mapped[i, 0] += np.dot(U, w_x)
        mapped[i, 1] += np.dot(U, w_y)
    return mapped


def warp_clothing(
    clothing_rgb: np.ndarray,
    clothing_mask: np.ndarray,
    clothing_contour: np.ndarray,
    body_contour: np.ndarray,
    out_shape: tuple[int, int],
    clothing_anchor: tuple[float, float] | None = None,
    body_anchor: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """将衣服 RGB 图和 mask 一起 warp 到人体坐标系。

    Args:
        clothing_rgb: 衣服 RGB 图 (Hc, Wc, 3)
        clothing_mask: 衣服二值 mask (Hc, Wc)
        clothing_contour: 衣服轮廓采样点 (n, 2)，在衣服图坐标系
        body_contour: 人体躯干轮廓 (n, 2)，在人体图坐标系
        out_shape: 输出图尺寸 (H, W)
        clothing_anchor: 衣服领口锚点 (x, y)，默认 None=用轮廓顶部中点
        body_anchor: 人体脖子锚点 (x, y)，默认 None=用 body_contour 顶部中点

    Returns:
        warped_rgb, warped_mask — 都对齐到 out_shape
    """
    n = min(len(clothing_contour), len(body_contour))
    src_pts = clothing_contour[:n].astype(np.float32)
    dst_pts = body_contour[:n].astype(np.float32)

    H, W = out_shape

    # 用外接矩形估算缩放（cover 身体区域）
    sx_min, sy_min = src_pts[:, 0].min(), src_pts[:, 1].min()
    sx_max, sy_max = src_pts[:, 0].max(), src_pts[:, 1].max()
    dx_min, dy_min = dst_pts[:, 0].min(), dst_pts[:, 1].min()
    dx_max, dy_max = dst_pts[:, 0].max(), dst_pts[:, 1].max()
    src_w = max(sx_max - sx_min, 1)
    src_h = max(sy_max - sy_min, 1)
    dst_w = max(dx_max - dx_min, 1)
    dst_h = max(dy_max - dy_min, 1)
    scale = max(dst_w / src_w, dst_h / src_h) * 1.05

    # 锚点定位：衣服领口 → 人体脖子
    if clothing_anchor is None:
        clothing_anchor = (float(sx_min + sx_max) / 2, float(sy_min))
    if body_anchor is None:
        body_anchor = (float(dx_min + dx_max) / 2, float(dy_min))

    cx, cy = clothing_anchor
    bx, by = body_anchor
    tx = bx - cx * scale
    ty = by - cy * scale
    M_affine = np.array([
        [scale, 0, tx],
        [0, scale, ty],
    ], dtype=np.float32)

    # 仿射 warp
    warped_rgb = cv2.warpAffine(clothing_rgb, M_affine, (W, H),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
    warped_mask = cv2.warpAffine(clothing_mask, M_affine, (W, H),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)

    return warped_rgb, warped_mask


def blend(
    person_rgb: np.ndarray,
    warped_rgb: np.ndarray,
    warped_mask: np.ndarray,
) -> np.ndarray:
    """将 warp 后的衣服叠加到人体图上（简单 alpha 混合）。

    mask 边缘做高斯模糊以减轻接缝。
    """
    mask_f = warped_mask.astype(np.float32) / 255.0
    # 边缘羽化
    kernel = max(3, int(min(mask_f.shape[:2]) * 0.02) // 2 * 2 + 1)
    mask_f = cv2.GaussianBlur(mask_f, (kernel, kernel), 0)

    out = person_rgb.astype(np.float32) * (1 - mask_f[:, :, np.newaxis]) \
        + warped_rgb.astype(np.float32) * mask_f[:, :, np.newaxis]
    return np.clip(out, 0, 255).astype(np.uint8)
