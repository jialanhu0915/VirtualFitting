"""TPS 变形 + 融合：将平铺服装 warp 到人体躯干区域。"""

from __future__ import annotations

import logging
from typing import cast

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# _tps_warp_dense 的分块大小：每块像素数 = 5 万，对应 ~224x224 的小方阵。
# 控制点 n=30 时单块内存峰值 ~24MB，足够 4K 图也不会爆。
_TPS_CHUNK_SIZE = 50_000


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


def _tps_warp_dense(
    src_ctl: np.ndarray,
    w_x: np.ndarray, w_y: np.ndarray,
    a_x: np.ndarray, a_y: np.ndarray,
    out_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """TPS 稠密 warp：把 out_shape 内每个像素从 src_ctl 空间映射到 dst 空间。

    与 _tps_warp_points 的区别：本函数对整张输出图一次性算映射场，
    整块算会爆内存（n=30, H*W=80万时 U 矩阵 ~192MB），所以按
    _TPS_CHUNK_SIZE 分块；块内整组向量化，无 Python 循环。

    Args:
        src_ctl: TPS 控制点 (n, 2)，必须与 _tps_coefficients 的 src 一致。
        w_x, w_y, a_x, a_y: _tps_coefficients 返回的系数。
        out_shape: 输出图尺寸 (H, W)。

    Returns:
        (map_x, map_y) 形状均为 (H, W) float32，可直接喂给 cv2.remap。
    """
    H, W = out_shape
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    flat = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
    M = flat.shape[0]

    out = np.empty((M, 2), dtype=np.float32)
    src_x = src_ctl[:, 0]
    src_y = src_ctl[:, 1]
    for start in range(0, M, _TPS_CHUNK_SIZE):
        end = min(start + _TPS_CHUNK_SIZE, M)
        chunk = flat[start:end]                          # (k, 2)
        dx = chunk[:, 0:1] - src_x[np.newaxis, :]        # (k, n)
        dy = chunk[:, 1:2] - src_y[np.newaxis, :]
        r2 = dx * dx + dy * dy
        with np.errstate(divide="ignore", invalid="ignore"):
            U = np.where(r2 > 1e-12, 0.5 * r2 * np.log(r2 + 1e-12), 0.0)
        out[start:end, 0] = (
            a_x[0] + a_x[1] * chunk[:, 0] + a_x[2] * chunk[:, 1] + U @ w_x
        )
        out[start:end, 1] = (
            a_y[0] + a_y[1] * chunk[:, 0] + a_y[2] * chunk[:, 1] + U @ w_y
        )

    return out[:, 0].reshape(H, W).astype(np.float32), \
           out[:, 1].reshape(H, W).astype(np.float32)


def _warp_tps(
    clothing_rgb: np.ndarray,
    clothing_mask: np.ndarray,
    semantic_pairs: np.ndarray,
    out_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """用 8 对语义关键点做 TPS warp。

    与早期版本的区别：早期用 30 个 arc-length 采样点做"对应"，
    但弧长第 i 个点在语义上不对应（衣服第 8 个点是袖口、人体第 8
    个点是肩下），TPS 强行拉扯会扭曲袖子。本版本用 8 对语义点
    （领口↔颈、肩↔肩、腋下↔肘、下摆↔髋 + 底中点↔髋中点），
    由 CORRESPONDENCE 字典给出，语义一一对应。

    Args:
        clothing_rgb: 衣服 RGB 图 (Hc, Wc, 3)
        clothing_mask: 衣服二值 mask (Hc, Wc)
        semantic_pairs: (n, 2, 2) 数组，semantic_pairs[i, 0] 是衣服关键点
            （衣服图坐标系），semantic_pairs[i, 1] 是人体关键点（人体图坐标系）。
        out_shape: 输出图尺寸 (H, W)

    Returns:
        warped_rgb, warped_mask — 都对齐到 out_shape
    """
    pairs = semantic_pairs.astype(np.float64)
    src = pairs[:, 0, :]   # clothing 坐标
    dst = pairs[:, 1, :]   # body 坐标

    # 系数以 dst 为 src_ctl：f(body 坐标) -> clothing 坐标。
    w_x, w_y, a_x, a_y = _tps_coefficients(dst, src)
    map_x, map_y = _tps_warp_dense(dst, w_x, w_y, a_x, a_y, out_shape)

    warped_rgb = cv2.remap(
        clothing_rgb, map_x, map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    # mask 用最近邻保持二值性，理由同 TPS 路径注释。
    warped_mask = cv2.remap(
        clothing_mask, map_x, map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return warped_rgb, warped_mask


def warp_clothing(
    clothing_rgb: np.ndarray,
    clothing_mask: np.ndarray,
    clothing_contour: np.ndarray,
    body_contour: np.ndarray,
    out_shape: tuple[int, int],
    clothing_anchor: tuple[float, float] | None = None,
    body_anchor: tuple[float, float] | None = None,
    method: str = "affine",
    semantic_pairs: np.ndarray | None = None,
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
        method: ``"affine"``（默认）只做等比缩放+对齐；``"tps"`` 用 semantic_pairs
            提供的语义对应点做薄板样条非线性形变。
        semantic_pairs: (n, 2, 2) 数组，semantic_pairs[i, 0] 是衣服点、
            semantic_pairs[i, 1] 是对应人体点。仅 method="tps" 时必填。

    Returns:
        warped_rgb, warped_mask — 都对齐到 out_shape
    """
    if method == "tps":
        if semantic_pairs is None:
            raise ValueError("method='tps' 需要提供 semantic_pairs")
        return _warp_tps(
            clothing_rgb, clothing_mask,
            cast(np.ndarray, semantic_pairs), out_shape,
        )

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
    """将 warp 后的衣服叠加到人体图上（带边缘羽化的 alpha 混合）。

    修复白边问题：
    1. mask 收回：warpAffine 对 mask 和 rgb 是独立插值的，插值会让 mask
       边缘比 warped_rgb 实际覆盖多出一圈，导致 blend 时透出黑色 border。
       用 "mask AND (warped_rgb != 黑色 border)" 把 mask 收回到实际衣服覆盖区。
    2. mask 内缩 ~1% min_dim：衣服图本身带的"白边"（如 qipao 白底 + 分割
       dilation 引入的白边）会被 warp 出来显示成白边。预先 erode 几步消掉。
    3. 边缘羽化：高斯模糊让 mask 边界平滑过渡。
    """
    # 1. mask 收回：只在 warped_rgb 有颜色的地方使用 mask
    has_color = (warped_rgb.max(axis=2) > 0).astype(np.uint8) * 255
    mask = cv2.min(warped_mask, has_color)
    # 2. mask 内缩，去掉衣服图的白边
    erode_k = max(3, int(min(mask.shape[:2]) * 0.01) // 2 * 2 + 1)
    mask = cv2.erode(mask, np.ones((erode_k, erode_k), np.uint8))
    # 3. 边缘羽化
    mask_f = mask.astype(np.float32) / 255.0
    kernel = max(3, int(min(mask_f.shape[:2]) * 0.02) // 2 * 2 + 1)
    mask_f = cv2.GaussianBlur(mask_f, (kernel, kernel), 0)

    out = person_rgb.astype(np.float32) * (1 - mask_f[:, :, np.newaxis]) \
        + warped_rgb.astype(np.float32) * mask_f[:, :, np.newaxis]
    return np.clip(out, 0, 255).astype(np.uint8)
