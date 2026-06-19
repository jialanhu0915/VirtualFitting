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


def _estimate_similarity_transform(
    src: np.ndarray, dst: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama 1991 闭式解：2D 相似变换 (scale, R 2x2, t 2,) 最小化
    sum ||s · R · src_i + t - dst_i||²。

    用于把衣服轮廓对齐到人体轮廓。Stage A 的核心：从 30 对
    (clothing_pts, body_pts) 估计一个旋转+缩放+平移变换，让衣服图
    按这个变换 warp 到人体坐标系，"按身体轮廓贴合"。

    Args:
        src: (N, 2) 源点（衣服图坐标系）
        dst: (N, 2) 目标点（人体图坐标系）

    Returns:
        (scale, R, t) —— 用法 dst_pred = scale * R @ src + t
    """
    assert src.shape == dst.shape and src.shape[1] == 2, \
        f"src/dst must be (N, 2); got {src.shape} / {dst.shape}"

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    # 协方差矩阵 H = src_c.T @ dst_c
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)

    # 处理反射：det(R) 必须为 +1（旋转），不是 -1（镜像）
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T

    var_src = float((src_c ** 2).sum())
    scale = float(S.sum()) / var_src if var_src > 1e-12 else 1.0
    t = dst_mean - scale * R @ src_mean
    return scale, R, t


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


def _body_silhouette_per_row(
    body_pts: np.ndarray, ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """对每个 y 求身体轮廓左右边（多边形边插值）。

    body_pts 是闭合多边形（N, 2）。对每条边 (x1,y1)→(x2,y2)，若 y 在该边
    的 y 跨度内，线性插值得到该 y 处的 x；所有跨越该 y 的边的 x 取 min/max
    即为左右边。

    y 在多边形 y 范围外的返回 NaN（应跳过）。
    """
    n = len(body_pts)
    body_left = np.full(len(ys), np.inf, dtype=np.float64)
    body_right = np.full(len(ys), -np.inf, dtype=np.float64)

    for i in range(n):
        x1, y1 = float(body_pts[i, 0]), float(body_pts[i, 1])
        x2, y2 = float(body_pts[(i + 1) % n, 0]), float(body_pts[(i + 1) % n, 1])
        if y1 == y2:
            continue
        y_lo, y_hi = (y1, y2) if y1 < y2 else (y2, y1)
        mask = (ys >= y_lo) & (ys <= y_hi)
        if not mask.any():
            continue
        t_edge = (ys[mask] - y1) / (y2 - y1)
        x_interp = x1 + t_edge * (x2 - x1)
        body_left[mask] = np.minimum(body_left[mask], x_interp)
        body_right[mask] = np.maximum(body_right[mask], x_interp)

    body_left[~np.isfinite(body_left)] = np.nan
    body_right[~np.isfinite(body_right)] = np.nan
    return body_left, body_right


def _build_body_regions(
    body_pts: np.ndarray, out_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """基于 body_pts 顶点构造 3 个区域 mask（躯干/左袖/右袖）。

    body_pts 顶点顺序（10 点 arm-aware）：
        0 neck_top, 1 lsh, 2 larmpit, 3 lelbow, 4 lwrist,
        5 bottom_center, 6 rwrist, 7 relbow, 8 rarmpit, 9 rsh

    区域：
      躯干: neck_top → lsh(顶点1) → lhip line 到底中 → rhip line 到 rsh(顶点9)
      左袖: 顶点 1-4 之间的"袖子带"（肩外 → 腋 → 腕内侧 → 腕外）
      右袖: 镜像

    Returns:
        (mask_torso, mask_lsleeve, mask_rsleeve) — 三个 (H, W) bool 数组。
        同一像素可能被多个 mask 覆盖（袖子/躯干交接处）；用 np.logical_or
        累加，最终优先级：lsleeve > rsleeve > torso。
    """
    H, W = out_shape
    # body_pts 是采样后的（n_points 个），不是 10 个原顶点——不能直接用
    # "第 1 个 = lsh"这种假设。改用极值法：

    pts = body_pts
    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())

    # 区域 1（躯干）：用 x 的"中间 60%" 作为躯干，y 范围 = 整个 body
    # （手臂外缘 x 偏离躯干 > 30px 才算袖子，所以袖子 = 身体外侧 20%）
    torso_x_lo = x_min + 0.20 * (x_max - x_min)
    torso_x_hi = x_min + 0.80 * (x_max - x_min)
    yy, xx = np.mgrid[0:H, 0:W]
    mask_torso = (xx >= torso_x_lo) & (xx <= torso_x_hi) \
        & (yy >= y_min) & (yy <= y_max)

    # 区域 2/3（袖子）：取身体 x 外侧 22% 范围。y 范围 = 腋下到腕。
    # 用 body_pts y 分布的 25% 和 95% 分位近似"腋下"和"腕"。
    ys_sorted = np.sort(pts[:, 1])
    armpit_y = float(ys_sorted[int(len(ys_sorted) * 0.30)])
    wrist_y = float(ys_sorted[int(len(ys_sorted) * 0.95)])

    # 左袖：图像坐标右侧（x 大），x >= torso_x_hi
    mask_lsleeve = (xx >= torso_x_hi) \
        & (yy >= armpit_y) & (yy <= wrist_y)
    # 右袖：图像坐标左侧（x 小），x <= torso_x_lo
    mask_rsleeve = (xx <= torso_x_lo) \
        & (yy >= armpit_y) & (yy <= wrist_y)

    return mask_torso, mask_lsleeve, mask_rsleeve


def _region_silhouette_per_row(
    body_pts: np.ndarray,
    cloth_mask: np.ndarray,
    region_mask: np.ndarray,
    ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """对每个 y，在 region_mask 内的 body 边界 + cloth 像素上取 silhouette。

    body_silhouette：先算 body_pts 全行 silhouette，再 mask 掉 region 外的 x。
    cloth_silhouette：在 region_mask 内扫 cloth 像素的左右边界。

    Args:
        body_pts: 身体轮廓 (N, 2)
        cloth_mask: 衣服 mask (H, W)
        region_mask: 区域 bool mask (H, W)
        ys: 采样 y 数组

    Returns:
        (body_left, body_right, cloth_left, cloth_right) — 长度 len(ys)。
        无效位置 NaN 或 -1。
    """
    h, w = cloth_mask.shape

    # body silhouette：先全行，再 mask
    body_left_full, body_right_full = _body_silhouette_per_row(body_pts, ys)

    # 在每个 y 的 body 范围内，限制到 region_mask 的 x 范围
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    body_left = np.full(len(ys), np.nan, dtype=np.float64)
    body_right = np.full(len(ys), np.nan, dtype=np.float64)
    for i, y in enumerate(yi):
        if not np.isfinite(body_left_full[i]) or not np.isfinite(body_right_full[i]):
            continue
        bl, br = int(body_left_full[i]), int(body_right_full[i])
        region_xs = np.where(region_mask[y, bl:br + 1])[0]
        if len(region_xs) == 0:
            continue
        body_left[i] = bl + region_xs[0]
        body_right[i] = bl + region_xs[-1]

    # cloth silhouette：在 region_mask 内扫描
    cloth_left = np.full(len(ys), -1.0, dtype=np.float64)
    cloth_right = np.full(len(ys), -1.0, dtype=np.float64)
    for i, y in enumerate(yi):
        row_cloth = (cloth_mask[y] > 0) & region_mask[y]
        if not row_cloth.any():
            continue
        cloth_left[i] = float(np.argmax(row_cloth))
        cloth_right[i] = float(w - 1 - np.argmax(row_cloth[::-1]))

    return body_left, body_right, cloth_left, cloth_right


def _warp_affine_stage_a(
    clothing_rgb: np.ndarray,
    clothing_mask: np.ndarray,
    clothing_pts: np.ndarray,
    body_pts: np.ndarray,
    clothing_anchor: tuple[float, float] | None,
    body_anchor: tuple[float, float] | None,
    out_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Stage A: 仿射粗定位（按身体外接矩形 + 领口锚点对齐）。

    旧"用 8 对语义关键点做 TPS warp"路径是错误的根本方向——TPS
    按点插值不按轮廓贴合，衣服被按点拉扯成"翼"/"舌头"。Stage A
    用简单的外接矩形 + 锚点对齐做"按整体轮廓贴合"：

      scale = max(body_w / cloth_w, body_h / cloth_h) * 1.05
      tx, ty 让 cloth 领口（clothing_anchor）映射到 body 脖子（body_anchor）

    这保证衣服覆盖身体区域（max 保证两个方向都不小于 1.0，加 1.05
    余量），且领口对齐人体脖子（保证纵向位置正确）。

    Args:
        clothing_rgb: 衣服 RGB 图 (Hc, Wc, 3)
        clothing_mask: 衣服二值 mask (Hc, Wc)
        clothing_pts / body_pts: 轮廓采样点，用于估算 bbox
        clothing_anchor: 衣服领口锚点 (x, y)
        body_anchor: 人体脖子锚点 (x, y)
        out_shape: 输出图尺寸 (H, W)

    Returns:
        warped_rgb, warped_mask — 都对齐到 out_shape（人体图坐标系）
    """
    n = min(len(clothing_pts), len(body_pts))
    src_pts = clothing_pts[:n].astype(np.float32)
    dst_pts = body_pts[:n].astype(np.float32)

    H, W = out_shape

    sx_min, sy_min = src_pts[:, 0].min(), src_pts[:, 1].min()
    sx_max, sy_max = src_pts[:, 0].max(), src_pts[:, 1].max()
    dx_min, dy_min = dst_pts[:, 0].min(), dst_pts[:, 1].min()
    dx_max, dy_max = dst_pts[:, 0].max(), dst_pts[:, 1].max()
    src_w = max(sx_max - sx_min, 1)
    src_h = max(sy_max - sy_min, 1)
    dst_w = max(dx_max - dx_min, 1)
    dst_h = max(dy_max - dy_min, 1)
    scale = max(dst_w / src_w, dst_h / src_h) * 1.05

    if clothing_anchor is None:
        clothing_anchor = (float(sx_min + sx_max) / 2, float(sy_min))
    if body_anchor is None:
        body_anchor = (float(dx_min + dx_max) / 2, float(dy_min))

    cx, cy = clothing_anchor
    bx, by = body_anchor
    tx = bx - cx * scale
    ty = by - cy * scale
    logger.info(
        "Stage A affine: scale=%.4f  tx=%.1f  ty=%.1f  "
        "(clothing_anchor=(%.0f,%.0f) → body_anchor=(%.0f,%.0f))",
        scale, tx, ty, cx, cy, bx, by,
    )
    M_aff = np.array([
        [scale, 0, tx],
        [0, scale, ty],
    ], dtype=np.float32)

    warped_rgb = cv2.warpAffine(
        clothing_rgb, M_aff, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        clothing_mask, M_aff, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return warped_rgb, warped_mask


def _warp_flow(
    stage_a_rgb: np.ndarray,
    stage_a_mask: np.ndarray,
    body_pts: np.ndarray,
    out_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Stage B: 流水式逐行 stretch（分块 fit）。

    Stage A 已经把衣服粗定位到身体区域（领口对齐 + bbox 缩放）。
    本函数按"流水"语义逐行把衣服 fit 到身体轮廓。**Step 3**：
    把人体分成 3 个独立区域（躯干/左袖/右袖），各区域分别算 (s, t)，
    避免 arm-aware silhouette 把衣服主体横向撑宽 + 袖子糊在躯干两侧。

    每个 y 对每个区域独立算：
      s(y, region) = body_w_region / cloth_w_region
      t(y, region) = body_left_region - s * cloth_left_region

    remap 时按 region_id 选 s, t。

    短袖场景：袖子区域在 y > shoulder 无 cloth 像素 → valid=False →
    s=1, t=0 → 袖子消失，等价旧行为。

    Args:
        stage_a_rgb: Stage A 仿射后的衣服 RGB
        stage_a_mask: Stage A 仿射后的衣服 mask
        body_pts: arm-aware 身体轮廓
        out_shape: 输出 (H, W)
    """
    H_out, W_out = out_shape

    rows_with_cloth = np.where(stage_a_mask.any(axis=1))[0]
    if len(rows_with_cloth) == 0:
        return (
            np.zeros((H_out, W_out, 3), dtype=np.uint8),
            np.zeros((H_out, W_out), dtype=np.uint8),
        )
    top_y = float(rows_with_cloth[0])
    body_bottom = float(body_pts[:, 1].max())

    # 采样 y：从 top_y 到 body_bottom
    n_samples = 30
    ys_query = np.linspace(top_y, max(body_bottom, top_y + 1), n_samples)

    # 3 个区域 mask (H, W) bool
    mask_torso, mask_lsleeve, mask_rsleeve = _build_body_regions(
        body_pts, (H_out, W_out),
    )

    # 为 region_id 提供优先级：lsleeve > rsleeve > torso
    # 不在任一 region 内：region_id = -1
    region_id = np.full((H_out, W_out), -1, dtype=np.int8)
    region_id[mask_torso] = 0
    region_id[mask_rsleeve] = 2   # 先覆盖右袖（图像左侧）
    region_id[mask_lsleeve] = 1   # 再覆盖左袖（图像右侧，优先级最高）

    # 对每个 region 独立算 silhouette 和 (s, t)
    # region_data: list of dict {name, mask, s, t}
    region_masks = [
        ("torso", mask_torso),
        ("lsleeve", mask_lsleeve),
        ("rsleeve", mask_rsleeve),
    ]
    region_s: dict[str, np.ndarray] = {}
    region_t: dict[str, np.ndarray] = {}
    region_valid: dict[str, np.ndarray] = {}

    for name, rmask in region_masks:
        bl, br, cl, cr = _region_silhouette_per_row(
            body_pts, stage_a_mask, rmask, ys_query,
        )
        s_r = np.ones(n_samples, dtype=np.float64)
        t_r = np.zeros(n_samples, dtype=np.float64)
        valid_r = (
            np.isfinite(bl) & np.isfinite(br)
            & (cl >= 0) & (cr > cl)
            & (br > bl)
        )
        bw = br - bl
        cw = cr - cl
        s_r[valid_r] = bw[valid_r] / cw[valid_r]
        # clamp-before-t（与 558cd0a 一致）
        s_r[valid_r] = np.clip(s_r[valid_r], 0.7, 1.5)
        t_r[valid_r] = bl[valid_r] - s_r[valid_r] * cl[valid_r]
        region_s[name] = s_r
        region_t[name] = t_r
        region_valid[name] = valid_r

    # V 领保留：s=1, t=0（仅 torso；袖子无 V 领概念）
    V_NECK_SAMPLES = 3
    for name in ("torso", "lsleeve", "rsleeve"):
        s_r = region_s[name]
        t_r = region_t[name]
        valid_r = region_valid[name]
        # 锚定 top_y
        s_r[0] = 1.0
        t_r[0] = 0.0
        # V 领保留仅作用于躯干（袖子无需保留）
        if name != "torso":
            region_valid[name] = valid_r | (np.arange(n_samples) == 0)
            continue
        # 50/50 平滑过渡（同 _warp_flow 旧逻辑）
        if V_NECK_SAMPLES < n_samples and bool(valid_r[V_NECK_SAMPLES]):
            s_trans_raw = float(s_r[V_NECK_SAMPLES])
            t_trans_raw = float(t_r[V_NECK_SAMPLES])
        else:
            s_trans_raw = 1.0
            t_trans_raw = 0.0
        for i in range(1, V_NECK_SAMPLES):
            s_r[i] = 1.0
            t_r[i] = 0.0
        if V_NECK_SAMPLES < n_samples:
            s_r[V_NECK_SAMPLES] = 0.5 + 0.5 * s_trans_raw
            t_r[V_NECK_SAMPLES] = 0.5 * t_trans_raw
        region_valid[name] = valid_r | (np.arange(n_samples) == 0)

    # 限制 s 范围到 [0.7, 1.5]
    for name in region_s:
        region_s[name] = np.clip(region_s[name], 0.7, 1.5)

    # 把 s(y), t(y) 在 y 方向 linear interp 到 H_out
    body_top = float(body_pts[:, 1].min())
    body_bot = float(body_pts[:, 1].max())
    ys_all = np.arange(H_out, dtype=np.float64)
    in_body = (ys_all >= body_top) & (ys_all <= body_bot)

    s_per_region: dict[str, np.ndarray] = {}
    t_per_region: dict[str, np.ndarray] = {}
    for name, _ in region_masks:
        s_r = region_s[name]
        t_r = region_t[name]
        valid_r = region_valid[name]
        ys_used = ys_query[valid_r]
        s_used = s_r[valid_r]
        t_used = t_r[valid_r]
        s_full = np.ones(H_out, dtype=np.float64)
        t_full = np.zeros(H_out, dtype=np.float64)
        if len(ys_used) > 1:
            s_in = np.interp(ys_all[in_body], ys_used, s_used)
            t_in = np.interp(ys_all[in_body], ys_used, t_used)
            s_full[in_body] = np.clip(s_in, 0.7, 1.5)
            t_full[in_body] = t_in
        s_per_region[name] = s_full
        t_per_region[name] = t_full

    # 构造 per-region map_x, map_y：每个 region 一张
    # 选最大：W_in 是 stage_a_rgb 的宽度
    W_in = stage_a_rgb.shape[1]
    grid_y, grid_x = np.mgrid[0:H_out, 0:W_out]

    # 合并：map_x[y, x] = (x - t[region_id[y, x]](y)) / s[region_id[y, x]](y)
    # 用 lookup 形式
    s_map = np.ones((H_out, W_out), dtype=np.float32)
    t_map = np.zeros((H_out, W_out), dtype=np.float32)
    s_map[region_id == 0] = s_per_region["torso"][region_id == 0][:, np.newaxis] \
        if False else np.broadcast_to(
            s_per_region["torso"][:, np.newaxis], (H_out, W_out)
        )[region_id == 0]
    t_map[region_id == 0] = np.broadcast_to(
        t_per_region["torso"][:, np.newaxis], (H_out, W_out)
    )[region_id == 0]
    s_map[region_id == 1] = np.broadcast_to(
        s_per_region["lsleeve"][:, np.newaxis], (H_out, W_out)
    )[region_id == 1]
    t_map[region_id == 1] = np.broadcast_to(
        t_per_region["lsleeve"][:, np.newaxis], (H_out, W_out)
    )[region_id == 1]
    s_map[region_id == 2] = np.broadcast_to(
        s_per_region["rsleeve"][:, np.newaxis], (H_out, W_out)
    )[region_id == 2]
    t_map[region_id == 2] = np.broadcast_to(
        t_per_region["rsleeve"][:, np.newaxis], (H_out, W_out)
    )[region_id == 2]

    # 默认（region_id == -1）: s=1, t=0（不变）
    s_map[region_id == -1] = 1.0
    t_map[region_id == -1] = 0.0

    safe_s = np.where(np.abs(s_map) > 1e-6, s_map, 1.0)
    valid_s = (np.abs(s_map) > 1e-6) & np.isfinite(s_map)
    map_x = np.where(
        valid_s,
        (grid_x - t_map) / safe_s,
        grid_x.astype(np.float32),
    ).astype(np.float32)
    map_x = np.clip(map_x, 0, W_in - 1).astype(np.float32)
    map_y = grid_y.astype(np.float32)

    # 调试日志：每个 region 的 s 范围
    log_lines = []
    for name, _ in region_masks:
        v = region_valid[name]
        if v.sum() > 1:
            log_lines.append(
                f"{name}=[{float(region_s[name][v].min()):.2f},{float(region_s[name][v].max()):.2f}]"
            )
    logger.info(
        "Stage B flow (region): top_y=%.0f body_bottom=%.0f valid=",
        top_y, body_bottom,
    )
    for line in log_lines:
        logger.info("  %s", line)

    warped_rgb = cv2.remap(
        stage_a_rgb, map_x, map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    warped_mask = cv2.remap(
        stage_a_mask, map_x, map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return warped_rgb, warped_mask


def _warp_tps(
    clothing_rgb: np.ndarray,
    clothing_mask: np.ndarray,
    semantic_pairs: np.ndarray,
    out_shape: tuple[int, int],
    clothing_pts: np.ndarray,
    body_pts: np.ndarray,
    clothing_anchor: tuple[float, float] | None = None,
    body_anchor: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage A: 仿射粗定位（按身体外接矩形 + 领口锚点对齐）。

    旧"用 8 对语义关键点做 TPS warp"路径是错误的根本方向——TPS
    按点插值不按轮廓贴合，衣服被按点拉扯成"翼"/"舌头"。Stage A
    用简单的外接矩形 + 锚点对齐做"按整体轮廓贴合"：

      scale = max(body_w / cloth_w, body_h / cloth_h) * 1.05
      tx, ty 让 cloth 领口（clothing_anchor）映射到 body 脖子（body_anchor）

    这保证衣服覆盖身体区域（max 保证两个方向都不小于 1.0，加 1.05
    余量），且领口对齐人体脖子（保证纵向位置正确）。

    Stage B（residual TPS 修正姿态偏差）将在后续 commit 加入。

    Args:
        clothing_rgb: 衣服 RGB 图 (Hc, Wc, 3)
        clothing_mask: 衣服二值 mask (Hc, Wc)
        semantic_pairs: 保留参数以兼容接口，Stage A 暂不使用
        out_shape: 输出图尺寸 (H, W)
        clothing_pts: 衣服轮廓采样点 (n, 2)，衣服图坐标系
        body_pts: 人体躯干轮廓 (n, 2)，人体图坐标系
        clothing_anchor: 衣服领口锚点 (x, y)，默认 None=用轮廓顶部中点
        body_anchor: 人体脖子锚点 (x, y)，默认 None=用 body_contour 顶部中点

    Returns:
        warped_rgb, warped_mask — 都对齐到 out_shape
    """
    n = min(len(clothing_pts), len(body_pts))
    src_pts = clothing_pts[:n].astype(np.float32)
    dst_pts = body_pts[:n].astype(np.float32)

    H, W = out_shape

    sx_min, sy_min = src_pts[:, 0].min(), src_pts[:, 1].min()
    sx_max, sy_max = src_pts[:, 0].max(), src_pts[:, 1].max()
    dx_min, dy_min = dst_pts[:, 0].min(), dst_pts[:, 1].min()
    dx_max, dy_max = dst_pts[:, 0].max(), dst_pts[:, 1].max()
    src_w = max(sx_max - sx_min, 1)
    src_h = max(sy_max - sy_min, 1)
    dst_w = max(dx_max - dx_min, 1)
    dst_h = max(dy_max - dy_min, 1)
    scale = max(dst_w / src_w, dst_h / src_h) * 1.05

    if clothing_anchor is None:
        clothing_anchor = (float(sx_min + sx_max) / 2, float(sy_min))
    if body_anchor is None:
        body_anchor = (float(dx_min + dx_max) / 2, float(dy_min))

    cx, cy = clothing_anchor
    bx, by = body_anchor
    tx = bx - cx * scale
    ty = by - cy * scale
    logger.info(
        "Stage A affine: scale=%.4f  tx=%.1f  ty=%.1f  "
        "(clothing_anchor=(%.0f,%.0f) → body_anchor=(%.0f,%.0f))",
        scale, tx, ty, cx, cy, bx, by,
    )
    M_aff = np.array([
        [scale, 0, tx],
        [0, scale, ty],
    ], dtype=np.float32)

    warped_rgb = cv2.warpAffine(
        clothing_rgb, M_aff, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        clothing_mask, M_aff, (W, H),
        flags=cv2.INTER_LINEAR,
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
        method: ``"affine"``（默认）只做 Stage A 仿射；``"flow"`` Stage A
            + 流水式逐行 stretch；``"tps"`` 用 semantic_pairs 做薄板样条
            非线性形变（已不推荐——见 _warp_tps 注释）。
        semantic_pairs: (n, 2, 2) 数组，仅 method="tps" 时必填。

    Returns:
        warped_rgb, warped_mask — 都对齐到 out_shape
    """
    if method == "tps":
        if semantic_pairs is None:
            raise ValueError("method='tps' 需要提供 semantic_pairs")
        return _warp_tps(
            clothing_rgb, clothing_mask,
            cast(np.ndarray, semantic_pairs), out_shape,
            clothing_contour, body_contour,
            clothing_anchor=clothing_anchor,
            body_anchor=body_anchor,
        )

    if method == "flow":
        # Stage A 粗定位 → Stage B 流水式逐行 fit
        stage_a_rgb, stage_a_mask = _warp_affine_stage_a(
            clothing_rgb, clothing_mask,
            clothing_contour, body_contour,
            clothing_anchor=clothing_anchor,
            body_anchor=body_anchor,
            out_shape=out_shape,
        )
        return _warp_flow(stage_a_rgb, stage_a_mask, body_contour, out_shape)

    # 默认 affine：仅 Stage A
    return _warp_affine_stage_a(
        clothing_rgb, clothing_mask,
        clothing_contour, body_contour,
        clothing_anchor=clothing_anchor,
        body_anchor=body_anchor,
        out_shape=out_shape,
    )


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
