"""平铺服装图的关键点检测（纯传统 CV，不依赖任何深度学习模型）。"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

from .keypoints import Keypoint

logger = logging.getLogger(__name__)

# 调试目录：默认指向一个永远不存在的占位路径，未通过 set_debug_dir() 启用时
# 所有 imwrite 调用都被短路；用 Path 而不是 None 是为了让 Pyright 在所有
# 调用点都不需要 None 守卫。
_DEBUG_DIR: Path = Path("/__virtual_tryon_debug_disabled__")


def set_debug_dir(path: Path | None) -> None:
    """设置中间产物输出目录。None 表示关闭调试输出。"""
    global _DEBUG_DIR
    _DEBUG_DIR = path if path is not None else Path("/__virtual_tryon_debug_disabled__")


class ClothingDetector:
    """从平铺服装图中提取 8 个关键点（纯传统 CV，单一方法：边缘检测）。

    分割策略：CLAHE 拉伸灰度 → 多通道 Canny → 闭运算封口 → 最大轮廓填充。
    适合所有颜色组合（含白衬衫 vs 白底），CLAHE 拉伸能放大弱边缘。
    若图像带 alpha 通道且含实际透明信息，优先用 alpha 通道抠图（成本低）。

    取最大外轮廓后，按轮廓几何特征自适应派生 8 个关键点
    （领口凹点、肩部/腋下左右极值、下摆左右端点内缩）。

    Attributes:
        keypoint_method: 关键点派生方法，"geometric"（V1，凹点+极值），
            "skeleton"（V2，骨架分叉+对称轴+曲率融合）。

    Raises:
        RuntimeError: 找不到合格的服装轮廓时抛出。
    """

    def __init__(self, keypoint_method: str = "geometric") -> None:
        if keypoint_method not in ("geometric", "skeleton"):
            raise ValueError(
                f"keypoint_method 必须是 'geometric' 或 'skeleton'，收到 {keypoint_method!r}"
            )
        self.keypoint_method = keypoint_method

    def sample_contour(
        self, image: np.ndarray, n_points: int = 30,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        """从服装图中提取轮廓并均匀采样 n_points 个点，同时返回 mask 和领口锚点。

        Returns:
            (points, mask, neck_anchor): points 是 (n_points, 2) int32 采样坐标，
            mask 是二值前景掩码。
        """
        if image.ndim == 3 and image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            mask = self._mask_from_alpha(image[:, :, 3])
            if mask is None:
                mask = self._segment_edge(rgb)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = self._segment_edge(rgb)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise RuntimeError("未找到服装轮廓")
        contour = max(contours, key=cv2.contourArea)

        # 按弧长均匀采样 n_points 个点。
        points = self._sample_evenly(contour, n_points)

        # 领口锚点：用两尖中点算法（与 _extract_keypoints 一致）
        raw_pts = contour.reshape(-1, 2)
        neck_anchor = self._find_neck_anchor(raw_pts)
        return points, mask, neck_anchor

    @staticmethod
    def _find_neck_anchor(pts: np.ndarray) -> tuple[float, float]:
        """从轮廓点中计算领口锚点（两尖中点算法）。

        取顶部 25% 点，按 x 中位分左右半，每半找最高点(y最小)，返回中点。
        与 _extract_keypoints 的领口逻辑一致，但不走完整的 8 点派生流程。
        """
        order = np.argsort(pts[:, 1])
        sorted_pts = pts[order]
        y_min = int(sorted_pts[0, 1])
        y_max = int(sorted_pts[-1, 1])
        ch = max(y_max - y_min, 1)
        band_bottom = int(y_min + ch * 0.25)
        top_band = sorted_pts[sorted_pts[:, 1] <= band_bottom]
        if len(top_band) < 2:
            return (float(pts[:, 0].mean()), float(pts[:, 1].min()))
        x_med = int(np.median(top_band[:, 0]))
        left = top_band[top_band[:, 0] <= x_med]
        right = top_band[top_band[:, 0] > x_med]
        if len(left) == 0 or len(right) == 0:
            return (float(top_band[:, 0].mean()), float(top_band[:, 1].max()))
        lp = left[np.argmin(left[:, 1])]
        rp = right[np.argmin(right[:, 1])]
        return (float((lp[0] + rp[0]) / 2), float((lp[1] + rp[1]) / 2))

    @staticmethod
    def _sample_evenly(contour: np.ndarray, n_points: int) -> np.ndarray:
        """沿轮廓弧长均匀采样 n_points 个点，返回 (n_points, 2)。"""
        pts = contour.reshape(-1, 2).astype(np.float32)
        # 首尾接合形成闭合轮廓
        if len(pts) > 1 and np.any(pts[0] != pts[-1]):
            pts = np.vstack([pts, pts[0:1]])
        # 计算段长和累计弧长
        diffs = np.diff(pts, axis=0)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = cum_len[-1]
        if total <= 0:
            # 退化：轮廓太小，直接返回原有点
            return contour.reshape(-1, 2)[:n_points]
        # 等弧长采样
        sample_lens = np.linspace(0, total, n_points, endpoint=False)
        indices = np.searchsorted(cum_len, sample_lens, side="right") - 1
        indices = np.clip(indices, 0, len(pts) - 2)
        # 在段内线性插值
        t = (sample_lens - cum_len[indices]) / np.maximum(seg_lens[indices], 1e-6)
        sampled = pts[indices] + t[:, np.newaxis] * diffs[indices]
        return sampled.astype(np.int32)

    def detect(self, image: np.ndarray) -> dict[str, Keypoint]:
        if image.ndim == 3 and image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            mask = self._mask_from_alpha(image[:, :, 3])
            if mask is None:
                # alpha 全不透明，alpha 没有任何分割信息，走边缘检测抠图。
                mask = self._segment_edge(rgb)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = self._segment_edge(rgb)

        # 把分割掩码落盘，便于调试分割质量。
        if _DEBUG_DIR != Path("/__virtual_tryon_debug_disabled__"):
            cv2.imwrite(str(_DEBUG_DIR / "clothing_1_mask.png"), mask)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise RuntimeError("未找到服装轮廓")
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 100:
            raise RuntimeError("服装轮廓过小（面积 < 100 像素），可能不是有效服装图")

        points = contour.reshape(-1, 2)

        # 把轮廓画在原图副本上落盘，便于直观对照几何派生。
        if _DEBUG_DIR != Path("/__virtual_tryon_debug_disabled__"):
            bgr = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            contour_vis = bgr.copy()
            cv2.drawContours(contour_vis, [contour], -1, (0, 255, 255), 2)
            cv2.imwrite(str(_DEBUG_DIR / "clothing_2_contour.png"), contour_vis)

        if self.keypoint_method == "skeleton":
            return self._extract_keypoints_skeleton(points, mask, image)
        return self._extract_keypoints(points, image)

    @staticmethod
    def _mask_from_alpha(alpha: np.ndarray) -> np.ndarray | None:
        """从 alpha 通道构造二值掩码；若 alpha 全不透明则返回 None。"""
        if alpha.min() >= 250:
            return None
        return ((alpha > 10).astype(np.uint8)) * 255

    def _segment_edge(self, rgb: np.ndarray) -> np.ndarray:
        """基于边缘检测的服装分割（CLAHE + 多通道 Canny + 闭运算 + 填充分割）。

        整个抠图流程的唯一方法。CLAHE 拉伸放大弱边缘，多通道 Canny 投票
        捕获色距相同但灰度突变的边界，最后取最大外轮廓填充。

        Args:
            rgb: 输入 RGB 图 (H, W, 3)。

        Returns:
            二值前景 mask (H, W)，dtype=uint8，全 0 表示放弃分割。
        """
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1. CLAHE 自适应直方图均衡——把窄灰度带拉开，暴露被压平的边缘。
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        # 2. 多通道 Canny 合并：灰度(均衡后) + B + G + R + Lab-L。
        #    多通道投票能捕获色距相同但灰度突变的边界（如浅色 vs 浅灰）。
        edges = np.zeros(bgr.shape[:2], dtype=np.uint8)
        edges = cv2.bitwise_or(edges, cv2.Canny(gray_eq, 50, 150))
        for ch in range(3):  # B, G, R 各自做 Canny
            edges = cv2.bitwise_or(edges, cv2.Canny(bgr[:, :, ch], 50, 150))
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
        edges = cv2.bitwise_or(edges, cv2.Canny(lab[:, :, 0], 50, 150))

        # 3. 闭运算 + 二次膨胀封口，把 Canny 边缘连成闭合环。
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        closed = cv2.dilate(closed, kernel, iterations=1)

        # 4. 取最大外轮廓并填充。
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return np.zeros(bgr.shape[:2], dtype=np.uint8)
        main = max(contours, key=cv2.contourArea)
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [main], 255)

        # 5. 内缩补偿：dilation 把 mask 撑到白底里 ~7 像素，回缩同等大小让 mask
        # 紧贴衣服实际边缘，避免 warp 后透出白底产生白边。
        mask = cv2.erode(mask, kernel, iterations=1)
        return mask

    def _extract_keypoints(self, points: np.ndarray, image: np.ndarray) -> dict[str, Keypoint]:
        """从轮廓点集自适应派生 8 个关键点。

        不依赖固定比例，而是利用服装轮廓的几何特征：

        - 领口 (top_center)：轮廓顶部附近最深的凹点
          （圆领/V 领都是轮廓线"内陷"的位置）。
          如果找不到凹点（例如高领无明显开口），退化为顶部中点。
        - 肩部 (left/right_shoulder)：领口往下扫描左右轮廓 x 坐标，
          第一次出现宽度明显放缓的拐点。
        - 腋下 (left/right_armpit)：肩部往下到下摆之间再次出现宽度
          明显收窄的位置（袖窿收口处）。
        - 下摆 (left/right_bottom)：下摆轮廓左右两端各内缩一点。

        对 T 恤/衬衫/旗袍/连衣裙都能适应，无需按版型调比例。
        """
        # 按 y 升序排序轮廓点，方便按行扫描。
        order = np.argsort(points[:, 1])
        pts_sorted = points[order]

        y_min = int(pts_sorted[0, 1])
        y_max = int(pts_sorted[-1, 1])
        ch: float = float(y_max - y_min)
        if ch <= 0:
            raise RuntimeError("服装轮廓高度为零，无法提取关键点")

        # 调试可视化用的画布：每次画一个新状态，覆盖到同一张图上方便对照。
        vis = None
        if _DEBUG_DIR != Path("/__virtual_tryon_debug_disabled__"):
            vis = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            vis = vis.copy()
            # 在画布左侧画 y 轴坐标，方便对照"在哪个 y 区间里"。
            for frac, label in [(0.0, "0%"), (0.15, "15%"), (0.50, "50%"),
                                (0.85, "85%"), (1.0, "100%")]:
                yy = int(y_min + frac * ch)
                cv2.line(vis, (0, yy), (vis.shape[1], yy), (180, 180, 180), 1)
                cv2.putText(vis, label, (4, yy - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        def _save_step(name: str, kp: tuple[int, int] | None,
                       band: tuple[int, int] | None,
                       color: tuple[int, int, int]) -> None:
            if vis is None:
                return
            v = vis.copy()
            if band is not None:
                # 半透明黄色矩形标注搜索带。
                overlay = v.copy()
                cv2.rectangle(overlay, (0, band[0]), (v.shape[1], band[1]),
                              (0, 255, 255), -1)
                cv2.addWeighted(overlay, 0.2, v, 0.8, 0, v)
            if kp is not None:
                cv2.circle(v, (int(kp[0]), int(kp[1])), 10, color, -1)
                cv2.circle(v, (int(kp[0]), int(kp[1])), 10, (0, 0, 0), 2)
            cv2.imwrite(str(_DEBUG_DIR / f"clothing_3_{name}.png"), v)

        # 1. 领口：取顶部 25% 轮廓点，按 x 中位分成左右两半，
        # 每半找最高点（y 最小），两点中点即为领口几何中心。
        #
        # 无论何种领型，领口总是由左右两个"尖"界定：圆领的领口
        # 左右边缘、立领的左右领角、翻领的左右翻领尖。取顶部 25%
        # 足以覆盖各种领型，同时不会包含袖子（袖端在肩部以下）。
        band_top = y_min
        band_bottom = int(y_min + ch * 0.25)
        top_band = pts_sorted[
            (pts_sorted[:, 1] >= band_top) & (pts_sorted[:, 1] <= band_bottom)
        ]
        if len(top_band) < 2:
            # 区间内点不够，退化到全图顶部中点。
            top_band = pts_sorted[pts_sorted[:, 1] <= band_bottom]
            neck_center = (
                int(top_band[:, 0].mean()),
                int(top_band[:, 1].max()),
            )
        else:
            # 按 x 中位分成左右两半，每半找最高点。
            x_med = int(np.median(top_band[:, 0]))
            left_half = top_band[top_band[:, 0] <= x_med]
            right_half = top_band[top_band[:, 0] > x_med]
            if len(left_half) == 0 or len(right_half) == 0:
                # 衣服不对称（如单侧遮挡），退化。
                neck_center = (
                    int(top_band[:, 0].mean()),
                    int(top_band[:, 1].max()),
                )
            else:
                left_peak = left_half[np.argmin(left_half[:, 1])]
                right_peak = right_half[np.argmin(right_half[:, 1])]
                neck_center = (
                    int((left_peak[0] + right_peak[0]) // 2),
                    int((left_peak[1] + right_peak[1]) // 2),
                )
        _save_step("neck", neck_center,
                   (band_top, band_bottom), (0, 0, 255))

        # 2. 肩部
        shoulder_band_top = int(neck_center[1] + ch * 0.12)
        shoulder_band_bottom = int(neck_center[1] + ch * 0.22)
        left_shoulder, right_shoulder = ClothingDetector._left_right_extrema(
            pts_sorted, shoulder_band_top, shoulder_band_bottom
        )
        _save_step("shoulder", None,
                   (shoulder_band_top, shoulder_band_bottom), (0, 255, 0))
        if vis is not None:
            v = vis.copy()
            cv2.rectangle(v, (0, shoulder_band_top),
                          (v.shape[1], shoulder_band_bottom), (0, 255, 0), 2)
            for p, c in [(left_shoulder, (255, 0, 0)), (right_shoulder, (0, 0, 255))]:
                cv2.circle(v, p, 10, c, -1)
                cv2.circle(v, p, 10, (0, 0, 0), 2)
            cv2.imwrite(str(_DEBUG_DIR / "clothing_4_shoulder_pts.png"), v)

        # 3. 腋下
        armpit_band_top = int(shoulder_band_bottom + ch * 0.03)
        armpit_band_bottom = int(neck_center[1] + ch * 0.50)
        left_armpit, right_armpit = ClothingDetector._left_right_extrema(
            pts_sorted, armpit_band_top, armpit_band_bottom
        )
        if vis is not None:
            v = vis.copy()
            cv2.rectangle(v, (0, armpit_band_top),
                          (v.shape[1], armpit_band_bottom), (0, 165, 255), 2)
            for p, c in [(left_armpit, (255, 0, 0)), (right_armpit, (0, 0, 255))]:
                cv2.circle(v, p, 10, c, -1)
                cv2.circle(v, p, 10, (0, 0, 0), 2)
            cv2.imwrite(str(_DEBUG_DIR / "clothing_5_armpit_pts.png"), v)

        # 4. 下摆
        bottom_band_top = int(y_max - ch * 0.05)
        left_bottom_raw, right_bottom_raw = ClothingDetector._left_right_extrema(
            pts_sorted, bottom_band_top, y_max
        )
        bw = right_bottom_raw[0] - left_bottom_raw[0]
        inset = int(bw * 0.08)
        left_bottom = (left_bottom_raw[0] + inset, y_max)
        right_bottom = (right_bottom_raw[0] - inset, y_max)
        if vis is not None:
            v = vis.copy()
            cv2.rectangle(v, (0, bottom_band_top), (v.shape[1], y_max),
                          (255, 0, 255), 2)
            for p, c in [(left_bottom, (255, 0, 0)), (right_bottom, (0, 0, 255))]:
                cv2.circle(v, p, 10, c, -1)
                cv2.circle(v, p, 10, (0, 0, 0), 2)
            cv2.imwrite(str(_DEBUG_DIR / "clothing_6_bottom_pts.png"), v)

        def mk(name: str, p: tuple[int, int]) -> Keypoint:
            return Keypoint(int(p[0]), int(p[1]), name=name)

        return {
            "top_center":     mk("top_center",     neck_center),
            "bottom_center":  mk("bottom_center",  ((left_bottom[0] + right_bottom[0]) // 2, y_max)),
            "left_shoulder":  mk("left_shoulder",  left_shoulder),
            "right_shoulder": mk("right_shoulder", right_shoulder),
            "left_armpit":    mk("left_armpit",    left_armpit),
            "right_armpit":   mk("right_armpit",   right_armpit),
            "left_bottom":    mk("left_bottom",    left_bottom),
            "right_bottom":   mk("right_bottom",   right_bottom),
        }

    @staticmethod
    def _left_right_extrema(
        pts: np.ndarray, y_lo: int, y_hi: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """在 y 区间 [y_lo, y_hi] 内的轮廓点中找最左和最右的点，返回 (x, y)。"""
        mask = (pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi)
        band = pts[mask]
        if len(band) == 0:
            # 退化：用全图的极值。
            band = pts
        left = band[np.argmin(band[:, 0])]
        right = band[np.argmax(band[:, 0])]
        return (int(left[0]), int(left[1])), (int(right[0]), int(right[1]))

    # ============================================================
    # V2: 骨架 + 对称轴 + 轮廓曲率 融合关键点
    # ============================================================

    @staticmethod
    def _find_symmetry_axis(mask: np.ndarray) -> float:
        """求 mask 的垂直对称轴 x 坐标。

        对每一行取左右像素的中点，整个 mask 的中点中位数就是对称轴 x。
        大部分衣服左右对称，对称轴天然穿过领口中心、底摆中点，
        可以作为 V1 "宽度谷值"的稳定 fallback 信号。
        """
        h, w = mask.shape
        midpoints: list[float] = []
        for y in range(h):
            xs = np.where(mask[y] > 0)[0]
            if len(xs) >= 2:
                midpoints.append((int(xs[0]) + int(xs[-1])) / 2.0)
        if not midpoints:
            return w / 2.0
        return float(np.median(midpoints))

    @staticmethod
    def _skeleton_features(
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """提取 mask 的骨架并按邻域像素数分类为端点/分叉/普通。

        Returns:
            (skeleton, junctions, endpoints) 都是 bool 数组，shape 与 mask 相同。
            - skeleton: 所有骨架像素
            - junctions: 骨架分叉点（≥3 个邻居），通常对应腋下/领口分叉
            - endpoints: 骨架端点（恰好 1 个邻居），通常对应领口尖端/下摆
        """
        skel_bool = skeletonize(mask > 127)
        # 3x3 邻域计数（中心点不算）
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        nb = convolve(skel_bool.astype(np.uint8), kernel, mode="constant", cval=0)
        endpoints = skel_bool & (nb == 1)
        junctions = skel_bool & (nb >= 3)
        return skel_bool, junctions, endpoints

    @staticmethod
    def _contour_curvature(contour: np.ndarray) -> np.ndarray:
        """沿闭合轮廓计算每点的曲率（带符号）。

        约定：凸角（外凸）为正，凹角（内凹）为负。
        使用 5 点中心差分；首尾循环连接。
        """
        pts = contour.reshape(-1, 2).astype(np.float64)
        n = len(pts)
        if n < 5:
            return np.zeros(n)

        # 用循环索引取邻居
        idx = np.arange(n)
        x_prev = pts[(idx - 2) % n, 0]
        x_curr = pts[idx, 0]
        x_next = pts[(idx + 2) % n, 0]
        y_prev = pts[(idx - 2) % n, 1]
        y_curr = pts[idx, 1]
        y_next = pts[(idx + 2) % n, 1]

        # 一阶导数
        dx = (x_next - x_prev) / 4.0
        dy = (y_next - y_prev) / 4.0
        # 二阶导数
        ddx = (x_next - 2 * x_curr + x_prev) / 4.0
        ddy = (y_next - 2 * y_curr + y_prev) / 4.0

        denom = (dx * dx + dy * dy) ** 1.5
        with np.errstate(divide="ignore", invalid="ignore"):
            curv = np.where(denom > 1e-6, (dx * ddy - dy * ddx) / denom, 0.0)
        return curv

    def _curvature_extrema_in_band(
        self,
        contour: np.ndarray,
        y_lo: int,
        y_hi: int,
        x_sym: float,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """在 y 区间内找曲率绝对值最大的左右一对点。

        适合检测肩峰（凸角）和腋下 V（凹角）这类轮廓转角。
        返回 ((left_x, left_y), (right_x, right_y))，按 x 排序。
        """
        pts_sorted = contour.reshape(-1, 2)
        band_mask = (pts_sorted[:, 1] >= y_lo) & (pts_sorted[:, 1] <= y_hi)
        band_pts = pts_sorted[band_mask]
        if len(band_pts) < 5:
            return None

        # 在整个 contour 上算曲率，再索引回 band
        curv = self._contour_curvature(contour)
        band_curv = curv[band_mask]
        if len(band_curv) == 0:
            return None
        abs_curv = np.abs(band_curv)

        # 左侧：x < x_sym 的点里取 |曲率| 最大
        left_mask = band_pts[:, 0] < x_sym
        right_mask = band_pts[:, 0] >= x_sym
        if not left_mask.any() or not right_mask.any():
            return None

        left_idx = np.argmax(np.where(left_mask, abs_curv, -1.0))
        right_idx = np.argmax(np.where(right_mask, abs_curv, -1.0))

        return (
            (int(band_pts[left_idx, 0]), int(band_pts[left_idx, 1])),
            (int(band_pts[right_idx, 0]), int(band_pts[right_idx, 1])),
        )

    @staticmethod
    def _find_symmetric_pair(
        peaks: list[tuple[int, float]],
        contour: np.ndarray,
        x_sym: float,
        y_lo: int,
        y_hi: int,
        exclude_idx: int | None = None,
        max_try: int = 12,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """从曲率极值候选里挑最强的左右对称一对。

        Args:
            peaks: 候选列表 [(contour_idx, strength), ...]，按强度降序。
            contour: (n, 2) 轮廓点。
            x_sym: 对称轴 x 坐标。
            y_lo, y_hi: y 区间。
            exclude_idx: 排除的索引（避免和已分配的关键点重复）。
            max_try: 只在前 max_try 个候选里搜索配对。

        评分：strength 之和 - 0.1*|y1-y2| - 0.05*|到 x_sym 距离之差|。
        排除规则：两点必须分列 x_sym 两侧。
        """
        cands = [
            (idx, s) for (idx, s) in peaks[:max_try]
            if idx != exclude_idx
            and y_lo <= int(contour[idx, 1]) <= y_hi
        ]
        best = None
        best_score = float("-inf")
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                i1, s1 = cands[i]
                i2, s2 = cands[j]
                y1, y2 = float(contour[i1, 1]), float(contour[i2, 1])
                x1, x2 = float(contour[i1, 0]), float(contour[i2, 0])
                # 必须左右分列
                if not ((x1 < x_sym) != (x2 < x_sym)):
                    continue
                y_diff = abs(y1 - y2)
                sym_diff = abs(abs(x1 - x_sym) - abs(x2 - x_sym))
                score = s1 + s2 - 0.10 * y_diff - 0.05 * sym_diff
                if score > best_score:
                    best_score = score
                    best = (i1, i2)
        if best is None:
            return None
        i1, i2 = best
        p1 = (int(contour[i1, 0]), int(contour[i1, 1]))
        p2 = (int(contour[i2, 0]), int(contour[i2, 1]))
        # 按 x 排序：左点在前
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        return p1, p2

    def _extract_keypoints_skeleton(
        self, points: np.ndarray, mask: np.ndarray, image: np.ndarray,
    ) -> dict[str, Keypoint]:
        """V2: 对称轴 + 轮廓曲率 + 宽度谷值 融合的关键点派生。

        设计动机：V1 在 V 形翻领衬衫上失败（详见 docs/notes/...）。
        V2 用两个互补信号：
            1. 领口：宽度谷值行（V1 稳定的部分）+ 骨架端点投票。
            2. 肩/腋/下摆：轮廓曲率 NMS 找所有结构转角，按"凸点对/凹点对"
               配对，再按 y 区间分配。比 V1 的固定比例带更鲁棒——只要衣服
               在那几个位置有转角，曲率就会冒尖，与长宽比/款式无关。
            3. 下摆中心：对称轴 x_sym 直接落在 y_max 行上。

        每个角色都有 V1 几何方法的 fallback。
        """
        ys_all, _ = np.where(mask > 0)
        if len(ys_all) == 0:
            raise RuntimeError("mask 为空，无法提取关键点")
        y_min, y_max = int(ys_all.min()), int(ys_all.max())
        ch = max(y_max - y_min, 1)

        x_sym = self._find_symmetry_axis(mask)

        # 1. 整条轮廓的曲率
        contour = points.reshape(-1, 2)
        n = len(contour)
        if n < 8:
            return self._extract_keypoints(points, image)
        curv = self._contour_curvature(contour)

        # 2. NMS 找凸/凹点候选
        nms_window = max(5, n // 12)
        convex_peaks: list[tuple[int, float]] = []  # (idx, strength)
        concave_peaks: list[tuple[int, float]] = []
        for i in range(n):
            is_max = is_min = True
            c_i = curv[i]
            for di in range(-nms_window, nms_window + 1):
                if di == 0:
                    continue
                j = (i + di) % n
                c_j = curv[j]
                if c_j > c_i:
                    is_max = False
                if c_j < c_i:
                    is_min = False
            if is_max and c_i > 0:
                convex_peaks.append((i, float(c_i)))
            elif is_min and c_i < 0:
                concave_peaks.append((i, float(-c_i)))
        convex_peaks.sort(key=lambda x: -x[1])
        concave_peaks.sort(key=lambda x: -x[1])

        # ---- 调试可视化 ----
        vis = None
        if _DEBUG_DIR != Path("/__virtual_tryon_debug_disabled__"):
            vis = (image if image.shape[2] == 3
                   else cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)).copy()
            cv2.line(vis, (int(x_sym), 0), (int(x_sym), vis.shape[0]),
                     (200, 200, 0), 1)
            # 凸点（蓝）/凹点（红）
            for idx, _ in convex_peaks[:30]:
                cv2.circle(vis, (int(contour[idx, 0]), int(contour[idx, 1])),
                           4, (255, 100, 0), -1)
            for idx, _ in concave_peaks[:30]:
                cv2.circle(vis, (int(contour[idx, 0]), int(contour[idx, 1])),
                           4, (0, 100, 255), -1)
            cv2.imwrite(str(_DEBUG_DIR / "clothing_v2_0_curv.png"), vis)

        # ---- top_center：上半区域最强凹点 ----
        # 用宽度谷值行作为"高度锚点"：通常谷值行 ≈ 领口高度。
        widths = (mask > 0).sum(axis=1).astype(np.float32)
        upper_limit = min(y_max, int(y_min + ch * 0.35))
        if upper_limit > y_min and (upper_limit - y_min) >= 4:
            wu = widths[y_min:upper_limit + 1].copy()
            k = np.array([1, 2, 3, 2, 1], dtype=np.float32); k /= k.sum()
            wu = np.convolve(wu, k, mode="same")
            neck_row = int(np.argmin(wu)) + y_min
        else:
            neck_row = y_min

        top_center: tuple[int, int] | None = None
        # 领口必须在 (a) 接近 neck_row ±容差、(b) 上半 40% 内、(c) 接近 x_sym
        # 三者同时满足；任一不满足则回退到 (b)+(c)，再不行回退到 V1。
        y_tol = max(int(ch * 0.06), 25)
        # 领口到对称轴的容许偏差：取 mask 顶部 5% 的平均宽度的一半
        upper_w = widths[y_min:min(y_max + 1, y_min + max(int(ch * 0.05), 5))]
        x_tol = max(int(upper_w.mean() * 0.6), 30) if len(upper_w) else 80

        # 第一优先：neck_row 附近 + 接近 x_sym
        for idx, _ in concave_peaks:
            cx, cy = int(contour[idx, 0]), int(contour[idx, 1])
            if (abs(cy - neck_row) <= y_tol
                    and cy <= y_min + ch * 0.40
                    and abs(cx - x_sym) <= x_tol):
                top_center = (cx, cy)
                break
        # 第二优先：上半 35% + 接近 x_sym
        if top_center is None:
            for idx, _ in concave_peaks:
                cx, cy = int(contour[idx, 0]), int(contour[idx, 1])
                if cy <= y_min + ch * 0.35 and abs(cx - x_sym) <= x_tol:
                    top_center = (cx, cy)
                    break
        # 第三优先：neck_row 行 mask 中点（最稳定的领口 y 高度）
        if top_center is None:
            row_xs = np.where(mask[neck_row] > 0)[0]
            if len(row_xs) > 0:
                top_center = (int(round((row_xs[0] + row_xs[-1]) / 2)), neck_row)
        # 极端 fallback：V1
        if top_center is None:
            v1_neck = self._find_neck_anchor(mask)
            top_center = (int(v1_neck[0]), int(v1_neck[1]))
        top_idx = None  # 不再用 top_idx 做排除，因为轮廓点索引和位置无关

        # ---- bottom_center：对称轴 + 底边 ----
        bottom_center = (int(round(x_sym)), y_max)

        # ---- shoulders：上半 10%~50% 内的最强凸点对 ----
        # 从 top_idx 之下开始找，确保不和领口凹点重复
        sh_y_top = int(top_center[1] + ch * 0.05)
        sh_y_bot = int(y_min + ch * 0.50)
        shoulder_pair = self._find_symmetric_pair(
            convex_peaks, contour, x_sym, sh_y_top, sh_y_bot,
        )
        if shoulder_pair is not None:
            left_shoulder, right_shoulder = shoulder_pair
        else:
            # Fallback：V1 band extrema
            band_top = int(top_center[1] + ch * 0.10)
            band_bot = int(top_center[1] + ch * 0.30)
            ls, rs = ClothingDetector._left_right_extrema(
                points, band_top, band_bot,
            )
            left_shoulder, right_shoulder = ls, rs

        # ---- armpits：肩膀 y 之下到 70% 内的最强凹点对 ----
        sh_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
        ar_y_top = int(sh_mid_y + ch * 0.03)
        ar_y_bot = int(y_min + ch * 0.70)
        armpit_pair = self._find_symmetric_pair(
            concave_peaks, contour, x_sym, ar_y_top, ar_y_bot,
            exclude_idx=top_idx,
        )
        if armpit_pair is not None:
            left_armpit, right_armpit = armpit_pair
        else:
            # Fallback A：V1 宽度收窄行
            widths_band = widths[ar_y_top:ar_y_bot + 1]
            if len(widths_band) >= 5:
                k = np.array([1, 2, 3, 2, 1], dtype=np.float32); k /= k.sum()
                widths_band_s = np.convolve(widths_band, k, mode="same")
                narrow_row = int(np.argmin(widths_band_s)) + ar_y_top
                row_xs = np.where(mask[narrow_row] > 0)[0]
                if len(row_xs) > 0:
                    left_armpit = (int(row_xs[0]), narrow_row)
                    right_armpit = (int(row_xs[-1]), narrow_row)
                else:
                    left_armpit, right_armpit = ClothingDetector._left_right_extrema(
                        points, ar_y_top, ar_y_bot,
                    )
            else:
                left_armpit, right_armpit = ClothingDetector._left_right_extrema(
                    points, ar_y_top, ar_y_bot,
                )

        # ---- bottom corners：底部 5% 内的最强凸点对 ----
        bm_y_top = int(y_max - ch * 0.10)
        bottom_pair = self._find_symmetric_pair(
            convex_peaks, contour, x_sym, bm_y_top, y_max,
        )
        if bottom_pair is not None:
            lb_raw, rb_raw = bottom_pair
            bw = rb_raw[0] - lb_raw[0]
            inset = int(bw * 0.08)
            left_bottom = (lb_raw[0] + inset, y_max)
            right_bottom = (rb_raw[0] - inset, y_max)
        else:
            # Fallback：V1 band extrema
            band_top = int(y_max - ch * 0.05)
            lb_raw, rb_raw = ClothingDetector._left_right_extrema(
                points, band_top, y_max,
            )
            bw = rb_raw[0] - lb_raw[0]
            inset = int(bw * 0.08)
            left_bottom = (lb_raw[0] + inset, y_max)
            right_bottom = (rb_raw[0] - inset, y_max)

        # ---- 调试可视化：标注最终 8 点 ----
        if vis is not None and _DEBUG_DIR != Path("/__virtual_tryon_debug_disabled__"):
            v = vis.copy()
            colors = {
                "top_center": (0, 0, 255),
                "bottom_center": (0, 255, 255),
                "left_shoulder": (255, 0, 0),
                "right_shoulder": (0, 0, 255),
                "left_armpit": (255, 100, 0),
                "right_armpit": (0, 100, 255),
                "left_bottom": (255, 0, 255),
                "right_bottom": (255, 255, 0),
            }
            for name, pt in [
                ("top_center", top_center), ("bottom_center", bottom_center),
                ("left_shoulder", left_shoulder), ("right_shoulder", right_shoulder),
                ("left_armpit", left_armpit), ("right_armpit", right_armpit),
                ("left_bottom", left_bottom), ("right_bottom", right_bottom),
            ]:
                cv2.circle(v, pt, 10, colors[name], -1)
                cv2.circle(v, pt, 10, (0, 0, 0), 2)
            cv2.imwrite(str(_DEBUG_DIR / "clothing_v2_final.png"), v)

        def mk(name: str, p: tuple[int, int]) -> Keypoint:
            return Keypoint(int(p[0]), int(p[1]), name=name)

        return {
            "top_center":     mk("top_center",     top_center),
            "bottom_center":  mk("bottom_center",  bottom_center),
            "left_shoulder":  mk("left_shoulder",  left_shoulder),
            "right_shoulder": mk("right_shoulder", right_shoulder),
            "left_armpit":    mk("left_armpit",    left_armpit),
            "right_armpit":   mk("right_armpit",   right_armpit),
            "left_bottom":    mk("left_bottom",    left_bottom),
            "right_bottom":   mk("right_bottom",   right_bottom),
        }
