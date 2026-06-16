"""平铺服装图的关键点检测（纯传统 CV，不依赖任何深度学习模型）。"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

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
        keypoint_method: 关键点派生方法，"geometric"（V1，宽度谷值 + 极值），
            "width"（V3，宽度剖面自适应：上半 argmax 肩部 / 肩部之下 argmin 腋下）。

    Raises:
        RuntimeError: 找不到合格的服装轮廓时抛出。
    """

    def __init__(self, keypoint_method: str = "geometric") -> None:
        if keypoint_method not in ("geometric", "width"):
            raise ValueError(
                f"keypoint_method 必须是 'geometric' 或 'width'，收到 {keypoint_method!r}"
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

        if self.keypoint_method == "width":
            return self._extract_keypoints_width(points, mask, image)
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
    # V3: 宽度剖面法（width profile analysis）
    # ============================================================

    def _extract_keypoints_width(
        self, points: np.ndarray, mask: np.ndarray, image: np.ndarray,
    ) -> dict[str, Keypoint]:
        """V3: 宽度剖面法自适应关键点。

        核心思想：衣服轮廓在每个 y 行的水平跨度 width(y) 是一条曲线，
        肩部对应上半的宽度极大值，腋下对应肩部之下的宽度极小值。V1 用
        固定比例带（12%~22% 等）选 y 区间，V3 完全交给曲线本身的极值决定。

        优势（相对 V1）：
            - 不假设"肩在 15%、腋在 40%"。长裙、超短款、宽袍等极端长宽比
              仍能正确定位。
            - 对 T 恤、衬衫、旗袍、背心等都基于同一物理信号
              （轮廓宽度变化），不依赖轮廓转角。

        实现细节：
            1. 用 `connectedComponents` 只保留最大连通分量，避免水印、
               阴影等污染把宽度剖面带偏。
            2. 在最大分量上重算 contour 和宽度。

        信号分工：
            - 肩部 y：上半 60% 区间内 widths 平滑后的 argmax。
            - 腋下 y：[肩部 y + 3%, y_min + 70%] 区间内 widths 的 argmin。
            - 顶/底：沿用 V1 宽度谷值 + 极值的成熟逻辑。
            - 关键点 x：落在选定的 y 行，contour 左右极值。

        退化：V3 内部多处回退到 V1 的 _left_right_extrema。
        """
        # `points` 当前未直接使用——V3 内部从 clean_mask 重算 clean_contour
        # 以避免 mask 阴影/水印污染。保留参数仅为签名对称。
        del points
        ys_all, _ = np.where(mask > 0)
        if len(ys_all) == 0:
            raise RuntimeError("mask 为空，无法提取关键点")
        y_min, y_max = int(ys_all.min()), int(ys_all.max())
        ch = max(y_max - y_min, 1)
        h_img = mask.shape[0]

        # 1. 提最大连通分量，去掉阴影/水印的污染
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8,
        )
        if num_labels > 1:
            # label 0 是背景，跳过
            sizes = stats[1:, cv2.CC_STAT_AREA]
            largest = 1 + int(np.argmax(sizes))
            clean_mask = (labels == largest).astype(np.uint8) * 255
        else:
            clean_mask = mask
        # 2. 在 clean_mask 上重算最大外轮廓
        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            raise RuntimeError("连通分量清洗后未找到轮廓")
        clean_contour = max(contours, key=cv2.contourArea)
        clean_points = clean_contour.reshape(-1, 2)

        # 3. 从 clean contour 算宽度剖面
        contour_xs = clean_points[:, 0]
        contour_ys = clean_points[:, 1]
        row_x_min: dict[int, int] = {}
        row_x_max: dict[int, int] = {}
        for y, x in zip(contour_ys, contour_xs):
            y_i = int(y)
            if y_i in row_x_min:
                if x < row_x_min[y_i]:
                    row_x_min[y_i] = x
                if x > row_x_max[y_i]:
                    row_x_max[y_i] = x
            else:
                row_x_min[y_i] = x
                row_x_max[y_i] = x
        widths = np.zeros(h_img, dtype=np.float32)
        for y_i, x_lo in row_x_min.items():
            widths[y_i] = float(row_x_max[y_i] - x_lo)

        # 2. 高斯平滑（消除轮廓抖动）
        smooth_k = max(5, ch // 30)
        if smooth_k % 2 == 0:
            smooth_k += 1
        widths_s = cv2.GaussianBlur(
            widths.reshape(-1, 1), (smooth_k, 1), 0
        ).flatten()

        # 3. 肩部 y：上半 60% 内宽度极大值（first max 通常是结构肩部）
        upper_end = min(y_max, int(y_min + ch * 0.60))
        shoulder_band = widths_s[y_min:upper_end + 1]
        if shoulder_band.size > 0 and shoulder_band.max() > 0:
            shoulder_y_offset = int(np.argmax(shoulder_band))
            shoulder_y = shoulder_y_offset + y_min
        else:
            shoulder_y = int(y_min + ch * 0.20)

        # 4. 腋下 y：肩部之下到 70% 内的宽度极小值
        armpit_top = max(y_min, shoulder_y + max(int(ch * 0.03), 5))
        armpit_bot = min(y_max, int(y_min + ch * 0.70))
        if armpit_top < armpit_bot and armpit_bot > armpit_top:
            armpit_band = widths_s[armpit_top:armpit_bot + 1]
            armpit_y_offset = int(np.argmin(armpit_band))
            armpit_y = armpit_y_offset + armpit_top
        else:
            armpit_y = int(y_min + ch * 0.45)

        # 5. 在 shoulder_y / armpit_y 行取 contour 左右极值
        def _row_extrema_from_contour(y: int):
            ys_match = (contour_ys == y)
            xs_at = contour_xs[ys_match]
            if len(xs_at) >= 2:
                return (int(xs_at.min()), int(y)), (int(xs_at.max()), int(y))
            return None

        sh_ext = _row_extrema_from_contour(shoulder_y)
        if sh_ext is not None:
            left_shoulder, right_shoulder = sh_ext
        else:
            band_top = max(0, shoulder_y - 20)
            band_bot = min(h_img - 1, shoulder_y + 20)
            left_shoulder, right_shoulder = ClothingDetector._left_right_extrema(
                clean_points, band_top, band_bot,
            )

        ar_ext = _row_extrema_from_contour(armpit_y)
        if ar_ext is not None:
            left_armpit, right_armpit = ar_ext
        else:
            band_top = max(0, armpit_y - 20)
            band_bot = min(h_img - 1, armpit_y + 20)
            left_armpit, right_armpit = ClothingDetector._left_right_extrema(
                clean_points, band_top, band_bot,
            )

        # 6. 领口：上半 30% 内宽度谷值（V1 宽度谷值逻辑，作用于 mask）
        #    这里 mask 宽度和 contour 宽度在领口处基本一致，用 mask 即可。
        widths_mask = (mask > 0).sum(axis=1).astype(np.float32)
        upper_limit = min(y_max, int(y_min + ch * 0.30))
        if upper_limit > y_min and (upper_limit - y_min) >= 4:
            wu = widths_mask[y_min:upper_limit + 1].copy()
            k = np.array([1, 2, 3, 2, 1], dtype=np.float32); k /= k.sum()
            wu = np.convolve(wu, k, mode="same")
            neck_row = int(np.argmin(wu)) + y_min
        else:
            neck_row = y_min
        row_xs = np.where(mask[neck_row] > 0)[0]
        if len(row_xs) >= 2:
            top_center = (int((row_xs[0] + row_xs[-1]) / 2), neck_row)
        else:
            mid_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
            top_center = (mid_x, neck_row)

        # 7. 底摆：底部 5% 极值 + 内缩（V1 逻辑）
        bottom_band_top = int(y_max - ch * 0.05)
        left_bottom_raw, right_bottom_raw = ClothingDetector._left_right_extrema(
            clean_points, bottom_band_top, y_max,
        )
        bw = right_bottom_raw[0] - left_bottom_raw[0]
        inset = int(bw * 0.08)
        left_bottom = (left_bottom_raw[0] + inset, y_max)
        right_bottom = (right_bottom_raw[0] - inset, y_max)
        bottom_center = ((left_bottom[0] + right_bottom[0]) // 2, y_max)

        # ---- 调试可视化：把宽度剖面画在主图右侧 ----
        if _DEBUG_DIR != Path("/__virtual_tryon_debug_disabled__"):
            vis = (image if image.shape[2] == 3
                   else cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)).copy()
            panel_w = 200
            profile_w = widths_s[y_min:y_max + 1]
            max_w = profile_w.max() if profile_w.max() > 0 else 1
            prof_img = np.full((y_max - y_min + 1, panel_w, 3), 255, dtype=np.uint8)
            for y in range(y_min, y_max + 1):
                w = int(widths_s[y] / max_w * (panel_w - 4))
                if w > 0:
                    cv2.line(prof_img, (0, y - y_min), (w, y - y_min), (0, 0, 0), 1)
            for name, y in [("shoulder", shoulder_y), ("armpit", armpit_y),
                            ("neck", neck_row)]:
                cv2.line(prof_img, (0, y - y_min), (panel_w, y - y_min),
                         (0, 0, 255), 1)
                cv2.putText(prof_img, name, (panel_w - 60, y - y_min + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            h_v = vis.shape[0]
            if prof_img.shape[0] < h_v:
                pad = np.full((h_v - prof_img.shape[0], panel_w, 3),
                              255, dtype=np.uint8)
                prof_img = np.vstack([prof_img, pad])
            elif prof_img.shape[0] > h_v:
                prof_img = prof_img[:h_v]
            vis = np.hstack([vis, prof_img])
            cv2.imwrite(str(_DEBUG_DIR / "clothing_v3_profile.png"), vis)

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
