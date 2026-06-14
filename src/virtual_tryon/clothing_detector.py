"""平铺服装图的关键点检测。"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from .keypoints import Keypoint

logger = logging.getLogger(__name__)

# rembg 模型缓存重定向到项目内 models/u2net/。
# rembg 的 BaseSession.u2net_home() 读 U2NET_HOME 环境变量（没有则用 ~/.u2net，
# Windows 下展开为 C:\Users\<user>\.u2net，会污染 C 盘）。
# 必须在 import rembg 之前执行；这里只设环境变量不实际 import rembg，
# 这样未安装 rembg 时本模块仍可正常加载。
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
_REMBG_CACHE_ROOT: Path = _PROJECT_ROOT / "models" / "u2net"
_REMBG_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["U2NET_HOME"] = str(_REMBG_CACHE_ROOT)

# 调试目录：默认指向一个永远不存在的占位路径，未通过 set_debug_dir() 启用时
# 所有 imwrite 调用都被短路；用 Path 而不是 None 是为了让 Pyright 在所有
# 调用点都不需要 None 守卫。
_DEBUG_DIR: Path = Path("/__virtual_tryon_debug_disabled__")


def set_debug_dir(path: Path | None) -> None:
    """设置中间产物输出目录。None 表示关闭调试输出。"""
    global _DEBUG_DIR
    _DEBUG_DIR = path if path is not None else Path("/__virtual_tryon_debug_disabled__")


class ClothingDetector:
    """从平铺服装图中提取 8 个关键点。

    策略：
      1. 若图像带 alpha 通道且含有实际透明信息，用 alpha 通道做掩码。
      2. 否则使用色彩距离 (max(R,G,B)-min(R,G,B)) + 形态学开闭运算分割。
      3. 取最大外轮廓。
      4. 按轮廓几何特征自适应派生 8 个关键点（领口凹点、肩部/腋下左右极值、
         下摆左右端点内缩）。

    Raises:
        RuntimeError: 找不到合格的服装轮廓时抛出。
    """

    def sample_contour(
        self, image: np.ndarray, n_points: int = 30,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        """从服装图中提取轮廓并均匀采样 n_points 个点，同时返回 mask 和领口锚点。

        复用 detect() 的 mask 提取流水线（alpha → 色彩距离 → rembg fallback）。

        Returns:
            (points, mask, neck_anchor): points 是 (n_points, 2) int32 采样坐标，
            mask 是二值前景掩码。
        """
        # 复用 mask 提取逻辑，与 detect() 前段完全一致。
        if image.ndim == 3 and image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            mask = self._mask_from_alpha(image[:, :, 3])
            if mask is None:
                mask = self._segment(rgb)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = self._segment(rgb)

        if not self._mask_is_usable(mask):
            logger.info("主分割前景占比过低，fallback 到 rembg")
            mask = self._segment_rembg(rgb)

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
                # alpha 全不透明，alpha 没有任何分割信息，走主分割 + fallback。
                mask = self._segment(rgb)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = self._segment(rgb)

        # Fallback：当主分割的 mask 不可用（前景占比过低）时，调用 rembg
        # 重新抠图。rembg 是可选依赖，未安装或调用失败时保持原 mask。
        if not self._mask_is_usable(mask):
            logger.info("主分割前景占比过低，fallback 到 rembg")
            mask = self._segment_rembg(rgb)

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

        return self._extract_keypoints(points, image)

    @staticmethod
    def _mask_from_alpha(alpha: np.ndarray) -> np.ndarray | None:
        """从 alpha 通道构造二值掩码；若 alpha 全不透明则返回 None。"""
        if alpha.min() >= 250:
            return None
        return ((alpha > 10).astype(np.uint8)) * 255

    @staticmethod
    def _segment(rgb: np.ndarray) -> np.ndarray:
        """从 RGB 图像分割出服装区域。

        策略：服装通常是彩色主体，背景是白色/灰色/纯色，所以
        用 "远离灰度" 的距离作为前景度量比 Otsu 更稳健：
        对每个像素，把 R、G、B 三通道的方差开根号，距离灰度越远
        越可能是彩色服装。阈值取自适应（最大距离的 30%）。

        然后做形态学闭运算填补花纹里的洞，再做开运算去掉孤立点。
        """
        # 各像素到灰度的"色彩距离"：max(R,G,B) - min(R,G,B)。
        # 白色背景为 0；彩色服装一般 > 30。
        r, g, b = rgb[:, :, 0].astype(int), rgb[:, :, 1].astype(int), rgb[:, :, 2].astype(int)
        color_dist = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

        # 把色彩距离落盘成伪彩色热力图，便于肉眼判断分割质量。
        if _DEBUG_DIR != Path("/__virtual_tryon_debug_disabled__"):
            norm = np.clip(color_dist, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            cv2.imwrite(str(_DEBUG_DIR / "clothing_0_color_distance.png"), heatmap)

        # 自适应阈值：取最大距离的 20% 作为分界，至少 5。
        # 下限 5 是为了排除纯白/纯灰背景的微小噪声；20% 比例让浅紫、浅粉
        # 等花纹也能被判为前景（不会被硬卡掉）。
        thresh = max(5, int(color_dist.max() * 0.20))
        mask = (color_dist > thresh).astype(np.uint8) * 255

        # 闭运算填补花纹内的洞，开运算去掉背景噪点。
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    @staticmethod
    def _mask_is_usable(mask: np.ndarray) -> bool:
        """判断主分割 mask 是否可用。

        两种失败模式：
        1. 前景占比 < 5%：整张图被判为背景（如纯白衣服 vs 浅灰背景）。
        2. 前景占比 > 70%：几乎全图都是前景（如浅粉衣服 vs 浅蓝背景，
           闭运算把洞填满后整体被判为前景）——这种 mask 找出的最大
           轮廓几乎等于整张图，关键点会跑到边界外。

        两种情况都交给 rembg fallback。
        """
        fg = int((mask > 0).sum())
        ratio = fg / mask.size
        return 0.05 < ratio < 0.70

    def _segment_rembg(self, rgb: np.ndarray) -> np.ndarray:
        """用 rembg 抠出服装前景，返回二值 mask。

        rembg 是可选依赖：未安装或首次调用失败时返回空 mask，
        由上层 findContours 抛"未找到服装轮廓"异常——行为退化为
        "无可用分割"，不会让程序崩溃。
        """
        try:
            from rembg import new_session, remove  # noqa: WPS433
        except ImportError:
            logger.warning("rembg 未安装，跳过 fallback。请运行 "
                           "`uv pip install rembg[cpu]` 后重试。")
            return np.zeros(rgb.shape[:2], dtype=np.uint8)

        if not hasattr(self, "_rembg_session"):
            logger.info("首次调用 rembg，按需下载模型到 %s", _REMBG_CACHE_ROOT)
            self._rembg_session = new_session("u2net")

        rgba = remove(rgb, session=self._rembg_session)
        # rembg 输出 RGBA（PIL Image，Pyright 不知道其支持 numpy 索引），alpha > 10
        # 即视为前景（避免纯白边缘被当背景）。
        return ((rgba[:, :, 3] > 10).astype(np.uint8)) * 255  # type: ignore[index]

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
