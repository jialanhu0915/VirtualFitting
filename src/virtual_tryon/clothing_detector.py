"""平铺服装图的关键点检测。"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .keypoints import Keypoint

logger = logging.getLogger(__name__)


class ClothingDetector:
    """从平铺服装图中提取 8 个关键点。

    策略：
      1. 若图像带 alpha 通道且含有实际透明信息，用 alpha 通道做掩码。
      2. 否则使用 Otsu 阈值 + 形态学开闭运算分割出服装区域。
      3. 取最大外轮廓。
      4. 根据轮廓的上下左右极值，按固定比例派生 8 个关键点。

    Raises:
        RuntimeError: 找不到合格的服装轮廓时抛出。
    """

    def detect(self, image: np.ndarray) -> dict[str, Keypoint]:
        if image.ndim == 3 and image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            mask = self._mask_from_alpha(image[:, :, 3])
            if mask is None:
                # alpha 全不透明，alpha 没有任何分割信息，退化为 Otsu。
                mask = self._segment(rgb)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = self._segment(rgb)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise RuntimeError("未找到服装轮廓")
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 100:
            raise RuntimeError("服装轮廓过小（面积 < 100 像素），可能不是有效服装图")

        return self._extract_keypoints(contour.reshape(-1, 2))

    @staticmethod
    def _mask_from_alpha(alpha: np.ndarray) -> np.ndarray | None:
        """从 alpha 通道构造二值掩码；若 alpha 全不透明则返回 None。"""
        if alpha.min() >= 250:
            return None
        return ((alpha > 10).astype(np.uint8)) * 255

    @staticmethod
    def _segment(rgb: np.ndarray) -> np.ndarray:
        """Otsu 阈值 + 形态学清理，得到服装掩码。"""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    @staticmethod
    def _extract_keypoints(points: np.ndarray) -> dict[str, Keypoint]:
        """从轮廓点集派生 8 个关键点。

        关键点的相对位置来自服装先验（领口偏上、下摆偏下、肩部约占顶部 20%），
        这些比例对常见 T 恤/衬衫效果尚可，但对长外套/连衣裙可能需要调整。
        """
        top = points[np.argmin(points[:, 1])]
        bottom = points[np.argmax(points[:, 1])]
        left = points[np.argmin(points[:, 0])]
        right = points[np.argmax(points[:, 0])]

        cw = float(right[0] - left[0])   # 服装宽度
        ch = float(bottom[1] - top[1])  # 服装高度

        def mk(name: str, x: float, y: float) -> Keypoint:
            return Keypoint(int(round(x)), int(round(y)), name=name)

        return {
            "top_center":     mk("top_center",     top[0],               top[1]),
            "bottom_center":  mk("bottom_center",  bottom[0],            bottom[1]),
            "left_shoulder":  mk("left_shoulder",  left[0]  + cw * 0.15, top[1] + ch * 0.20),
            "right_shoulder": mk("right_shoulder", right[0] - cw * 0.15, top[1] + ch * 0.20),
            "left_armpit":    mk("left_armpit",    left[0]  + cw * 0.10, top[1] + ch * 0.35),
            "right_armpit":   mk("right_armpit",   right[0] - cw * 0.10, top[1] + ch * 0.35),
            "left_bottom":    mk("left_bottom",    left[0]  + cw * 0.15, bottom[1] - ch * 0.05),
            "right_bottom":   mk("right_bottom",   right[0] - cw * 0.15, bottom[1] - ch * 0.05),
        }
