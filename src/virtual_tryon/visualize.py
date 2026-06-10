"""可视化工具。"""

from __future__ import annotations

import cv2
import numpy as np

from .keypoints import Keypoint


def draw_keypoints(
    image: np.ndarray,
    keypoints: dict[str, Keypoint],
    color: tuple[int, int, int] = (0, 255, 0),
    radius: int = 8,
    font_scale: float = 0.45,
    thickness: int = 1,
) -> np.ndarray:
    """在图像副本上绘制关键点和名称标签。

    同时处理 BGR（3 通道）和 BGRA（4 通道）输入：对 BGRA 图像
    会先合成到黑色背景得到 BGR，确保结果能直接保存为 JPEG。

    Args:
        image: BGR 或 BGRA 格式的 numpy 数组。
        keypoints: 关键点字典。
        color: 圆点的 BGR 颜色，默认绿色。
        radius: 圆点半径（像素）。
        font_scale: 标签字号缩放。
        thickness: 标签笔画粗细。

    Returns:
        绘制后的 BGR 图像。
    """
    annotated = image.copy()
    if annotated.ndim == 3 and annotated.shape[2] == 4:
        # BGRA 合成到黑色背景再转 BGR，便于保存为 JPEG。
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGRA2BGR)

    for kp in keypoints.values():
        # 圆点。
        cv2.circle(annotated, (kp.x, kp.y), radius, color, -1, cv2.LINE_AA)
        # 文字描边：先画白色粗边，再画黑色正文，提升在不同背景上的可读性。
        cv2.putText(
            annotated, kp.name, (kp.x - 30, kp.y - 12),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
            thickness + 1, cv2.LINE_AA,
        )
        cv2.putText(
            annotated, kp.name, (kp.x - 30, kp.y - 12),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
            thickness, cv2.LINE_AA,
        )
    return annotated
