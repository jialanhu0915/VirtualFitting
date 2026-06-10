"""关键点数据结构与服装-人体对应关系表。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Keypoint:
    """二维关键点，附带可选的置信度。

    Attributes:
        x: 像素 x 坐标。
        y: 像素 y 坐标。
        confidence: 检测置信度，范围 [0, 1]，未启用置信度时为 1.0。
        name: 关键点名称，便于调试和日志输出。
    """

    x: int
    y: int
    confidence: float = 1.0
    name: str = ""

    def to_array(self) -> np.ndarray:
        """转换为形状为 (2,) 的 float32 数组。"""
        return np.array([self.x, self.y], dtype=np.float32)


# MediaPipe Pose 33 点关键点索引。MediaPipe 的官方定义不可更改，
# 这里以"名称 -> 索引"的形式提供以便按名访问。
MEDIAPIPE_POSE_INDICES: dict[str, int] = {
    "nose": 0,
    "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8,
    "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}


# 试衣流程中实际使用的人体关键点集合。
# `neck` 是在 human_detector 中由双肩中点派生出来的虚拟关键点。
HUMAN_KEYPOINTS_USED: set[str] = {
    "neck",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_hip", "right_hip",
}


# 从平铺服装图中提取的 8 个关键点。
CLOTHING_KEYPOINTS: set[str] = {
    "top_center", "bottom_center",
    "left_shoulder", "right_shoulder",
    "left_armpit", "right_armpit",
    "left_bottom", "right_bottom",
}


# 服装关键点 -> 人体关键点的对应关系。
# 该表是试衣流水线的核心配置：它决定了 TPS/仿射变换如何把服装
# 上的每个语义点对齐到人体的对应部位。
CORRESPONDENCE: dict[str, str] = {
    "top_center":     "neck",           # 领口中心  -> 脖子（双肩中点）
    "left_shoulder":  "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_armpit":    "left_elbow",     # 腋下  -> 肘
    "right_armpit":   "right_elbow",
    "left_bottom":    "left_hip",       # 下摆  -> 髋
    "right_bottom":   "right_hip",
}
