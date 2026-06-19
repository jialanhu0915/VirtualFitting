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


# 从平铺服装图中提取的 7 个关键点。
# 旧版 8 点里有 `bottom_center` / `left_bottom` / `right_bottom` 三点，对应
# 衣服图的最底端——但这语义错位：长款衣服（旗袍/连衣裙到脚踝）的最底端
# 不应对应人体的髋。现在改成 `left_hip` / `right_hip`，由检测器在**衣身
# 最大宽度处**派生（短款 T 恤 ≈ 下摆；长款旗袍 ≈ 衣身中段），自适应。
CLOTHING_KEYPOINTS: set[str] = {
    "top_center",
    "left_shoulder", "right_shoulder",
    "left_armpit", "right_armpit",
    "left_hip", "right_hip",
}


# 服装关键点 -> 人体关键点的对应关系。
# 该表是试衣流水线的核心配置：它决定了 TPS/仿射变换如何把服装
# 上的每个语义点对齐到人体的对应部位。
#
# `armpit` 对应人体的 `shoulder`：MediaPipe Pose 33 点里没有"腋下"关键点。
# 之前用 `elbow` 是"穿上后大致到肘"的近似，但这就是袖子两侧"翼"的根因
# 之一（袖窿被强行拉到肘外侧，袖子跟着扭曲）。改为 `shoulder` 后语义更清晰：
# 衣服腋下 ≈ 身体肩部水平位置；TPS 控制点的实际 dst 位置由 main.py 的
# _cover_outpush 沿"neck→hip中点"法向外推，覆盖肩部外缘。
CORRESPONDENCE: dict[str, str] = {
    "top_center":     "neck",           # 领口中心  -> 脖子（双肩中点）
    "left_shoulder":  "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_armpit":    "left_shoulder",  # 衣服腋下 -> 身体肩部（同高）
    "right_armpit":   "right_shoulder",
    "left_hip":       "left_hip",       # 衣服胯部  -> 人体髋
    "right_hip":      "right_hip",
}
