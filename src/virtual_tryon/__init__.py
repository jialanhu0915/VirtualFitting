"""虚拟试衣流水线：关键点检测 -> 几何变形 -> 图像融合。"""

from .clothing_detector import ClothingDetector
from .human_detector import (
    HaarHumanDetector,
    HeuristicHumanDetector,
    HumanDetector,
    MediaPipeHumanDetector,
    RobustHumanDetector,
)
from .keypoints import CLOTHING_KEYPOINTS, CORRESPONDENCE, Keypoint

__all__ = [
    "Keypoint",
    "CORRESPONDENCE",
    "CLOTHING_KEYPOINTS",
    "HumanDetector",
    "MediaPipeHumanDetector",
    "HaarHumanDetector",
    "HeuristicHumanDetector",
    "RobustHumanDetector",
    "ClothingDetector",
]
