"""人体关键点检测器，含三级降级链。

优先级：MediaPipe Pose (Tasks API) -> Haar 级联 -> 启发式估算。
`RobustHumanDetector` 把三者串起来，调用者始终能拿到结果，
只有在三者全部失败时才会抛异常。
"""

from __future__ import annotations

import logging
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# mediapipe 已列入 pyproject.toml 依赖，作为模块级 import。
# 若环境确实没装 mediapipe，本文件加载会失败——这是有意的，避免无意义地降级到
# 精度很低的 Haar 路径。
from .keypoints import Keypoint, MEDIAPIPE_POSE_INDICES
from .visualize import draw_keypoints

logger = logging.getLogger(__name__)


# 当前文件位于 src/virtual_tryon/human_detector.py，
# 向上三级即为仓库根目录，模型缓存到根目录的 models/ 下。
_MODELS_DIR: Path = Path(__file__).resolve().parent.parent.parent / "models"
_MODEL_PATH: Path = _MODELS_DIR / "pose_landmarker_full.task"

# 调试目录：默认指向一个永远不存在的占位路径，未通过 set_debug_dir() 启用时
# 所有 imwrite 调用都被短路；用 Path 而不是 None 是为了让 Pyright 在所有
# 调用点都不需要 None 守卫。
_DEBUG_DIR: Path = Path("/__virtual_tryon_debug_disabled__")


def set_debug_dir(path: Path | None) -> None:
    """设置中间产物输出目录。None 表示关闭调试输出。"""
    global _DEBUG_DIR
    _DEBUG_DIR = path if path is not None else Path("/__virtual_tryon_debug_disabled__")
# MediaPipe 官方公共存储，提供 MediaPipe Solutions / Tasks API 的模型权重。
# 使用 full 模型而非 lite：full 在长袖/交叉臂/侧身姿态下识别更稳定，
# lite 在这些场景会出现左右镜像翻转。模型约 12MB。
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def _ensure_mediapipe_model() -> Path:
    """按需下载 MediaPipe Pose 模型到本地缓存。

    首次调用时若模型文件不存在，从 Google 公共存储下载至
    `models/pose_landmarker_lite.task`（约 5MB）。后续调用直接复用本地文件。

    Returns:
        本地 .task 模型文件的路径。

    Raises:
        RuntimeError: 下载失败时抛出。
    """
    if _MODEL_PATH.exists():
        return _MODEL_PATH

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("正在下载 MediaPipe Pose 模型到 %s ...", _MODEL_PATH)
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    except Exception as e:
        # 清理未完成的下载，下次运行可以重新尝试。
        if _MODEL_PATH.exists():
            _MODEL_PATH.unlink()
        raise RuntimeError(f"下载 MediaPipe 模型失败: {e}") from e

    size_mb = _MODEL_PATH.stat().st_size / 1024 / 1024
    logger.info("模型下载完成（%.1f MB）", size_mb)
    return _MODEL_PATH


class HumanDetector(ABC):
    """人体关键点检测器抽象基类。

    实现类必须返回至少包含 `HUMAN_KEYPOINTS_USED` 中名称的字典；
    失败时应抛出 RuntimeError，由上层 `RobustHumanDetector` 捕获并降级。
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> dict[str, Keypoint]:
        """在 BGR 图像中检测关键点。返回 关键点名 -> Keypoint。"""


class MediaPipeHumanDetector(HumanDetector):
    """基于 MediaPipe Pose Tasks API 的检测器（精度最高）。"""

    def __init__(self) -> None:
        model_path = _ensure_mediapipe_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        logger.info("MediaPipeHumanDetector 就绪（模型=%s）", model_path.name)

    def detect(self, image: np.ndarray) -> dict[str, Keypoint]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # type: ignore[call-arg]  # mediapipe 缺 type stub，data= 是合法参数
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb
        )
        results = self._landmarker.detect(mp_image)
        if not results.pose_landmarks:
            raise RuntimeError("MediaPipe 未检测到任何人体姿态")

        # 多人体检测时取第一个；本项目目前只处理单人（num_poses=1）。
        landmarks = results.pose_landmarks[0]
        h, w = image.shape[:2]

        out: dict[str, Keypoint] = {}
        for name, idx in MEDIAPIPE_POSE_INDICES.items():
            lm = landmarks[idx]
            out[name] = Keypoint(
                x=int(lm.x * w),
                y=int(lm.y * h),
                confidence=float(getattr(lm, "visibility", 1.0)),
                name=name,
            )

        # MediaPipe Pose 的 left/right 是 subject 解剖学的左右，与图像坐标轴的
        # 左右没有固定关系——取决于相机是否水平翻转（自拍模式会把 subject 的
        # 右侧映射到图像左侧）。因此 MediaPipe 原始输出可能"看起来左右反了"，
        # 但命名（left/right）始终对应 subject 解剖学方向。
        #
        # 历史教训：之前在这里加了"自动水平翻转"逻辑，把对的输出翻成了错的。
        # MediaPipe Pose full 模型对新模特的输出已经是正确的，不再做任何翻转。
        if (
            "right_eye" in out
            and "left_eye" in out
            and out["right_eye"].x < out["left_eye"].x
        ):
            logger.warning(
                "MediaPipe 输出在图像坐标上看似镜像（right_eye 在左）；"
                "这是相机水平翻转（自拍）导致的，是正确信号，不再做自动翻转。"
            )

        # 把 MediaPipe 原始关键点（不再做任何翻转）落盘，便于对照排查。
        if _DEBUG_DIR != Path("/__virtual_tryon_debug_disabled__"):
            raw_vis = draw_keypoints(image, out, color=(0, 0, 255))
            cv2.imwrite(str(_DEBUG_DIR / "human_1_raw.png"), raw_vis)
            r_eye: Keypoint | None = out.get("right_eye")
            l_eye: Keypoint | None = out.get("left_eye")
            logger.info(
                "right_eye.x=%s left_eye.x=%s（不做自动翻转）",
                r_eye.x if r_eye is not None else None,
                l_eye.x if l_eye is not None else None,
            )

        # 派生关键点：neck = 双肩中点。MediaPipe 不直接提供 neck 关键点，
        # 用双肩中点近似颈部位置，对 T 恤/衬衫的对齐已经足够。
        if "left_shoulder" in out and "right_shoulder" in out:
            ls, rs = out["left_shoulder"], out["right_shoulder"]
            out["neck"] = Keypoint(
                x=(ls.x + rs.x) // 2,
                y=(ls.y + rs.y) // 2,
                confidence=min(ls.confidence, rs.confidence),
                name="neck",
            )

            # 把"派生 neck"的图单独落盘，画一条双肩连线 + neck 标记。
            if _DEBUG_DIR != Path("/__virtual_tryon_debug_disabled__"):
                v = draw_keypoints(image, out, color=(0, 255, 0))
                cv2.line(v, (ls.x, ls.y), (rs.x, rs.y), (255, 255, 0), 2)
                cv2.circle(v, (out["neck"].x, out["neck"].y), 12,
                           (0, 255, 255), 2)
                cv2.imwrite(str(_DEBUG_DIR / "human_3_neck.png"), v)
        return out


class HaarHumanDetector(HumanDetector):
    """基于 Haar 级联的人脸/人体检测器（精度较低，作为 MediaPipe 失败的备选）。

    思路：先尝试检测人脸；若失败再尝试检测全身。基于检测到的区域
    按人体比例粗略推算其他关键点，精度不如 MediaPipe，但完全不依赖深度学习。
    """

    def __init__(self) -> None:
        cascade_names = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt.xml",
            "haarcascade_fullbody.xml",
        ]
        # OpenCV 的 cv2.data 子模块在 type stub 中未暴露，用 getattr 绕过。
        haar_dir = getattr(cv2, "data").haarcascades
        self._classifiers: list[tuple[str, cv2.CascadeClassifier]] = []
        for name in cascade_names:
            path = haar_dir + name
            clf = cv2.CascadeClassifier(path)
            if not clf.empty():
                self._classifiers.append((name, clf))
        if not self._classifiers:
            raise RuntimeError("未找到任何 Haar 级联 XML 文件")
        logger.info("HaarHumanDetector 就绪（%d 个级联）", len(self._classifiers))

    def detect(self, image: np.ndarray) -> dict[str, Keypoint]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detectMultiScale 返回 Sequence[Rect]，与空列表合并赋值给同一变量
        # 时 Pyright 会报类型不兼容，不显式标注即可。
        faces = []
        cascade_used = ""
        for name, clf in self._classifiers:
            detected = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            if len(detected) > 0:
                faces = detected
                cascade_used = name
                break
        if len(faces) == 0:
            raise RuntimeError("Haar 检测器未找到人脸或人体")
        logger.info("Haar 通过 %s 检测到 %d 个区域", cascade_used, len(faces))

        # 取第一个检测框，按人体比例推算关键点。
        (fx, fy, fw, fh) = faces[0]
        cx, cy = fx + fw // 2, fy + fh // 2
        head = fh                # 头部高度（用于按比例推算其他部位）
        body_top = fy + fh       # 肩部大致位置

        out: dict[str, Keypoint] = {
            "nose":          Keypoint(cx, cy, name="nose"),
            "left_shoulder": Keypoint(
                int(fx - fw * 0.3), int(body_top + head * 0.5), name="left_shoulder"
            ),
            "right_shoulder": Keypoint(
                int(fx + fw * 1.3), int(body_top + head * 0.5), name="right_shoulder"
            ),
            "left_elbow":     Keypoint(
                int(fx - fw * 0.4), int(body_top + head * 1.5), name="left_elbow"
            ),
            "right_elbow":    Keypoint(
                int(fx + fw * 1.4), int(body_top + head * 1.5), name="right_elbow"
            ),
            "left_hip":       Keypoint(
                int(fx + fw * 0.2), int(body_top + head * 3.0), name="left_hip"
            ),
            "right_hip":      Keypoint(
                int(fx + fw * 0.8), int(body_top + head * 3.0), name="right_hip"
            ),
        }
        # 派生 neck = 双肩中点。
        ls, rs = out["left_shoulder"], out["right_shoulder"]
        out["neck"] = Keypoint(
            (ls.x + rs.x) // 2, (ls.y + rs.y) // 2, name="neck"
        )
        return out


class HeuristicHumanDetector(HumanDetector):
    """最后备选：基于图像中心按固定比例放置关键点。

    适用场景：输入图根本不含人脸/人体，或仅作 sanity check。
    关键点位置只是图像中心的简单缩放，精度很差。
    """

    def detect(self, image: np.ndarray) -> dict[str, Keypoint]:
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        logger.warning(
            "HeuristicHumanDetector: 关键点为图像中心的粗略估算，精度较差"
        )
        return {
            "nose":           Keypoint(cx, cy - h // 4, name="nose"),
            "left_shoulder":  Keypoint(cx - w // 6, cy - h // 8, name="left_shoulder"),
            "right_shoulder": Keypoint(cx + w // 6, cy - h // 8, name="right_shoulder"),
            "left_elbow":     Keypoint(cx - w // 4, cy + h // 8, name="left_elbow"),
            "right_elbow":    Keypoint(cx + w // 4, cy + h // 8, name="right_elbow"),
            "left_hip":       Keypoint(cx - w // 8, cy + h // 4, name="left_hip"),
            "right_hip":      Keypoint(cx + w // 8, cy + h // 4, name="right_hip"),
            "neck":           Keypoint(cx, cy - h // 8, name="neck"),
        }


class RobustHumanDetector(HumanDetector):
    """按 MediaPipe -> Haar -> 启发式 顺序尝试的检测器包装器。"""

    def __init__(self) -> None:
        self._detectors: list[HumanDetector] = []
        for cls in (
            MediaPipeHumanDetector,
            HaarHumanDetector,
            HeuristicHumanDetector,
        ):
            try:
                self._detectors.append(cls())
            except Exception as e:
                # 初始化失败（如 mediapipe 未安装、模型下载失败）不算致命错误，
                # 跳过该检测器继续尝试下一个。
                logger.warning("%s 不可用，已跳过: %s", cls.__name__, e)
        if not self._detectors:
            raise RuntimeError("没有任何人体检测器可初始化")

    def detect(self, image: np.ndarray) -> dict[str, Keypoint]:
        last_err: Exception | None = None
        for det in self._detectors:
            try:
                result = det.detect(image)
                logger.info("使用 %s 检测成功", type(det).__name__)
                return result
            except Exception as e:
                logger.warning("%s 检测失败: %s", type(det).__name__, e)
                last_err = e
        raise RuntimeError(f"所有人体检测器均失败，最后错误: {last_err}")
