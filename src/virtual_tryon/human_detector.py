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


def body_region_contour(
    kpts: dict[str, Keypoint],
    n_points: int = 30,
    expand_ratio: float = 0.05,
) -> np.ndarray:
    """从 MediaPipe 关键点提取躯干 + 手臂区域轮廓。

    多边形按真实人体形状设计（旧 v3 版的形状有几何错误，会让流水式 fit
    出沙漏轮廓）：
    - 顶部 neck_top 在肩线以上（不再是平直肩线）
    - 腋下与肩同 x（不再向内凹——臂从此处开始离开躯干）
    - 髋部向外扩到外缘（MediaPipe hip 是关节中心，不是髋外缘）
    - 腋下之后沿手臂外缘继续向下到腕（让长袖 fit 到手臂而非被压扁）
    - 底部仍是单点（不模拟腿）

    顶点（顺时针，从 neck_top 沿左半身体到右半）：
        neck_top → 左肩 → 左腋 → 左肘 → 左腕 → 底中 → 右腕 → 右肘 → 右腋 → 右肩

    当 MediaPipe 缺少 wrist 关键点（手被裁出画外等），肘/腕顶点退回到
    髋部位置，silhouette 在腋下后向内收到腰线——等价于无手臂的旧版
    本，保证短袖/裁手场景不破坏。

    Args:
        kpts: MediaPipe / RobustHumanDetector 返回的 33 关键点。
        n_points: 采样点数，应与衣服轮廓采样点数一致。
        expand_ratio: 轮廓外扩比例，防止 warp 后衣服边缘露肉。
    """
    required = ["left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow",
                "left_hip", "right_hip", "nose"]
    for name in required:
        if name not in kpts:
            raise ValueError(f"缺少人体关键点: {name}")

    ls_x, ls_y = kpts["left_shoulder"].x, kpts["left_shoulder"].y
    rs_x, rs_y = kpts["right_shoulder"].x, kpts["right_shoulder"].y
    le_x, le_y = kpts["left_elbow"].x, kpts["left_elbow"].y
    re_x, re_y = kpts["right_elbow"].x, kpts["right_elbow"].y
    lh_x, lh_y = kpts["left_hip"].x, kpts["left_hip"].y
    rh_x, rh_y = kpts["right_hip"].x, kpts["right_hip"].y
    nose_y = kpts["nose"].y

    # 腕关键点可选：缺失时手臂段退化（肘/腕顶点在腋下-髋连线上）。
    has_lw = "left_wrist" in kpts
    has_rw = "right_wrist" in kpts
    lw_x = kpts["left_wrist"].x if has_lw else None
    lw_y = kpts["left_wrist"].y if has_lw else None
    rw_x = kpts["right_wrist"].x if has_rw else None
    rw_y = kpts["right_wrist"].y if has_rw else None

    cx = (ls_x + rs_x) / 2
    shoulder_y = (ls_y + rs_y) / 2

    # neck_top：肩线以上（衣服领口覆盖上界）。0.30 与主流程 body_anchor 偏移一致。
    neck_top_y = shoulder_y - (shoulder_y - nose_y) * 0.30

    # 腋下 y：肩到肘的中点（臂开始脱离躯干的高度）
    armpit_y_l = ls_y + (le_y - ls_y) * 0.50
    armpit_y_r = rs_y + (re_y - rs_y) * 0.50

    # 关节中心→身体外缘 padding。MediaPipe 给的是关节中心，肩/髋外缘
    # 都在更外侧（subject 的解剖学左右 → image 坐标系相反方向）：
    #   左肩/左髋（image 右侧）= 关节 x + pad
    #   右肩/右髋（image 左侧）= 关节 x - pad
    shoulder_pad = 8      # 三角肌厚度
    hip_pad = 30          # 髋骨/臀部厚度
    elbow_pad = 5         # 肘关节中心 → 上臂外缘
    wrist_pad = 3         # 腕关节中心 → 前臂外缘

    lsh_x = ls_x + shoulder_pad
    rsh_x = rs_x - shoulder_pad
    larmpit_x = lsh_x
    rarmpit_x = rsh_x
    lhip_x = lh_x + hip_pad
    rhip_x = rh_x - hip_pad

    # 肘/腕：直接用 MediaPipe 关键点 + pad。如果腕缺失则把肘/腕顶点都
    # 放在腋下到髋连线上（约 60% 高度），让 silhouette 在腋下后向内收。
    if has_lw:
        lelbow_x = le_x + elbow_pad
        lwrist_x = lw_x + wrist_pad
        lwrist_y = lw_y
    else:
        # 退化：肘在腋下到髋的 60% y，腕在 90% y
        leg_y = larmpit_y_l + (lh_y - larmpit_y_l) * 0.60
        leg_y2 = larmpit_y_l + (lh_y - larmpit_y_l) * 0.90
        lelbow_x = (larmpit_x + lhip_x) / 2
        lwrist_x = (larmpit_x + lhip_x) / 2
        lwrist_y = leg_y2
        le_y = leg_y  # 同时改 le_y 不影响其他计算
    if has_rw:
        relbow_x = re_x - elbow_pad
        rwrist_x = rw_x - wrist_pad
        rwrist_y = rw_y
    else:
        reg_y = rarmpit_y_r + (rh_y - rarmpit_y_r) * 0.60
        reg_y2 = rarmpit_y_r + (rh_y - rarmpit_y_r) * 0.90
        relbow_x = (rarmpit_x + rhip_x) / 2
        rwrist_x = (rarmpit_x + rhip_x) / 2
        rwrist_y = reg_y2
        re_y = reg_y

    bottom_y = max(lh_y, rh_y)
    bottom_x = (lhip_x + rhip_x) / 2

    verts = np.array([
        (cx, neck_top_y),       # 0  neck_top（肩线以上）
        (lsh_x, ls_y),          # 1  左肩外缘
        (larmpit_x, armpit_y_l),# 2  左腋下（与肩同 x）
        (lelbow_x, le_y),       # 3  左肘外缘
        (lwrist_x, lwrist_y),   # 4  左腕外缘
        (bottom_x, bottom_y),   # 5  底部中点（不模拟腿）
        (rwrist_x, rwrist_y),   # 6  右腕外缘
        (relbow_x, re_y),       # 7  右肘外缘
        (rarmpit_x, armpit_y_r),# 8  右腋下
        (rsh_x, rs_y),          # 9  右肩外缘
    ], dtype=np.float32)

    # 外扩 expand_ratio（防止 warp 后边缘露肉）
    if expand_ratio > 0:
        cx_v = verts[:, 0].mean()
        cy_v = verts[:, 1].mean()
        for i in range(len(verts)):
            dx, dy = verts[i][0] - cx_v, verts[i][1] - cy_v
            dist = np.sqrt(dx * dx + dy * dy) + 1e-6
            verts[i][0] += dx / dist * expand_ratio * (abs(dx) + abs(dy))
            verts[i][1] += dy / dist * expand_ratio * (abs(dx) + abs(dy))

    # 计算段长和累计弧长，均匀采样 n_points。
    diffs = np.diff(verts, axis=0, append=verts[0:1])
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_len[-1]
    if total <= 0:
        return verts.astype(np.int32)
    sample_lens = np.linspace(0, total, n_points, endpoint=False)
    indices = np.searchsorted(cum_len, sample_lens, side="right") - 1
    indices = np.clip(indices, 0, len(verts) - 1)
    t = (sample_lens - cum_len[indices]) / np.maximum(seg_lens[indices], 1e-6)
    sampled = verts[indices] + t[:, np.newaxis] * diffs[indices]
    return sampled.astype(np.int32)
