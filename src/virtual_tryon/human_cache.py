"""人体关键点检测结果的本地缓存。

把 `RobustHumanDetector.detect()` 的输出（含 MediaPipe / Haar / 启发式
的最终结果，以及从 MediaPipe 派生的 `neck` 关键点）落到 person 图像
所在目录。后续 `run` 直接读取，跳过模型推理。

缓存文件命名规则：以 person 路径去掉原后缀为 stem，再加 `.keypoints.json`
（结构化数据）和 `.keypoints.jpg`（带关键点标注的可视化图）。
例：`data_picture/people/image.png`
    → `data_picture/people/image.keypoints.json`
    → `data_picture/people/image.keypoints.jpg`

失效策略：比较 person 图像的 mtime 与缓存里记录的 mtime，
差值 > 1e-3 秒即视为失效、返回 None。容差是为了应对不同文件系统
对 mtime 精度的差异。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .keypoints import Keypoint

logger = logging.getLogger(__name__)

# mtime 比较的容差（秒）。Windows NTFS mtime 精度 ~1ms，Linux ext4 ~ns。
_MTIME_TOLERANCE = 1e-3

# 缓存 schema 版本号。未来字段语义变化时递增，load 时不匹配即视为无效。
_CACHE_VERSION = 1


def cache_paths_for(person_path: Path) -> tuple[Path, Path]:
    """根据 person 图像路径返回 (json 路径, 可视化 jpg 路径)。

    例：`image.png` → `image.keypoints.json`, `image.keypoints.jpg`。
    """
    stem = person_path.with_suffix("")  # 去后缀
    return (
        stem.with_suffix(".keypoints.json"),
        stem.with_suffix(".keypoints.jpg"),
    )


def save_human_cache(
    person_path: Path,
    keypoints: dict[str, Keypoint],
    visualization: np.ndarray,
) -> tuple[Path, Path]:
    """把 keypoints 和带标注的可视化图写到 person 图像同目录。

    Args:
        person_path: 人物图像路径，缓存写到其同目录。
        keypoints: RobustHumanDetector 返回的关键点字典。
        visualization: 已经在原图上画好关键点的 BGR 图像。

    Returns:
        (json_path, jpg_path)。
    """
    json_path, jpg_path = cache_paths_for(person_path)
    payload = {
        "version": _CACHE_VERSION,
        "image_mtime": person_path.stat().st_mtime,
        "keypoints": {
            name: {
                "x": kp.x,
                "y": kp.y,
                "confidence": kp.confidence,
                "name": kp.name,
            }
            for name, kp in keypoints.items()
        },
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not cv2.imwrite(str(jpg_path), visualization):
        raise OSError(f"写入可视化缓存失败: {jpg_path}")
    logger.info("人体关键点缓存已写入: %s", json_path)
    return json_path, jpg_path


def load_human_cache(person_path: Path) -> Optional[dict[str, Keypoint]]:
    """从 person 图像同目录读缓存。

    失效条件（任一满足即返回 None）：
    - json 文件不存在
    - json 损坏 / version 不匹配
    - person 图像的 mtime 与缓存记录的 mtime 不一致（>1e-3s）

    Returns:
        有效时返回 keypoints 字典；否则返回 None。
    """
    json_path, _ = cache_paths_for(person_path)
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("人体缓存 JSON 不可读，将重建: %s (%s)", json_path, e)
        return None
    if payload.get("version") != _CACHE_VERSION:
        logger.warning(
            "人体缓存版本不匹配（%s vs %s），将重建",
            payload.get("version"), _CACHE_VERSION,
        )
        return None
    cached_mtime = float(payload.get("image_mtime", 0.0))
    current_mtime = person_path.stat().st_mtime
    if abs(current_mtime - cached_mtime) > _MTIME_TOLERANCE:
        logger.info(
            "person 图 mtime 变化（%s → %s），缓存失效",
            cached_mtime, current_mtime,
        )
        return None
    return {
        name: Keypoint(
            x=int(d["x"]),
            y=int(d["y"]),
            confidence=float(d.get("confidence", 1.0)),
            name=str(d.get("name", name)),
        )
        for name, d in payload["keypoints"].items()
    }
