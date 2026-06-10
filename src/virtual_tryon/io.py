"""图像读写工具。"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image(path: str | Path, with_alpha: bool = False) -> np.ndarray:
    """从磁盘读取图像。

    Args:
        path: 图像文件路径。
        with_alpha: 是否保留 alpha 通道。仅在处理带透明背景的 PNG 时设为 True。

    Returns:
        BGR（无 alpha）或 BGRA（有 alpha）格式的 numpy 数组。

    Raises:
        FileNotFoundError: 文件不存在或无法被 OpenCV 解码。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"图像文件不存在: {p}")
    flag = cv2.IMREAD_UNCHANGED if with_alpha else cv2.IMREAD_COLOR
    img = cv2.imread(str(p), flag)
    if img is None:
        raise FileNotFoundError(f"无法解码图像: {p}")
    return img


def save_image(path: str | Path, image: np.ndarray) -> None:
    """保存图像到磁盘，必要时自动创建父目录。

    Args:
        path: 输出文件路径。
        image: BGR 或 BGRA 格式的 numpy 数组。

    Raises:
        OSError: 写入失败时抛出。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(p), image):
        raise OSError(f"写入图像失败: {p}")
    logger.info("已保存图像: %s", p)


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，不存在则递归创建。返回路径对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
