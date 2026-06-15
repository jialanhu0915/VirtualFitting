"""曲率极值 + MediaPipe 关键点 + RANSAC 仿射 —— 试穿原型。

流程：
  衣服：灰度 + 边缘检测 → 等弧长采样 → 曲率选 5 个解剖点 (collar/sh-L/sh-R/waist/hem)
  人体：MediaPipe Pose 关键点 → 取 5 个解剖点 (neck/sh-L/sh-R/hip/foot)
  两者一一对应，RANSAC 拟合 2x3 仿射矩阵
  warp 衣服到人体坐标系，多面板可视化

用法：
  uv run python scripts/test_curvature_ransac.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

from virtual_tryon.human_detector import RobustHumanDetector, body_region_contour
from virtual_tryon.io import ensure_dir, load_image
from virtual_tryon.tps_warp import blend, warp_clothing

OUT = ensure_dir(Path(__file__).resolve().parent.parent / "output" / "curvature_ransac")
CLOTHING = "data_picture/clothes/image.png"
HUMAN = "data_picture/people/image.png"


# ---------- 边缘检测（与 clothing_detector._segment_edge 同款） ----------
def segment_edge(rgb: np.ndarray) -> np.ndarray:
    """CLAHE + 多通道 Canny + 小核闭运算 + 最大轮廓填充（保细节版）。

    与 clothing_detector 的差别：kernel 从 15x15 缩到 5x5，去掉 dilation iterations，
    过滤掉 < 5% 图像面积的噪声轮廓，保留领口弧度、袖窿等曲线细节。
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    edges = np.zeros(bgr.shape[:2], dtype=np.uint8)
    edges = cv2.bitwise_or(edges, cv2.Canny(gray_eq, 30, 120))
    for ch in range(3):
        edges = cv2.bitwise_or(edges, cv2.Canny(bgr[:, :, ch], 30, 120))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    edges = cv2.bitwise_or(edges, cv2.Canny(lab[:, :, 0], 30, 120))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(bgr.shape[:2], dtype=np.uint8)
    # 过滤掉小噪声轮廓（< 5% 图像面积）
    min_area = 0.05 * bgr.shape[0] * bgr.shape[1]
    main = max((c for c in contours if cv2.contourArea(c) >= min_area),
               key=cv2.contourArea, default=max(contours, key=cv2.contourArea))
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [main], 255)
    return mask


# ---------- 轮廓采样 ----------
def largest_contour(mask: np.ndarray) -> np.ndarray:
    """最大外轮廓 → (n, 2) 浮点。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)
    main = max(contours, key=cv2.contourArea)
    return main.reshape(-1, 2).astype(np.float32)


def uniform_sample(pts: np.ndarray, n: int) -> np.ndarray:
    """等弧长采样 n 个点。"""
    if len(pts) < 2:
        return np.tile(pts[0] if len(pts) else np.zeros(2), (n, 1)).astype(np.float32)
    diffs = np.diff(pts, axis=0)
    seglens = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seglens)])
    total = float(cumlen[-1])
    if total < 1e-6:
        return np.tile(pts[0], (n, 1)).astype(np.float32)
    targets = np.linspace(0, total, n, endpoint=False)
    out = np.zeros((n, 2), dtype=np.float32)
    j = 0
    for i, t in enumerate(targets):
        while j < len(cumlen) - 1 and cumlen[j + 1] < t:
            j += 1
        seg = cumlen[j + 1] - cumlen[j]
        if seg < 1e-6:
            out[i] = pts[j]
        else:
            alpha = (t - cumlen[j]) / seg
            out[i] = (1 - alpha) * pts[j] + alpha * pts[j + 1]
    return out


# ---------- 曲率 ----------
def compute_curvature(pts: np.ndarray, k: int = 5) -> np.ndarray:
    """用前后 k 邻域的切向量夹角余弦偏离量当曲率（0=直, 2=反向折）。"""
    N = len(pts)
    curv = np.zeros(N, dtype=np.float32)
    for i in range(N):
        v1 = pts[i] - pts[(i - k) % N]
        v2 = pts[(i + k) % N] - pts[i]
        n1 = np.linalg.norm(v1) + 1e-8
        n2 = np.linalg.norm(v2) + 1e-8
        cos = float(np.dot(v1, v2) / (n1 * n2))
        curv[i] = 1 - np.clip(cos, -1.0, 1.0)
    return curv


def compute_sign(pts: np.ndarray, k: int = 5) -> np.ndarray:
    """凸/凹符号：相邻切向量叉积 z 分量。

    OpenCV findContours 返回的是顺时针轮廓（y 轴向下）：>0=凸点，<0=凹点。
    """
    N = len(pts)
    sign = np.zeros(N, dtype=np.float32)
    for i in range(N):
        v1 = pts[(i + k) % N] - pts[i]
        v2 = pts[i] - pts[(i - k) % N]
        sign[i] = v1[0] * v2[1] - v1[1] * v2[0]
    return sign


def smooth_circular(arr: np.ndarray, window: int = 7) -> np.ndarray:
    """循环卷积平滑。"""
    if window < 3 or len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(arr, kernel, mode="same")


def select_features(curv: np.ndarray, sign: np.ndarray | None = None,
                    k: int = 12, min_sep: int = 12,
                    prefer_convex: bool = True,
                    convex_weight: float = 1.0,
                    concave_weight: float = 0.7) -> list[int]:
    """按曲率降序选 k 个特征，特征间最小循环间隔 min_sep。

    sign: 凸/凹符号（>0 凸，<0 凹）。提供时按 prefer_convex 加权：
      - prefer_convex=True：凸点原曲率，凹点按 concave_weight 降权。
      - prefer_convex=False：凸点按 convex_weight，凹点原曲率（基本等同原版）。
    """
    N = len(curv)
    if sign is None:
        score = curv.astype(np.float32)
    else:
        weight = np.where(sign > 0,
                          convex_weight if prefer_convex else convex_weight,
                          concave_weight if prefer_convex else 1.0)
        score = curv.astype(np.float32) * weight
    order = np.argsort(-score).tolist()
    selected: list[int] = []
    for idx in order:
        ok = all(min((idx - s) % N, (s - idx) % N) >= min_sep for s in selected)
        if ok:
            selected.append(int(idx))
            if len(selected) >= k:
                break
    return sorted(selected)


# ---------- 解剖学先验选点（锚点驱动） ----------
def _lateral(pts: np.ndarray, idxs: np.ndarray) -> tuple[int, int]:
    """在 idxs 里找 x 最小和 x 最大。"""
    xs = pts[idxs, 0]
    return int(idxs[np.argmin(xs)]), int(idxs[np.argmax(xs)])


def _top_curv(curv: np.ndarray, idxs: np.ndarray, want: str,
              sign: np.ndarray) -> int | None:
    """在 idxs 里找曲率最高点。want: 'convex' (sign>0) / 'concave' (sign<0) / 'any'."""
    if want == "convex":
        cand = idxs[sign[idxs] > 0]
    elif want == "concave":
        cand = idxs[sign[idxs] < 0]
    else:
        cand = idxs
    if len(cand) == 0:
        return None
    return int(cand[np.argmax(curv[cand])])


def select_features_by_anchors(pts: np.ndarray, curv: np.ndarray, sign: np.ndarray
                               ) -> tuple[list[int], list[np.ndarray], list[str]]:
    """衣服 5 个解剖点，返回 (idx, xy, names) 三元组。

    collar = 双肩中点（跟人体 neck 一致，避免 mask 顶端误判）
    sh-L / sh-R = 0-0.18 H 窗口 lateral x 极值
    waist = 0.40-0.60 H 窗口 lateral x 极值中 y 居中点
    hem = 轮廓 y 最大点

    idx 版：每个 xy 投影到最近轮廓点（collar 用双肩 y 中位投影），用于曲率图。
    xy 版：双肩中点真正位置（collar 不在轮廓上），用于 warp。
    """
    ys = pts[:, 1]
    H = float(ys.max())
    hem_idx = int(np.argmax(ys))
    y_hem = float(ys[hem_idx])
    neck_idx = int(np.argmin(ys))
    y_neck = float(ys[neck_idx])

    def _band(off_lo: float, off_hi: float):
        lo = y_neck + off_lo * H
        hi = min(y_neck + off_hi * H, y_hem)
        return (ys >= lo) & (ys <= hi)

    # 双肩：0-0.18 H 窗口 lateral
    band = _band(0.00, 0.18)
    idxs = np.where(band)[0]
    if len(idxs) >= 2:
        sh_L_idx, sh_R_idx = _lateral(pts, idxs)
    else:
        sh_L_idx = sh_R_idx = hem_idx
    sh_L = pts[sh_L_idx]
    sh_R = pts[sh_R_idx]

    # 领口 = 双肩中点（跟人体 neck 定义一致）
    collar_xy = (sh_L + sh_R) / 2

    # 腰：0.40-0.60 H 窗口 lateral 中位点
    band = _band(0.40, 0.60)
    idxs = np.where(band)[0]
    if len(idxs) >= 2:
        xs = pts[idxs, 0]
        y_mid = (ys[int(idxs[np.argmin(xs)])] + ys[int(idxs[np.argmax(xs)])]) / 2
        waist_idx = int(idxs[np.argmin(np.abs(ys[idxs] - y_mid))])
    else:
        waist_idx = hem_idx
    waist_xy = pts[waist_idx]

    hem_xy = pts[hem_idx]

    # collar 投影到轮廓（y 中位最近点）—— 用于曲率图绘制
    collar_target_y = (sh_L[1] + sh_R[1]) / 2
    collar_idx = int(np.argmin(np.abs(ys - collar_target_y)))

    feats_idx = [collar_idx, sh_L_idx, sh_R_idx, waist_idx, hem_idx]
    feats_xy = [collar_xy, sh_L, sh_R, waist_xy, hem_xy]
    names = ["collar", "sh-L", "sh-R", "waist", "hem"]
    return feats_idx, feats_xy, names


# ---------- 人体 MediaPipe 关键点 ----------
def detect_human_keypoints(human_bgr: np.ndarray,
                           detector: RobustHumanDetector) -> tuple[list[np.ndarray], list[str]]:
    """从 MediaPipe 关键点取 5 个解剖点 (x, y)，按"图像左/右"排序。

    返回 (pts, names)，5 个点：neck / sh-L / sh-R / hip / foot。
    sh-L 是图像 x 较小的那侧，sh-R 是 x 较大的那侧。
    """
    kpts = detector.detect(human_bgr)
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
                "left_ankle", "right_ankle")
    for name in required:
        if kpts.get(name) is None:
            raise RuntimeError(f"MediaPipe 缺关键点: {name}")
    ls = kpts["left_shoulder"]
    rs = kpts["right_shoulder"]
    lh = kpts["left_hip"]
    rh = kpts["right_hip"]
    la = kpts["left_ankle"]
    ra = kpts["right_ankle"]

    # 双肩：按 x 排序
    if ls.x < rs.x:
        sh_L, sh_R = ls, rs
    else:
        sh_L, sh_R = rs, ls
    # 双髋/双踝中点（不区分左右）
    hip_mid = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2], dtype=np.float32)
    foot_mid = np.array([(la.x + ra.x) / 2, (la.y + ra.y) / 2], dtype=np.float32)
    # neck 优先用派生点，否则用鼻子
    neck_kp = kpts.get("neck") or kpts.get("nose")
    if neck_kp is None:
        raise RuntimeError("MediaPipe 缺关键点: neck/nose")
    neck_pt = np.array([neck_kp.x, neck_kp.y], dtype=np.float32)

    pts = [neck_pt,
           np.array([sh_L.x, sh_L.y], dtype=np.float32),
           np.array([sh_R.x, sh_R.y], dtype=np.float32),
           hip_mid,
           foot_mid]
    names = ["neck", "sh-L", "sh-R", "hip", "foot"]
    return pts, names


# ---------- 描述子 ----------
def build_descriptor(pts: np.ndarray, idx: int, radius: int = 8) -> np.ndarray:
    """特征点周围 2*radius 个邻域点的相对坐标 → 32 维向量，L2 归一。"""
    center = pts[idx]
    desc: list[float] = []
    for offset in list(range(-radius, 0)) + list(range(1, radius + 1)):
        rel = pts[(idx + offset) % len(pts)] - center
        desc.extend(rel.tolist())
    arr = np.array(desc, dtype=np.float32)
    return arr / (np.linalg.norm(arr) + 1e-8)


# ---------- 匹配 ----------
def match_lowe_ratio(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.75
                     ) -> list[tuple[int, int, float]]:
    """Lowe ratio 双向最近邻。返回 [(i_in_1, j_in_2, dist), ...]。"""
    matches: list[tuple[int, int, float]] = []
    for i, d1 in enumerate(desc1):
        dists = np.linalg.norm(desc2 - d1, axis=1)
        order = np.argsort(dists)
        if len(order) < 2:
            continue
        best, second = int(order[0]), int(order[1])
        if dists[best] / (dists[second] + 1e-8) < ratio:
            matches.append((i, best, float(dists[best])))
    return matches


# ---------- RANSAC 仿射 ----------
def ransac_affine(pts1: np.ndarray, pts2: np.ndarray, n_iter: int = 1000,
                  thresh: float = 15.0,
                  det_min: float = 0.5) -> tuple[np.ndarray | None, np.ndarray]:
    """2x3 仿射矩阵：pts2 = M @ pts1。

    det_min: 最小行列式绝对值。低于此值视为三点共线 / 矩阵退化，跳过。
    """
    n = len(pts1)
    if n < 3:
        return None, np.array([], dtype=int)
    best_inliers = np.array([], dtype=int)
    best_M = None
    rng = np.random.default_rng(0)
    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        try:
            M = cv2.getAffineTransform(pts1[idx].astype(np.float32),
                                        pts2[idx].astype(np.float32))
        except cv2.error:
            continue
        if M is None:
            continue
        # 拒绝接近共线 / 退化矩阵（det ≈ 0）
        if abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) < det_min:
            continue
        proj = (M[:, :2] @ pts1.T + M[:, 2:3]).T
        dists = np.linalg.norm(proj - pts2, axis=1)
        inliers = np.where(dists < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_M = M
    # 末尾再校验一次最终矩阵
    if best_M is not None and abs(best_M[0, 0] * best_M[1, 1]
                                   - best_M[0, 1] * best_M[1, 0]) < det_min:
        return None, np.array([], dtype=int)
    return best_M, best_inliers


# ---------- 主流程 ----------
def run_one(cloth_path: str, human_path: str, n: int = 64) -> dict:
    """跑一套返回诊断 dict。

    衣服：边缘检测 + 曲率选 5 个解剖点。
    人体：MediaPipe Pose 关键点取 5 个解剖点。
    Warp：tps_warp.warp_clothing（外接矩形 + 锚点对齐的缩放覆盖）。
    """
    cloth_rgb = load_image(cloth_path, with_alpha=True)
    if cloth_rgb.ndim == 3 and cloth_rgb.shape[2] == 4:
        cloth_rgb = cv2.cvtColor(cloth_rgb, cv2.COLOR_BGRA2RGB)
    else:
        cloth_rgb = cv2.cvtColor(cloth_rgb, cv2.COLOR_BGR2RGB)
    human_bgr = load_image(human_path)
    human_rgb = cv2.cvtColor(human_bgr, cv2.COLOR_BGR2RGB)

    # --- 衣服：边缘检测 + 曲率 ---
    cloth_mask = segment_edge(cloth_rgb)
    print(f"  衣服 mask 占比: {100 * (cloth_mask > 0).sum() / cloth_mask.size:.2f}%")
    cloth_pts = uniform_sample(largest_contour(cloth_mask), n)
    if len(cloth_pts) < n:
        print("  ! 衣服轮廓采样不足")
        return {"ok": False}
    cloth_curv = smooth_circular(compute_curvature(cloth_pts))
    cloth_sign = compute_sign(cloth_pts)
    cloth_feats_idx, cloth_feats_xy, cloth_names = select_features_by_anchors(
        cloth_pts, cloth_curv, cloth_sign)
    print(f"  衣服特征点  {len(cloth_feats_idx)} {cloth_names}")
    for name, xy in zip(cloth_names, cloth_feats_xy):
        print(f"    {name:<6}  ({xy[0]:.1f}, {xy[1]:.1f})")

    # --- 人体：MediaPipe ---
    detector = RobustHumanDetector()
    human_kpts, human_names = detect_human_keypoints(human_bgr, detector)
    print(f"  人体特征点  {len(human_kpts)} {human_names}")
    for name, pt in zip(human_names, human_kpts):
        print(f"    {name:<6}  ({pt[0]:.0f}, {pt[1]:.0f})")

    # --- 缩放覆盖（tps_warp.warp_clothing）---
    cloth_kpts = np.array(cloth_feats_xy, dtype=np.float32)
    human_kpts_arr = np.array(human_kpts, dtype=np.float32)
    if len(cloth_kpts) != len(human_kpts_arr):
        print(f"  ! 特征点数量不匹配 衣={len(cloth_kpts)} 人={len(human_kpts_arr)}")
        return {"ok": False}
    H, W = human_rgb.shape[:2]
    # 优先锚点：前 3 个 (collar / sh-L / sh-R) 是定位最准的点，
    # 用来决定缩放/旋转/平移；waist/hem 让相似变换自由带过去。
    pri_cloth = np.array(cloth_feats_xy[:3], dtype=np.float32)
    pri_body = np.array(human_kpts[:3], dtype=np.float32)
    warped_rgb, warped_mask = warp_clothing(
        cloth_rgb, cloth_mask, cloth_kpts, human_kpts_arr, (H, W),
        clothing_priority_anchors=pri_cloth,
        body_priority_anchors=pri_body,
    )
    overlay = blend(human_rgb, warped_rgb, warped_mask)
    print(f"  Warp 完成: warped_mask 非零={int((warped_mask > 0).sum())} 像素")

    diag: dict = {
        "ok": True,
        "cloth_rgb": cloth_rgb,
        "human_rgb": human_rgb,
        "cloth_mask": cloth_mask,
        "cloth_pts": cloth_pts,
        "cloth_feats": cloth_feats_xy,   # (x, y) 浮点对
        "cloth_feats_idx": cloth_feats_idx,  # 轮廓 idx（用于曲率图）
        "cloth_names": cloth_names,
        "cloth_curv": cloth_curv,
        "cloth_sign": cloth_sign,
        "human_kpts": human_kpts,
        "human_names": human_names,
        "warped_rgb": warped_rgb,
        "warped_mask": warped_mask,
        "overlay": overlay,
    }
    return diag


def visualize(diag: dict, save_path: Path) -> None:
    cloth_rgb = diag["cloth_rgb"]
    human_rgb = diag["human_rgb"]
    cloth_pts = diag["cloth_pts"]
    cloth_feats = diag["cloth_feats"]
    human_kpts = diag["human_kpts"]
    warped_rgb = diag["warped_rgb"]
    overlay = diag["overlay"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # 1. 衣服 + 5 个解剖点
    axes[0, 0].imshow(cloth_rgb)
    feat_xs = [pt[0] for pt in cloth_feats]
    feat_ys = [pt[1] for pt in cloth_feats]
    axes[0, 0].scatter(feat_xs, feat_ys,
                       c="red", s=80, marker="o", edgecolors="white", linewidths=1.5)
    axes[0, 0].plot(cloth_pts[:, 0], cloth_pts[:, 1], "-", color="cyan",
                    alpha=0.4, linewidth=1)
    for pt, name in zip(cloth_feats, diag["cloth_names"]):
        axes[0, 0].annotate(name, (pt[0], pt[1]),
                             xytext=(8, -4), textcoords="offset points",
                             fontsize=8, color="red", fontweight="bold")
    axes[0, 0].set_title(f"衣服 + {len(cloth_feats)} 特征点")
    axes[0, 0].axis("off")

    # 2. 人体 + 5 个 MediaPipe 关键点
    axes[0, 1].imshow(human_rgb)
    for pt, name in zip(human_kpts, diag["human_names"]):
        axes[0, 1].scatter([pt[0]], [pt[1]], c="blue", s=80, marker="o",
                           edgecolors="white", linewidths=1.5)
        axes[0, 1].annotate(name, (pt[0], pt[1]),
                            xytext=(8, -4), textcoords="offset points",
                            fontsize=8, color="blue", fontweight="bold")
    axes[0, 1].set_title(f"人体 + {len(human_kpts)} MediaPipe 关键点")
    axes[0, 1].axis("off")

    # 3. Warp 后
    axes[1, 0].imshow(warped_rgb)
    axes[1, 0].set_title("TPS warp 后 (衣服)")
    axes[1, 0].axis("off")

    # 4. 叠加
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("人体 + 衣服叠加 (TPS)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# ---------- 中间过程分阶段落盘 ----------
def _two_panel(imgs: list, titles: list[str], fname: str,
               figsize: tuple = (14, 6), cmap: str | None = None) -> Path:
    """两张图左右拼接存盘。"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, img, t in zip(axes, imgs, titles):
        ax.imshow(img, cmap=cmap)
        ax.set_title(t)
        ax.axis("off")
    plt.tight_layout()
    out = OUT / fname
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    return out


def _one_panel(img, title: str, fname: str, figsize: tuple = (8, 10),
               cmap: str | None = None) -> Path:
    """单张图存盘。"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    out = OUT / fname
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    return out


def save_intermediates(diag: dict) -> list[Path]:
    """把每一阶段产物单独存为 PNG，方便逐步排查。

    衣服：原图 / mask / 采样点 / 曲率 / 特征点
    人体：原图 / MediaPipe 关键点叠加（不再有 mask / 曲率 / 描述子）
    """
    cloth_rgb = diag["cloth_rgb"]
    human_rgb = diag["human_rgb"]
    cloth_mask = diag["cloth_mask"]
    cloth_pts = diag["cloth_pts"]
    cloth_feats = diag["cloth_feats"]
    cloth_feats_idx = diag["cloth_feats_idx"]
    cloth_curv = diag["cloth_curv"]
    cloth_sign = diag["cloth_sign"]
    human_kpts = diag["human_kpts"]

    saved: list[Path] = []

    # 01 输入原图
    saved.append(_two_panel(
        [cloth_rgb, human_rgb],
        ["衣服原图", "人体原图"],
        "01_inputs.png",
    ))

    # 02 衣服 mask
    saved.append(_one_panel(cloth_mask, "衣服 mask (边缘检测)", "02_cloth_mask.png", cmap="gray"))

    # 03 衣服等弧长采样点
    cloth_sampled = cloth_rgb.copy()
    cv2.polylines(cloth_sampled,
                  [cloth_pts.astype(np.int32).reshape(-1, 1, 2)],
                  isClosed=True, color=(0, 255, 255), thickness=2)
    for x, y in cloth_pts:
        cv2.circle(cloth_sampled, (int(x), int(y)), 3, (255, 0, 255), -1)
    saved.append(_one_panel(cloth_sampled, f"衣服 {len(cloth_pts)} 采样点", "03_cloth_sampled.png"))

    # 04 衣服曲率曲线
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cloth_curv, color="red", label="cloth", linewidth=1.5)
    ax.set_title("衣服平滑后曲率曲线")
    ax.set_xlabel("轮廓索引 (0-{})".format(len(cloth_curv) - 1))
    ax.set_ylabel("曲率")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT / "04_curvature.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    saved.append(out)

    # 09 衣服曲率详细诊断
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax = axes[0]
    ax.plot(cloth_curv, color="gray", linewidth=1.0, alpha=0.5)
    convex_mask = cloth_sign > 0
    concave_mask = ~convex_mask
    ax.fill_between(np.arange(len(cloth_curv)), 0, cloth_curv,
                    where=convex_mask, color="red", alpha=0.4, label="凸")
    ax.fill_between(np.arange(len(cloth_curv)), 0, cloth_curv,
                    where=concave_mask, color="blue", alpha=0.4, label="凹")
    for i, name in zip(cloth_feats_idx, diag["cloth_names"]):
        col = "red" if cloth_sign[i] > 0 else "blue"
        ax.axvline(i, color=col, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.scatter([i], [cloth_curv[i]], color=col, s=60, zorder=5,
                   edgecolors="black", linewidths=0.8)
        ax.annotate(name, (i, cloth_curv[i]),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=8, color=col, fontweight="bold")
    ax.set_title(f"衣服 曲率  ({len(cloth_feats_idx)} 特征点)")
    ax.set_ylabel("曲率")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sign_norm = np.sign(cloth_sign)
    ax.fill_between(np.arange(len(cloth_curv)), 0, sign_norm,
                    where=convex_mask, color="red", alpha=0.5, label="凸 (+1)")
    ax.fill_between(np.arange(len(cloth_curv)), 0, sign_norm,
                    where=concave_mask, color="blue", alpha=0.5, label="凹 (-1)")
    for i in cloth_feats_idx:
        col = "red" if cloth_sign[i] > 0 else "blue"
        ax.axvline(i, color=col, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("衣服 凸/凹符号")
    ax.set_xlabel("轮廓索引")
    ax.set_ylabel("符号")
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = OUT / "09_curvature_detail.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    saved.append(out)

    # 05 衣服特征点（cv2 英文标注）+ 人体 MediaPipe 关键点（matplotlib 标注）
    cloth_feat_img = cloth_rgb.copy()
    for pt, name in zip(cloth_feats, diag["cloth_names"]):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(cloth_feat_img, (x, y), 12, (0, 255, 0), 3)
        cv2.putText(cloth_feat_img, name, (x + 14, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    human_kp_img = human_rgb.copy()
    for pt, name in zip(human_kpts, diag["human_names"]):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(human_kp_img, (x, y), 12, (255, 0, 0), 3)
        cv2.putText(human_kp_img, name, (x + 14, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    saved.append(_two_panel(
        [cloth_feat_img, human_kp_img],
        [f"衣服 {len(cloth_feats)} 解剖学特征点",
         f"人体 {len(human_kpts)} MediaPipe 关键点"],
        "05_features.png",
    ))

    # 06 对应关系可视化（5↔5 确定性对应）
    match_img = human_rgb.copy()
    src = np.array(cloth_feats, dtype=np.float32)
    dst = np.array(human_kpts, dtype=np.float32)
    for k_m in range(len(src)):
        cv2.line(match_img, tuple(src[k_m].astype(int)),
                 tuple(dst[k_m].astype(int)), (0, 255, 0), 2, cv2.LINE_AA)
    for (x, y), name in zip(src, diag["cloth_names"]):
        cv2.circle(match_img, (int(x), int(y)), 7, (255, 0, 255), -1)
    for (x, y), name in zip(dst, diag["human_names"]):
        cv2.circle(match_img, (int(x), int(y)), 7, (255, 255, 0), -1)
    saved.append(_one_panel(
        match_img,
        f"5↔5 一一对应",
        "06_correspondence.png",
    ))

    # 07 / 08 warp + 叠加（直接用 diag 里的 warped_rgb / overlay）
    saved.append(_one_panel(diag["warped_rgb"], "TPS warp 后 (衣服)", "07_warped.png"))
    saved.append(_one_panel(diag["overlay"], "人体 + 衣服叠加 (TPS)", "08_overlay.png"))

    return saved


def main() -> None:
    print(f"=== 曲率 + TPS warp 原型 ===")
    print(f"  衣服: {CLOTHING}")
    print(f"  人体: {HUMAN}\n")

    diag = run_one(CLOTHING, HUMAN)
    intermediates = save_intermediates(diag)
    save = OUT / "result_image.png"
    visualize(diag, save)
    print("\n中间过程:")
    for p in intermediates:
        print(f"  - {p.relative_to(OUT.parent.parent)}")
    print(f"\n总览面板: {save.relative_to(OUT.parent.parent)}")
    print(f"目录:     {OUT}")


if __name__ == "__main__":
    main()
