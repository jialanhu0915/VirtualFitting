"""3 人 × 4 衣 = 12 张组合，全跑一遍 + 生成 contact sheet。

每个组合落盘到 output/grid3x4/{person}__{clothing}/，最后生成一张 3x4 拼接
的 contact sheet 方便对比。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PEOPLE_DIR = ROOT / "data_picture" / "people"
CLOTHES_DIR = ROOT / "data_picture" / "clothes"
GRID = ROOT / "output" / "grid3x4"
GRID.mkdir(parents=True, exist_ok=True)

# 过滤掉缓存文件
people = sorted(p for p in PEOPLE_DIR.glob("image*")
                if ".keypoints" not in p.name and p.suffix.lower() in (".png", ".jpg"))
clothes = sorted(c for c in CLOTHES_DIR.glob("image*")
                 if c.suffix.lower() in (".png", ".jpg"))

print(f"people ({len(people)}): {[p.name for p in people]}")
print(f"clothes ({len(clothes)}): {[c.name for c in clothes]}")


def run_one(person: Path, cloth: Path) -> Path:
    out_dir = GRID / f"{person.stem}__{cloth.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        str(ROOT / "main.py"),
        "run",
        "--person", str(person),
        "--clothing", str(cloth),
        "--output", str(out_dir),
        "--warp-method", "flow",
    ]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       timeout=60)
    if r.returncode != 0:
        print(f"  [FAIL rc={r.returncode}] {r.stderr[-400:]}")
    else:
        print(f"  [ok]  {out_dir.name}")
    return out_dir / "result.jpg"


def make_contact_sheet() -> Path:
    """3 行（人）× 4 列（衣）拼接 result.jpg。"""
    import cv2
    import numpy as np
    rows = []
    for p in people:
        cells = []
        for c in clothes:
            out_dir = GRID / f"{p.stem}__{c.stem}"
            res = out_dir / "result.jpg"
            img = cv2.imread(str(res)) if res.exists() else None
            if img is None:
                # 占位灰块
                img = np.full((400, 300, 3), 64, dtype=np.uint8)
                cv2.putText(img, "FAIL", (100, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # 缩小到 300x400
            img = cv2.resize(img, (300, 400))
            # 标人 / 衣名
            cv2.putText(img, c.stem, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cells.append(img)
        row = np.hstack(cells)
        # 行左侧标人
        labeled = np.full((row.shape[0], 100, 3), 32, dtype=np.uint8)
        cv2.putText(labeled, p.stem, (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        rows.append(np.hstack([labeled, row]))

    sheet = np.vstack(rows)
    out = GRID / "contact_sheet.jpg"
    cv2.imwrite(str(out), sheet)
    print(f"\ncontact sheet: {out}  shape={sheet.shape}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-run", action="store_true",
                    help="只生成 contact sheet，不重跑")
    args = ap.parse_args()

    if not args.skip_run:
        for p in people:
            for c in clothes:
                print(f"\n>>> {p.stem} × {c.stem}")
                run_one(p, c)

    make_contact_sheet()


if __name__ == "__main__":
    main()
