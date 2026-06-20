"""测试 s(y) median 平滑的效果。3 人 × qipao，跑 flow 命令，输出到
output/test_smooth/，并生成一个 before/after 对比 contact sheet。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PEOPLE = ROOT / "data_picture" / "people"
QIPAO = ROOT / "data_picture" / "clothes" / "image.png"
OUT = ROOT / "output" / "test_smooth"
OUT.mkdir(parents=True, exist_ok=True)

people = sorted(p for p in PEOPLE.glob("image*")
                if ".keypoints" not in p.name and p.suffix.lower() in (".png", ".jpg"))

for p in people:
    out_dir = OUT / f"{p.stem}__qipao"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        str(ROOT / "main.py"), "run",
        "--person", str(p),
        "--clothing", str(QIPAO),
        "--output", str(out_dir),
        "--warp-method", "flow",
    ]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"[FAIL {p.stem}] {r.stderr[-300:]}")
    else:
        print(f"[ok] {p.stem}")

# 生成 contact sheet
import cv2
import numpy as np
sheet_cells = []
for p in people:
    res = OUT / f"{p.stem}__qipao" / "result.jpg"
    img = cv2.imread(str(res))
    if img is None:
        img = np.full((400, 300, 3), 64, dtype=np.uint8)
    img = cv2.resize(img, (300, 400))
    cv2.putText(img, p.stem, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)
    sheet_cells.append(img)
row = np.hstack(sheet_cells)
cv2.imwrite(str(OUT / "contact_sheet.jpg"), row)
print(f"\ncontact_sheet: {OUT / 'contact_sheet.jpg'}")
