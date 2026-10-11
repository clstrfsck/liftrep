"""Diagnostic: contact sheets of the scene video, one labelled thumbnail every STEP_S seconds.

usage: contact_sheet.py VIDEO OUT_DIR [STEP_S]   (STEP_S defaults to 10)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from videoinfo import open_video

video, out_dir = sys.argv[1], Path(sys.argv[2])
step = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
if step <= 0:
    raise SystemExit("STEP_S must be greater than 0")
cols, rows, tw = 8, 8, 240
out_dir.mkdir(parents=True, exist_ok=True)

cap, fps, n_frames = open_video(video)
dur = n_frames / fps
thumbs, t = [], 0.0
while t < dur:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    if not ok:
        break
    th = cv2.resize(f, (tw, tw * 9 // 16))
    cv2.putText(th, f"{int(t//60):02d}:{t%60:04.1f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(th, f"{int(t//60):02d}:{t%60:04.1f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1)
    thumbs.append(th)
    t += step

per = cols * rows
for i in range(0, len(thumbs), per):
    chunk = thumbs[i:i + per]
    chunk += [np.zeros_like(thumbs[0])] * (per - len(chunk))
    sheet = np.vstack([np.hstack(chunk[r * cols:(r + 1) * cols]) for r in range(rows)])
    cv2.imwrite(str(out_dir / f"sheet_{i // per:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
print(len(thumbs), "thumbs")
