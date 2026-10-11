"""Diagnostic: detect the projector-screen quadrilateral in sample frames and draw it.

usage: screen_probe.py VIDEO OUT_DIR T1 [T2 ...]   (times in seconds)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from videoinfo import open_video

video, out_dir = sys.argv[1], Path(sys.argv[2])
times = [float(t) for t in sys.argv[3:]]
out_dir.mkdir(parents=True, exist_ok=True)
cap, fps, n_frames = open_video(video)
duration = n_frames / fps

def find_screen(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    thr, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None, thr, m
    c = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    for eps in np.linspace(0.01, 0.08, 15):
        q = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(q) == 4:
            return q.reshape(4, 2), thr, m
    return None, thr, m

for t in times:
    if not 0 <= t <= duration:
        print(f"{t}: outside the video (0-{duration:.1f} s), skipped")
        continue
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    if not ok:
        print(f"{t}: frame could not be read, skipped")
        continue
    q, thr, m = find_screen(f)
    vis = f.copy()
    if q is not None:
        cv2.polylines(vis, [q.astype(np.int32)], True, (0, 0, 255), 3)
    label = f"t={t} otsu={thr:.0f} quad={'ok' if q is not None else 'none'}"
    cv2.putText(vis, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    mask_bgr = cv2.cvtColor(cv2.resize(m, (960, 540)), cv2.COLOR_GRAY2BGR)
    small = np.hstack([cv2.resize(vis, (960, 540)), mask_bgr])
    cv2.imwrite(str(out_dir / f"probe_{t:07.1f}.jpg"), small)
    print(t, thr, None if q is None else q.tolist())
