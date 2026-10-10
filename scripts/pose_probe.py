"""Probe: run YOLO pose on whole frames in a time window and report speed and all detections
(unfiltered: people in the room are included; extract_frames.py keeps only the athlete on screen).

usage: pose_probe.py VIDEO T0 T1 MODEL OUT_PREFIX [DEVICE]   (DEVICE defaults to auto)
"""
import sys
import time

import cv2
from device import describe, pick_device
from ultralytics import YOLO

video, t0, t1, model_name, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
device = pick_device(sys.argv[6] if len(sys.argv) > 6 else "auto")
print("device:", describe(device))
model = YOLO(model_name)
cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
fps = cap.get(cv2.CAP_PROP_FPS)
n, tic, saved = 0, time.time(), 0
while cap.get(cv2.CAP_PROP_POS_MSEC) < t1 * 1000:
    ok, f = cap.read()
    if not ok:
        break
    r = model.predict(f, imgsz=1280, device=device, verbose=False, conf=0.25)[0]
    n += 1
    if n % 125 == 1:
        cv2.imwrite(f"{out}_{saved}.jpg", cv2.resize(r.plot(), (960, 540))); saved += 1
        b = r.boxes
        print(f"frame {n}: {len(b)} persons", [(round(float(c),2), [int(v) for v in xy]) for c, xy in zip(b.conf, b.xyxy)])
dt = time.time() - tic
print(f"{n} frames in {dt:.1f}s = {n/dt:.1f} fps")
