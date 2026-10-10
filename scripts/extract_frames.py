"""Pass 1: per-frame extraction from the head-mounted scene video.

For every frame:
  * find the bright projected-clip region (Otsu threshold, largest component)
  * run YOLO pose on a crop around that region and keep the best person inside it
  * estimate a homography from this frame to a per-segment reference frame using
    ORB features on the screen plane (lifter masked out), which cancels head motion

Writes one row per frame to a parquet/CSV for the analysis pass.

usage: extract_frames.py VIDEO OUT.csv [--start S] [--end S] [--model yolo11x-pose.pt]
                         [--device auto|cuda|cuda:N|mps|cpu]
"""
import argparse, time
import cv2, numpy as np, pandas as pd
from ultralytics import YOLO
from device import pick_device, describe

KP_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb", "r_elb",
            "l_wri", "r_wri", "l_hip", "r_hip", "l_knee", "r_knee", "l_ank", "r_ank"]

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out")
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=1e9)
ap.add_argument("--model", default="models/yolo11x-pose.pt")
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--device", default="auto",
                help="auto (default: NVIDIA GPU, else Apple GPU, else CPU), cuda, cuda:N, mps or cpu")
args = ap.parse_args()

device = pick_device(args.device)
# 16-bit inference: faster on NVIDIA GPUs; MPS and CPU stay at full precision
precision = {"quantize": 16} if device.startswith("cuda") else {}
print(f"device: {describe(device)}{', fp16' if precision else ''}", flush=True)
model = YOLO(args.model)
orb = cv2.ORB_create(nfeatures=1500, fastThreshold=10)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(round(args.start * fps))
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)


def screen_region(gray):
    """Bounding box (x0,y0,x1,y1) and mask of the largest bright component, or None."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    thr, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thr < 60:  # nothing meaningfully bright (looking at floor/ceiling)
        return None, None
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None, None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[i]
    if area < 0.02 * gray.size:
        return None, None
    return (int(x), int(y), int(x + w), int(y + h)), (lab == i)


def match_h(kp, des, kp_ref, des_ref):
    """RANSAC homography mapping (kp) -> (kp_ref); returns (H, inliers) or (None, 0)."""
    ms = bf.knnMatch(des, des_ref, k=2)
    good = [m[0] for m in ms if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 25:
        return None, 0
    src = np.float32([kp[m.queryIdx].pt for m in good])
    dst = np.float32([kp_ref[m.trainIdx].pt for m in good])
    H, im = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return None, 0
    return H, int(im.sum())


ref = None          # current stabilisation reference: kp, des, base (ref -> segment coords)
prev = None         # previous frame's features and its frame -> segment homography
ref_id = -1         # increments on every re-anchor
seg_id = -1         # increments only when the coordinate system genuinely restarts
rows = []
tic = time.time()
fidx = start_frame
while True:
    ok, frame = cap.read()
    if not ok or fidx / fps > args.end:
        break
    t = fidx / fps
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    row = {"frame": fidx, "t": t}
    box, mask = screen_region(gray)
    if box is not None:
        x0, y0, x1, y1 = box
        row.update(scr_x0=x0, scr_y0=y0, scr_x1=x1, scr_y1=y1)

        # --- pose on a padded crop of the bright region
        pw, ph = int(0.08 * (x1 - x0)), int(0.15 * (y1 - y0))
        cx0, cy0 = max(0, x0 - pw), max(0, y0 - ph)
        cx1, cy1 = min(frame.shape[1], x1 + pw), min(frame.shape[0], y1 + ph)
        crop = frame[cy0:cy1, cx0:cx1]
        r = model.predict(crop, imgsz=args.imgsz, device=device, verbose=False, **precision, conf=0.25)[0]
        best = None
        if r.keypoints is not None and len(r.boxes):
            for j in range(len(r.boxes)):
                bx = r.boxes.xyxy[j].cpu().numpy() + [cx0, cy0, cx0, cy0]
                mx, my = int((bx[0] + bx[2]) / 2), int((bx[1] + bx[3]) / 2)
                inside = 0 <= my < mask.shape[0] and 0 <= mx < mask.shape[1] and \
                    x0 <= mx <= x1 and y0 <= my <= y1
                size = max(bx[2] - bx[0], bx[3] - bx[1]) / max(1, (y1 - y0))
                score = float(r.boxes.conf[j]) * (1.0 if inside else 0.0) * (1.0 if size > 0.15 else 0.0)
                if score > 0 and (best is None or score > best[0]):
                    best = (score, j, bx)
        if best is not None:
            _, j, bx = best
            row.update(p_conf=best[0], p_x0=bx[0], p_y0=bx[1], p_x1=bx[2], p_y1=bx[3])
            kxy = r.keypoints.xy[j].cpu().numpy() + [cx0, cy0]
            kc = r.keypoints.conf[j].cpu().numpy()
            for name, (kx, ky), c in zip(KP_NAMES, kxy, kc):
                row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_c"] = kx, ky, c

        # --- stabilisation: homography frame -> reference, on screen-plane features
        fmask = np.zeros_like(gray)
        fmask[y0:y1, x0:x1] = 255
        # include the dark media-player margins / taskbar around the clip (same plane)
        fmask[max(0, y0 - ph):min(gray.shape[0], y1 + ph), max(0, x0 - pw):min(gray.shape[1], x1 + pw)] = 255
        if best is not None:
            bx = best[2].astype(int)
            fmask[max(0, bx[1] - 10):bx[3] + 10, max(0, bx[0] - 10):bx[2] + 10] = 0
        kp, des = orb.detectAndCompute(gray, fmask)
        H, inl = None, 0
        if ref is not None and des is not None and len(kp) > 30:
            H, inl = match_h(kp, des, ref["kp"], ref["des"])
            if H is not None:
                H = ref["base"] @ H
        if H is None or inl < 40:
            # Re-anchor: try linking this frame to the previous frame so the new
            # reference shares the old coordinate system (same clip, tracking drift).
            linked = False
            if des is not None and len(kp) > 30:
                if prev is not None and prev["H"] is not None:
                    Hp, inl_p = match_h(kp, des, prev["kp"], prev["des"])
                    if Hp is not None and inl_p >= 60:
                        base = prev["H"] @ Hp
                        ref = {"kp": kp, "des": des, "base": base}
                        H, inl, linked = base, inl_p, True
                if not linked:
                    # genuine content change: new segment, new coordinate system
                    seg_id += 1
                    ref = {"kp": kp, "des": des, "base": np.eye(3)}
                    H, inl = np.eye(3), len(kp)
                ref_id += 1
            else:
                H = None
        prev = {"kp": kp, "des": des, "H": H} if des is not None else None
        if H is not None:
            row["seg_id"] = seg_id
            row["ref_id"] = ref_id
            row["h_inliers"] = inl
            for a in range(3):
                for b in range(3):
                    row[f"h{a}{b}"] = H[a, b]
    else:
        prev = None
    rows.append(row)
    fidx += 1
    if fidx % 500 == 0:
        el = time.time() - tic
        done = fidx - start_frame
        print(f"t={t:7.1f}s  frames={done}  {done / el:5.1f} fps  ref={ref_id}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(args.out, index=False)
print(f"wrote {len(df)} rows to {args.out} in {time.time() - tic:.0f}s")
