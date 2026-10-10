"""Pass 1b: re-stabilise each lift clip by matching frames directly to a clip reference frame.

Pass 1 chains its stabilisation from frame to frame, so one bad link (motion blur, a
glance away) corrupts every later frame in the chain. This pass replaces those transforms:

  * clips are found from the pass-1 person detections (runs with gaps under 1.5 s)
  * per clip, a sharp reference frame is picked from the middle half
  * every frame is matched directly to the reference (ORB + RANSAC homography); a frame
    that can't be is matched to the nearest registered keyframe instead, so a transform
    is never more than two links from the reference
  * transforms with implausible rotation, shear or scale at the athlete are rejected

Clips can also contain back-to-back videos (the next video starting within 1.5 s). Possible
changes of on-screen video are detected - frames ~0.25 s apart are aligned and their static
content compared, and a change is confirmed only if no sharp frame within 2 s before matches
one within 2 s after - and flagged in the `video_change` column. They are not used to split
clips: fast head movement and the screen leaving the view produce the same signal, so they are
for review. seg_id identifies the clips; analyse_reps.py uses them as its clips.

Writes a copy of the pass-1 per-frame data with the stabilisation columns (h00..h22,
h_inliers, ref_id, seg_id) replaced: OUT_DIR/frames_restab.csv. Feed that to
analyse_reps.py instead of the pass-1 CSVs.

usage: restabilise.py VIDEO PASS1_DIR OUT_DIR [--jobs N]
"""
import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from videoinfo import open_video

MIN_CLIP_S, MAX_GAP_S = 8.0, 1.5      # same clip rule as analyse_reps.py
MIN_INLIERS = 40
KEYFRAME_EVERY = 12                   # frames between keyframe candidates
CUT_GAP = 6                           # frames apart when comparing on-screen content (~0.25 s)
CUT_SIM = 0.5                         # content similarity below this = different content
SAME_SIM = 0.8                        # ... above this = same video (rejects a candidate change)
THUMB_SCALE = 4                       # content comparisons run at 1/4 resolution
CONFIRM_S = 2.0                       # a change is confirmed only if no sharp frame within this
                                      # many seconds before it matches one within it after
H_COLS = [f"h{a}{b}" for a in range(3) for b in range(3)]


def find_clips(df, fps):
    has = (df.p_conf.fillna(0) >= 0.4).values
    frames = df.frame.values
    runs, cur = [], None
    for f, hp in zip(frames, has, strict=True):
        if not hp:
            continue
        if cur and f - cur[1] <= MAX_GAP_S * fps:
            cur[1] = f
        else:
            if cur:
                runs.append(cur)
            cur = [f, f]
    if cur:
        runs.append(cur)
    return [(a, b) for a, b in runs if (b - a) / fps >= MIN_CLIP_S]


def make_mask(shape, r):
    """Screen region (padded) minus the athlete box, as in pass 1."""
    m = np.zeros(shape, np.uint8)
    if not np.isfinite(r["scr_x0"]):
        return None
    x0, y0, x1, y1 = (int(r[k]) for k in ("scr_x0", "scr_y0", "scr_x1", "scr_y1"))
    pw, ph = int(0.08 * (x1 - x0)), int(0.15 * (y1 - y0))
    m[max(0, y0 - ph):min(shape[0], y1 + ph), max(0, x0 - pw):min(shape[1], x1 + pw)] = 255
    if np.isfinite(r["p_x0"]):
        bx = [int(r[k]) for k in ("p_x0", "p_y0", "p_x1", "p_y1")]
        m[max(0, bx[1] - 10):bx[3] + 10, max(0, bx[0] - 10):bx[2] + 10] = 0
    return m


def content_thumb(g, r):
    """Small grey image of the frame and a mask of the video's static content: the bright
    projected area minus the player-controls strip at its bottom and minus the athlete."""
    cm = np.zeros(g.shape, np.uint8)
    x0, y0, x1, y1 = (int(r[k]) for k in ("scr_x0", "scr_y0", "scr_x1", "scr_y1"))
    cm[max(0, y0):max(0, y0 + int(0.82 * (y1 - y0))), max(0, x0):max(0, x1)] = 255
    if np.isfinite(r["p_x0"]):
        bx = [int(r[k]) for k in ("p_x0", "p_y0", "p_x1", "p_y1")]
        mx, my = int(0.15 * (bx[2] - bx[0])), int(0.1 * (bx[3] - bx[1]))
        cm[max(0, bx[1] - my):max(0, bx[3] + my), max(0, bx[0] - mx):max(0, bx[2] + mx)] = 0
    small = cv2.resize(g, None, fx=1 / THUMB_SCALE, fy=1 / THUMB_SCALE, interpolation=cv2.INTER_AREA)
    cms = cv2.resize(cm, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    return small, cms


def plausible(H, pt):
    """Local Jacobian of H at pt: rotation < 15 deg, shear/anisotropy < 1.25, scale 0.6-1.7."""
    x, y = pt
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    if abs(w) < 1e-6:
        return False
    u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
    v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
    J = np.array([[H[0, 0] - u * H[2, 0], H[0, 1] - u * H[2, 1]],
                  [H[1, 0] - v * H[2, 0], H[1, 1] - v * H[2, 1]]]) / w
    sv = np.linalg.svd(J, compute_uv=False)
    rot = np.degrees(np.arctan2(J[1, 0] - J[0, 1], J[0, 0] + J[1, 1]))
    return (abs(rot) < 15 and sv[0] / max(sv[1], 1e-9) < 1.25 and 0.6 < np.sqrt(sv[0] * sv[1]) < 1.7
            and np.linalg.det(J) > 0)


def process_clip(job):
    video, clip_idx, a, b, rows, fps = job
    cv2.setRNGSeed(clip_idx)  # RANSAC is randomised: seed it so re-runs give the same result
    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=10)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    rows = rows.set_index("frame")
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    feats, thumbs = {}, {}
    for f in range(a, b + 1):
        ok, img = cap.read()
        if not ok:
            break
        if f not in rows.index:
            continue
        r = rows.loc[f]
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m = make_mask(g.shape, r)
        if m is None:
            continue
        kp, des = orb.detectAndCompute(g, m)
        if des is None or len(kp) < 30:
            continue
        small, cmask = content_thumb(g, r)
        cx = r["p_x0"] + r["p_x1"] if np.isfinite(r["p_x0"]) else r["scr_x0"] + r["scr_x1"]
        cy = r["p_y0"] + r["p_y1"] if np.isfinite(r["p_y0"]) else r["scr_y0"] + r["scr_y1"]
        sy0, sy1 = int(max(0, r["scr_y0"])), int(r["scr_y1"])
        sx0, sx1 = int(max(0, r["scr_x0"])), int(r["scr_x1"])
        sharp = cv2.Laplacian(g[sy0:sy1, sx0:sx1], cv2.CV_64F).var()
        feats[f] = (np.float32([k.pt for k in kp]), des, (cx / 2, cy / 2), sharp)
        thumbs[f] = (small, cmask)
    cap.release()
    # every frame of the detected clip belongs to it, registered or not; frames that can't be
    # registered keep an empty transform (analyse_reps.py treats them as tracking gaps)
    clip_frames = [{"frame": int(f), "seg_id": clip_idx} for f in rows.index]
    if not feats:
        return clip_frames, []

    def match(fa, fb, min_inliers=MIN_INLIERS, min_good=25):
        pa, da, ctr, _ = feats[fa]
        pb, db, _, _ = feats[fb]
        ms = bf.knnMatch(da, db, k=2)
        good = [m[0] for m in ms if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < min_good:
            return None, 0
        H, inl = cv2.findHomography(pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]],
                                    cv2.RANSAC, 3.0)
        if H is None:
            return None, 0
        n = int(inl.sum())
        return (H, n) if n >= min_inliers and plausible(H, ctr) else (None, n)

    def content_sim(fa, fb):
        """Similarity (NCC) of the static on-screen content of two frames, after aligning
        fa onto fb; NaN when they can't be aligned (e.g. different content on screen)."""
        H, _ = match(fa, fb, min_inliers=20, min_good=20)
        if H is None:
            return np.nan
        (sa, ma), (sb, mb) = thumbs[fa], thumbs[fb]
        S = np.diag([1 / THUMB_SCALE, 1 / THUMB_SCALE, 1.0])
        Hs = S @ H @ np.linalg.inv(S)
        size = (sb.shape[1], sb.shape[0])
        wa = cv2.warpPerspective(sa, Hs, size, flags=cv2.INTER_LINEAR)
        wm = cv2.warpPerspective(ma, Hs, size, flags=cv2.INTER_NEAREST)
        m = (wm > 0) & (mb > 0)
        if m.sum() < 200:
            return np.nan
        x, y = wa[m].astype(float), sb[m].astype(float)
        x, y = x - x.mean(), y - y.mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        return float((x * y).sum() / den) if den > 0 else np.nan

    # --- changes of on-screen video within the clip. Frames CUT_GAP apart are compared; a
    # change shows as a run of frames whose comparison fails or scores low. It is confirmed
    # only if no frame shortly before the run matches any frame shortly after it (motion
    # blur also breaks comparisons briefly, but the content either side is the same).
    def sharpest(frames):
        """The (up to) four sharpest of the given frames."""
        return sorted(frames, key=lambda f: -feats[f][3])[:4]

    fr = sorted(feats)
    bad = [f for f in fr if f - CUT_GAP in feats and not (content_sim(f, f - CUT_GAP) >= CUT_SIM)]
    cuts, run = [], []
    for f in bad + [None]:
        if f is not None and run and f - run[-1] <= 2:
            run.append(f)
            continue
        if len(run) >= CUT_GAP // 2:
            s_, e_ = run[0], run[-1]
            # the sharpest frames up to CONFIRM_S either side (motion blur next to a fast
            # head movement would otherwise make the same video look different)
            win = round(CONFIRM_S * fps)
            before = sharpest([x for x in fr if s_ - CUT_GAP - win <= x <= s_ - CUT_GAP])
            after = sharpest([x for x in fr if e_ + 1 <= x <= e_ + win])
            same = any(content_sim(x, y) >= SAME_SIM for x in after for y in before)
            if before and after and not same:
                cuts.append(max(s_, e_ - CUT_GAP + 1))
        run = [f] if f is not None else []

    def register(fr):
      """Register frames fr (one video) to a sharp reference frame from their middle."""
      mid = fr[len(fr) // 4: max(len(fr) // 4 + 1, 3 * len(fr) // 4)]
      ref = max(mid, key=lambda f: len(feats[f][0]) * feats[f][3])  # many features, sharp
      reg = {ref: (np.eye(3), len(feats[ref][0]), 0)}  # frame -> (H to ref, inliers, keyframe id)
      keyframes = [ref]
      # register outward from the reference so neighbouring keyframes are available
      order = sorted(fr, key=lambda f: abs(f - ref))
      for f in order:
        if f == ref:
            continue
        H, n = match(f, ref)
        kf_id = 0
        if H is None:
            # nearest registered keyframes (by time), then two-link transform
            for kf in sorted(keyframes, key=lambda k: abs(k - f))[:4]:
                if kf == ref:
                    continue
                Hk, nk = match(f, kf)
                if Hk is not None:
                    H2 = reg[kf][0] @ Hk
                    if plausible(H2, feats[f][2]):
                        H, n, kf_id = H2, nk, keyframes.index(kf)
                        break
        if H is None:
            continue
        reg[f] = (H / H[2, 2], n, kf_id)
        # promote well-registered, sharp frames to keyframes (direct matches only)
        if kf_id == 0 and n >= 120 and all(abs(f - k) >= KEYFRAME_EVERY for k in keyframes):
            keyframes.append(f)
      return reg

    # --- register the clip. Detected changes are reported, not used to split the clip: on
    # this kind of footage the same signal (frames that can't be matched) also comes from fast
    # head movement and the screen leaving the view, so splitting would cut real clips apart.
    reg = register(fr)
    out = [r for r in clip_frames if r["frame"] not in reg]
    for f, (H, n, kf_id) in reg.items():
        rec = {"frame": f, "seg_id": clip_idx, "ref_id": clip_idx * 1000 + kf_id, "h_inliers": n}
        rec.update({c: H.ravel()[i] for i, c in enumerate(H_COLS)})
        out.append(rec)
    return out, [(clip_idx, c) for c in cuts]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("pass1_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 2)))
    args = ap.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    # 3: read only the chunks of the latest run_extract_all.py run when it left a list of them
    manifest = Path(args.pass1_dir) / "chunks.txt"
    if manifest.exists():
        files = [Path(args.pass1_dir) / n for n in manifest.read_text().split()]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"chunk files listed in {manifest} are missing: {missing}")
    else:
        files = sorted(Path(args.pass1_dir).glob("frames_*.csv"))
    if not files:
        raise SystemExit(f"no frames_*.csv in {args.pass1_dir}")
    df = pd.concat([pd.read_csv(f) for f in files]).sort_values("frame").drop_duplicates("frame")
    cap, fps, _ = open_video(args.video)
    cap.release()
    clips = find_clips(df, fps)
    print(f"{len(clips)} clips, {sum(b - a + 1 for a, b in clips)} frames, {args.jobs} jobs", flush=True)

    cols = ["frame", "scr_x0", "scr_y0", "scr_x1", "scr_y1", "p_x0", "p_y0", "p_x1", "p_y1"]
    dfi = df.set_index("frame", drop=False)
    jobs = [(args.video, i, a, b, dfi.loc[a:b, cols].reset_index(drop=True), fps)
            for i, (a, b) in enumerate(clips, start=1)]
    jobs.sort(key=lambda j: -(j[3] - j[2]))  # longest first, for better load balance
    tic, results, all_cuts = time.time(), [], []
    with Pool(args.jobs) as pool:
        for k, (res, cuts) in enumerate(pool.imap_unordered(process_clip, jobs), start=1):
            results += res
            all_cuts += cuts
            print(f"  {k}/{len(jobs)} clips done ({time.time() - tic:.0f}s)", flush=True)
    print(f"{len(all_cuts)} possible video change(s) inside clips (flagged for review, not split)",
          flush=True)

    out = df.set_index("frame", drop=False)
    for c in H_COLS + ["h_inliers", "ref_id", "seg_id"]:
        out[c] = np.nan
    if results:
        new = pd.DataFrame(results).set_index("frame")
        out.loc[new.index, new.columns] = new.values
    else:
        print("no lift clips found: writing the data without stabilisation")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    # marker checked by analyse_reps.py: pass-1 transforms and segment IDs are chunk-local
    # and must not be analysed directly
    out["stabilisation"] = "restab"
    # possible changes of on-screen video, per frame (1 = a candidate change starts here)
    out["video_change"] = 0
    for _, c in all_cuts:
        out.loc[c, "video_change"] = 1
    out_path = Path(args.out_dir) / "frames_restab.csv"
    out.to_csv(out_path, index=False)
    in_clips = sum(b - a + 1 for a, b in clips)
    n_reg = sum("h00" in r for r in results)
    pct = 100 * n_reg / max(1, in_clips)
    print(f"registered {n_reg} of {in_clips} clip frames ({pct:.1f}%) "
          f"-> {out_path} in {time.time() - tic:.0f}s")
