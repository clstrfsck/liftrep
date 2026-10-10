"""Pass 2: find lift clips, reps, top/bottom of travel, dwell and effort timing.

Reads the per-frame CSV(s) written by extract_frames.py and writes:
  out/clips.csv   one row per detected lift clip
  out/reps.csv    one row per rep (timings, durations, pixels)
  out/events.csv  one row per top / bottom of travel event
  out/plots/clip_NN.png  QA plot per clip

Coordinates:
  *_scene_x/y  pixel in the scene video at that frame (same space as an eye tracker's gaze point)
  *_stab_x/y   pixel in the clip's stabilised frame (head motion removed); use for distances/speeds
Time:
  *_t          seconds from the start of the video
  *_frame      video frame index
  *_rec_us     video time in microseconds (for joining with eye-tracker timestamps)

usage: analyse_reps.py OUT_DIR FRAMES.csv|FOLDER|PATTERN [...]
       e.g. analyse_reps.py work/out work   (all work/frames_*.csv)
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import savgol_filter, find_peaks, medfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = 25.0                # fallback only: replaced by the rate measured from the input below
KP_MIN_CONF = 0.5
MIN_CLIP_S = 8.0          # shortest run of lifter detections treated as a clip
MAX_GAP_S = 1.5           # detection gaps bridged inside a clip
DWELL_BAND = 0.10         # within 10% of rep range of the top/bottom level = "dwelling"
V_STILL_FRAC = 0.12       # moving = faster than 12% of the phase's peak speed...
V_STILL_MIN = 15.0        # ... and faster than 15 stab px/s; otherwise stationary
STILL_FRAMES = 2          # consecutive stationary frames that end a movement
NEAR_TOP_FRAC = 0.3       # ... and only when within 30% of the travel from the top,
MAX_STALL_S = 1.0         # ... unless stationary this long (longer than a sticking point)
REF_DEV_FRAC = 0.12       # stationary body reference this far (x body height) from its
                          # local median = stabilisation glitch in that frame
HEAD_FAST_PX = 12.0       # observer head speed (scene px/frame) treated as fast
BAR_JUMP_FRAC = 0.35      # bar moving more than 35% of body height in one frame = glitch
MIN_REP_RANGE_FRAC = 0.4  # a rep must travel >= 40% of the clip's robust range
MIN_ECCENTRIC_S = 0.2     # faster lowering than this is a tracking spike, not a rep
MIN_EFFORT_S = 0.12       # ... likewise for the effort (lifting) phase (3 frames)
PARTIAL_RANGE_FRAC = 0.5  # reps with < 50% of the clip's median range are partial
DEBUG_CLIP = int(os.environ.get("ANALYSE_DEBUG_CLIP", "0"))  # print rep candidates for one clip
MAX_SPEED_RATIO = 3.0     # a rep moving > 3x faster than the clip's typical rep is a glitch

out_dir = Path(sys.argv[1])
(out_dir / "plots").mkdir(parents=True, exist_ok=True)
# inputs may be files, folders (all frames_*.csv inside) or wildcard patterns; the patterns
# are expanded here because Windows shells don't expand them
inputs = []
for a in sys.argv[2:]:
    pa = Path(a)
    if pa.is_dir():
        inputs += sorted(pa.glob("frames_*.csv"))
    elif any(ch in a for ch in "*?["):
        inputs += sorted(Path().glob(a))
    else:
        inputs.append(pa)
if not inputs:
    raise SystemExit("no per-frame CSV files found")
df = pd.concat([pd.read_csv(f) for f in inputs]).sort_values("frame").drop_duplicates("frame")
df = df.set_index("frame", drop=False)
# video frame rate, from the per-frame data (pass 1 records each frame's number and time)
_ft = df[["frame", "t"]].dropna()
if len(_ft) > 1:
    _rate = np.diff(_ft.frame.values) / np.diff(_ft.t.values)
    _rate = _rate[np.isfinite(_rate)]
    if len(_rate):
        FPS = float(np.median(_rate))


def stabilise(d, x, y):
    w = d.h20 * x + d.h21 * y + d.h22
    return (d.h00 * x + d.h01 * y + d.h02) / w, (d.h10 * x + d.h11 * y + d.h12) / w


H_COLS = [f"h{a}{b}" for a in range(3) for b in range(3)]


def head_speed(d):
    """Observer head speed per frame: scene-pixel motion of a fixed clip point (screen centre)."""
    Hs = d[H_COLS].values.reshape(-1, 3, 3)
    c = np.array([np.nanmedian((d.scr_x0 + d.scr_x1) / 2), np.nanmedian((d.scr_y0 + d.scr_y1) / 2), 1.0])
    pts = np.full((len(d), 2), np.nan)
    for i, h in enumerate(Hs):
        if np.all(np.isfinite(h)):
            q = np.linalg.solve(h, c)
            pts[i] = q[:2] / q[2]
    return np.r_[np.nan, np.hypot(*np.diff(pts, axis=0).T)]


def rebase(d):
    """Re-express each frame's homography relative to one reference frame inside the clip.

    Pass 1 chains references across the whole session, so the stored transforms carry
    drift accumulated over minutes. inv(H_ref) @ H_t keeps only the within-clip motion.
    Frames whose re-based transform is implausible (big scale change, rotation or shear at
    the athlete, or a large jump) are marked invalid (NaN) rather than trusted.
    """
    d = d.copy()
    Hs = d[H_COLS].values.reshape(-1, 3, 3)
    n = len(d)
    mid = np.arange(n)[n // 4: max(n // 4 + 1, 3 * n // 4)]
    inl = d.h_inliers.values[mid]
    if np.all(np.isnan(inl)):
        d[H_COLS] = np.nan
        return d, 0.0
    r = mid[int(np.nanargmax(inl))]
    Hb = np.linalg.inv(Hs[r]) @ Hs
    Hb = Hb / Hb[:, 2:3, 2:3]
    # local scale and position where it matters: at the athlete (person box centre).
    # (The 2x2 determinant is not the local scale once perspective terms are present.)
    cx = ((d.p_x0 + d.p_x1) / 2).fillna((d.scr_x0 + d.scr_x1) / 2).values
    cy = ((d.p_y0 + d.p_y1) / 2).fillna((d.scr_y0 + d.scr_y1) / 2).values

    def apply(px, py):
        w = Hb[:, 2, 0] * px + Hb[:, 2, 1] * py + Hb[:, 2, 2]
        return ((Hb[:, 0, 0] * px + Hb[:, 0, 1] * py + Hb[:, 0, 2]) / w,
                (Hb[:, 1, 0] * px + Hb[:, 1, 1] * py + Hb[:, 1, 2]) / w)
    mx, my = apply(cx, cy)
    ex, ey = apply(cx + 20, cy)
    fx, fy = apply(cx, cy + 20)
    scale = np.sqrt(np.abs((ex - mx) * (fy - my) - (ey - my) * (fx - mx))) / 20
    rel = scale / np.nanmedian(scale)
    dev = np.hypot(mx - np.nanmedian(mx), my - np.nanmedian(my))
    scr_w = np.nanmedian((d.scr_x1 - d.scr_x0).values)
    # rotation and shear at the athlete (a level line must stay roughly level)
    rot = np.degrees(np.arctan2(ey - my, ex - mx))
    le, lf = np.hypot(ex - mx, ey - my), np.hypot(fx - mx, fy - my)
    aniso = np.maximum(le, lf) / np.maximum(np.minimum(le, lf), 1e-9)
    good = (rel > 0.75) & (rel < 1.33) & (dev < 0.5 * scr_w) & (np.abs(rot) < 15) & (aniso < 1.25)
    Hb[~good] = np.nan
    d[H_COLS] = Hb.reshape(-1, 9)
    return d, float(np.mean(good))


def kp_mean(d, names):
    """Confidence-gated mean of a left/right keypoint pair -> (x, y) Series in scene px.

    When both sides are confident but at very different heights (one is a misdetection,
    e.g. a wrist snapped onto the bench), keep the side closest to the previous accepted
    position.
    """
    xs = np.stack([d[f"{n}_x"].where(d[f"{n}_c"] >= KP_MIN_CONF).values for n in names], 1)
    ys = np.stack([d[f"{n}_y"].where(d[f"{n}_c"] >= KP_MIN_CONF).values for n in names], 1)
    scale = (d.p_y1 - d.p_y0).values  # person box height, scene px
    ox, oy = np.full(len(d), np.nan), np.full(len(d), np.nan)
    last = None
    for i in range(len(d)):
        ok = ~np.isnan(xs[i])
        if not ok.any():
            continue
        # the bar is horizontal, so a genuine pair sits at about the same height; in a
        # front view the sides are legitimately far apart horizontally (wide grip)
        if ok.all() and abs(ys[i, 1] - ys[i, 0]) > 0.15 * scale[i]:
            if last is None:
                continue
            j = int(np.argmin(np.hypot(xs[i] - last[0], ys[i] - last[1])))
            ox[i], oy[i] = xs[i, j], ys[i, j]
        else:
            ox[i], oy[i] = np.nanmean(xs[i]), np.nanmean(ys[i])
        last = (ox[i], oy[i])
    return pd.Series(ox, index=d.index), pd.Series(oy, index=d.index)


# ---------------------------------------------------------------- clip detection
has_person = (df.p_conf.fillna(0) >= 0.4) & df.seg_id.notna()
frames = df.frame.values
runs, cur = [], None
for f, hp, seg in zip(frames, has_person.values, df.seg_id.values):
    if hp:
        if cur and seg == cur["seg"] and f - cur["end"] <= MAX_GAP_S * FPS:
            cur["end"] = f
        else:
            if cur:
                runs.append(cur)
            cur = {"start": f, "end": f, "seg": seg}
if cur:
    runs.append(cur)
runs = [r for r in runs if (r["end"] - r["start"]) / FPS >= MIN_CLIP_S]

clip_rows, rep_rows, event_rows = [], [], []
frame_tables = []  # per-frame rep/phase + stabilisation, for annotate_video.py

for ci, r in enumerate(runs, start=1):
    d = df.loc[r["start"]:r["end"]].copy()
    d = d[d.seg_id == r["seg"]]
    t = d.t.values

    d, stab_valid = rebase(d)

    # --- lift type: in a squat the athlete is upright, so the hips sit well below the
    # shoulders (relative to body height); lying on a bench they are level.
    hx, hy = kp_mean(d, ["l_hip", "r_hip"])
    sx, sy = kp_mean(d, ["l_sho", "r_sho"])
    box_h = (d.p_y1 - d.p_y0)
    hip_below_sho = float(np.nanmedian((hy - sy) / box_h)) if (hy - sy).notna().any() else np.nan
    aspect = float(np.nanmedian(box_h / (d.p_x1 - d.p_x0)))
    if np.isfinite(hip_below_sho):
        lift = "squat" if hip_below_sho > 0.2 else "bench_press"
    else:
        lift = "squat" if aspect > 1.45 else "bench_press"
    # bar proxy: hands for bench press, shoulders (bar on upper back) for squat
    track = ["l_wri", "r_wri"] if lift == "bench_press" else ["l_sho", "r_sho"]
    bx, by = kp_mean(d, track)
    bxs, bys = stabilise(d, bx, by)

    # --- glitch gate. During fast head movement (motion blur) the stabilisation can be wrong
    # for a few frames, which moves *everything* in the frame, so a body part that should be
    # stationary (hips/knees lying on a bench; ankles in a squat) moves with the bar. Mask
    # frames where that reference leaves its local median, where it is missing while the head
    # moves fast, or where the bar jumps implausibly far in one frame.
    ref_names = ["l_hip", "r_hip", "l_knee", "r_knee"] if lift == "bench_press" else ["l_ank", "r_ank"]
    ref_y = []
    for n_ in ref_names:
        okc = d[f"{n_}_c"] >= KP_MIN_CONF
        _, ry = stabilise(d, d[f"{n_}_x"].where(okc), d[f"{n_}_y"].where(okc))
        ref_y.append(ry)
    ref_y = pd.concat(ref_y, axis=1).mean(axis=1)
    body_h = float(np.nanmedian(box_h))
    ref_med = ref_y.rolling(int(2 * FPS) | 1, center=True, min_periods=5).median()
    ref_bad = (ref_y - ref_med).abs() > REF_DEV_FRAC * body_h
    head = head_speed(d)
    bar_jump = pd.Series(bys.values).diff().abs().values > BAR_JUMP_FRAC * body_h
    glitch = ref_bad.values | (ref_y.isna().values & (head > HEAD_FAST_PX)) | bar_jump
    bys = bys.where(~glitch)
    n_glitch = int(np.count_nonzero(glitch & np.isfinite(by.values)))

    # --- clean signal: median despike, bridge short gaps, smooth
    y = pd.Series(bys.values, index=d.frame.values)
    x = pd.Series(bxs.values, index=d.frame.values)
    valid_frac = y.notna().mean()
    kp_present = y.notna().values  # before gap bridging
    track_ok = d.h00.notna().values & kp_present
    y = y.interpolate(limit=int(0.5 * FPS), limit_area="inside")
    x = x.interpolate(limit=int(0.5 * FPS), limit_area="inside")
    ok = y.notna().values
    yy = y.values.copy()
    raw_y = pd.Series(bys.values).values  # un-interpolated, for noise measurement
    if ok.sum() < 3 * FPS:
        clip_rows.append({"clip": ci, "start_t": round(t[0], 3), "end_t": round(t[-1], 3),
                          "start_frame": int(d.frame.values[0]), "end_frame": int(d.frame.values[-1]),
                          "duration_s": round(t[-1] - t[0], 2), "lift": lift, "tracked": "+".join(track),
                          "n_reps": 0, "valid_kp_frac": round(float(valid_frac), 3),
                          "hip_below_shoulder": round(hip_below_sho, 3), "box_aspect": round(aspect, 2),
                          "stab_valid_frac": round(stab_valid, 3),
                          "skipped": "too few frames with a tracked bar proxy"})
        continue
    yy[ok] = medfilt(yy[ok], 7)
    win = 9 if ok.sum() > 9 else (ok.sum() // 2) * 2 - 1
    yy[ok] = savgol_filter(yy[ok], win, 2)
    vy = np.full_like(yy, np.nan)
    vy[ok] = savgol_filter(y.values[ok], win, 2, deriv=1) * FPS  # px/s, +ve = moving down

    robust = np.nanpercentile(yy, 95) - np.nanpercentile(yy, 5)
    # bottoms = largest y (image y grows downward), tops = smallest y
    yfill = pd.Series(yy).interpolate(limit_direction="both").to_numpy(copy=True)
    bots, _ = find_peaks(yfill, prominence=MIN_REP_RANGE_FRAC * robust, distance=int(0.8 * FPS))
    bots = [b for b in bots if ok[b]]

    vfill = pd.Series(vy).interpolate(limit_direction="both").fillna(0).to_numpy(copy=True)  # px/s, +ve down

    def walk(i, step, stop, moving, near=lambda j: True):
        """From i, step while the bar keeps moving; returns the last moving index.

        The movement ends after STILL_FRAMES stationary frames in a row if the bar is
        `near` the top there, or after MAX_STALL_S of stillness anywhere (a sticking point
        is brief; a long stationary period is set-up, a hold or paused playback), or at
        `stop`."""
        j, still, last = i, 0, i
        max_stall = int(round(MAX_STALL_S * FPS))
        while (j + step - stop) * step <= 0:
            j += step
            if moving(j):
                still, last = 0, j
            else:
                still += 1
                if (still >= STILL_FRAMES and near(j)) or still >= max_stall:
                    break
        return last

    # moving / stationary is judged on a 5-frame rolling median of the velocity, so single
    # jittery frames can't break up a stationary period
    vmed = pd.Series(vfill).rolling(5, center=True, min_periods=1).median().to_numpy()
    v_floor = V_STILL_MIN
    real_y = np.where(ok, yfill, np.nan)  # data only where the bar was seen (or briefly bridged)

    def argext(a, b, fn):
        """Index of the extreme of the real data in a..b, or None if there is none."""
        seg = real_y[a:b + 1]
        return None if np.all(np.isnan(seg)) else a + int(fn(seg))

    def band_run(i, inside, lo, hi):
        """Contiguous run of frames around i (within lo..hi) where inside[j] holds."""
        s_, e_ = i, i
        while s_ > lo and inside[s_ - 1]:
            s_ -= 1
        while e_ < hi and inside[e_ + 1]:
            e_ += 1
        return s_, e_

    def build(bots):
      """Reps from bottoms.

      Velocity locates each rep's movement: lowering begins where the bar starts moving
      down (speed above V_STILL_FRAC of that lowering's peak speed) and the effort phase
      ends where the upward movement stops. The top levels are taken from the bar just
      before and just after that movement, so a stationary period at another height (bar
      in the rack, set-up) or a glitch can't define the top. Dwells and phase boundaries
      are then measured with the position band (DWELL_BAND of the rep's range) around the
      top and bottom levels, which captures short pauses that velocity smoothing hides.
      """
      reps = []
      for k, b in enumerate(bots):
        lo = bots[k - 1] if k > 0 else 0
        hi = bots[k + 1] if k + 1 < len(bots) else len(yfill) - 1
        # peak speeds of the descent leading into this bottom and the ascent leaving it:
        # the contiguous downward (upward) movement next to the bottom, not the fastest
        # movement anywhere in the window
        # (step over the bottom dwell first: the bar is momentarily still at the bottom)
        max_skip = int(round(0.5 * FPS))
        d0, skip = b, 0
        while d0 > lo and vmed[d0 - 1] <= v_floor and skip < max_skip:
            d0, skip = d0 - 1, skip + 1
        while d0 > lo and vmed[d0 - 1] > v_floor:
            d0 -= 1
        u1, skip = b, 0
        while u1 < hi and -vmed[u1 + 1] <= v_floor and skip < max_skip:
            u1, skip = u1 + 1, skip + 1
        while u1 < hi and -vmed[u1 + 1] > v_floor:
            u1 += 1
        i_pd = d0 + int(np.argmax(vfill[d0:b + 1]))      # peak lowering speed
        i_pu = b + int(np.argmin(vfill[b:u1 + 1]))       # peak lifting speed (most negative)
        if vfill[i_pd] <= v_floor or -vfill[i_pu] <= v_floor:
            continue
        thr_d = max(V_STILL_FRAC * vfill[i_pd], v_floor)
        thr_u = max(V_STILL_FRAC * -vfill[i_pu], v_floor)
        # On the way up the bar only counts as stopped when it is stationary *near the top*
        # (within NEAR_TOP_FRAC of the travel from the highest point after the bottom), so a
        # brief stall partway up (a sticking point) stays inside the effort phase.
        hi_a = float(yfill[b:hi + 1].min())
        near_a = lambda j: yfill[j] <= hi_a + NEAR_TOP_FRAC * (yfill[b] - hi_a)
        # descent onset: where the bar was last stationary (sticking points happen on the
        # way up, so no near-the-top requirement here)
        ecc_s = walk(i_pd, -1, lo, lambda j: vmed[j] > thr_d)
        con_e = walk(i_pu, +1, hi, lambda j: -vmed[j] > thr_u, near_a)  # ascent stop
        # top levels: the bar just outside the movement (a few frames' median)
        pre = yfill[max(lo, ecc_s - 3):ecc_s + 1]
        post = yfill[con_e:min(hi, con_e + 3) + 1]
        top_lv_b, top_lv_a = float(np.median(pre)), float(np.median(post))
        bottom = argext(ecc_s, con_e, np.nanargmax)
        if bottom is None:
            continue
        bot_lv = yfill[bottom]
        rep_range = bot_lv - max(top_lv_b, top_lv_a)
        if DEBUG_CLIP == ci:
            print(f"  bot {t[b]:.2f}: onset {t[ecc_s]:.2f} stop {t[con_e]:.2f} top_b {top_lv_b:.0f} top_a {top_lv_a:.0f} "
                  f"bot {bot_lv:.0f} range {rep_range:.0f} (min {MIN_REP_RANGE_FRAC * robust:.0f})")
        if rep_range < MIN_REP_RANGE_FRAC * robust:
            continue
        band = DWELL_BAND * rep_range
        tb_s, tb_e = band_run(ecc_s, yfill <= top_lv_b + band, lo, bottom)
        ta_s, ta_e = band_run(con_e, yfill <= top_lv_a + band, bottom, hi)
        bo_s, bo_e = band_run(bottom, yfill >= bot_lv - band, tb_e, ta_s)
        top_before = argext(tb_s, tb_e, np.nanargmin)
        top_after = argext(ta_s, ta_e, np.nanargmin)
        if top_before is None or top_after is None:
            continue
        spike = (t[bo_s] - t[tb_e] < MIN_ECCENTRIC_S) or (t[ta_s] - t[bo_e] < MIN_EFFORT_S)
        reps.append(dict(top_before=top_before, tb=(tb_s, tb_e), bottom=bottom, bo=(bo_s, bo_e),
                         top_after=top_after, ta=(ta_s, ta_e), range=rep_range, spike=spike,
                         speed=max(vfill[i_pd], -vfill[i_pu])))
      # implausibly fast compared with the clip's other reps: a tracking glitch
      if len(reps) >= 3:
          med_speed = np.median([rp["speed"] for rp in reps])
          for rp in reps:
              if rp["speed"] > MAX_SPEED_RATIO * med_speed:
                  rp["spike"] = True
      return reps

    fr = d.frame.values
    def ev(i):
        """Pixel/time record for index i in this clip."""
        return {"t": round(t[i], 3), "frame": int(fr[i]), "rec_us": int(round(t[i] * 1e6)),
                "stab_x": round(float(x.values[i]), 1) if np.isfinite(x.values[i]) else np.nan,
                "stab_y": round(float(yy[i]), 1),
                "scene_x": round(float(bx.iloc[i]), 1) if np.isfinite(bx.iloc[i]) else np.nan,
                "scene_y": round(float(by.iloc[i]), 1) if np.isfinite(by.iloc[i]) else np.nan}

    # Tracking spikes (implausibly fast dips) are removed and the reps rebuilt, so a spike
    # in the middle of a top dwell doesn't cut the neighbouring reps' dwell short.
    reps = build(bots)
    spike_reps = [rp for rp in reps if rp["spike"]]
    spikes = [b for b in bots if any(abs(b - rp["bottom"]) <= 3 for rp in spike_reps)]
    if spikes:
        # bridge each glitch dip in the trace (straight line across it) so it can't extend
        # the neighbouring reps' phases, then rebuild without it
        for rp in spike_reps:
            a_, b_ = rp["tb"][1], rp["ta"][0]
            if b_ > a_ + 1:
                yfill[a_:b_ + 1] = np.linspace(yfill[a_], yfill[b_], b_ - a_ + 1)
                vfill[a_:b_ + 1] = (yfill[b_] - yfill[a_]) / ((b_ - a_) / FPS)
                yy[a_:b_ + 1] = yfill[a_:b_ + 1]
        reps = [rp for rp in build([b for b in bots if b not in spikes]) if not rp["spike"]]
    med_range = np.median([rp["range"] for rp in reps]) if reps else np.nan
    for rp in reps:
        rp["status"] = "partial" if rp["range"] < PARTIAL_RANGE_FRAC * med_range else "ok"
    n_spikes = len(spikes)

    n_ok = 0
    ok_reps = [rp for rp in reps if rp["status"] == "ok"]
    for rp in reps:
        if rp["status"] == "ok":
            n_ok += 1
        n = n_ok if rp["status"] == "ok" else np.nan
        tb_s, tb_e = rp["tb"]; bo_s, bo_e = rp["bo"]; ta_s, ta_e = rp["ta"]
        ecc = slice(tb_e, bo_s + 1)        # lowering: leave top -> arrive bottom
        con = slice(bo_e, ta_s + 1)        # effort (concentric): leave bottom -> arrive top
        top_ev, bot_ev, top2_ev = ev(rp["top_before"]), ev(rp["bottom"]), ev(rp["top_after"])
        con_v = -vy[con]                   # +ve = upward
        rec = {
            "clip": ci, "rep": n, "rep_status": rp["status"], "lift": lift,
            "rep_start_t": round(t[tb_e], 3), "rep_end_t": round(t[ta_s], 3),
            "rep_duration_s": round(t[ta_s] - t[tb_e], 3),
            "top_t": top_ev["t"], "top_frame": top_ev["frame"], "top_rec_us": top_ev["rec_us"],
            "top_scene_x": top_ev["scene_x"], "top_scene_y": top_ev["scene_y"],
            "top_stab_x": top_ev["stab_x"], "top_stab_y": top_ev["stab_y"],
            "bottom_t": bot_ev["t"], "bottom_frame": bot_ev["frame"], "bottom_rec_us": bot_ev["rec_us"],
            "bottom_scene_x": bot_ev["scene_x"], "bottom_scene_y": bot_ev["scene_y"],
            "bottom_stab_x": bot_ev["stab_x"], "bottom_stab_y": bot_ev["stab_y"],
            "top_dwell_before_s": round(t[tb_e] - t[tb_s], 3),
            "bottom_dwell_s": round(t[bo_e] - t[bo_s], 3),
            "top_dwell_after_s": round(t[ta_e] - t[ta_s], 3),
            "eccentric_start_t": round(t[tb_e], 3), "eccentric_s": round(t[bo_s] - t[tb_e], 3),
            "effort_start_t": round(t[bo_e], 3), "effort_end_t": round(t[ta_s], 3),
            "effort_s": round(t[ta_s] - t[bo_e], 3),
            "range_stab_px": round(rp["range"], 1),
            "effort_mean_vel_px_s": round(float(np.nanmean(con_v)), 1) if con.stop > con.start else np.nan,
            "effort_peak_vel_px_s": round(float(np.nanmax(con_v)), 1) if con.stop > con.start else np.nan,
            "first_top_dwell_truncated": rp["tb"][0] == 0,
            # tracking quality across the rep (top dwell end -> top dwell start after)
            "track_ok_frac": round(float(track_ok[tb_e:ta_s + 1].mean()), 2),
            "reanchors": int(np.count_nonzero(np.diff(d.ref_id.values[tb_e:ta_s + 1]))),
            "kp_valid_frac": round(float(kp_present[tb_e:ta_s + 1].mean()), 2),
            # scatter of raw points around the smoothed trace, as a fraction of rep range
            "noise_frac": round(float(np.nanmedian(np.abs(raw_y[tb_s:ta_e + 1] - yy[tb_s:ta_e + 1]))
                                      / rp["range"]), 3),
        }
        # good = stabilisation valid, bar proxy seen for at least half the rep, and the raw
        # points hug the smoothed trace (scatter < 5% of the rep's range)
        rec["tracking_quality"] = "good" if (rec["kp_valid_frac"] >= 0.5 and rec["noise_frac"] <= 0.05
                                             and float(np.mean(d.h00.notna().values[tb_e:ta_s + 1])) >= 0.9) else "check"
        rep_rows.append(rec)
        if rp["status"] != "ok":
            continue  # events only for counted reps
        for kind, e, (ds, de) in (("top", top_ev, rp["tb"]), ("bottom", bot_ev, rp["bo"])):
            event_rows.append({"clip": ci, "rep": n, "event": kind, **e,
                               "dwell_start_t": round(t[ds], 3), "dwell_end_t": round(t[de], 3),
                               "dwell_s": round(t[de] - t[ds], 3)})
        if rp is ok_reps[-1]:
            event_rows.append({"clip": ci, "rep": n, "event": "top", **top2_ev,
                               "dwell_start_t": round(t[ta_s], 3), "dwell_end_t": round(t[ta_e], 3),
                               "dwell_s": round(t[ta_e] - t[ta_s], 3)})

    durs = [rr["rep_duration_s"] for rr in rep_rows if rr["clip"] == ci and rr["rep_status"] == "ok"]
    clip_rows.append({
        "clip": ci, "start_t": round(t[0], 3), "end_t": round(t[-1], 3),
        "start_frame": int(fr[0]), "end_frame": int(fr[-1]),
        "duration_s": round(t[-1] - t[0], 2), "lift": lift, "tracked": "+".join(track),
        "n_reps": len(ok_reps), "n_partial": len(reps) - len(ok_reps), "n_spikes_removed": n_spikes,
        "mean_rep_s": round(float(np.mean(durs)), 3) if durs else np.nan,
        "robust_range_stab_px": round(float(robust), 1), "valid_kp_frac": round(float(valid_frac), 3),
        "hip_below_shoulder": round(hip_below_sho, 3), "box_aspect": round(aspect, 2),
        "stab_valid_frac": round(stab_valid, 3), "glitch_frames_masked": n_glitch,
    })

    # --- per-frame table for annotate_video.py: which rep / phase each frame is in, the
    # smoothed bar position, and the re-based stabilising homography (frame -> clip coords)
    phase = np.full(len(d), "", dtype=object)
    cur_rep = np.zeros(len(d), dtype=int)  # rep in progress (0 = before the first rep)
    for k, rp in enumerate(ok_reps, start=1):
        (tb_s, tb_e), (bo_s, bo_e), (ta_s, ta_e) = rp["tb"], rp["bo"], rp["ta"]
        phase[tb_s:tb_e + 1] = "top_dwell"
        phase[tb_e + 1:bo_s] = "lowering"
        phase[bo_s:bo_e + 1] = "bottom_dwell"
        phase[bo_e + 1:ta_s] = "effort"
        phase[ta_s:ta_e + 1] = "top_dwell"
        cur_rep[tb_e + 1:] = k
    if ok_reps:
        first = ok_reps[0]["tb"][0]
        phase[:first][phase[:first] == ""] = "setup"  # before the first rep: no phase timers
    ft = pd.DataFrame({"frame": fr, "t": t, "clip": ci, "rep": cur_rep, "n_reps": len(ok_reps),
                       "phase": phase, "bar_stab_x": x.values, "bar_stab_y": yy,
                       "bar_scene_x": bx.values, "bar_scene_y": by.values,
                       "person_w": (d.p_x1 - d.p_x0).values})
    for k, col in enumerate(H_COLS):
        ft[col] = d[col].values
    frame_tables.append(ft)

    # --- QA plot
    fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax[0].plot(t, y.values, ".", ms=2, color="0.6", label="raw (stabilised)")
    ax[0].plot(t, yy, color="C0", label=f"bar proxy y ({'+'.join(track)})")
    for rp in reps:
        if rp["status"] != "ok":
            ax[0].plot(t[rp["bottom"]], yy[rp["bottom"]], "v", mfc="none", color="0.4")
            continue
        for (s, e), c in ((rp["tb"], "C2"), (rp["bo"], "C3"), (rp["ta"], "C2")):
            ax[0].axvspan(t[s], t[e], color=c, alpha=0.2)
        ax[0].plot(t[rp["bottom"]], yy[rp["bottom"]], "v", color="C3")
        ax[0].plot(t[rp["top_before"]], yy[rp["top_before"]], "^", color="C2")
        ax[0].plot(t[rp["top_after"]], yy[rp["top_after"]], "^", color="C2")
        ax[1].axvspan(t[rp["bo"][1]], t[rp["ta"][0]], color="C1", alpha=0.25)
    ax[0].invert_yaxis(); ax[0].legend(loc="upper right"); ax[0].set_ylabel("stabilised y px (up = top)")
    ax[0].set_title(f"clip {ci}: {lift}, {len(ok_reps)} reps (+{len(reps) - len(ok_reps)} partial, hollow), t={t[0]:.1f}-{t[-1]:.1f}s "
                    "(green = top dwell, red = bottom dwell, orange = effort)")
    ax[1].plot(t, -vy, color="C1"); ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("vertical velocity px/s (+ up)"); ax[1].set_xlabel("video time (s)")
    plt.tight_layout(); plt.savefig(out_dir / "plots" / f"clip_{ci:02d}.png", dpi=80); plt.close(fig)

# --- timing outliers within a clip. Informational only: a genuinely fatigued (slow) rep
# will also show up here, so these are never dropped; paused playback is the usual false case.
reps_df = pd.DataFrame(rep_rows)
if len(reps_df):
    reps_df["timing_outlier"] = ""
    for col in ("eccentric_s", "effort_s"):
        g = reps_df.groupby("clip")[col]
        med = g.transform("median")
        mad = g.transform(lambda v: (v - v.median()).abs().median()).clip(lower=0.04)
        out = (reps_df[col] - med).abs() > 4 * 1.4826 * mad
        reps_df.loc[out, "timing_outlier"] += col + ";"
    good = reps_df[(reps_df.tracking_quality == "good") & (reps_df.rep_status == "ok")].groupby("clip").size()
    for c in clip_rows:
        c["n_reps_tracking_good"] = int(good.get(c["clip"], 0))
block, prev_lift = 0, None
for c in clip_rows:
    if c["lift"] != prev_lift:
        block, prev_lift = block + 1, c["lift"]
    c["block"] = block
if len(reps_df):
    reps_df.insert(1, "block", reps_df["clip"].map({c["clip"]: c["block"] for c in clip_rows}))
reps_df.to_csv(out_dir / "reps.csv", index=False)
pd.DataFrame(clip_rows).to_csv(out_dir / "clips.csv", index=False)
pd.DataFrame(event_rows).to_csv(out_dir / "events.csv", index=False)
if frame_tables:
    pd.concat(frame_tables).to_csv(out_dir / "frames_annotation.csv.gz", index=False, float_format="%.6g")
print(pd.DataFrame(clip_rows).to_string(index=False))
