"""Pass 3: draw the rep analysis onto the scene video.

For every frame inside a detected lift clip it draws:
  * top of travel line (green) and bottom of travel line (red) for the rep in progress,
    fixed to the screen content (they follow the observer's head movement)
  * dwell time next to each line: a live timer while dwelling, otherwise the last dwell
  * effort (lifting) arrow from the bottom line to the top line with its duration, and a
    thinner lowering arrow; live while in progress, otherwise the last value
  * the bar proxy position
Lines, arrows and the bar marker are drawn through a time-smoothed version of the
stabilising transform (--geom-smooth) to remove jitter; labels are smoothed separately
(--label-smooth).
  * a panel with clip, lift, rep number / total reps in the clip, and the last rep's duration

Frames outside clips are copied unchanged, so the output keeps the original timing (frame
N of the output is frame N of the input when rendering from the start).

usage: annotate_video.py VIDEO ANALYSIS_DIR OUT.mp4 [--start S] [--end S] [--clip N]

ANALYSIS_DIR is the output folder of analyse_reps.py (needs frames_annotation.csv.gz,
clips.csv and reps.csv). Uses ffmpeg (H.264, with the original audio) if it is on the PATH, otherwise
OpenCV's mp4v encoder (no audio). With ffmpeg, chapter markers for each clip / lift / rep
are added (see add_chapters.py) unless --no-chapters is given.
"""
import argparse, shutil, subprocess, time
from pathlib import Path
import cv2, numpy as np, pandas as pd
from add_chapters import write_with_chapters

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("analysis_dir")
ap.add_argument("out")
ap.add_argument("--start", type=float, default=0.0, help="seconds")
ap.add_argument("--end", type=float, default=None, help="seconds (default: end of video)")
ap.add_argument("--clip", type=int, default=None, help="render just this clip (with 1 s either side)")
ap.add_argument("--no-chapters", action="store_true",
                help="don't add clip / lift / rep chapter markers (added by default when ffmpeg is available)")
ap.add_argument("--geom-smooth", type=float, default=0.4,
                help="seconds over which the stabilising transform used for lines, arrows and the bar "
                     "marker is smoothed, removing jitter (0 = off)")
ap.add_argument("--label-smooth", type=float, default=0.8,
                help="seconds over which label positions are smoothed (0 = off); lines are never smoothed")
args = ap.parse_args()

adir = Path(args.analysis_dir)
fa = pd.read_csv(adir / "frames_annotation.csv.gz").set_index("frame")
clips = pd.read_csv(adir / "clips.csv").set_index("clip")
H_COLS = [f"h{a}{b}" for a in range(3) for b in range(3)]

cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start, end = args.start, args.end if args.end is not None else n_frames / fps
if args.clip is not None:
    c = clips.loc[args.clip]
    start, end = max(0.0, c["start_t"] - 1), c["end_t"] + 1
f0, f1 = int(round(start * fps)), min(n_frames - 1, int(round(end * fps)))

# ------------------------------------------------------------- per-frame phase timers
# For each phase, the live elapsed time while in it and the duration of the last completed
# occurrence (within the same clip) otherwise.
PHASES = ("top_dwell", "lowering", "bottom_dwell", "effort")
fa["phase"] = fa["phase"].fillna("")
live = {p: pd.Series(np.nan, index=fa.index) for p in PHASES}
last = {p: pd.Series(np.nan, index=fa.index) for p in PHASES}
for clip_id, g in fa.groupby("clip"):
    frames, phases = g.index.values, g["phase"].values
    run_start = 0
    last_val = dict.fromkeys(PHASES, np.nan)
    for i in range(len(g)):
        if i > 0 and phases[i] != phases[i - 1]:
            if phases[i - 1] in PHASES:  # the previous run just finished
                last_val[phases[i - 1]] = (frames[i - 1] - frames[run_start] + 1) / fps
            run_start = i
        if phases[i] in PHASES:
            live[phases[i]][frames[i]] = (frames[i] - frames[run_start] + 1) / fps
        for q in PHASES:
            last[q][frames[i]] = last_val[q]

# last completed rep duration (top dwell end -> next top dwell start) per frame
rep_dur = pd.Series(np.nan, index=fa.index)
for clip_id, g in fa.groupby("clip"):
    t_eff_end, t_ecc_start, val = {}, {}, np.nan
    ph, reps, ts = g["phase"].values, g["rep"].values, g["t"].values
    for i in range(len(g)):
        if i > 0 and ph[i - 1] == "top_dwell" and ph[i] == "lowering":
            t_ecc_start[reps[i]] = ts[i]
        if i > 0 and ph[i - 1] == "effort" and ph[i] != "effort":
            k = reps[i - 1]
            if k in t_ecc_start:
                val = ts[i] - t_ecc_start[k]
        rep_dur[g.index[i]] = val

# rep levels: stabilised y of the top and bottom of travel for each counted rep
reps = pd.read_csv(adir / "reps.csv")
reps = reps[reps.rep_status == "ok"]
clip_cx = fa.groupby("clip").bar_stab_x.median()  # fallback when a rep has no x position
levels = {}
for _, r in reps.iterrows():
    xs = [v for v in (r["top_stab_x"], r["bottom_stab_x"]) if np.isfinite(v)]
    cx = np.mean(xs) if xs else clip_cx.get(int(r["clip"]), np.nan)
    levels[(int(r["clip"]), int(r["rep"]))] = (r["top_stab_y"], r["bottom_stab_y"], cx)
half_len = fa.groupby("clip").person_w.median() * 0.6  # half line length, px

# ------------------------------------------------------- smoothed scene <- clip transform
# The per-frame homography carries high-frequency jitter from feature matching. Smooth its
# inverse (clip coords -> scene pixels) over time, element-wise after normalising h22 = 1,
# with a centred Gaussian so the overlay doesn't lag the observer's head movement. Only the
# mapping is smoothed: rep-to-rep changes in line level stay instant. Frames with no valid
# transform are filled by interpolation within the clip (the overlay is drawn there too).
HINV_COLS = [f"i{a}{b}" for a in range(3) for b in range(3)]


def _inv(row):
    m = row.reshape(3, 3)
    if not np.all(np.isfinite(m)):
        return np.full(9, np.nan)
    mi = np.linalg.inv(m)
    return (mi / mi[2, 2]).ravel()


fa[HINV_COLS] = (np.apply_along_axis(_inv, 1, fa[H_COLS].values.astype(float)) if len(fa)
                 else np.empty((0, 9)))  # empty session: nothing to draw
win = int(round(args.geom_smooth * fps)) | 1
for clip_id, idx in fa.groupby("clip").groups.items():
    g = fa.loc[idx, HINV_COLS]
    full = pd.RangeIndex(idx.min(), idx.max() + 1)
    g = g.reindex(full).interpolate(limit_direction="both")
    if win >= 3:
        g = g.rolling(win, center=True, min_periods=1, win_type="gaussian").mean(std=win / 4)
    fa.loc[idx, HINV_COLS] = g.loc[idx].values

# ------------------------------------------------------------------------ drawing helpers
GREEN, RED, ORANGE, BLUE, WHITE = (60, 200, 60), (60, 60, 230), (0, 160, 255), (230, 160, 60), (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, org, scale=0.55, color=WHITE, thick=1, box=True):
    """Text with its baseline-left corner at org; on a dark box so it reads on any background."""
    x, y = int(org[0]), int(org[1])
    (w, h), base = cv2.getTextSize(s, FONT, scale, thick + 1)
    # keep the label inside the frame (lines may run off the edge when the screen does)
    x = min(max(x, 4), img.shape[1] - w - 4)
    y = min(max(y, h + 4), img.shape[0] - base - 4)
    if box:
        cv2.rectangle(img, (x - 3, y - h - 3), (x + w + 3, y + base + 1), (25, 25, 25), -1)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick + 1 if box else thick, cv2.LINE_AA)


def to_scene(Hinv, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ Hinv.T
    return p[:, :2] / p[:, 2:3]


def fmt(v):
    return "--" if not np.isfinite(v) else f"{v:.2f} s"


def geometry(f):
    """Scene-pixel geometry for frame f (unsmoothed), or None outside a usable clip frame."""
    if f not in fa.index:
        return None
    row = fa.loc[f]
    Hinv = row[HINV_COLS].values.astype(float).reshape(3, 3)  # smoothed clip -> scene
    if not np.all(np.isfinite(Hinv)):
        return None
    clip_id, k = int(row["clip"]), int(row["rep"])
    lv = levels.get((clip_id, max(k, 1)))
    if lv is None:
        return None
    top_y, bot_y, cx = lv
    if not np.all(np.isfinite([top_y, bot_y, cx])):
        return None  # no measured level for this rep: draw nothing rather than guess
    L = float(half_len.get(clip_id, 120))
    ex = cx - L - 18
    g = {"clip": clip_id}
    g["tl"], g["tr"] = to_scene(Hinv, [(cx - L, top_y), (cx + L, top_y)])
    g["bl"], g["br"] = to_scene(Hinv, [(cx - L, bot_y), (cx + L, bot_y)])
    g["eb"], g["et"] = to_scene(Hinv, [(ex, bot_y), (ex, top_y)])
    g["lt"], g["lb"] = to_scene(Hinv, [(ex - 22, top_y), (ex - 22, bot_y)])
    g["bp"] = None
    if np.isfinite(row["bar_stab_y"]):
        bx = row["bar_stab_x"] if np.isfinite(row["bar_stab_x"]) else cx
        (g["bp"],) = to_scene(Hinv, [(bx, row["bar_stab_y"])])
    # label anchors: right end of each line, and the middle of the effort arrow
    g["top_lab"], g["bot_lab"], g["eff_lab"] = g["tr"], g["br"], (g["eb"] + g["et"]) / 2
    return g


LABELS = ("top_lab", "bot_lab", "eff_lab")


def smooth_labels(geo):
    """Centred Gaussian smoothing of label anchors over time, within each clip.

    Only the label positions are smoothed; the lines and arrows keep their per-frame
    positions. Centring (rather than a running average) means labels don't lag behind.
    """
    win = int(round(args.label_smooth * fps))
    if win < 2 or not geo:
        return {f: {k: g[k] for k in LABELS} for f, g in geo.items()}
    win |= 1  # odd
    frames = np.array(sorted(geo))
    clip_of = pd.Series([geo[f]["clip"] for f in frames], index=frames)
    out = {f: {} for f in frames}
    for _, idx in clip_of.groupby(clip_of).groups.items():
        full = pd.RangeIndex(idx.min(), idx.max() + 1)  # include gaps, then interpolate
        for k in LABELS:
            xy = pd.DataFrame([geo[f][k] for f in idx], index=idx, columns=["x", "y"]).reindex(full)
            xy = xy.interpolate(limit_direction="both")
            sm = xy.rolling(win, center=True, min_periods=1, win_type="gaussian").mean(std=win / 4)
            for f in idx:
                out[f][k] = sm.loc[f].values
    return out


def label(img, s, anchor, actual, offset, scale, color):
    """Text at a smoothed anchor + offset, with a thin leader to the real line end if apart."""
    pos = anchor + offset
    if np.hypot(*(anchor - actual)) > 6:
        cv2.line(img, tuple(actual.astype(int)), tuple(anchor.astype(int)), color, 1, cv2.LINE_AA)
    text(img, s, pos, scale, color)


def draw(img, f, g, lab):
    row = fa.loc[f]
    clip_id, k, n = int(row["clip"]), int(row["rep"]), int(row["n_reps"])
    ph = row["phase"]
    lift = clips.loc[clip_id, "lift"].replace("_", " ")

    # --- panel (fixed position)
    x0, y0 = 20, 20
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + 330, y0 + 112), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    text(img, f"Clip {clip_id}  ({lift})", (x0 + 12, y0 + 30), 0.7, WHITE, 1, box=False)
    text(img, f"Rep {k} of {n}" if k else f"Rep - of {n}", (x0 + 12, y0 + 62), 0.75, ORANGE, 1, box=False)
    text(img, f"last rep {fmt(rep_dur[f])}" + (f"   {ph.replace('_', ' ')}" if ph else ""),
         (x0 + 12, y0 + 94), 0.55, box=False)
    if g is None:
        return
    P = lambda v: tuple(np.asarray(v).astype(int))
    PINK = (90, 90, 255)

    # --- top and bottom of travel lines (per-frame, not smoothed)
    cv2.line(img, P(g["tl"]), P(g["tr"]), GREEN, 4 if ph == "top_dwell" else 2, cv2.LINE_AA)
    cv2.line(img, P(g["bl"]), P(g["br"]), RED, 4 if ph == "bottom_dwell" else 2, cv2.LINE_AA)

    # --- effort (up) and lowering (down) arrows
    eff_on, low_on = ph == "effort", ph == "lowering"
    cv2.arrowedLine(img, P(g["eb"]), P(g["et"]), ORANGE, 4 if eff_on else 2, cv2.LINE_AA, tipLength=0.15)
    cv2.arrowedLine(img, P(g["lt"]), P(g["lb"]), BLUE, 3 if low_on else 1, cv2.LINE_AA, tipLength=0.15)

    # --- bar proxy
    if g["bp"] is not None:
        cv2.circle(img, P(g["bp"]), 7, ORANGE if eff_on else WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, P(g["bp"]), 7, (0, 0, 0), 1, cv2.LINE_AA)

    # --- labels at smoothed positions
    top_dw = live["top_dwell"][f] if ph == "top_dwell" else last["top_dwell"][f]
    bot_dw = live["bottom_dwell"][f] if ph == "bottom_dwell" else last["bottom_dwell"][f]
    eff = live["effort"][f] if eff_on else last["effort"][f]
    low = live["lowering"][f] if low_on else last["lowering"][f]
    label(img, "TOP", lab["top_lab"], g["tr"], (8, -6), 0.55, GREEN)
    text(img, f"dwell {fmt(top_dw)}", lab["top_lab"] + (8, 18), 0.5, GREEN)
    label(img, "BOTTOM", lab["bot_lab"], g["br"], (8, -6), 0.55, PINK)
    text(img, f"dwell {fmt(bot_dw)}", lab["bot_lab"] + (8, 18), 0.5, PINK)
    em = lab["eff_lab"]
    text(img, f"EFFORT {fmt(eff)}", em + (-185, -8), 0.6 if eff_on else 0.55, ORANGE)
    text(img, f"lowering {fmt(low)}", em + (-185, 18), 0.5, BLUE)


# ------------------------------------------------------------------------------- output
ffmpeg = shutil.which("ffmpeg")
final_out = args.out
if ffmpeg and not args.no_chapters:  # render to a temporary file, then remux with chapters
    args.out = str(Path(final_out).with_suffix(".nochapters.mp4"))
if ffmpeg:
    cmd = [ffmpeg, "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
           "-ss", f"{f0 / fps:.3f}", "-t", f"{(f1 - f0 + 1) / fps:.3f}", "-i", args.video,
           "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(buf):
        try:
            proc.stdin.write(buf)
        except BrokenPipeError:
            proc.wait()
            raise SystemExit(f"ffmpeg stopped unexpectedly (exit code {proc.returncode})")
else:
    print("ffmpeg not found: writing mp4v without audio")
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not vw.isOpened():
        raise SystemExit(f"could not open {args.out} for writing")
    write = None

geo = {f: g for f in range(f0, f1 + 1) if (g := geometry(f)) is not None}
labels = smooth_labels(geo)

cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
tic = time.time()
read_fail = None
for f in range(f0, f1 + 1):
    ok, img = cap.read()
    if not ok:
        read_fail = f
        break
    if f in fa.index:
        draw(img, f, geo.get(f), labels.get(f))
    if write:
        write(img.tobytes())
    else:
        vw.write(img)
    if (f - f0) % 1000 == 999:
        print(f"{(f - f0 + 1)} / {f1 - f0 + 1} frames, {(f - f0 + 1) / (time.time() - tic):.0f} fps", flush=True)

if ffmpeg:
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed (exit code {proc.returncode}); {args.out} is incomplete")
else:
    vw.release()
if read_fail is not None:
    # The container's frame count can exceed the decodable frames (e.g. a video cut without
    # re-encoding), and a damaged file looks the same, so warn rather than fail.
    print(f"WARNING: the video could only be read up to frame {read_fail - 1}, not frame {f1} as "
          f"its header states; the output stops there", flush=True)
    f1 = read_fail - 1
if ffmpeg and not args.no_chapters:
    chs = write_with_chapters(args.out, adir, final_out, offset=f0 / fps)
    Path(args.out).unlink()
    print(f"added {len(chs)} chapters")
print(f"wrote {final_out}: frames {f0}-{f1} ({(f1 - f0 + 1) / fps:.1f} s) in {time.time() - tic:.0f}s")
