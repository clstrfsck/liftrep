"""Add chapter markers (clip / lift / rep) to an annotated video, without re-encoding.

Chapters cover the whole timeline and don't overlap (MP4 chapters can't be nested):
  "Between clips"                                   outside lift clips
  "Clip N - bench press (block B) - set-up"        clip start -> first rep
  "Clip N - bench press (block B) - rep k of M"   one per counted rep
Reps whose tracking quality is "check" get " [check]" appended.

usage: add_chapters.py VIDEO ANALYSIS_DIR OUT.mp4 [--offset S]

--offset is the source time (s) of the video's first frame, for videos rendered from a
time range (annotate_video.py --start/--clip); 0 for a whole-video render.
OUT may not be the same file as VIDEO. Requires ffmpeg on the PATH.
"""
import argparse, shutil, subprocess, tempfile
from pathlib import Path
import cv2, pandas as pd


def chapters(analysis_dir, offset, duration):
    """List of (start_s, end_s, title) in the output video's time, covering [0, duration]."""
    adir = Path(analysis_dir)
    clips = pd.read_csv(adir / "clips.csv")
    reps = pd.read_csv(adir / "reps.csv")
    reps = reps[reps.rep_status == "ok"]
    marks = []  # (source start time, title); each chapter runs to the next mark
    for _, c in clips.sort_values("start_t").iterrows():
        cid = int(c["clip"])
        name = f"Clip {cid} - {c['lift'].replace('_', ' ')} (block {int(c['block'])})"
        r = reps[reps["clip"] == cid].sort_values("rep")
        marks.append((c["start_t"], f"{name} - set-up" if len(r) else f"{name} - no reps"))
        for _, rp in r.iterrows():
            flag = " [check]" if rp["tracking_quality"] != "good" else ""
            marks.append((rp["rep_start_t"], f"{name} - rep {int(rp['rep'])} of {len(r)}{flag}"))
        marks.append((c["end_t"] + 1 / 25, "Between clips"))
    marks.sort(key=lambda m: m[0])
    if not marks or marks[0][0] > offset:
        marks.insert(0, (offset, "Between clips"))

    out = []
    for i, (t, title) in enumerate(marks):
        s = t - offset
        e = (marks[i + 1][0] - offset) if i + 1 < len(marks) else duration
        s, e = max(0.0, s), min(duration, e)
        if e - s >= 0.1:
            if out and out[-1][2] == title == "Between clips":
                out[-1] = (out[-1][0], e, title)  # merge adjacent gaps
            else:
                out.append((s, e, title))
    return out


def esc(s):
    """Escape a value for ffmpeg's FFMETADATA format."""
    for ch in ("\\", "=", ";", "#", "\n"):
        s = s.replace(ch, "\\" + ch)
    return s


def write_with_chapters(video, analysis_dir, out, offset=0.0):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on the PATH")
    if Path(video).resolve() == Path(out).resolve():
        raise SystemExit("OUT must be a different file from VIDEO")
    cap = cv2.VideoCapture(str(video))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    chs = chapters(analysis_dir, offset, duration)
    meta = [";FFMETADATA1",
            "title=" + esc(Path(video).stem + " - lift rep annotations"),
            "comment=" + esc("Chapters: clip / lift / rep from analyse_reps.py")]
    for s, e, title in chs:
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(round(s * 1000))}",
                 f"END={int(round(e * 1000))}", "title=" + esc(title)]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(meta) + "\n")
        meta_path = fh.name
    try:
        subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(video), "-i", meta_path,
                        "-map", "0", "-map_metadata", "1", "-map_chapters", "1",
                        "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    finally:
        Path(meta_path).unlink(missing_ok=True)
    return chs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("analysis_dir")
    ap.add_argument("out")
    ap.add_argument("--offset", type=float, default=0.0)
    args = ap.parse_args()
    chs = write_with_chapters(args.video, args.analysis_dir, args.out, args.offset)
    n_rep = sum("- rep " in t for _, _, t in chs)
    print(f"wrote {args.out}: {len(chs)} chapters ({n_rep} reps)")
