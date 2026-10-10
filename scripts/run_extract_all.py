"""Run pass-1 extraction over a whole video in parallel chunks (any OS), then pass 1b.

Splits the video into JOBS equal time chunks, runs extract_frames.py on each in its own
process, and waits for them all. Outputs work/frames_<start>.csv and work/log_<start>.txt.
Then runs restabilise.py (pass 1b) into work/restab/, which analyse_reps.py requires: each
pass-1 chunk numbers its segments and chains its transforms independently.

usage: run_extract_all.py VIDEO [--out-dir work] [--jobs N] [--device auto|cuda|mps|cpu]
                          [--model models/yolo11x-pose.pt] [--overwrite] [--no-restabilise]
"""
import argparse
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import cv2
from device import describe, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out-dir", default="work")
ap.add_argument("--jobs", type=int, default=0,
                help="parallel processes (default: 4 on a GPU, about one per 4 CPU cores on CPU)")
ap.add_argument("--device", default="auto")
ap.add_argument("--model", default="models/yolo11x-pose.pt")
ap.add_argument("--overwrite", action="store_true",
                help="delete pass-1 chunk files and logs left in --out-dir by an earlier run")
ap.add_argument("--no-restabilise", action="store_true",
                help="skip pass 1b (re-stabilisation); it must then be run before analyse_reps.py")
args = ap.parse_args()

if args.jobs < 0:
    raise SystemExit("--jobs must be at least 1 (or 0 for the default)")
device = pick_device(args.device)
cores = os.cpu_count() or 4
jobs = args.jobs or (4 if device != "cpu" else max(1, cores // 4))
print(f"device: {describe(device)}")
if device == "cpu":
    print("note: CPU inference is slow; consider --model models/yolo11m-pose.pt or yolo11s-pose.pt")

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise SystemExit(f"cannot open video: {args.video}")
duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
cap.release()

# every chunk at least MIN_CHUNK_S long, so chunk start times (and file names) stay distinct
MIN_CHUNK_S = 10
capped = max(1, min(jobs, int(duration // MIN_CHUNK_S)))
if capped < jobs:
    print(f"using {capped} job(s): each chunk must be at least {MIN_CHUNK_S} s")
jobs = capped
print(f"{jobs} parallel job(s)")

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
# files from an earlier run (possibly split differently) would otherwise be read by pass 1b
stale = sorted(out_dir.glob("frames_*.csv")) + sorted(out_dir.glob("log_*.txt"))
if stale:
    if not args.overwrite:
        raise SystemExit(f"{out_dir} already holds pass-1 files from an earlier run "
                         f"({', '.join(p.name for p in stale[:4])}{', ...' if len(stale) > 4 else ''}). "
                         "Use another --out-dir, or --overwrite to delete them.")
    for p in stale:
        p.unlink()
    print(f"deleted {len(stale)} file(s) from an earlier run")
extractor = Path(__file__).with_name("extract_frames.py")
env = dict(os.environ)
if device == "cpu":  # share the cores between processes instead of oversubscribing
    env["OMP_NUM_THREADS"] = str(max(1, cores // jobs))

step = duration / jobs
failed = 0
with ExitStack() as logs:  # each chunk's log file stays open until its process ends
    procs = []
    for i in range(jobs):
        start = int(i * step)
        end = int((i + 1) * step) if i < jobs - 1 else int(duration) + 1
        log = logs.enter_context(open(out_dir / f"log_{start}.txt", "w"))
        cmd = [sys.executable, str(extractor), args.video, str(out_dir / f"frames_{start}.csv"),
               "--start", str(start), "--end", str(end), "--device", device, "--model", args.model]
        procs.append((start, end, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)))
        print(f"  chunk {start:>5}-{end:<5}s -> {out_dir / f'frames_{start}.csv'}")

    # list of this run's chunks, read by restabilise.py
    (out_dir / "chunks.txt").write_text("\n".join(f"frames_{s}.csv" for s, _, _ in procs) + "\n")

    tic = time.time()
    for start, end, p in procs:
        rc = p.wait()
        if rc != 0:
            failed += 1
            print(f"chunk {start}-{end}s FAILED (exit {rc}); see {out_dir / f'log_{start}.txt'}")
print(f"{'pass 1 done' if not failed else f'{failed} chunk(s) failed'} in {(time.time() - tic) / 60:.1f} min")
if failed:
    sys.exit(1)

# pass 1b: required before analysis (pass-1 chunks have independent coordinate systems)
restab_dir = out_dir / "restab"
if args.no_restabilise:
    print(f"skipped re-stabilisation; run: restabilise.py \"{args.video}\" {out_dir} {restab_dir}")
    sys.exit(0)
print("pass 1b: re-stabilising clips ...", flush=True)
rc = subprocess.run([sys.executable, str(Path(__file__).with_name("restabilise.py")),
                     args.video, str(out_dir), str(restab_dir)], check=False).returncode
if rc != 0:
    sys.exit(rc)
print(f"ALL DONE. Next: analyse_reps.py {out_dir / 'out'} {restab_dir}")
