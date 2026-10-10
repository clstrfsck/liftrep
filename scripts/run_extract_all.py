"""Run pass-1 extraction over a whole video in parallel chunks (any OS).

Splits the video into JOBS equal time chunks, runs extract_frames.py on each in its own
process, and waits for them all. Outputs work/frames_<start>.csv and work/log_<start>.txt.

usage: run_extract_all.py VIDEO [--out-dir work] [--jobs N] [--device auto|cuda|mps|cpu]
                          [--model models/yolo11x-pose.pt]
"""
import argparse, os, subprocess, sys, time
from pathlib import Path
import cv2
from device import pick_device, describe

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out-dir", default="work")
ap.add_argument("--jobs", type=int, default=0,
                help="parallel processes (default: 4 on a GPU, about one per 4 CPU cores on CPU)")
ap.add_argument("--device", default="auto")
ap.add_argument("--model", default="models/yolo11x-pose.pt")
args = ap.parse_args()

device = pick_device(args.device)
cores = os.cpu_count() or 4
jobs = args.jobs or (4 if device != "cpu" else max(1, cores // 4))
print(f"device: {describe(device)}, {jobs} parallel jobs")
if device == "cpu":
    print("note: CPU inference is slow; consider --model models/yolo11m-pose.pt or yolo11s-pose.pt")

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise SystemExit(f"cannot open video: {args.video}")
duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
cap.release()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
extractor = Path(__file__).with_name("extract_frames.py")
env = dict(os.environ)
if device == "cpu":  # share the cores between processes instead of oversubscribing
    env["OMP_NUM_THREADS"] = str(max(1, cores // jobs))

step = duration / jobs
procs = []
for i in range(jobs):
    start = int(i * step)
    end = int((i + 1) * step) if i < jobs - 1 else int(duration) + 1
    log = open(out_dir / f"log_{start}.txt", "w")
    cmd = [sys.executable, str(extractor), args.video, str(out_dir / f"frames_{start}.csv"),
           "--start", str(start), "--end", str(end), "--device", device, "--model", args.model]
    procs.append((start, end, log, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)))
    print(f"  chunk {start:>5}-{end:<5}s -> {out_dir / f'frames_{start}.csv'}")

tic = time.time()
failed = 0
for start, end, log, p in procs:
    rc = p.wait()
    log.close()
    if rc != 0:
        failed += 1
        print(f"chunk {start}-{end}s FAILED (exit {rc}); see {out_dir / f'log_{start}.txt'}")
print(f"{'ALL DONE' if not failed else f'{failed} chunk(s) failed'} in {(time.time() - tic) / 60:.1f} min")
sys.exit(1 if failed else 0)
