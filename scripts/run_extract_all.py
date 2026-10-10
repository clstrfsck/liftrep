"""Run pass-1 extraction over a whole video in parallel chunks (any OS), then pass 1b.

Splits the video into JOBS equal time chunks, runs extract_frames.py on each in its own
process, and waits for them all. Outputs work/frames_<start>.csv and work/log_<start>.txt.
Then runs restabilise.py (pass 1b) into work/restab/, which analyse_reps.py requires: each
pass-1 chunk numbers its segments and chains its transforms independently.

usage: run_extract_all.py VIDEO [--out-dir work] [--jobs N] [--device auto|cuda|mps|cpu]
                          [--model models/yolo11x-pose.pt] [--no-restabilise]
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
ap.add_argument("--no-restabilise", action="store_true",
                help="skip pass 1b (re-stabilisation); it must then be run before analyse_reps.py")
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
                     args.video, str(out_dir), str(restab_dir)]).returncode
if rc != 0:
    sys.exit(rc)
print(f"ALL DONE. Next: analyse_reps.py {out_dir / 'out'} {restab_dir}")
