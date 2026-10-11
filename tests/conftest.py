"""Shared helpers for the tests: synthetic per-frame data, a tiny video, and script runners."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

FPS = 25.0
H_COLS = [f"h{a}{b}" for a in range(3) for b in range(3)]


def run_script(name, *args):
    """Run scripts/<name> with this interpreter; returns the CompletedProcess."""
    return subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, args)],
                          capture_output=True, text=True, check=False)


def bar_trace(setup_s=2.0, n_reps=6, top=400.0, bottom=500.0, lower_s=0.6, bottom_s=0.2,
              lift_s=0.4, top_s=1.0, tail_s=1.0):
    """Bar height per frame (image y: larger = lower) for a set of identical reps."""
    def hold(level, secs):
        return [level] * round(secs * FPS)

    def ramp(a, b, secs):
        n = round(secs * FPS)
        return list(np.linspace(a, b, n + 1)[1:])

    y = hold(top, setup_s)
    for _ in range(n_reps):
        y += ramp(top, bottom, lower_s) + hold(bottom, bottom_s) + ramp(bottom, top, lift_s)
        y += hold(top, top_s)
    return np.array(y + hold(top, tail_s))


def session_frames(bar_y, lift="bench_press"):
    """Per-frame table in pass-1b format for one clip, identity stabilisation.

    bench_press: athlete lying (hips level with shoulders), bar proxy = wrists.
    squat: athlete upright (hips well below shoulders), bar proxy = shoulders.
    """
    n = len(bar_y)
    d = {"frame": np.arange(n), "t": np.arange(n) / FPS, "p_conf": 0.9,
         "scr_x0": 500.0, "scr_y0": 200.0, "scr_x1": 1400.0, "scr_y1": 950.0}
    if lift == "bench_press":
        d.update(p_x0=800.0, p_y0=500.0, p_x1=1000.0, p_y1=600.0)  # wide, short box
        pts = {"wri": (900, bar_y), "sho": (850, 550.0), "hip": (950, 550.0),
               "knee": (980, 555.0), "ank": (1000, 560.0)}
    else:
        d.update(p_x0=850.0, p_y0=250.0, p_x1=950.0, p_y1=650.0)  # tall, narrow box
        pts = {"sho": (900, bar_y), "wri": (900, bar_y), "hip": (900, bar_y + 120),
               "knee": (900, 520.0), "ank": (900, 640.0)}
    for name, (x, y) in pts.items():
        for side in ("l", "r"):
            d[f"{side}_{name}_x"], d[f"{side}_{name}_y"], d[f"{side}_{name}_c"] = float(x), y, 0.9
    df = pd.DataFrame(d)
    for c in H_COLS:
        df[c] = 1.0 if c in ("h00", "h11", "h22") else 0.0
    df["h_inliers"], df["ref_id"], df["seg_id"] = 300, 1, 1
    df["stabilisation"], df["video_change"] = "restab", 0
    return df


def analyse(tmp_path, frames, name="frames_restab.csv"):
    """Write frames to a CSV, run 02analyse_reps.py on it; returns (process, out_dir)."""
    src = tmp_path / "in"
    src.mkdir(exist_ok=True)
    frames.to_csv(src / name, index=False)
    out = tmp_path / "out"
    return run_script("02analyse_reps.py", out, src), out


@pytest.fixture
def tiny_video(tmp_path):
    """A 2-second 320x240 25 fps video file."""
    path = tmp_path / "tiny.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240))
    for i in range(int(2 * FPS)):
        img = np.full((240, 320, 3), 40, np.uint8)
        cv2.rectangle(img, (40 + i, 60), (200 + i, 180), (230, 230, 230), -1)
        vw.write(img)
    vw.release()
    return path
