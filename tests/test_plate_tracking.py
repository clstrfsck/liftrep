"""Tests of plate tracking (plate_tracking.py) and of pass 2 following the plate."""
import cv2
import numpy as np
import pandas as pd
from conftest import analyse, bar_trace, session_frames
from plate_tracking import clip_plate, plate_candidates, track_plate

IDENTITY = np.eye(3)


def test_candidates_find_a_dark_plate_on_a_bright_screen():
    img = np.full((600, 900), 235, np.uint8)
    cv2.circle(img, (420, 260), 40, 30, -1)
    cv2.circle(img, (420, 260), 8, 200, -1)  # bright hub, as on the real plates
    c = plate_candidates(img, (100, 50, 800, 550))
    best = c[np.argmin(np.hypot(c[:, 0] - 420, c[:, 1] - 260))]
    assert np.hypot(best[0] - 420, best[1] - 260) < 3 and abs(best[2] - 40) < 4


def moving_plate(n_reps=6, hide_every=0):
    """Circles per frame: the plate (r 30, following a bar trace) and a stored plate of the
    same size that never moves, as on a rack post nearby; wrists just below the plate."""
    y = bar_trace(n_reps=n_reps)
    cands, wrists = [], np.full((len(y), 2, 2), np.nan)
    for i, yi in enumerate(y):
        c = [(1050.0, 250.0, 30.0)]  # stored plate
        if not (hide_every and i % hide_every == 0):
            c.append((900.0, yi, 30.0))
        cands.append(np.array(c))
        wrists[i, 0] = (905.0, yi + 20)
    return y, cands, wrists


def test_track_follows_the_plate_not_a_stored_one():
    y, cands, wrists = moving_plate(hide_every=4)
    out = track_plate(cands, [IDENTITY] * len(y), wrists)
    found = np.isfinite(out[:, 0])
    assert found.mean() > 0.7
    assert np.allclose(out[found, 0], 900) and np.allclose(out[found, 1], y[found])


def test_track_skips_frames_without_the_plate_instead_of_jumping():
    # the plate is missed for 0.3 s around every bottom (it overlaps the body); a nearby
    # stored plate is always found, but the track must not hop onto it
    y, cands, wrists = moving_plate()
    for i in np.flatnonzero(y > 480):
        cands[i] = cands[i][:1]
    out = track_plate(cands, [IDENTITY] * len(y), wrists)
    assert not np.any(out[:, 0] == 1050)


def rows_with(wrists):
    return pd.DataFrame({"l_wri_x": wrists[:, 0, 0], "l_wri_y": wrists[:, 0, 1], "l_wri_c": 0.9,
                         "r_wri_x": np.nan, "r_wri_y": np.nan, "r_wri_c": 0.0})


def test_clip_plate_accepts_a_plate_moving_with_the_wrists():
    y, cands, wrists = moving_plate()
    track, info = clip_plate(cands, [IDENTITY] * len(y), rows_with(wrists))
    assert track is not None and info["plate"] == "ok"


def test_clip_plate_rejects_a_circle_that_does_not_move_with_the_wrists():
    # front view: the plate is edge-on; the only circle near the hands is a still one below
    # them (the head), and the hands move
    y = bar_trace()
    cands = [np.array([(900.0, 560.0, 30.0)]) for _ in y]
    wrists = np.full((len(y), 2, 2), np.nan)
    wrists[:, 0, 0], wrists[:, 0, 1] = 900.0, y + 60
    track, info = clip_plate(cands, [IDENTITY] * len(y), rows_with(wrists))
    assert track is None and info["plate"] in ("rejected", "not found at the hands")


def with_plate(frames, bar_y):
    frames = frames.copy()
    frames["plate_x"], frames["plate_y"], frames["plate_r"] = 900.0, bar_y, 30.0
    return frames


def test_bench_press_follows_the_plate_when_the_wrists_are_misplaced(tmp_path):
    y = bar_trace(n_reps=6)
    frames = session_frames(y)
    # the wrists are on the chest whenever the bar is low (hidden behind the plate)
    low = y > 450
    for side in ("l", "r"):
        frames.loc[low, f"{side}_wri_y"] = 560.0
    proc, out = analyse(tmp_path, with_plate(frames, y))
    assert proc.returncode == 0, proc.stderr
    clip = pd.read_csv(out / "clips.csv").iloc[0]
    reps = pd.read_csv(out / "reps.csv")
    assert clip.tracked == "plate" and clip.plate_frac == 1.0
    assert clip.n_reps == 6 and (reps.tracking_quality == "good").all()
    assert np.allclose(reps.bottom_stab_y, 500, atol=3)


def test_sparse_plate_track_falls_back_to_the_wrists(tmp_path):
    y = bar_trace(n_reps=6)
    frames = with_plate(session_frames(y), y)
    frames.loc[frames.index % 3 != 0, ["plate_x", "plate_y", "plate_r"]] = np.nan  # 1 frame in 3
    _, out = analyse(tmp_path, frames)
    clip = pd.read_csv(out / "clips.csv").iloc[0]
    assert clip.tracked == "l_wri+r_wri" and clip.n_reps == 6


def test_squat_keeps_the_shoulders(tmp_path):
    y = bar_trace(n_reps=5)
    _, out = analyse(tmp_path, with_plate(session_frames(y, lift="squat"), y))
    assert pd.read_csv(out / "clips.csv").loc[0, "tracked"] == "l_sho+r_sho"


def test_bench_press_is_unchanged_without_plate_columns(tmp_path):
    _, out = analyse(tmp_path, session_frames(bar_trace(n_reps=6)))
    clip = pd.read_csv(out / "clips.csv").iloc[0]
    assert clip.tracked == "l_wri+r_wri" and clip.plate_frac == 0 and clip.n_reps == 6

