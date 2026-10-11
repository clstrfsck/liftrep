"""End-to-end tests of 02analyse_reps.py on synthetic sessions with known answers."""
import numpy as np
import pandas as pd
import pytest
from conftest import FPS, analyse, bar_trace, run_script, session_frames


def read(out, name):
    return pd.read_csv(out / name)


def test_bench_press_reps_and_phases(tmp_path):
    proc, out = analyse(tmp_path, session_frames(bar_trace(n_reps=6)))
    assert proc.returncode == 0, proc.stderr
    clips, reps = read(out, "clips.csv"), read(out, "reps.csv")
    assert len(clips) == 1 and clips.loc[0, "lift"] == "bench_press"
    ok = reps[reps.rep_status == "ok"]
    assert len(ok) == 6 and clips.loc[0, "n_reps"] == 6
    assert (ok.tracking_quality == "good").all()
    # 0.6 s lowering, 0.2 s pause, 0.4 s lift, measured from leaving / reaching the 10% bands
    assert ok.eccentric_s.between(0.4, 0.6).all()
    assert ok.effort_s.between(0.25, 0.42).all()
    assert ok.bottom_dwell_s.between(0.2, 0.45).all()
    assert ok.rep_duration_s.between(0.9, 1.4).all()
    # top and bottom of travel at the bar's real levels
    assert np.allclose(ok.top_stab_y, 400, atol=3) and np.allclose(ok.bottom_stab_y, 500, atol=3)
    # two events (top, bottom) per rep, plus the final top
    assert len(read(out, "events.csv")) == 2 * 6 + 1


def test_squat_is_classified_and_counted(tmp_path):
    proc, out = analyse(tmp_path, session_frames(bar_trace(n_reps=5), lift="squat"))
    assert proc.returncode == 0, proc.stderr
    clips = read(out, "clips.csv")
    assert clips.loc[0, "lift"] == "squat" and clips.loc[0, "n_reps"] == 5


def test_set_up_has_no_phase(tmp_path):
    _, out = analyse(tmp_path, session_frames(bar_trace(setup_s=3.0)))
    fa = pd.read_csv(out / "frames_annotation.csv.gz")
    first_lowering = fa.t[fa.phase == "lowering"].min()
    # before the first rep the bar is still: top dwell or set-up, never lowering or effort
    assert first_lowering > 2.9
    assert set(fa.phase[fa.t < 2.0].fillna("")) <= {"setup", "top_dwell"}


def hide_wrists(frames, start_s, secs):
    start = round(start_s * FPS)
    for side in ("l", "r"):
        frames.loc[start:start + round(secs * FPS) - 1, f"{side}_wri_c"] = 0.0
    return frames


THIRD_REP_LOWERING = 2.0 + 2 * 2.2  # set-up, then two 2.2 s reps


def test_rep_spanning_a_long_gap_is_indeterminate(tmp_path):
    # a 0.56 s gap (longer than the 0.5 s that is bridged) inside the third rep's lowering
    frames = hide_wrists(session_frames(bar_trace(n_reps=6)), THIRD_REP_LOWERING + 0.02, 0.56)
    proc, out = analyse(tmp_path, frames)
    assert proc.returncode == 0, proc.stderr
    reps, clips = read(out, "reps.csv"), read(out, "clips.csv")
    assert (reps.rep_status == "indeterminate").sum() == 1
    assert clips.loc[0, "n_reps"] == 5 and clips.loc[0, "n_indeterminate"] == 1
    assert reps.loc[reps.rep_status == "indeterminate", "max_gap_s"].iloc[0] > 0.5
    assert (reps.loc[reps.rep_status == "ok", "max_gap_s"] == 0).all()


def test_short_gap_is_bridged_and_counted(tmp_path):
    frames = hide_wrists(session_frames(bar_trace(n_reps=6)), THIRD_REP_LOWERING + 0.1, 0.2)
    _, out = analyse(tmp_path, frames)
    clip = read(out, "clips.csv").iloc[0]
    assert clip.n_reps == 6 and clip.unbridged_gap_s == 0


def test_gap_hiding_a_whole_rep_is_reported(tmp_path):
    # the gap covers the third rep's bottom: that rep can't be seen, so it isn't counted, but the
    # clip reports the unobserved time so the count can be read as a lower bound
    frames = hide_wrists(session_frames(bar_trace(n_reps=6)), THIRD_REP_LOWERING + 0.1, 0.8)
    _, out = analyse(tmp_path, frames)
    clip = read(out, "clips.csv").iloc[0]
    assert clip.n_reps == 5 and clip.unbridged_gap_s == pytest.approx(0.8, abs=0.05)


def test_unregistered_edge_frames_keep_the_clip_extent(tmp_path):
    frames = session_frames(bar_trace(n_reps=6))
    # pass 1b couldn't register the first and last second: transform missing, clip id kept
    edge = round(1.0 * FPS)
    for c in ("h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"):
        frames.loc[:edge, c] = np.nan
        frames.loc[len(frames) - edge:, c] = np.nan
    _, out = analyse(tmp_path, frames)
    clip = read(out, "clips.csv").iloc[0]
    assert clip.start_t == 0.0 and clip.end_t == pytest.approx(frames.t.iloc[-1])
    assert clip.n_reps == 6


def test_session_without_clips_writes_complete_empty_outputs(tmp_path):
    frames = session_frames(bar_trace())
    frames["p_conf"] = np.nan  # nobody on screen
    proc, out = analyse(tmp_path, frames)
    assert proc.returncode == 0, proc.stderr
    for name in ("clips.csv", "reps.csv", "events.csv"):
        df = read(out, name)
        assert len(df) == 0 and len(df.columns) > 5  # header present
    assert "rep_status" in read(out, "reps.csv").columns
    assert len(pd.read_csv(out / "frames_annotation.csv.gz")) == 0


def test_raw_pass1_input_is_refused(tmp_path):
    frames = session_frames(bar_trace()).drop(columns=["stabilisation"])
    proc, _ = analyse(tmp_path, frames, name="frames_0.csv")
    assert proc.returncode != 0 and "01restabilise.py" in proc.stderr + proc.stdout


def test_absolute_wildcard_input(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    session_frames(bar_trace()).to_csv(src / "frames_restab.csv", index=False)
    proc = run_script("02analyse_reps.py", tmp_path / "out", str(src.resolve() / "frames_*.csv"))
    assert proc.returncode == 0, proc.stderr


def test_old_plots_are_removed(tmp_path):
    out = tmp_path / "out"
    (out / "plots").mkdir(parents=True)
    (out / "plots" / "clip_99.png").write_bytes(b"stale")
    analyse(tmp_path, session_frames(bar_trace()))
    assert not (out / "plots" / "clip_99.png").exists()
    assert (out / "plots" / "clip_01.png").exists()


def test_shallow_rep_spanning_a_gap_is_indeterminate_not_partial(tmp_path):
    # four full reps, then a shallow one (40% of the range) with a gap inside its lowering
    y = list(bar_trace(n_reps=4, tail_s=0.0))
    shallow_start = len(y)
    y += list(bar_trace(setup_s=0.0, n_reps=1, bottom=440.0))
    frames = session_frames(np.array(y))
    gap_s = shallow_start / FPS + 0.04
    frames = hide_wrists(frames, gap_s, 0.56)
    _, out = analyse(tmp_path, frames)
    reps = read(out, "reps.csv")
    shallow = reps[reps.bottom_t > shallow_start / FPS]
    assert len(shallow) == 1 and shallow.rep_status.iloc[0] == "indeterminate"


def test_edge_gaps_count_as_unbridged_time(tmp_path):
    # the bar isn't seen for the clip's first second (e.g. registration failed there)
    frames = hide_wrists(session_frames(bar_trace(n_reps=6)), 0.0, 1.0)
    _, out = analyse(tmp_path, frames)
    clip = read(out, "clips.csv").iloc[0]
    assert clip.unbridged_gap_s == pytest.approx(1.0, abs=0.05)
    assert clip.n_reps == 6
