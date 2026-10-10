"""Input validation and safety checks of the command-line scripts."""
import pytest
from conftest import analyse, bar_trace, run_script, session_frames
from videoinfo import check_time, open_video


def test_open_video_rejects_a_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="cannot open video"):
        open_video(tmp_path / "missing.mp4")


def test_open_video_reads_properties(tiny_video):
    cap, fps, n = open_video(tiny_video)
    cap.release()
    assert fps == pytest.approx(25.0) and n == 50


def test_check_time():
    check_time(1.0, 2.0)
    with pytest.raises(SystemExit, match="outside the video"):
        check_time(3.0, 2.0, "--start")


def empty_analysis(tmp_path):
    frames = session_frames(bar_trace())
    frames["p_conf"] = float("nan")
    _, out = analyse(tmp_path, frames)
    return out


def test_annotation_refuses_to_overwrite_its_source(tmp_path, tiny_video):
    out = empty_analysis(tmp_path)
    before = tiny_video.read_bytes()
    proc = run_script("annotate_video.py", tiny_video, out, tiny_video)
    assert proc.returncode != 0 and "source video" in proc.stderr + proc.stdout
    assert tiny_video.read_bytes() == before


def test_annotation_refuses_when_the_temporary_render_is_the_source(tmp_path, tiny_video):
    out = empty_analysis(tmp_path)
    src = tmp_path / "x.nochapters.mp4"
    src.write_bytes(tiny_video.read_bytes())
    proc = run_script("annotate_video.py", src, out, tmp_path / "x.mp4")
    assert proc.returncode != 0 and "source video" in proc.stderr + proc.stdout


def test_annotation_of_an_empty_session(tmp_path, tiny_video):
    out = empty_analysis(tmp_path)
    proc = run_script("annotate_video.py", tiny_video, out, tmp_path / "annotated.mp4", "--no-chapters")
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "annotated.mp4").stat().st_size > 0


def test_annotation_rejects_an_unknown_clip(tmp_path, tiny_video):
    out = empty_analysis(tmp_path)
    proc = run_script("annotate_video.py", tiny_video, out, tmp_path / "a.mp4", "--clip", "7")
    assert proc.returncode != 0 and "no such clip" in proc.stderr + proc.stdout


def test_launcher_rejects_negative_jobs(tmp_path, tiny_video):
    proc = run_script("run_extract_all.py", tiny_video, "--out-dir", tmp_path / "w", "--jobs", "-1")
    assert proc.returncode != 0 and "--jobs" in proc.stderr + proc.stdout


def test_launcher_refuses_stale_files(tmp_path, tiny_video):
    work = tmp_path / "w"
    work.mkdir()
    (work / "frames_0.csv").write_text("frame\n")
    proc = run_script("run_extract_all.py", tiny_video, "--out-dir", work)
    assert proc.returncode != 0 and "earlier run" in proc.stderr + proc.stdout
    assert (work / "frames_0.csv").exists()


def test_restabilise_rejects_zero_jobs(tmp_path, tiny_video):
    proc = run_script("restabilise.py", tiny_video, tmp_path, tmp_path / "r", "--jobs", "0")
    assert proc.returncode != 0 and "--jobs" in proc.stderr + proc.stdout


def test_contact_sheet_rejects_non_positive_step(tmp_path, tiny_video):
    proc = run_script("contact_sheet.py", tiny_video, tmp_path / "cs", "0")
    assert proc.returncode != 0 and "STEP_S" in proc.stderr + proc.stdout


def test_contact_sheet_rejects_unreadable_video(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    proc = run_script("contact_sheet.py", bad, tmp_path / "cs", "1")
    assert proc.returncode != 0 and "cannot open video" in proc.stderr + proc.stdout


def test_screen_probe_skips_times_outside_the_video(tmp_path, tiny_video):
    proc = run_script("screen_probe.py", tiny_video, tmp_path / "sp", "1", "99")
    assert proc.returncode == 0, proc.stderr
    assert "99.0: outside the video" in proc.stdout
