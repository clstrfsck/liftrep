"""Input validation and safety checks of the command-line scripts."""
import os
import subprocess

import cv2
import pytest
import torch
import videoinfo
from conftest import ROOT, analyse, bar_trace, run_script, session_frames
from videoinfo import check_time, open_video, same_file


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


def test_annotation_refuses_a_hard_link_to_its_source(tmp_path, tiny_video):
    out = empty_analysis(tmp_path)
    link = tmp_path / "link.mp4"
    os.link(tiny_video, link)  # different path, same file
    before = tiny_video.read_bytes()
    proc = run_script("annotate_video.py", tiny_video, out, link)
    assert proc.returncode != 0 and "source video" in proc.stderr + proc.stdout
    assert tiny_video.read_bytes() == before


def test_add_chapters_refuses_a_hard_link_to_its_input(tmp_path, tiny_video):
    out = empty_analysis(tmp_path)
    link = tmp_path / "link.mp4"
    os.link(tiny_video, link)
    proc = run_script("add_chapters.py", tiny_video, out, link)
    assert proc.returncode != 0 and "different file" in proc.stderr + proc.stdout


def test_same_file():
    assert same_file(__file__, __file__)
    assert same_file(__file__, os.path.join(os.path.dirname(__file__), ".", os.path.basename(__file__)))
    assert not same_file(__file__, "no/such/file.mp4")


def test_open_video_rejects_a_video_whose_frames_dont_decode(monkeypatch, tiny_video):
    class Undecodable:
        """Valid-looking metadata, but no frame can be read (as with a damaged file)."""
        def isOpened(self):
            return True
        def get(self, prop):
            return 25.0 if prop == cv2.CAP_PROP_FPS else 100.0
        def read(self):
            return False, None
        def release(self):
            pass
    monkeypatch.setattr(videoinfo.cv2, "VideoCapture", lambda *_: Undecodable())
    with pytest.raises(SystemExit, match="no frame could be decoded"):
        videoinfo.open_video(tiny_video)


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without an NVIDIA GPU")
def test_pipeline_overwrite_removes_the_old_annotated_video(tmp_path, tiny_video):
    work = tmp_path / "w"
    (work / "out").mkdir(parents=True)
    old = work / "annotated.mp4"
    old.write_bytes(b"old video")
    # --device cuda stops pass 1 straight away on this machine; the old video must already be gone
    proc = subprocess.run([str(ROOT / "run_pipeline.sh"), str(tiny_video), "--out-dir", str(work),
                           "--overwrite", "--no-video", "--device", "cuda"],
                          capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert not old.exists()


def test_pipeline_refuses_earlier_results_without_overwrite(tmp_path, tiny_video):
    work = tmp_path / "w"
    work.mkdir()
    (work / "annotated.mp4").write_bytes(b"old video")
    proc = subprocess.run([str(ROOT / "run_pipeline.sh"), str(tiny_video), "--out-dir", str(work)],
                          capture_output=True, text=True, check=False)
    assert proc.returncode != 0 and "earlier run" in proc.stderr
    assert (work / "annotated.mp4").read_bytes() == b"old video"
