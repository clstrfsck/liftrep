"""Open a video and check its basic properties before any script relies on them."""
import math

import cv2


def open_video(path):
    """Return (capture, fps, frame_count) for the video at path.

    Stops with a clear message if the file can't be opened or reports an unusable frame rate
    or frame count (zero, negative or not a number), instead of failing later with a
    division by zero or an empty loop.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not (math.isfinite(fps) and fps > 0):
        cap.release()
        raise SystemExit(f"{path}: unusable frame rate ({fps})")
    if not (math.isfinite(n_frames) and n_frames > 0):
        cap.release()
        raise SystemExit(f"{path}: unusable frame count ({n_frames})")
    return cap, fps, int(n_frames)


def check_time(t, duration, name="time"):
    """Stop with a clear message if t (seconds) is outside 0..duration."""
    if not (0 <= t <= duration):
        raise SystemExit(f"{name} {t} s is outside the video (0-{duration:.1f} s)")
