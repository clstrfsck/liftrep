"""Open a video and check its basic properties before any script relies on them."""
import math
import os
from pathlib import Path

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
    # a damaged file can have valid metadata: make sure a frame actually decodes
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise SystemExit(f"{path}: no frame could be decoded (damaged or unsupported video?)")
    # rewind; if the backend can't seek back, reopen so callers always start at frame 0
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0) or cap.get(cv2.CAP_PROP_POS_FRAMES) != 0:
        cap.release()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f"cannot reopen video: {path}")
    return cap, fps, int(n_frames)


def same_file(a, b):
    """True if paths a and b name the same file: the same path, a symlink to it, or a hard
    link to it (an existing hard link has a different path but the same inode)."""
    if os.path.exists(a) and os.path.exists(b):
        return os.path.samefile(a, b)
    return Path(a).resolve() == Path(b).resolve()


def check_time(t, duration, name="time"):
    """Stop with a clear message if t (seconds) is outside 0..duration."""
    if not (0 <= t <= duration):
        raise SystemExit(f"{name} {t} s is outside the video (0-{duration:.1f} s)")
