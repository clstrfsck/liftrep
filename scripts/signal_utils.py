"""Signal helpers for the rep analysis: tracking gaps, continuous runs and per-run smoothing."""
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


def bridge_short_gaps(s, max_len):
    """Linearly fill interior gaps of at most max_len samples; leave longer gaps entirely
    unfilled (pandas' interpolate(limit=...) would fill the first max_len of every gap)."""
    filled = s.interpolate(limit_area="inside")
    gap = s.isna()
    run_id = (gap != gap.shift()).cumsum()
    run_len = gap.groupby(run_id).transform("sum")
    return s.where(~gap | (run_len > max_len), filled)


def n_changes(ids):
    """Number of changes between consecutive known values (missing values are skipped)."""
    v = np.asarray(ids, dtype=float)
    v = v[~np.isnan(v)]
    return int(np.count_nonzero(np.diff(v))) if len(v) > 1 else 0


def runs_of(mask):
    """(start, end) index pairs of contiguous True runs in a boolean array."""
    m = np.r_[False, mask, False].astype(int)
    starts, ends = np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1) - 1
    return list(zip(starts, ends, strict=True))


def smooth_segments(y, ok, fps):
    """Despike, smooth and differentiate each contiguous stretch of data separately, so no
    filter window or velocity spans an unbridged gap. Returns (smoothed y, velocity px/s)."""
    yy, vy = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for a, b in runs_of(ok):
        seg = y[a:b + 1]
        n = len(seg)
        if n == 1:
            yy[a], vy[a] = seg[0], 0.0
            continue
        med = median_filter(seg, size=min(7, n), mode="nearest")
        w = min(9, n if n % 2 else n - 1)
        if w >= 3:
            yy[a:b + 1] = savgol_filter(med, w, 2)
            vy[a:b + 1] = savgol_filter(seg, w, 2, deriv=1) * fps
        else:
            yy[a:b + 1] = med
            vy[a:b + 1] = np.gradient(seg) * fps
    return yy, vy
