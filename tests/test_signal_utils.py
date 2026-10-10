"""Unit tests for the signal helpers used by the rep analysis."""
import numpy as np
import pandas as pd
from conftest import FPS
from signal_utils import bridge_short_gaps, n_changes, runs_of, smooth_segments


def test_short_gaps_are_bridged_completely():
    s = pd.Series([0.0, np.nan, np.nan, 3.0])
    assert bridge_short_gaps(s, max_len=2).tolist() == [0.0, 1.0, 2.0, 3.0]


def test_long_gaps_are_left_entirely_unfilled():
    # pandas' interpolate(limit=2) would fill the first two samples of this gap
    s = pd.Series([0.0, np.nan, np.nan, np.nan, 4.0])
    out = bridge_short_gaps(s, max_len=2)
    assert out.isna().tolist() == [False, True, True, True, False]


def test_edge_gaps_are_not_filled():
    s = pd.Series([np.nan, 1.0, 2.0, np.nan])
    assert bridge_short_gaps(s, max_len=5).isna().tolist() == [True, False, False, True]


def test_runs_of():
    mask = np.array([False, True, True, False, True, False, False, True])
    assert [(int(a), int(b)) for a, b in runs_of(mask)] == [(1, 2), (4, 4), (7, 7)]
    assert runs_of(np.zeros(3, bool)) == []


def test_n_changes_ignores_missing_values():
    assert n_changes([1, np.nan, 1]) == 0          # a gap is not a change
    assert n_changes([1, 1, 2, np.nan, 2, 3]) == 2
    assert n_changes([np.nan, np.nan]) == 0


def test_smoothing_never_spans_a_gap():
    # two flat stretches at very different heights, separated by a gap
    y = np.r_[np.full(20, 100.0), np.full(10, np.nan), np.full(20, 300.0)]
    ok = ~np.isnan(y)
    yy, vy = smooth_segments(y, ok, FPS)
    assert np.allclose(yy[:20], 100.0) and np.allclose(yy[30:], 300.0)  # no blending across
    assert np.allclose(vy[ok], 0.0)                                      # no jump read as motion
    assert np.isnan(yy[20:30]).all() and np.isnan(vy[20:30]).all()


def test_smoothing_keeps_segment_ends():
    # an edge-padding median filter would flatten the end of a steady ramp
    y = np.arange(30, dtype=float)
    yy, vy = smooth_segments(y, np.ones(30, bool), FPS)
    assert np.allclose(yy, y, atol=1e-6)
    assert np.allclose(vy, FPS)  # 1 px per frame


def test_very_short_segments():
    y = np.array([5.0, np.nan, 7.0, 8.0])
    yy, vy = smooth_segments(y, ~np.isnan(y), FPS)
    assert yy[0] == 5.0 and vy[0] == 0.0
    assert np.isfinite(yy[2:]).all()
