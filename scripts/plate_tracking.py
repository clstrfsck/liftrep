"""Plate tracking: follow the bar's end plate, a large circle, through a clip.

In a bench press seen from the side the plate hides the hands, so the wrist keypoints are
often missing or misplaced; the plate itself is a large, high-contrast circle centred on the
bar. Used by 01restabilise.py (pass 1b), which decodes every clip frame anyway:

  * plate_candidates() finds circles in a frame's screen region (Hough transform)
  * track_plate() picks one circle per frame, in the clip's stabilised coordinates: the
    radius and horizontal position of the circles found at the wrists (the hands hold the
    bar at the plate's centre) say which circles are the plate, and a shortest-path search
    over the frames chooses a smooth path through them, skipping frames without one
  * the track is kept only if it follows the wrists where they are seen (plate_check())

A front view shows the plate edge-on, and a squat's bar is followed by the shoulders, so
02analyse_reps.py uses the plate only for bench press clips with a validated track.
"""
import cv2
import numpy as np

KP_MIN_CONF = 0.5         # wrist keypoints below this confidence are ignored (as in pass 2)
R_MIN_FRAC, R_MAX_FRAC = 0.02, 0.2  # plate radius search range, x the screen region's height
ANCHOR_DIST = 1.2         # a circle within this many radii of a wrist is "at the hands"
MIN_ANCHORS = 15          # frames with a circle at the hands needed to identify the plate
R_TOL = 0.25              # the plate's radius stays within 25% of its typical radius
X_TOL = 2.5               # ... and its centre within 2.5 radii of its typical x (stab px;
                          # the bar moves sideways between the rack and the chest)
WRIST_FAR = 2.0           # a circle this many radii from the wrists gets no reward for them
STEP_MAX = 0.8            # largest plausible move between frames, x radius
MAX_SKIP = 12             # frames without the plate the path may skip (0.5 s at 25 fps)
BREAK_COST = 3.0          # cost of starting the path again elsewhere
MIN_OFFSET, MAX_OFFSET = -0.25, 1.5  # validation: the wrists sit at, or up to 1.5 radii
                          # below, the plate's centre (in a front view the circles found are
                          # below the hands: the head, the bench)
MAX_SPREAD = 0.3          # ... and move with it: the spread (MAD) of the wrist-plate height
                          # difference is under 0.3 radii


def plate_candidates(gray, scr):
    """Circles (x, y, r) in scene px inside the screen region scr = (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = (int(v) for v in scr)
    x0, y0 = max(0, x0), max(0, y0)
    h = y1 - y0
    if h < 20 or x1 - x0 < 20:
        return np.empty((0, 3))
    g = cv2.GaussianBlur(gray[y0:y1, x0:x1], (5, 5), 0)
    cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=10, param1=60, param2=0.6,
                          minRadius=max(5, int(R_MIN_FRAC * h)), maxRadius=int(R_MAX_FRAC * h))
    if cs is None:
        return np.empty((0, 3))
    return cs[0].astype(float) + [x0, y0, 0]


def to_stab(H, pts):
    """Apply homography H to (n, 2) points."""
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, :2] / q[:, 2:3]


def local_scale(H, pts):
    """Scale factor of homography H at each of (n, 2) points (stabilised px per scene px)."""
    a, b = to_stab(H, pts), to_stab(H, pts + [1.0, 0.0])
    c = to_stab(H, pts + [0.0, 1.0])
    return np.sqrt(np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                          - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])))


def track_plate(cands, Hs, wrists):
    """One plate position per frame of a clip.

    cands:  per frame, an (n, 3) array of circles (x, y, r) in scene px
    Hs:     per frame, the 3x3 scene -> stabilised homography, or None if unregistered
    wrists: (n_frames, 2, 2) wrist positions (left, right; x, y) in scene px, NaN if unseen
    Returns (n_frames, 3) of plate x, y, r in scene px (NaN where not found), or None if
    no plate could be identified at the hands.
    """
    n = len(cands)
    # stabilised candidates (centre and radius: the observer moves closer and further away, so
    # the radius in scene px changes); frames without a transform have none
    st = []
    for i in range(n):
        c = cands[i]
        if Hs[i] is None or len(c) == 0:
            st.append(np.empty((0, 3)))
            continue
        st.append(np.c_[to_stab(Hs[i], c[:, :2]), c[:, 2] * local_scale(Hs[i], c[:, :2])])
    # --- identify the plate: circles at the hands give its radius and horizontal position
    anchors = []
    for i in range(n):
        c = cands[i]
        if len(st[i]) == 0:
            continue
        for w in wrists[i]:
            if np.all(np.isfinite(w)):
                dist = np.hypot(c[:, 0] - w[0], c[:, 1] - w[1])
                near = dist < ANCHOR_DIST * c[:, 2]
                if near.any():
                    j = int(np.argmin(np.where(near, dist / c[:, 2], np.inf)))
                    anchors.append((st[i][j, 2], st[i][j, 0]))
    if len(anchors) < MIN_ANCHORS:
        return None
    a = np.array(anchors)
    R = float(np.median(a[:, 0]))
    X = float(np.median(a[np.abs(a[:, 0] / R - 1) < R_TOL, 1]))

    # --- shortest path through the plate-like circles. Each circle chosen earns a reward (up
    # to 1, less the further its radius and position are from the plate's, and from the
    # wrists where they are seen); consecutive choices up to MAX_SKIP frames apart are joined
    # if the move between them is plausible, and starting again anywhere costs BREAK_COST,
    # so the path can't hop onto another circle (a stored plate) by skipping a frame.
    wst = []
    for i in range(n):
        w = wrists[i][np.isfinite(wrists[i][:, 0])]
        wst.append(to_stab(Hs[i], w) if Hs[i] is not None and len(w) else np.empty((0, 2)))
    nodes = []
    for i in range(n):
        s_ = st[i]
        k = (np.abs(s_[:, 2] / R - 1) < R_TOL) & (np.abs(s_[:, 0] - X) < X_TOL * R)
        s_ = s_[k]
        cost = -1 + 0.4 * np.abs(s_[:, 2] / R - 1) / R_TOL + 0.3 * np.abs(s_[:, 0] - X) / (X_TOL * R)
        if len(wst[i]) and len(s_):
            dw = np.min(np.hypot(s_[:, None, 0] - wst[i][None, :, 0],
                                 s_[:, None, 1] - wst[i][None, :, 1]), axis=1) / R
            cost += 0.3 * np.minimum(dw / WRIST_FAR, 1)
        nodes.append((np.flatnonzero(k), s_[:, :2], cost))
    best = [None] * n          # per frame: best path cost ending at each node
    back = [None] * n          # ... and its predecessor (frame, node), or None (path start)
    done_best, done_arg = 0.0, None  # best cost of any path ending before the current frame
    for i in range(n):
        _, pts, cost = nodes[i]
        if i > 0 and best[i - 1] is not None and len(best[i - 1]):
            m = int(np.argmin(best[i - 1]))
            if best[i - 1][m] < done_best:
                done_best, done_arg = float(best[i - 1][m]), (i - 1, m)
        if not len(pts):
            best[i], back[i] = np.empty(0), []
            continue
        b = np.full(len(pts), done_best + BREAK_COST)
        bk = [done_arg] * len(pts)
        for j in range(max(0, i - MAX_SKIP), i):
            if best[j] is None or not len(best[j]):
                continue
            pj = nodes[j][1]
            step = np.hypot(pj[:, None, 0] - pts[None, :, 0],
                            pj[:, None, 1] - pts[None, :, 1]) / (STEP_MAX * R * (i - j))
            tot = best[j][:, None] + np.where(step <= 1, step ** 2, np.inf)
            arg = np.argmin(tot, axis=0)
            val = tot[arg, np.arange(len(pts))]
            for q in np.flatnonzero(val < b):
                b[q], bk[q] = val[q], (j, int(arg[q]))
        best[i], back[i] = b + cost, bk
    # trace back from the best path end
    end, end_cost = None, 0.0
    for i in range(n):
        if best[i] is not None and len(best[i]) and best[i].min() < end_cost:
            end, end_cost = (i, int(np.argmin(best[i]))), float(best[i].min())
    out = np.full((n, 3), np.nan)
    while end is not None:
        i, q = end
        out[i] = cands[i][nodes[i][0][q]]
        end = back[i][q]
    return out


def plate_check(plate_y, wrist_y, radius):
    """Does the plate track move with the wrists? (stabilised y of each, NaN where unseen).

    The hands hold the bar at the plate's centre, so where both are seen the wrists sit at a
    roughly constant height relative to the plate (just below its centre, depending on the
    grip and the plate size). A circle that doesn't move with them (a stored plate, a head,
    the player's buttons) leaves the wrists' movement in the difference.
    Returns (ok, median offset / radius, spread of the offset / radius).
    """
    both = np.isfinite(plate_y) & np.isfinite(wrist_y)
    if both.sum() < MIN_ANCHORS:
        return False, np.nan, np.nan
    diff = wrist_y[both] - plate_y[both]
    offset = float(np.median(diff))
    spread = float(np.median(np.abs(diff - offset)))
    return (MIN_OFFSET * radius <= offset <= MAX_OFFSET * radius and spread <= MAX_SPREAD * radius,
            offset / radius, spread / radius)


def clip_plate(cands, Hs, rows):
    """Plate track for one clip, kept only if it follows the wrists.

    cands, Hs: per frame, as for track_plate(); rows: the clip's per-frame data, with the
    pass-1 wrist keypoints (l_wri_x/y/c, r_wri_x/y/c).
    Returns ((n_frames, 3) plate x, y, r in scene px or None, a dict describing the result).
    """
    wr = np.stack([np.c_[np.where(rows[f"{s}_wri_c"] >= KP_MIN_CONF, rows[f"{s}_wri_x"], np.nan),
                         np.where(rows[f"{s}_wri_c"] >= KP_MIN_CONF, rows[f"{s}_wri_y"], np.nan)]
                   for s in ("l", "r")], axis=1)
    track = track_plate(cands, Hs, wr)
    if track is None:
        return None, {"plate": "not found at the hands"}
    ok_h = np.array([h is not None for h in Hs])
    py, wy = np.full(len(Hs), np.nan), np.full(len(Hs), np.nan)
    for i in np.flatnonzero(ok_h):
        if np.isfinite(track[i, 0]):
            py[i] = to_stab(Hs[i], track[i:i + 1, :2])[0, 1]
        w = wr[i][np.isfinite(wr[i][:, 0])]
        if len(w) and np.isfinite(track[i, 0]):
            # the wrist nearer the plate (the other may be misplaced, e.g. on the chest)
            near = np.argmin(np.hypot(w[:, 0] - track[i, 0], w[:, 1] - track[i, 1]))
            wy[i] = to_stab(Hs[i], w[near:near + 1])[0, 1]
    found = np.isfinite(py)
    radius = float(np.median([track[i, 2] * local_scale(Hs[i], track[i:i + 1, :2])[0]
                              for i in np.flatnonzero(found)])) if found.any() else np.nan
    ok, offset, spread = plate_check(py, wy, radius)
    info = {"plate": "ok" if ok else "rejected", "frac": float(found[ok_h].mean()) if ok_h.any() else 0.0,
            "radius": radius, "wrist_offset": offset, "wrist_spread": spread}
    return (track if ok else None), info
