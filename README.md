# Lift rep feature extraction

Tools to extract lift features from the scene video of a head-mounted camera (such as an
eye tracker's), so they can be compared with where the wearer was looking.

For every lift clip shown on the projector screen the tools report:

1. Top of travel (time + pixel)
2. Bottom of travel (time + pixel)
3. Dwell time at top and bottom
4. Effort (concentric / lifting) phase timing
5. Rep count per clip
6. Duration of each rep

No manual labelling is needed: a pretrained YOLO pose model finds the athlete, and the
rep features are derived from the bar's vertical motion.

## How it works

1. **Pass 1 – per-frame extraction** (`scripts/extract_frames.py`, slow, GPU)
   - Finds the bright projected clip on the screen in each scene-video frame.
   - Runs YOLO pose (`yolo11x-pose`) on a crop around it and keeps the best person
     *inside* the screen (people in the room are ignored).
   - Matches ORB features on the screen plane (athlete masked out) to a reference frame,
     giving a homography that cancels the observer's head movement ("stabilised" coordinates).
   - Writes one CSV row per frame.

   Pass 1 builds each frame's transform by chaining from earlier frames, so a single bad
   link (motion blur, a glance away) corrupts every later frame. Pass 1b replaces it.

1b. **Pass 1b – re-stabilisation** (`scripts/restabilise.py`, CPU, ≈ 5–10 min)
   - Finds the lift clips from the pass-1 person detections.
   - Per clip, picks a sharp reference frame from the middle and matches every frame
     **directly** to it. A frame that can't be matched directly is matched to the nearest
     registered keyframe, so a transform is never more than two links from the reference.
   - Rejects transforms whose rotation, shear or scale at the athlete is implausible.
   - Writes a copy of the per-frame data with the stabilisation replaced.

2. **Pass 2 – rep analysis** (`scripts/analyse_reps.py`, fast, seconds)
   - Splits the session into lift clips (continuous runs where an athlete is on screen),
     and groups consecutive clips of the same lift into blocks.
   - Re-bases the stabilisation on a frame inside each clip and rejects frames whose
     transform is implausible (scale, position, rotation or shear at the athlete).
   - Classifies each clip as **bench press** or **squat**: an upright athlete's hips sit well
     below the shoulders (relative to body height); lying on a bench they are level.
   - Uses a bar proxy: **wrists** for bench press, **shoulders** for squat (bar on upper back).
   - Smooths the stabilised vertical position, finds bottoms/tops, dwell periods, eccentric
     (lowering) and effort (lifting) phases, removes tracking spikes, marks partial reps,
     and writes CSVs plus a QA plot per clip.

## Setup (once)

Works on macOS, Linux and Windows (tested with Python 3.12 on an Apple Silicon Mac). A GPU is strongly recommended:
an NVIDIA GPU (CUDA) or an Apple Silicon Mac. CPU-only works but is very slow (see
[Choosing a device](#choosing-a-device)). ffmpeg is optional (used by the annotated video for H.264 output with audio).

Create a virtual environment in the project folder (shown with `uv`; `python -m venv .venv`
then `pip` works too):

```bash
cd <project folder>
uv venv --python 3.12 .venv
```

**PyTorch with GPU support** – install this first:

| Machine | Command |
|---|---|
| Apple Silicon Mac | Nothing extra; the default PyTorch includes the Apple GPU (MPS). |
| Linux + NVIDIA GPU | Nothing extra; the default PyTorch on Linux includes CUDA. |
| Windows + NVIDIA GPU | `uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu124` (pick the `cuXXX` index matching your CUDA driver from pytorch.org; the default Windows install is CPU-only) |
| No GPU | Nothing extra. |

Then the rest:

```bash
uv pip install --python .venv -r requirements.txt
```

On a Linux server without a display, replace `opencv-python` with `opencv-python-headless`
in `requirements.txt` to avoid `libGL` errors.

The Python interpreter in the environment is `.venv/bin/python` on macOS and Linux, and
`.venv\Scripts\python` on Windows. Commands below use the macOS/Linux form; on Windows
substitute the path (forward slashes in the script arguments work on Windows too).

The pose model `models/yolo11x-pose.pt` is already downloaded. If it is missing,
Ultralytics downloads it automatically the first time it is used; place it in `models/`
(or pass `--model` with its path).

### Choosing a device

All scripts that run the pose model take `--device`, defaulting to `auto`:

| `--device` | Uses |
|---|---|
| `auto` (default) | NVIDIA GPU if available, else Apple Silicon GPU, else CPU |
| `cuda` / `cuda:N` | NVIDIA GPU (first, or number N). Runs in 16-bit precision for speed. |
| `mps` | Apple Silicon GPU |
| `cpu` | CPU only |

The chosen device is printed at the start of each run and in each chunk's log. Asking for
a device that isn't available stops with an error instead of silently falling back.

Pass 1 is the slow step: roughly 1.5–2× the video's duration on a recent laptop or desktop
GPU, and many hours on CPU. On CPU, use a smaller model such as
`--model models/yolo11m-pose.pt` or `yolo11s-pose.pt` (downloaded automatically), at some
cost in keypoint accuracy.

## Running

All commands are run from the project directory.

### Pass 1 – extract per-frame data (slow)

Whole video, split into parallel chunks (outputs `work/frames_<start>.csv`, logs
`work/log_<start>.txt`):

```bash
.venv/bin/python scripts/run_extract_all.py "<video>.mp4"
```

The video's length is read automatically and split into equal chunks. Options:

| Option | Default | Meaning |
|---|---|---|
| `--out-dir` | `work` | Where the per-frame CSVs and logs go |
| `--jobs` | 4 on a GPU; about one per 4 CPU cores on CPU | Number of parallel processes |
| `--device` | `auto` | See [Choosing a device](#choosing-a-device) |
| `--model` | `models/yolo11x-pose.pt` | Pose model |

Several processes help even on one GPU, because video decoding and feature matching run on
the CPU while the GPU runs the pose model; beyond about 4 the gain is small. To watch
progress, look at the last line of each log (macOS/Linux: `tail -n1 work/log_*.txt`;
Windows PowerShell: `Get-Content work\log_*.txt -Tail 1`).

To process a single time window (useful for testing a change):

```bash
.venv/bin/python scripts/extract_frames.py "<video>.mp4" work/test.csv --start 60 --end 120
```

Options: `--start` / `--end` (seconds), `--device` (default `auto`), `--model` (default
`models/yolo11x-pose.pt`), `--imgsz` (pose inference size, default 960).

### Pass 1b – re-stabilise (CPU, ≈ 5–10 min)

```bash
.venv/bin/python scripts/restabilise.py "<video>.mp4" work work/restab
```

Reads the pass-1 CSVs in `work/`, writes `work/restab/frames_restab.csv`. Option: `--jobs N`
(parallel processes, default: CPU cores − 2, at most 8). Progress is printed per clip.

### Pass 2 – rep analysis (fast, seconds)

```bash
.venv/bin/python scripts/analyse_reps.py work/out work/restab
```

The first argument is the output directory. The rest are the per-frame CSVs from pass 1b
(or directly from pass 1, which works but with the less reliable chained stabilisation),
given as files, a folder (all `frames_*.csv` in it) or a wildcard pattern such as
`"work/frames_*.csv"` (expanded by the script, so it also works in Windows shells). Chunks
are merged automatically. Pass 2 runs on the CPU and can be re-run freely after changing
thresholds without repeating pass 1.

### Pass 3 – annotated video (optional, ≈ 5 min for the whole video)

Draws the analysis onto the scene video:

```bash
# one clip (with 1 s either side), for checking
.venv/bin/python scripts/annotate_video.py "<video>.mp4" work/out work/annotated_clip1.mp4 --clip 1

# whole video
.venv/bin/python scripts/annotate_video.py "<video>.mp4" work/out work/annotated.mp4
```

Options: `--clip N` (one clip), or `--start` / `--end` (seconds) for any time range;
two smoothing settings (0 turns either off):

| Option | Default | Smooths |
|---|---|---|
| `--geom-smooth S` | 0.4 | The stabilising transform used to draw the lines, arrows and bar marker, over `S` seconds. This removes the high-frequency jitter of the per-frame transform while still following the observer's head movement. |
| `--label-smooth S` | 0.8 | The text labels' positions, over `S` seconds, so the numbers stay easy to read. When a label lags noticeably behind its line, a thin leader line joins them. |

Both use a centred Gaussian (past and future frames equally), so the overlay doesn't trail
behind the picture. Only the *mapping onto the frame* is smoothed: the analysis results
themselves (line levels, timings, rep changes) are drawn exactly as computed, and a new
rep's lines appear at once. With the 0.4 s default, frame-to-frame shake of the lines falls
about tenfold, and the lines stay within a median 2 px
(95th percentile 12 px, during fast head movement) of the unsmoothed position. Longer
windows are steadier but drift further from the picture when the head moves quickly.
Frames outside lift clips are copied unchanged, so a whole-video render keeps the original
timing (output frame N = input frame N) and can be lined up with other data recorded on the
same timeline (e.g. gaze).

What is drawn in each lift clip:

| Element | Meaning |
|---|---|
| Green line, `TOP` + `dwell` | Top of travel for the rep in progress, with the top dwell time |
| Red line, `BOTTOM` + `dwell` | Bottom of travel for the rep in progress, with the bottom dwell time |
| Orange arrow, `EFFORT` | Effort (lifting) phase from bottom to top, with its duration; thick while lifting |
| Thin blue arrow, `lowering` | Lowering phase, with its duration; thick while lowering |
| Dot | Bar proxy (wrists or shoulders); orange during the effort phase |
| Panel (top left) | Clip number and lift, `Rep k of N` for the clip, last rep's duration, current phase |

Times count up live while a phase is in progress; otherwise they show the last completed
value. During set-up (before a clip's first rep) no timer runs. The lines are drawn in the clip's stabilised coordinates and mapped back into each
frame (through the smoothed transform), so they stay on the screen content as the
observer's head moves.

The script uses ffmpeg (H.264, original audio kept) if it is installed and on the PATH;
otherwise it falls back to OpenCV's `mp4v` encoder without audio.

**Chapters.** With ffmpeg, the output also gets chapter markers that cover the whole
timeline, for jumping between clips and reps in QuickTime, VLC, IINA or mpv. MP4 chapters
can't be nested, so the list is flat:

| Chapter title | Spans |
|---|---|
| `Between clips` | Outside lift clips |
| `Clip N - bench press (block B) - set-up` | Clip start → first rep |
| `Clip N - bench press (block B) - rep k of M` | One per counted rep, from the end of the top dwell to the start of the next rep |

Reps with `tracking_quality = check` have ` [check]` appended. Use `--no-chapters` to
skip them. Chapters can also be added to an existing video without re-encoding (a few
seconds):

```bash
.venv/bin/python scripts/add_chapters.py work/annotated.mp4 work/out work/annotated_chapters.mp4
```

Use `--offset S` there if the video was rendered from a time range starting at `S` seconds.

## Outputs (`work/out/`)

### `clips.csv` – one row per lift clip

| Column | Meaning |
|---|---|
| `clip` | Clip number, in playing order |
| `block` | Block number: consecutive clips of the same lift (alternates bench press / squat) |
| `start_t`, `end_t`, `start_frame`, `end_frame`, `duration_s` | When the athlete was on screen |
| `lift` | `bench_press` or `squat` |
| `tracked` | Keypoints used as the bar proxy |
| `n_reps` | Reps counted (`rep_status = ok`) |
| `n_partial` | Partial reps (not counted) |
| `n_spikes_removed` | Implausibly fast dips discarded as tracking spikes |
| `n_reps_tracking_good` | Counted reps whose tracking quality is `good` |
| `mean_rep_s` | Mean duration of counted reps |
| `robust_range_stab_px` | Typical vertical movement range in the clip (stabilised px) |
| `valid_kp_frac` | Fraction of frames with a confident bar-proxy keypoint |
| `stab_valid_frac` | Fraction of frames with a valid stabilising transform |
| `glitch_frames_masked` | Frames masked as stabilisation glitches before rep detection |
| `hip_below_shoulder`, `box_aspect` | Lift classification features (squat if `hip_below_shoulder` > 0.2) |
| `skipped` | Reason, if the clip could not be analysed (otherwise empty) |

### `reps.csv` – one row per rep

| Column | Meaning |
|---|---|
| `clip`, `block`, `rep`, `lift` | Identifiers. `rep` is numbered over counted reps only |
| `rep_status` | `ok` (counted) or `partial` (< 50% of the clip's median range; not counted, no events) |
| `rep_start_t`, `rep_end_t`, `rep_duration_s` | Rep = leave top → arrive back at top |
| `top_*`, `bottom_*` | Top / bottom of travel: `_t`, `_frame`, `_rec_us`, `_scene_x/y`, `_stab_x/y` |
| `top_dwell_before_s`, `bottom_dwell_s`, `top_dwell_after_s` | Dwell durations |
| `eccentric_start_t`, `eccentric_s` | Lowering phase |
| `effort_start_t`, `effort_end_t`, `effort_s` | Effort (lifting) phase: leave bottom → arrive top |
| `effort_mean_vel_px_s`, `effort_peak_vel_px_s` | Bar speed during the effort phase (stabilised px/s) |
| `range_stab_px` | Rep range of motion (stabilised px) |
| `first_top_dwell_truncated` | `True` if the top dwell runs into the clip start |
| `track_ok_frac`, `reanchors`, `kp_valid_frac`, `noise_frac` | Tracking diagnostics (`noise_frac` = scatter of raw points around the smoothed trace ÷ rep range) |
| `tracking_quality` | `good` if stabilisation is valid for ≥ 90% of the rep, the bar proxy is seen for ≥ 50% of it, and `noise_frac` ≤ 0.05; otherwise `check` |
| `timing_outlier` | Phase(s) far from the clip's median. **Informational only** – a genuinely fatigued, slow rep also appears here |

### `events.csv` – one row per top / bottom of travel

`clip`, `rep`, `event` (`top`/`bottom`), time and pixel columns as above, and the dwell
period around it (`dwell_start_t`, `dwell_end_t`, `dwell_s`). Counted reps only.

### `frames_annotation.csv.gz` – one row per frame inside a clip

Used by `annotate_video.py`, and also handy for joining with other per-frame data (e.g. gaze):
`frame`, `t`, `clip`, `rep` (rep in progress, 0 before the first), `n_reps`, `phase`
(`setup`, `top_dwell`, `lowering`, `bottom_dwell`, `effort`, or empty), bar proxy position
(`bar_stab_x/y` smoothed, `bar_scene_x/y`), `person_w`, and the re-based homography
`h00`…`h22` (scene frame → clip stabilised coordinates).

### `plots/clip_NN.png` – QA plot per clip

Top panel: stabilised bar position (up = top of travel) with tops (▲), bottoms (▼),
top dwell (green), bottom dwell (red); partial reps are hollow grey ▽. Bottom panel: vertical velocity with the effort
phase shaded orange. Check these before trusting a clip's numbers.

## Coordinates and time

- **`*_scene_x/y`** – pixel in the scene video at that frame. This is the same space as an
  eye tracker's gaze point on its scene video, so use it to compare with where the wearer looked.
- **`*_stab_x/y`** – pixel in the clip's stabilised frame (head motion removed). Use it for
  distances and speeds. It is comparable *within* a clip, not between clips.
- **`*_t`** – seconds from the start of the video. **`*_frame`** – frame index.
- **`*_rec_us`** – video time in microseconds, for joining with an eye tracker's recording
  timestamps when the recording and the scene video start together.

## Definitions

Each rep is found in two steps. **Velocity** locates the movement: lowering begins where the
bar starts moving down, and the effort phase ends where the upward movement stops. **Position**
then measures the phases and dwells around the top and bottom levels found that way.

- **Moving / stationary** – the bar is moving when its speed exceeds 12% of that phase's peak
  speed and 15 stabilised px/s (`V_STILL_FRAC`, `V_STILL_MIN`); a movement ends after 2
  stationary frames (`STILL_FRAMES`), but only within 30% of the travel from the top
  (`NEAR_TOP_FRAC`). A pause partway up or down (a sticking point) therefore stays inside
  the effort or lowering phase, and a stationary bar at another height (in the rack, during
  set-up) never becomes part of a rep.
- **Top of travel** – the bar's level just before the lowering starts (and just after the
  effort phase ends, for the top after the rep); the reported point is the highest point of
  the top dwell.
- **Bottom of travel** – the lowest point between the start of lowering and the end of the
  effort phase.
- **Dwell** – time the bar stays within 10% of the rep's range of the top or bottom level
  (`DWELL_BAND`).
- **Lowering (eccentric)** – end of top dwell → start of bottom dwell.
- **Effort (concentric)** – end of bottom dwell → start of the next top dwell. Both bench
  press and squat start at the top, so effort is the upward movement for both.
- **Rep** – end of top dwell → start of the next top dwell. A rep must travel at least 40%
  of the clip's typical range (`MIN_REP_RANGE_FRAC`).
- **Set-up** – everything in a clip before the first rep's top dwell (unracking, settling,
  the clip starting). No phase timers run during set-up.
- **Glitch frames** – frames masked before reps are detected, because the stabilisation
  is wrong for a moment (usually fast head movement and motion blur). A frame is masked when
  a body part that should be stationary (hips/knees in a bench press, ankles in a squat)
  moves more than 12% of body height from its local median (`REF_DEV_FRAC`), when that
  reference is missing while the observer's head moves fast (`HEAD_FAST_PX`), or when the
  bar jumps more than 35% of body height in one frame (`BAR_JUMP_FRAC`).
- **Tracking spike** – a "rep" whose lowering takes < 0.2 s (`MIN_ECCENTRIC_S`) or lifting
  < 0.12 s (`MIN_EFFORT_S`), or which moves more than 3× faster than the clip's typical rep
  (`MAX_SPEED_RATIO`). Physically implausible, so it is bridged over in the trace and the
  neighbouring reps are rebuilt without it.
- **Partial rep** – a rep with < 50% of the clip's median range (`PARTIAL_RANGE_FRAC`),
  typically at the start or end of a clip. Listed, but not counted.

Thresholds are constants at the top of `scripts/analyse_reps.py`. Setting the environment
variable `ANALYSE_DEBUG_CLIP=N` prints, for each rep candidate in clip `N`, its movement
onset and stop times, top and bottom levels, and range against the minimum range.

## Limitations

- **Timing resolution is one frame** (0.04 s at 25 fps). Effort phases are often only ~0.3 s
  (≈ 7 frames at 25 fps), so effort durations carry roughly ±15% uncertainty.
- **Paused playback** – the operator sometimes pauses a clip. Head movement during a
  pause can create false reps, and a pause at the top inflates the top dwell. These often
  show as `timing_outlier`; review them on the QA plot.
- **Clips are numbered in playing order**, not by condition. Clips replayed or scrubbed by
  the operator can appear as extra or split clips.
- **Pose errors** – a wrist occasionally snaps onto the bench or taskbar; the analysis keeps
  the wrist nearest the previous position, but fast head motion (motion blur) still causes
  noise. Reps with unreliable tracking are marked `tracking_quality = check`.
- **Side-view bench press with heavy plates** – the plate hides the hands, so the wrist
  keypoints are often missing or misplaced. Reps in these clips are
  frequently marked `check`; tracking the plate itself would be more reliable here.
- **Blocks are numbered, not named.** Mapping blocks and clips to the athlete and load conditions has to come from the study protocol.

## Other scripts (diagnostics)

| Script | Purpose |
|---|---|
| `scripts/contact_sheet.py VIDEO OUT_DIR [STEP_S]` | Labelled thumbnail sheets every `STEP_S` seconds (session overview) |
| `scripts/screen_probe.py VIDEO OUT_DIR T1 T2 …` | Show the detected screen region at given times |
| `scripts/pose_probe.py VIDEO T0 T1 MODEL OUT_PREFIX [DEVICE]` | Pose speed / detection check on a time window (`DEVICE` defaults to `auto`) |
| `scripts/plot_test.py FRAMES.csv OUT.png` | Raw vs stabilised wrist trace and tracking diagnostics |
| `scripts/restabilise.py VIDEO PASS1_DIR OUT_DIR [--jobs N]` | Pass 1b: direct-to-reference re-stabilisation (see above) |
| `scripts/annotate_video.py` | Pass 3: annotated video (see above) |
| `scripts/add_chapters.py VIDEO ANALYSIS_DIR OUT.mp4 [--offset S]` | Add clip / lift / rep chapters to a video without re-encoding |
| `scripts/device.py` | Shared helper: picks the compute device (not run directly) |
