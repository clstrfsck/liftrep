# Lift rep feature extraction

Tools that measure weightlifting reps in the scene video of a head-mounted camera (such as an
eye tracker's) watching lift clips on a projector screen, so the reps can be compared with
where the wearer was looking. For every lift clip they report:

1. Top of travel (time and pixel position)
2. Bottom of travel (time and pixel position)
3. Dwell time at the top and the bottom
4. Timing of the effort (lifting) phase
5. Rep count per clip
6. Duration of each rep

No manual labelling is needed: a pretrained YOLO pose model finds the athlete, and the rep
features come from the bar's vertical movement.

## Quick start

After [setup](#setup), run the whole pipeline on a video:

```bash
./run_pipeline.sh "<video>.mp4"
```

Results go to `work/<video name>/`: the analysis in `out/` (see [Outputs](#outputs)) and an
annotated copy of the video, `annotated.mp4`. Expect roughly 1.5–2× the video's length on a
recent GPU.

## Setup

Requirements: macOS, Linux or Windows; Python 3.10 or newer (tested with 3.12 on an Apple
Silicon Mac); a GPU is strongly recommended (NVIDIA or Apple Silicon); ffmpeg is optional (for
H.264 output with audio and chapters in the annotated video).

1. Create a virtual environment in the project folder (shown with `uv`;
   `python -m venv .venv` and `pip` work too):

   ```bash
   cd <project folder>
   uv venv --python 3.12 .venv
   ```

2. **Windows with an NVIDIA GPU only:** install the CUDA build of PyTorch first, as the
   default Windows install is CPU-only. Pick the `cuXXX` index matching your driver from
   pytorch.org:

   ```bash
   uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

   On a Mac (Apple GPU) or Linux (CUDA) the default PyTorch already supports the GPU.

3. Install the requirements:

   ```bash
   uv pip install --python .venv -r requirements.txt
   ```

   On a Linux server without a display, use `opencv-python-headless` instead of
   `opencv-python` in `requirements.txt` (avoids `libGL` errors).

Notes:

- The environment's Python is `.venv/bin/python` (macOS, Linux) or `.venv\Scripts\python`
  (Windows). Commands below use the first form; forward slashes in arguments work on Windows.
- The pose model (`models/yolo11x-pose.pt` by default) isn't in the repository. Ultralytics
  downloads it the first time it's used; to keep it in `models/`, download it there or pass
  `--model` with its path.

### Choosing a device

Scripts that run the pose model take `--device` (default `auto`):

| `--device` | Uses |
|---|---|
| `auto` | NVIDIA GPU if available, else Apple Silicon GPU, else CPU |
| `cuda`, `cuda:N` | NVIDIA GPU (the first, or number N), in 16-bit precision for speed |
| `mps` | Apple Silicon GPU |
| `cpu` | CPU only: many hours for a long video; consider a smaller model such as `--model models/yolo11m-pose.pt` or `yolo11s-pose.pt` (downloaded automatically, slightly less accurate) |

The device in use is printed at the start of each run and in each chunk's log. A device that
isn't available stops the run with an error rather than falling back.

## Running

### Whole pipeline: `run_pipeline.sh`

```bash
./run_pipeline.sh [options] "<video>.mp4"
```

Runs the passes below in order; `./run_pipeline.sh --help` lists its options. It can be run
from any folder and needs bash (macOS, Linux, or Git Bash/WSL on Windows); the passes themselves
run on any OS.

| Option | Meaning |
|---|---|
| `-o`, `--out-dir DIR` | Working folder (default `work/<video name>/` in the project) |
| `-j`, `--jobs N` | Parallel pose-extraction processes (default: automatic) |
| `-d`, `--device DEV` | See [Choosing a device](#choosing-a-device) |
| `-m`, `--model PATH` | Pose model |
| `--overwrite` | Replace an earlier run's results in the working folder, including an earlier annotated video even with `--no-video`. Without it the script stops if results exist. The input video is never deleted. |
| `--no-video` | Skip the annotated video (pass 3) |

### Pass by pass

The pipeline scripts are numbered in the order they run. Run them from the project folder.

| Script | Pass | Time (51-min video) |
|---|---|---|
| `scripts/00run_extract_all.py` | 1 and 1b: pose extraction, then re-stabilisation | ≈ 1.5 h + 6 min |
| `scripts/01restabilise.py` | 1b: re-stabilisation (run by `00`; can be run alone) | ≈ 6 min (CPU) |
| `scripts/02analyse_reps.py` | 2: rep analysis | seconds (CPU) |
| `scripts/03annotate_video.py` | 3: annotated video with chapters (optional) | ≈ 5 min |

**Pass 1 + 1b – pose extraction and re-stabilisation**

```bash
.venv/bin/python scripts/00run_extract_all.py "<video>.mp4"
```

Splits the video into equal chunks (at least 10 s each), runs `extract_frames.py` on each in
parallel, then runs pass 1b. Writes `work/frames_<start>.csv`, `work/log_<start>.txt`,
`work/chunks.txt` (this run's chunk list, read by pass 1b) and `work/restab/`.

| Option | Default | Meaning |
|---|---|---|
| `--out-dir` | `work` | Output folder |
| `--jobs` | 4 on a GPU; about 1 per 4 cores on CPU | Parallel processes. Several help even on one GPU (decoding and feature matching run on the CPU meanwhile); beyond about 4 the gain is small |
| `--device` | `auto` | See [Choosing a device](#choosing-a-device) |
| `--model` | `models/yolo11x-pose.pt` | Pose model |
| `--overwrite` | off | Delete chunk files, logs and `restab/` left by an earlier run (otherwise the run stops, so old results can't be mixed in). Never deletes the input video |
| `--no-restabilise` | off | Skip pass 1b; run `01restabilise.py` yourself before pass 2 |

Progress: the last line of each log (`tail -n1 work/log_*.txt`; in PowerShell
`Get-Content work\log_*.txt -Tail 1`).

To process one time window only (useful for testing), run the worker directly:

```bash
.venv/bin/python scripts/extract_frames.py "<video>.mp4" work/test.csv --start 60 --end 120
```

Options: `--start` / `--end` (seconds), `--device`, `--model`, `--imgsz` (pose inference size,
default 960).

**Pass 1b – re-stabilisation on its own**

```bash
.venv/bin/python scripts/01restabilise.py "<video>.mp4" work work/restab
```

Needed only after `--no-restabilise`, or after re-running pass 1 for part of the video. Reads
the pass-1 CSVs in `work/` and writes `work/restab/frames_restab.csv`. Option: `--jobs N`
(default: CPU cores − 2, at most 8).

**Pass 2 – rep analysis**

```bash
.venv/bin/python scripts/02analyse_reps.py work/out work/restab
```

The first argument is the output folder; the rest are pass-1b CSVs, given as files, a folder
(all `frames_*.csv` in it) or a wildcard pattern (expanded by the script, so it also works in
Windows shells). Raw pass-1 CSVs are refused. It can be re-run freely, e.g. after changing a
threshold; QA plots from an earlier run are removed first.

**Pass 3 – annotated video**

```bash
# one clip (with 1 s either side), for checking
.venv/bin/python scripts/03annotate_video.py "<video>.mp4" work/out work/annotated_clip1.mp4 --clip 1
# the whole video
.venv/bin/python scripts/03annotate_video.py "<video>.mp4" work/out work/annotated.mp4
```

| Option | Default | Meaning |
|---|---|---|
| `--clip N` | | Render one clip only |
| `--start`, `--end` | whole video | Render a time range (seconds) |
| `--geom-smooth S` | 0.4 | Smooths the stabilising transform used to draw lines, arrows and the bar marker over `S` seconds, removing jitter while following the observer's head (0 = off) |
| `--label-smooth S` | 0.8 | Smooths the text labels' positions over `S` seconds so the numbers are easy to read (0 = off) |
| `--no-chapters` | | Don't add chapter markers |

The output must be a different file from the source video (the script refuses otherwise).
Frames outside lift clips are copied unchanged, so a whole-video render keeps the original
timing (output frame N = input frame N) and lines up with other data on the same timeline,
such as gaze.

What is drawn in each lift clip:

| Element | Meaning |
|---|---|
| Green line, `TOP` + `dwell` | Top of travel for the rep in progress, with the top dwell time |
| Red line, `BOTTOM` + `dwell` | Bottom of travel, with the bottom dwell time |
| Orange arrow, `raising` | Effort (lifting) phase from bottom to top, with its duration |
| Blue arrow, `lowering` | Lowering phase, with its duration |
| Dot | Bar proxy (wrists or shoulders); orange while lifting |
| Panel, top left | Clip number and lift, `Rep k of N`, last rep's duration, current phase |

Times count up while a phase is in progress and otherwise show the last completed value; the
phase in progress has a thicker arrow and a slightly larger label. No timer runs during set-up
(before a clip's first rep). Lines are drawn in the clip's stabilised coordinates, so they
stay on the screen content as the observer's head moves. Both smoothings are centred in time
(no lag) and affect only where things are drawn, never the results; at the default
`--geom-smooth`, frame-to-frame shake falls about tenfold while the lines stay within a median
2 px (95th percentile 12 px) of the unsmoothed position. A label that lags behind its line is
joined to it by a thin leader line.

With ffmpeg the output is H.264 with the original audio, plus chapter markers for jumping
between clips and reps (in VLC, IINA, mpv or QuickTime). Without ffmpeg it falls back to
OpenCV's `mp4v` encoder, without audio or chapters. MP4 chapters can't be nested, so the list
is flat:

| Chapter title | Spans |
|---|---|
| `Between clips` | Outside lift clips |
| `Clip N - bench press (block B) - set-up` | Clip start → first rep |
| `Clip N - bench press (block B) - rep k of M` | Each counted rep; ` [check]` is appended when its tracking quality is `check` |

To add chapters to an existing video without re-encoding (a few seconds), use
`scripts/add_chapters.py VIDEO ANALYSIS_DIR OUT.mp4`, with `--offset S` if the video was
rendered from a time range starting at `S` seconds.

## Outputs

The analysis is written to `out/` in the working folder (`work/out/` pass by pass,
`work/<video name>/out/` with `run_pipeline.sh`). Every file is always written with its full
set of columns, even for a video with no lift clips.

**Coordinates and time**, used throughout:

- `*_scene_x/y` – pixel in the scene video at that frame: the same space as an eye tracker's
  gaze point on its scene video, so use it to compare with where the wearer looked.
- `*_stab_x/y` – pixel in the clip's stabilised frame (head movement removed): use it for
  distances and speeds, comparing *within* a clip only.
- `*_t` – seconds from the start of the video; `*_frame` – frame index. Any frame rate works
  (it is read from the video and the per-frame data).
- `*_rec_us` – video time in microseconds, for joining with an eye tracker's recording
  timestamps when the recording and the scene video start together.

### `clips.csv` – one row per lift clip

| Column | Meaning |
|---|---|
| `clip` | Clip number, in playing order |
| `block` | Block number: a run of consecutive clips of the same lift |
| `start_t`, `end_t`, `start_frame`, `end_frame`, `duration_s` | When the athlete was on screen |
| `lift`, `tracked` | `bench_press` or `squat`, and the keypoints used as the bar proxy |
| `n_reps` | Counted (fully observed) reps. A lower bound where there are tracking gaps, since a rep can hide in one |
| `n_partial`, `n_indeterminate` | Reps listed but not counted (see `rep_status`) |
| `n_reps_tracking_good` | Counted reps with `tracking_quality = good` |
| `mean_rep_s` | Mean duration of counted reps |
| `n_spikes_removed` | Implausibly fast dips discarded as tracking spikes |
| `robust_range_stab_px` | Typical vertical movement range in the clip (stabilised px) |
| `valid_kp_frac` | Fraction of frames with a confident bar-proxy keypoint |
| `stab_valid_frac` | Fraction of frames with a valid stabilising transform |
| `glitch_frames_masked` | Frames masked as stabilisation glitches |
| `unbridged_gap_s` | Seconds without an observed (or briefly bridged) bar position, including at the clip's start and end; where this is large, `n_reps` may undercount |
| `hip_below_shoulder`, `box_aspect` | Lift classification features (squat if `hip_below_shoulder` > 0.2) |
| `possible_video_change_t` | Times (s) of possible changes of on-screen video inside the clip, for review |
| `skipped` | Why the clip wasn't analysed (too few frames with a tracked bar proxy, or mostly unregistered); empty otherwise |

### `reps.csv` – one row per rep

| Column | Meaning |
|---|---|
| `clip`, `block`, `rep`, `lift` | Identifiers; `rep` numbers counted reps only |
| `rep_status` | `ok` (counted); `partial` (less than 50% of the clip's median range); `indeterminate` (the lowering or effort phase spans an unbridged tracking gap, so its timings are interpolated; takes precedence over `partial`). Only `ok` reps are counted, numbered, and get events, chapters and overlays |
| `rep_start_t`, `rep_end_t`, `rep_duration_s` | The rep: leaving the top until arriving back at the top |
| `top_*`, `bottom_*` | Top / bottom of travel: `_t`, `_frame`, `_rec_us`, `_scene_x/y`, `_stab_x/y` |
| `top_dwell_before_s`, `bottom_dwell_s`, `top_dwell_after_s` | Dwell durations |
| `eccentric_start_t`, `eccentric_s` | Lowering phase |
| `effort_start_t`, `effort_end_t`, `effort_s` | Effort (lifting) phase: leaving the bottom until arriving at the top |
| `effort_mean_vel_px_s`, `effort_peak_vel_px_s` | Bar speed during the effort phase (stabilised px/s) |
| `range_stab_px` | Range of movement (stabilised px) |
| `first_top_dwell_truncated` | `True` if the top dwell runs into the clip start |
| `max_gap_s` | Longest unbridged tracking gap (over 0.5 s) in the lowering or effort phase; 0 if none |
| `track_ok_frac`, `reanchors`, `kp_valid_frac`, `noise_frac` | Tracking diagnostics (`noise_frac` = scatter of raw points around the smoothed trace ÷ rep range) |
| `tracking_quality` | `good` if, over the rep, the stabilisation is valid ≥ 90% of the time, the bar proxy is seen ≥ 50% of the time, `noise_frac` ≤ 0.05 and `max_gap_s` = 0; otherwise `check` |
| `timing_outlier` | Phases far from the clip's median. Informational only: a genuinely fatigued, slow rep also appears here |

### Other outputs

- **`events.csv`** – one row per top or bottom of travel of a counted rep: `clip`, `rep`,
  `event` (`top` / `bottom`), the time and pixel columns above, and the dwell around it
  (`dwell_start_t`, `dwell_end_t`, `dwell_s`).
- **`frames_annotation.csv.gz`** – one row per clip frame, used by pass 3 and handy for joining
  with other per-frame data such as gaze: `frame`, `t`, `clip`, `rep` (rep in progress; 0
  before the first), `n_reps`, `phase` (`setup`, `top_dwell`, `lowering`, `bottom_dwell`,
  `effort`, or empty), bar position (`bar_stab_x/y` smoothed, `bar_scene_x/y`), `person_w`,
  and the homography `h00`…`h22` (scene frame → clip stabilised coordinates).
- **`plots/clip_NN.png`** – QA plot per clip. Top: stabilised bar position (up = top of
  travel) with tops ▲, bottoms ▼, top dwell green, bottom dwell red; partial and indeterminate
  reps hollow grey ▽. Bottom: vertical speed, effort phase shaded orange. Check these before
  trusting a clip's numbers.

## How it works

**Pass 1 – pose extraction** (`extract_frames.py`, started in parallel chunks by
`00run_extract_all.py`). In each frame it finds the bright projected clip, runs YOLO pose
(`yolo11x-pose`) on a crop around it, and keeps the best person whose centre is on the screen
(inside the bright region's convex outline), ignoring people in the room. It also estimates a
transform (homography) that cancels the observer's head movement, by matching ORB features on
the screen with the athlete masked out. These transforms are chained from frame to frame, so
one bad link (motion blur, a glance away) would corrupt every later frame; pass 1b replaces
them.

**Pass 1b – re-stabilisation** (`01restabilise.py`; required, and run automatically).

- Finds the lift clips: runs of athlete detections with gaps under 1.5 s. These are the clips
  used from here on; every frame of a clip carries its clip ID, even if it can't be registered.
- Picks a sharp reference frame in the middle of each clip and matches every frame **directly**
  to it, or failing that to the nearest registered keyframe, so a transform is never more than
  two links from the reference. Transforms with implausible rotation, shear or scale at the
  athlete are rejected; frames left without one become tracking gaps in pass 2.
- Joins the pass-1 chunks, which number their segments and chain their transforms
  independently (analysing them directly could split a lift at a chunk boundary or merge it
  across unrelated coordinate systems, so pass 2 refuses raw pass-1 files).
- Flags possible changes of on-screen video inside a clip (back-to-back videos): frames
  ~0.25 s apart are aligned and their static content compared, and a change is confirmed only
  if no sharp frame within 2 s before it matches one within 2 s after. Flags are not used to
  split clips, because fast head movement and the screen leaving the view look the same.
- Seeds its random sampling (RANSAC) per clip, so re-runs give identical results.

**Pass 2 – rep analysis** (`02analyse_reps.py`).

- Uses pass 1b's clips and groups consecutive clips of the same lift into blocks. Clips with
  less than half their frames registered are skipped (`MIN_REGISTERED_FRAC`).
- Re-bases each clip's transforms on a frame inside it and rejects frames whose transform is
  implausible (scale, position, rotation or shear at the athlete).
- Classifies the lift: in a **squat** the athlete is upright, so the hips sit well below the
  shoulders (relative to body height); on a **bench press** they are level.
- Follows a bar proxy: the **wrists** for bench press, the **shoulders** for squat (bar on the
  upper back).
- Finds the reps and their phases (below), and writes the outputs and QA plots.

**Pass 3 – annotated video** (`03annotate_video.py`, optional): draws the results onto the
scene video and adds chapters.

### Definitions

Reps are found in two steps: **velocity** locates each movement (lowering starts where the bar
starts moving down; the effort phase ends where the upward movement stops), then **position**
measures the phases and dwells around the top and bottom levels found that way. The
thresholds below are constants at the top of `scripts/02analyse_reps.py`.

- **Moving / stationary** – moving means faster than 12% of that phase's peak speed and
  15 stabilised px/s (`V_STILL_FRAC`, `V_STILL_MIN`), judged on a 5-frame rolling median of the
  speed. A movement ends after 2 stationary frames (`STILL_FRAMES`). A rep's peak speeds come
  from the descent into its bottom and the ascent out of it. On the way **up**, a movement only
  ends within 30% of the travel from the top (`NEAR_TOP_FRAC`), so a brief pause partway up (a
  sticking point) stays inside the effort phase; stillness over 1 s (`MAX_STALL_S`) ends it
  anyway. A bar held still at another height (in the rack, during set-up, paused playback)
  never becomes part of a rep.
- **Top of travel** – the bar's level just before lowering starts (or, for the top after a
  rep, just after the effort phase ends); the reported point is the highest point of the top
  dwell.
- **Bottom of travel** – the lowest point between the start of lowering and the end of the
  effort phase. Tops and bottoms only come from frames where the bar was seen (or bridged over
  a short gap).
- **Dwell** – time the bar stays within 10% of the rep's range of the top or bottom level
  (`DWELL_BAND`).
- **Lowering (eccentric)** – end of the top dwell → start of the bottom dwell.
- **Effort (concentric)** – end of the bottom dwell → start of the next top dwell: the upward
  movement for both lifts. The annotated video labels it "raising"; the data uses `effort_*`.
- **Rep** – end of the top dwell → start of the next top dwell; it must travel at least 40% of
  the clip's typical range (`MIN_REP_RANGE_FRAC`).
- **Set-up** – everything in a clip before the first rep's top dwell (unracking, settling, the
  clip starting).
- **Tracking gaps** – frames without a bar position (keypoints missing, or masked as glitches).
  Gaps up to 0.5 s (`MAX_BRIDGE_S`) are bridged with a straight line; longer ones are left
  empty, and smoothing and speeds are computed separately for each continuous stretch, so
  nothing is measured across them. Rep detection still draws a straight line through a long
  gap to find reps either side of it.
- **Glitch frames** – frames masked because the stabilisation is briefly wrong (usually fast
  head movement and motion blur): a body part that should be still (hips and knees in a bench
  press, ankles in a squat) moves more than 12% of body height from its local median
  (`REF_DEV_FRAC`), or is missing while the head moves fast (`HEAD_FAST_PX`), or the bar jumps
  more than 35% of body height in one frame (`BAR_JUMP_FRAC`).
- **Tracking spike** – a "rep" whose lowering takes under 0.2 s (`MIN_ECCENTRIC_S`), lifting
  under 0.12 s (`MIN_EFFORT_S`), or that moves over 3× faster than the clip's typical rep
  (`MAX_SPEED_RATIO`). It is bridged over and the neighbouring reps rebuilt without it.
- **Partial rep** – under 50% of the clip's median range (`PARTIAL_RANGE_FRAC`), typically at
  a clip's start or end. Listed, not counted.
- **Indeterminate rep** – its lowering or effort phase spans an unbridged gap, so its timings
  would be interpolated rather than observed. Listed, not counted.

To see why reps were or weren't found in clip `N`, set `ANALYSE_DEBUG_CLIP=N` when running
pass 2: it prints each candidate's movement onset and stop times, top and bottom levels, and
range against the minimum.

## Limitations

- **Timing resolution is one frame** (0.04 s at 25 fps). Effort phases are often only ~0.3 s
  (≈ 7 frames), so their durations carry roughly ±15% uncertainty.
- **Paused playback** by the operator: head movement during a pause can create false reps, and
  a pause at the top lengthens the top dwell. These often show up as `timing_outlier`; check
  the QA plot.
- **Clips are numbered in playing order**, not by condition; replayed or scrubbed clips can
  appear as extra or split clips. Blocks are numbered, not named: mapping clips and blocks to
  athletes and loads has to come from the study protocol.
- **Back-to-back videos may form one clip** if the next video starts within 1.5 s; rep
  numbering and per-clip thresholds then span both. Check the `possible_video_change_t` times
  against the video.
- **Pose errors**: a wrist sometimes snaps onto the bench or taskbar (the analysis keeps the
  wrist nearest its previous position), and motion blur adds noise. Such reps are marked
  `tracking_quality = check`.
- **Side-view bench press with heavy plates**: the plate hides the hands, so wrist keypoints
  are often missing or misplaced and many reps are marked `check`. Tracking the plate itself
  would be more reliable.

## Other scripts

| Script | Purpose |
|---|---|
| `scripts/add_chapters.py VIDEO ANALYSIS_DIR OUT.mp4 [--offset S]` | Add chapters to a video without re-encoding (run automatically by pass 3) |
| `scripts/contact_sheet.py VIDEO OUT_DIR [STEP_S]` | Diagnostic: labelled thumbnail sheets every `STEP_S` seconds (default 10) |
| `scripts/screen_probe.py VIDEO OUT_DIR T1 T2 …` | Diagnostic: the detected screen region at given times |
| `scripts/pose_probe.py VIDEO T0 T1 MODEL OUT_PREFIX [DEVICE]` | Diagnostic: pose speed and detections in a time window (`DEVICE` defaults to `auto`) |
| `scripts/plot_test.py FRAMES.csv OUT.png` | Diagnostic: raw vs stabilised wrist trace and tracking diagnostics |
| `scripts/device.py`, `videoinfo.py`, `signal_utils.py` | Shared helpers, imported by the other scripts: device choice; opening and checking videos; gap bridging and smoothing |

Scripts that other scripts import aren't numbered, because Python can't import a module whose
name starts with a digit.

## Development

```bash
uv pip install --python .venv -r requirements-dev.txt   # pytest and ruff
.venv/bin/python -m pytest                               # run the tests
ruff check scripts tests                                 # lint (rules pinned in ruff.toml)
```

The tests (in `tests/`; `pytest.ini` limits discovery to that folder) need no real recording
or GPU. They cover the signal helpers; end-to-end runs of pass 2 on synthetic sessions with
known answers (rep counts, phase timings, lift classification, set-up, short and long gaps,
indeterminate reps, clips with unregistered edges, empty sessions); and the scripts' safety
checks (refusing raw pass-1 data; never overwriting or deleting the source video, including
through hard links; stale files; invalid `--jobs`; unreadable or undecodable videos). Passes 1
and 1b need real footage and aren't covered. Without ffmpeg, one test is skipped.

## Licence

The code is released under the MIT licence (see [LICENSE.md](LICENSE.md)).

Pass 1 depends on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and its
pretrained `yolo11*-pose` weights, which are licensed under **AGPL-3.0**, so using or
distributing the full pipeline is also subject to the AGPL's terms (Ultralytics sells a
commercial licence for uses that can't meet them). The other dependencies (OpenCV, NumPy,
SciPy, pandas, matplotlib, PyTorch) use permissive licences.

No study data is included: video, eye-tracker exports and generated outputs are excluded by
`.gitignore`.
