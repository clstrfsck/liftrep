#!/usr/bin/env bash
# Run the whole pipeline on one scene video:
#   pass 1   pose extraction (parallel chunks)          scripts/run_extract_all.py
#   pass 1b  re-stabilisation (run by the launcher)     scripts/restabilise.py
#   pass 2   rep analysis                                scripts/analyse_reps.py
#   pass 3   annotated video with chapters (optional)   scripts/annotate_video.py
#
# usage: ./run_pipeline.sh [options] VIDEO
#
# options:
#   -o, --out-dir DIR   working folder (default: work/<video name without extension>, in the project)
#   -j, --jobs N        parallel pose-extraction processes (default: automatic)
#   -d, --device DEV    auto | cuda | cuda:N | mps | cpu (default: auto)
#   -m, --model PATH    pose model (default: models/yolo11x-pose.pt)
#       --overwrite     replace results of an earlier run in the working folder
#       --no-video      skip the annotated video (pass 3)
#   -h, --help          show this help
#
# Results: DIR/out/ (clips.csv, reps.csv, events.csv, plots/) and DIR/annotated.mp4.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scripts="$here/scripts"

usage() { sed -n '2,19s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"; }
die() { echo "error: $*" >&2; exit 1; }

out_dir="" jobs="" device="auto" model="" overwrite=0 make_video=1 video=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--out-dir) [[ $# -ge 2 ]] || die "$1 needs a value"; out_dir="$2"; shift 2 ;;
        -j|--jobs)    [[ $# -ge 2 ]] || die "$1 needs a value"; jobs="$2"; shift 2 ;;
        -d|--device)  [[ $# -ge 2 ]] || die "$1 needs a value"; device="$2"; shift 2 ;;
        -m|--model)   [[ $# -ge 2 ]] || die "$1 needs a value"; model="$2"; shift 2 ;;
        --overwrite)  overwrite=1; shift ;;
        --no-video)   make_video=0; shift ;;
        -h|--help)    usage; exit 0 ;;
        -*)           die "unknown option: $1 (see --help)" ;;
        *)            [[ -z "$video" ]] || die "only one video can be given"; video="$1"; shift ;;
    esac
done
[[ -n "$video" ]] || { usage; exit 1; }
[[ -f "$video" ]] || die "video not found: $video"
[[ -z "$jobs" || "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive whole number"

# Paths given on the command line are relative to where the script was run; make them
# absolute, then work from the project folder so the defaults (work/, models/) are found there.
abspath() { [[ "$1" = /* ]] && echo "$1" || echo "$PWD/$1"; }
video="$(abspath "$video")"
[[ -n "$out_dir" ]] && out_dir="$(abspath "$out_dir")"
[[ -n "$model" ]] && model="$(abspath "$model")"
cd "$here"

# Python: the project's virtual environment if there is one (macOS/Linux or Windows layout)
if [[ -x "$here/.venv/bin/python" ]]; then
    py="$here/.venv/bin/python"
elif [[ -x "$here/.venv/Scripts/python.exe" ]]; then
    py="$here/.venv/Scripts/python.exe"
else
    py="$(command -v python3 || command -v python)" || die "no Python found; see README.md (Setup)"
fi

name="$(basename "$video")"
name="${name%.*}"
out_dir="${out_dir:-work/$name}"
analysis_dir="$out_dir/out"
annotated="$out_dir/annotated.mp4"

if [[ -e "$analysis_dir" || -e "$annotated" ]] && [[ $overwrite -eq 0 ]]; then
    die "$out_dir already has results from an earlier run; use --overwrite or another --out-dir"
fi
if [[ $overwrite -eq 1 ]]; then
    # results of the earlier run that this run may not replace (e.g. with --no-video); never
    # the input itself (-ef: same file, including a hard link or symlink to it)
    for old in "$annotated" "${annotated%.mp4}.nochapters.mp4"; do
        if [[ -e "$old" && "$old" -ef "$video" ]]; then
            die "the input video is $old, which --overwrite would delete; move it or use another --out-dir"
        fi
    done
    rm -f "$annotated" "${annotated%.mp4}.nochapters.mp4"
fi

step() { echo; echo "=== $* ($(date +%H:%M:%S))"; }
start=$SECONDS

step "Pass 1 + 1b: pose extraction and re-stabilisation"
extract_args=("$video" --out-dir "$out_dir" --device "$device")
[[ -n "$jobs" ]] && extract_args+=(--jobs "$jobs")
[[ -n "$model" ]] && extract_args+=(--model "$model")
[[ $overwrite -eq 1 ]] && extract_args+=(--overwrite)
"$py" "$scripts/run_extract_all.py" "${extract_args[@]}"

step "Pass 2: rep analysis"
"$py" "$scripts/analyse_reps.py" "$analysis_dir" "$out_dir/restab" > "$out_dir/analysis_summary.txt"
echo "clip summary: $out_dir/analysis_summary.txt"

if [[ $make_video -eq 1 ]]; then
    step "Pass 3: annotated video"
    "$py" "$scripts/annotate_video.py" "$video" "$analysis_dir" "$annotated"
fi

elapsed=$((SECONDS - start))
step "Done in $((elapsed / 60)) min $((elapsed % 60)) s"
echo "analysis:        $analysis_dir/ (clips.csv, reps.csv, events.csv, plots/)"
[[ $make_video -eq 1 ]] && echo "annotated video: $annotated"
exit 0
