#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

to_csv_list() {
  local raw="$1"
  raw="${raw// /,}"
  raw="${raw//,,/,}"
  echo "$raw"
}

STAMP="$(date +%Y%m%d_%H%M%S)"
RELEASE_ID="${RELEASE_ID:-static_cameraready_${STAMP}}"
RELEASE_ROOT="${RELEASE_ROOT:-results/static/camera_ready}"
RELEASE_DIR="${RELEASE_ROOT}/${RELEASE_ID}"

ONLY="${ONLY:-lines,circles,ellipses}"
CIRCLES="${CIRCLES:-25}"
ELLIPSES="${ELLIPSES:-25}"
LINES="${LINES:-25}"
RESOLUTIONS="${RESOLUTIONS:-0.32,0.50,0.64,1.00,1.28,1.50}"
WIGGLES="${WIGGLES:-0.0,0.05,0.1,0.2,0.3}"
SEEDS="${SEEDS:-0}"

OUT_CSV="${OUT_CSV:-${RELEASE_DIR}/csv/perturbed_sweep.csv}"
SUMMARY_DIR="${SUMMARY_DIR:-${RELEASE_DIR}/summary_plots}"
LOG_DIR="${LOG_DIR:-${RELEASE_DIR}/logs/perturbed_runs}"
DRY_RUN="${DRY_RUN:-0}"
NOTIFY="${NOTIFY:-1}"

mkdir -p "$(dirname "$OUT_CSV")" "$SUMMARY_DIR" "$LOG_DIR"

RESOLUTIONS_CSV="$(to_csv_list "$RESOLUTIONS")"
WIGGLES_CSV="$(to_csv_list "$WIGGLES")"
SEEDS_CSV="$(to_csv_list "$SEEDS")"

CMD=(
  python3 -m experiments.static.run_perturbed_sweeps
  --only "$ONLY"
  --circles "$CIRCLES"
  --ellipses "$ELLIPSES"
  --lines "$LINES"
  --resolutions "$RESOLUTIONS_CSV"
  --wiggles "$WIGGLES_CSV"
  --seeds "$SEEDS_CSV"
  --out_csv "$OUT_CSV"
  --summary_dir "$SUMMARY_DIR"
  --log_dir "$LOG_DIR"
)

if [[ "$NOTIFY" == "1" ]]; then
  CMD+=(--notify)
fi

echo "=========================================="
echo "Camera-ready perturbed static sweep"
echo "=========================================="
echo "Release ID:   $RELEASE_ID"
echo "Only:         $ONLY"
echo "Resolutions:  $RESOLUTIONS_CSV"
echo "Wiggles:      $WIGGLES_CSV"
echo "Seeds:        $SEEDS_CSV"
echo "CSV:          $OUT_CSV"
echo "Summary dir:  $SUMMARY_DIR"
echo "Logs:         $LOG_DIR"
echo "Slack notify: $NOTIFY"
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run command:"
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

"${CMD[@]}"

echo ""
echo "Done: perturbed static camera-ready sweep."
