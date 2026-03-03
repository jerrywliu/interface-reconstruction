#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RELEASE_ID="${RELEASE_ID:-static_cameraready_${STAMP}}"
RELEASE_ROOT="${RELEASE_ROOT:-results/static/camera_ready}"
RELEASE_DIR="${RELEASE_ROOT}/${RELEASE_ID}"

ONLY="${ONLY:-squares,zalesak}"
CIRCLES="${CIRCLES:-25}"
ELLIPSES="${ELLIPSES:-25}"
LINES="${LINES:-25}"
SQUARES="${SQUARES:-25}"
ZALESAK="${ZALESAK:-25}"

LOG_DIR="${LOG_DIR:-${RELEASE_DIR}/logs/cartesian_runs}"
DRY_RUN="${DRY_RUN:-0}"
NOTIFY="${NOTIFY:-1}"

mkdir -p "$LOG_DIR"

CMD=(
  python3 -m experiments.static.run_linear_sweeps
  --subprocess
  --only "$ONLY"
  --circles "$CIRCLES"
  --ellipses "$ELLIPSES"
  --lines "$LINES"
  --squares "$SQUARES"
  --zalesak "$ZALESAK"
  --log_dir "$LOG_DIR"
)

if [[ "$NOTIFY" == "1" ]]; then
  CMD+=(--notify)
fi

echo "=========================================="
echo "Camera-ready Cartesian static sweep"
echo "=========================================="
echo "Release ID:   $RELEASE_ID"
echo "Only:         $ONLY"
echo "Squares:      $SQUARES"
echo "Zalesak:      $ZALESAK"
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
echo "Done: Cartesian static camera-ready sweep."
