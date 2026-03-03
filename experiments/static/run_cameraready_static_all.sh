#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RELEASE_ID="${RELEASE_ID:-static_cameraready_${STAMP}}"
RELEASE_ROOT="${RELEASE_ROOT:-results/static/camera_ready}"
RELEASE_DIR="${RELEASE_ROOT}/${RELEASE_ID}"

RUN_PERTURBED="${RUN_PERTURBED:-1}"
RUN_CARTESIAN="${RUN_CARTESIAN:-1}"
SYNC_OVERLEAF="${SYNC_OVERLEAF:-0}"
OVERLEAF_DIR="${OVERLEAF_DIR:-../overleaf/interface-reconstruction-paper/figs/cameraready}"
NOTIFY="${NOTIFY:-1}"

OUT_CSV="${OUT_CSV:-${RELEASE_DIR}/csv/perturbed_sweep.csv}"
SUMMARY_DIR="${SUMMARY_DIR:-${RELEASE_DIR}/summary_plots}"

echo "=========================================="
echo "Static camera-ready pipeline"
echo "=========================================="
echo "Release ID:    $RELEASE_ID"
echo "Release dir:   $RELEASE_DIR"
echo "Run perturbed: $RUN_PERTURBED"
echo "Run cartesian: $RUN_CARTESIAN"
echo "Sync overleaf: $SYNC_OVERLEAF"
echo "Slack notify:  $NOTIFY"
echo ""

if [[ "$RUN_PERTURBED" == "1" ]]; then
  RELEASE_ID="$RELEASE_ID" \
  RELEASE_ROOT="$RELEASE_ROOT" \
  OUT_CSV="$OUT_CSV" \
  SUMMARY_DIR="$SUMMARY_DIR" \
  NOTIFY="$NOTIFY" \
  LOG_DIR="${RELEASE_DIR}/logs/perturbed_runs" \
  "$SCRIPT_DIR/run_cameraready_static_perturbed.sh"
fi

if [[ "$RUN_CARTESIAN" == "1" ]]; then
  RELEASE_ID="$RELEASE_ID" \
  RELEASE_ROOT="$RELEASE_ROOT" \
  NOTIFY="$NOTIFY" \
  LOG_DIR="${RELEASE_DIR}/logs/cartesian_runs" \
  "$SCRIPT_DIR/run_cameraready_static_cartesian.sh"
fi

RELEASE_ID="$RELEASE_ID" \
RELEASE_ROOT="$RELEASE_ROOT" \
SUMMARY_DIR="$SUMMARY_DIR" \
PERTURBED_CSV="$OUT_CSV" \
SYNC_OVERLEAF="$SYNC_OVERLEAF" \
OVERLEAF_DIR="$OVERLEAF_DIR" \
NOTIFY="$NOTIFY" \
"$SCRIPT_DIR/bundle_static_cameraready_release.sh"

echo ""
echo "Done: static camera-ready pipeline finished."
