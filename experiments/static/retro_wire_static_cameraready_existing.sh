#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RELEASE_ID="${RELEASE_ID:-static_cameraready_existing_${STAMP}}"
RELEASE_ROOT="${RELEASE_ROOT:-results/static/camera_ready}"
SUMMARY_DIR="${SUMMARY_DIR:-results/static/perturbed_plots}"
SYNC_OVERLEAF="${SYNC_OVERLEAF:-0}"
OVERLEAF_DIR="${OVERLEAF_DIR:-../overleaf/interface-reconstruction-paper/figs/cameraready}"
NOTIFY="${NOTIFY:-1}"

LATEST_PERTURBED_CSV="$(ls -t results/static/perturbed_sweep_*.csv 2>/dev/null | head -n 1 || true)"

echo "=========================================="
echo "Retro-wire existing static camera-ready assets"
echo "=========================================="
echo "Release ID:    $RELEASE_ID"
echo "Summary dir:   $SUMMARY_DIR"
echo "CSV hint:      ${LATEST_PERTURBED_CSV:-<none>}"
echo "Sync overleaf: $SYNC_OVERLEAF"
echo "Slack notify:  $NOTIFY"
echo ""

RELEASE_ID="$RELEASE_ID" \
RELEASE_ROOT="$RELEASE_ROOT" \
SUMMARY_DIR="$SUMMARY_DIR" \
PERTURBED_CSV="$LATEST_PERTURBED_CSV" \
SYNC_OVERLEAF="$SYNC_OVERLEAF" \
OVERLEAF_DIR="$OVERLEAF_DIR" \
NOTIFY="$NOTIFY" \
"$SCRIPT_DIR/bundle_static_cameraready_release.sh"

echo ""
echo "Done: retro-wire complete."
