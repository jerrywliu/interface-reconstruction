#!/bin/bash
########################################################
# Run perturbed static-ellipse experiments for linear methods:
#   - Youngs
#   - LVIRA
#   - safe_linear
#   - linear
#
# This script runs the full grid:
#   resolution x wiggle x seed x method
#
# Usage:
#   ./experiments/static/run_perturbed_ellipses_linear_methods.sh
#
# Optional environment overrides:
#   NUM_ELLIPSES=25
#   RESOLUTIONS="0.32 0.50 0.64 1.00 1.28 1.50"
#   WIGGLES="0.0 0.05 0.1 0.2 0.3"
#   SEEDS="0"
#   FIX_BOUNDARY=1
#   CONFIG=static/ellipse
########################################################

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

NUM_ELLIPSES="${NUM_ELLIPSES:-25}"
CONFIG="${CONFIG:-static/ellipse}"
FIX_BOUNDARY="${FIX_BOUNDARY:-1}"

RESOLUTIONS_STR="${RESOLUTIONS:-0.32 0.50 0.64 1.00 1.28 1.50}"
WIGGLES_STR="${WIGGLES:-0.0 0.05 0.1 0.2 0.3}"
SEEDS_STR="${SEEDS:-0}"

read -r -a RESOLUTIONS <<< "$RESOLUTIONS_STR"
read -r -a WIGGLES <<< "$WIGGLES_STR"
read -r -a SEEDS <<< "$SEEDS_STR"

METHODS=("Youngs" "LVIRA" "safe_linear" "linear")

method_to_tag() {
  case "$1" in
    Youngs) echo "youngs" ;;
    LVIRA) echo "lvira" ;;
    safe_linear) echo "safe_linear" ;;
    linear) echo "linear" ;;
    *) echo "$1" ;;
  esac
}

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/perturbed_ellipse_linear_methods_${STAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Perturbed linear-method sweep (ellipses)"
echo "=========================================="
echo "Repo root:    $REPO_ROOT"
echo "Config:       $CONFIG"
echo "Num ellipses: $NUM_ELLIPSES"
echo "Resolutions:  ${RESOLUTIONS[*]}"
echo "Wiggles:      ${WIGGLES[*]}"
echo "Seeds:        ${SEEDS[*]}"
echo "Methods:      ${METHODS[*]}"
echo "Logs:         $LOG_DIR"
echo ""

total_runs=0
failed_runs=0

for method in "${METHODS[@]}"; do
  for resolution in "${RESOLUTIONS[@]}"; do
    for wiggle in "${WIGGLES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        total_runs=$((total_runs + 1))
        method_tag="$(method_to_tag "$method")"
        rtag="${resolution//./p}"
        wtag="${wiggle//./p}"
        save_name="perturb_sweep_ellipses_${method_tag}_r${rtag}_w${wtag}_s${seed}"
        log_path="${LOG_DIR}/${save_name}.log"

        echo "[${total_runs}] ${save_name}"

        cmd=(
          python3 -m experiments.static.ellipses
          --config "$CONFIG"
          --facet_algo "$method"
          --num_ellipses "$NUM_ELLIPSES"
          --resolution "$resolution"
          --mesh_type perturbed_quads
          --perturb_wiggle "$wiggle"
          --perturb_seed "$seed"
          --perturb_fix_boundary "$FIX_BOUNDARY"
          --save_name "$save_name"
        )

        if "${cmd[@]}" >"$log_path" 2>&1; then
          echo "  -> OK"
        else
          echo "  -> FAILED (log: $log_path)"
          failed_runs=$((failed_runs + 1))
        fi
      done
    done
  done
done

echo ""
echo "=========================================="
echo "Sweep complete"
echo "=========================================="
echo "Total runs:  $total_runs"
echo "Failed runs: $failed_runs"
echo "Logs:        $LOG_DIR"

if [ "$failed_runs" -gt 0 ]; then
  exit 1
fi

