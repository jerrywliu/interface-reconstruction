#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-static_paper_affected_diagnostics_${STAMP}}"
RUN_ROOT="${RUN_ROOT:-results/static/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_WORKERS="${MAX_WORKERS:-5}"
NOTIFY="${NOTIFY:-1}"
DRY_RUN="${DRY_RUN:-0}"

OUT_CSV="${OUT_CSV:-${RUN_ROOT}/perturbed_sweep.csv}"
DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-${RUN_ROOT}/diagnostics}"
SUMMARY_DIR="${SUMMARY_DIR:-${RUN_ROOT}/summary_plots}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"

CMD=(
  "$PYTHON_BIN" -m experiments.static.run_perturbed_sweeps
  --only lines,circles,ellipses,squares,zalesak
  --algos linear,circular,linear+corner,circular+corner
  --wiggles 0.0,0.05,0.1,0.2,0.3
  --seeds 0
  --plic_fallback LVIRA
  --rescue_profile exact_linear_support_only
  --corner_behavior_profile pre_f8_corner
  --max_workers "$MAX_WORKERS"
  --out_csv "$OUT_CSV"
  --diagnostics_dir "$DIAGNOSTICS_DIR"
  --summary_dir "$SUMMARY_DIR"
  --log_dir "$LOG_DIR"
)

if [[ "$NOTIFY" == "1" ]]; then
  CMD+=(--notify)
else
  CMD+=(--no-notify)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry_run)
fi

echo "=========================================="
echo "Affected paper-results perturbed sweep"
echo "=========================================="
echo "Run ID:       $RUN_ID"
echo "Run root:     $RUN_ROOT"
echo "Workers:      $MAX_WORKERS"
echo "PLIC fallback: LVIRA"
echo "Corner behavior: pre_f8_corner"
echo "Rescue profile: exact_linear_support_only"
echo "Slack notify: $NOTIFY"
echo "Dry run:      $DRY_RUN"
echo ""
printf 'Command: '
printf '%q ' "${CMD[@]}"
echo ""
echo ""

"${CMD[@]}"

if [[ "$DRY_RUN" != "1" ]]; then
  echo ""
  echo "Done: affected paper-results sweep completed successfully."
  echo "Run bundle: $RUN_ROOT"
fi
