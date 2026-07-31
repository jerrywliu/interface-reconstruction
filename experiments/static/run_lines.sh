#!/usr/bin/env bash
set -euo pipefail

# Legacy convenience sweep. Use submission/run_final_static_sweep.sh for the
# frozen paper result set, or invoke experiments.static.lines directly for a
# recorded targeted run. Oriented-method PLIC fallback is LVIRA.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Examples:
# python3 -m experiments.static.lines --config static/line --num_lines 25 --facet_algo Youngs --save_name line_youngs
# python3 -m experiments.static.lines --config static/line --num_lines 25 --facet_algo ELVIRA --save_name line_elvira
# python3 -m experiments.static.lines --config static/line --num_lines 25 --facet_algo LVIRA --save_name line_lvira
# python3 -m experiments.static.lines --config static/line --num_lines 25 --facet_algo safe_linear --plic_fallback LVIRA --save_name line_safelinear
# python3 -m experiments.static.lines --config static/line --num_lines 25 --facet_algo linear --plic_fallback LVIRA --save_name line_mergelinear

python3 -m experiments.static.lines \
  --config static/line \
  --sweep \
  --plic_fallback LVIRA

# Historical tracked summary replay; this is not the audited final release.
python3 -m experiments.static.lines \
  --config static/line \
  --plot_only \
  --results_file results/static/line_reconstruction_results.txt
