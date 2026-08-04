#!/usr/bin/env bash
set -euo pipefail

# Legacy convenience sweep. Use submission/run_final_static_sweep.sh for the
# frozen paper result set, or invoke experiments.static.squares directly for a
# recorded targeted run. Oriented-method PLIC fallback is LVIRA.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Examples:
# python3 -m experiments.static.squares --config static/square --num_squares 15 --facet_algo Youngs --save_name square_youngs
# python3 -m experiments.static.squares --config static/square --num_squares 15 --facet_algo ELVIRA --save_name square_elvira
# python3 -m experiments.static.squares --config static/square --num_squares 15 --facet_algo LVIRA --save_name square_lvira
# python3 -m experiments.static.squares --config static/square --num_squares 15 --facet_algo linear --plic_fallback LVIRA --save_name square_linear
# python3 -m experiments.static.squares --config static/square --num_squares 15 --facet_algo linear+corner --plic_fallback LVIRA --save_name square_linear_corner
# python3 -m experiments.static.squares --config static/square --num_squares 15 --facet_algo safe_circle --plic_fallback LVIRA --save_name square_safecircle
# python3 -m experiments.static.squares --config static/square --num_squares 15 --facet_algo circular --plic_fallback LVIRA --save_name square_mergecircle

python3 -m experiments.static.squares \
  --config static/square \
  --sweep \
  --num_squares 15 \
  --plic_fallback LVIRA

# Historical tracked summary replay; this is not the audited final release.
python3 -m experiments.static.squares \
  --config static/square \
  --plot_only \
  --results_file results/static/square_reconstruction_results.txt
