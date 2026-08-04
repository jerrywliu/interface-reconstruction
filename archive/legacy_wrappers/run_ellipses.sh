#!/usr/bin/env bash
set -euo pipefail

# Legacy convenience sweep. Use submission/run_final_static_sweep.sh for the
# frozen paper result set, or invoke experiments.static.ellipses directly for a
# recorded targeted run. Oriented-method PLIC fallback is LVIRA.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Examples:
# python3 -m experiments.static.ellipses --config static/ellipse --num_ellipses 15 --facet_algo Youngs --save_name ellipse_youngs
# python3 -m experiments.static.ellipses --config static/ellipse --num_ellipses 15 --facet_algo ELVIRA --save_name ellipse_elvira
# python3 -m experiments.static.ellipses --config static/ellipse --num_ellipses 15 --facet_algo LVIRA --save_name ellipse_lvira
# python3 -m experiments.static.ellipses --config static/ellipse --num_ellipses 15 --facet_algo safe_linear --plic_fallback LVIRA --save_name ellipse_safelinear
# python3 -m experiments.static.ellipses --config static/ellipse --num_ellipses 15 --facet_algo linear --plic_fallback LVIRA --save_name ellipse_linear
# python3 -m experiments.static.ellipses --config static/ellipse --num_ellipses 15 --facet_algo safe_circle --plic_fallback LVIRA --save_name ellipse_safecircle
# python3 -m experiments.static.ellipses --config static/ellipse --num_ellipses 15 --facet_algo circular --plic_fallback LVIRA --save_name ellipse_mergecircle

python3 -m experiments.static.ellipses \
  --config static/ellipse \
  --sweep \
  --plic_fallback LVIRA

# Historical tracked summary replay; this is not the audited final release.
python3 -m experiments.static.ellipses \
  --config static/ellipse \
  --plot_only \
  --results_file results/static/ellipse_reconstruction_results.txt
