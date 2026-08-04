#!/usr/bin/env bash
set -euo pipefail

# Legacy convenience sweep. Use submission/run_final_static_sweep.sh for the
# frozen paper result set, or invoke experiments.static.zalesak directly for a
# recorded targeted run. Oriented-method PLIC fallback is LVIRA.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Examples:
# python3 -m experiments.static.zalesak --config static/zalesak --num_cases 15 --facet_algo Youngs --save_name zalesak_youngs
# python3 -m experiments.static.zalesak --config static/zalesak --num_cases 15 --facet_algo ELVIRA --save_name zalesak_elvira
# python3 -m experiments.static.zalesak --config static/zalesak --num_cases 15 --facet_algo LVIRA --save_name zalesak_lvira
# python3 -m experiments.static.zalesak --config static/zalesak --num_cases 15 --facet_algo safe_linear --plic_fallback LVIRA --save_name zalesak_safelinear
# python3 -m experiments.static.zalesak --config static/zalesak --num_cases 15 --facet_algo linear --plic_fallback LVIRA --save_name zalesak_linear
# python3 -m experiments.static.zalesak --config static/zalesak --num_cases 15 --facet_algo safe_circle --plic_fallback LVIRA --save_name zalesak_safecircle
# python3 -m experiments.static.zalesak --config static/zalesak --num_cases 15 --facet_algo circular --plic_fallback LVIRA --save_name zalesak_mergecircle
# python3 -m experiments.static.zalesak --config static/zalesak --num_cases 15 --facet_algo 'circular+corner' --plic_fallback LVIRA --corner_behavior_profile pre_f8_corner --rescue_profile exact_linear_support_only --save_name zalesak_circular_corner

python3 -m experiments.static.zalesak \
  --config static/zalesak \
  --sweep \
  --num_cases 15 \
  --plic_fallback LVIRA

# Historical tracked summary replay; this is not the audited final release.
python3 -m experiments.static.zalesak \
  --config static/zalesak \
  --plot_only \
  --results_file results/static/zalesak_reconstruction_results.txt
