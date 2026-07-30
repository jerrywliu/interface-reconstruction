#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python submission/check_submission_freeze.py --source-only

source_commit="$(git rev-parse HEAD)"
short_commit="$(git rev-parse --short=12 HEAD)"
timestamp="$(date -u +%Y%m%d_%H%M%S)"
run_namespace="${2:-submission_static_${timestamp}_${short_commit}}"
result_root="${1:-results/static/${run_namespace}}"

if [[ ! "$run_namespace" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Refusing to launch: invalid run namespace '$run_namespace'." >&2
  exit 1
fi
if [[ -e "$result_root" ]]; then
  echo "Refusing to launch: result root already exists: $result_root" >&2
  exit 1
fi
if find plots -maxdepth 1 -type d -name "${run_namespace}_perturb_sweep_*" -print -quit 2>/dev/null | grep -q .; then
  echo "Refusing to launch: run namespace already exists under plots/: $run_namespace" >&2
  exit 1
fi
mkdir -p "$result_root"

resolved_config="$result_root/submission_config.resolved.json"
python - "$source_commit" "$resolved_config" <<'PY'
import json
import sys
from pathlib import Path

source_commit, output = sys.argv[1:]
config_path = Path("submission/submission_config.json")
config = json.loads(config_path.read_text(encoding="utf-8"))
if config.get("launch_approved") is not True:
    raise SystemExit("Refusing to launch: submission_config.json is not approved")
config["status"] = "frozen"
config["source"]["target_commit"] = source_commit
Path(output).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

python submission/check_submission_freeze.py \
  --config "$resolved_config" \
  --allow-generated-path "$result_root"

python submission/capture_environment.py \
  --output "$result_root/environment.json"

python -m experiments.static.run_perturbed_sweeps \
  --only lines,circles,ellipses,squares,zalesak \
  --wiggles 0.0,0.05,0.1,0.2,0.3 \
  --seeds 0 \
  --lines 25 \
  --circles 25 \
  --ellipses 25 \
  --squares 25 \
  --zalesak 25 \
  --plic_fallback LVIRA \
  --rescue_profile exact_linear_support_only \
  --corner_behavior_profile pre_f8_corner \
  --max_workers 5 \
  --run_namespace "$run_namespace" \
  --raw_bundle_dir "$result_root/raw_runs" \
  --out_csv "$result_root/perturbed_sweep.csv" \
  --diagnostics_dir "$result_root/diagnostics" \
  --summary_dir "$result_root/summary_plots" \
  --log_dir "$result_root/logs" \
  --notify

echo "Completed final static sweep: $result_root"
