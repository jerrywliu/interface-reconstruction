#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RELEASE_ID="${RELEASE_ID:-static_cameraready_${STAMP}}"
RELEASE_ROOT="${RELEASE_ROOT:-results/static/camera_ready}"
RELEASE_DIR="${RELEASE_ROOT}/${RELEASE_ID}"

SUMMARY_DIR="${SUMMARY_DIR:-${RELEASE_DIR}/summary_plots}"
PERTURBED_CSV="${PERTURBED_CSV:-}"
SYNC_OVERLEAF="${SYNC_OVERLEAF:-0}"
OVERLEAF_DIR="${OVERLEAF_DIR:-../overleaf/interface-reconstruction-paper/figs/cameraready}"
NOTIFY="${NOTIFY:-1}"

CSV_DIR="${RELEASE_DIR}/csv"
PLOTS_DIR="${RELEASE_DIR}/summary_plots"
PAPER_DIR="${RELEASE_DIR}/paper_figs"
LOGS_DIR="${RELEASE_DIR}/logs"

mkdir -p "$CSV_DIR" "$PLOTS_DIR" "$PAPER_DIR" "$LOGS_DIR"

copy_if_exists() {
  local src="$1"
  local dest="$2"
  if [[ -f "$src" ]]; then
    cp -f "$src" "$dest"
    echo "Copied: $src -> $dest"
    return 0
  fi
  return 1
}

copy_if_new() {
  local src="$1"
  local dest="$2"
  if [[ -f "$dest" ]]; then
    return 0
  fi
  copy_if_exists "$src" "$dest"
}

copy_first_match() {
  local dest="$1"
  shift
  local src
  for src in "$@"; do
    if copy_if_exists "$src" "$dest"; then
      return 0
    fi
  done
  echo "Missing (all candidates): $dest"
  return 1
}

if [[ -d "$SUMMARY_DIR" && "$SUMMARY_DIR" != "$PLOTS_DIR" ]]; then
  for pattern in "lines_*.png" "circles_*.png" "ellipses_*.png" "squares_*.png" "zalesak_*.png"; do
    find "$SUMMARY_DIR" -maxdepth 1 -type f -name "$pattern" -exec cp -f {} "$PLOTS_DIR/" \;
  done
fi

if [[ -n "$PERTURBED_CSV" ]]; then
  copy_if_new "$PERTURBED_CSV" "${CSV_DIR}/$(basename "$PERTURBED_CSV")" || true
fi

LATEST_GENERIC_PERTURBED="$(ls -t results/static/perturbed_sweep_*.csv 2>/dev/null | head -n 1 || true)"
if [[ -n "$LATEST_GENERIC_PERTURBED" ]]; then
  copy_if_new "$LATEST_GENERIC_PERTURBED" "${CSV_DIR}/$(basename "$LATEST_GENERIC_PERTURBED")" || true
fi

for merged in $(ls -t results/static/perturbed_sweep_*_merged_*.csv 2>/dev/null || true); do
  copy_if_new "$merged" "${CSV_DIR}/$(basename "$merged")" || true
done

copy_first_match \
  "${PAPER_DIR}/line_reconstruction_perturbed_all_methods_2x2.png" \
  "${PLOTS_DIR}/lines_all_methods_2x2.png" \
  "results/static/perturbed_plots/lines_all_methods_2x2.png" || true

copy_first_match \
  "${PAPER_DIR}/circle_reconstruction_perturbed_all_methods_5x2_axes.png" \
  "${PLOTS_DIR}/circles_all_methods_5x2_axes.png" \
  "results/static/perturbed_plots/circles_all_methods_5x2_axes.png" || true

copy_first_match \
  "${PAPER_DIR}/ellipse_reconstruction_perturbed_all_methods_5x2_axes.png" \
  "${PLOTS_DIR}/ellipses_all_methods_5x2_axes.png" \
  "results/static/perturbed_plots/ellipses_all_methods_5x2_axes.png" || true

copy_first_match \
  "${PAPER_DIR}/ellipse_reconstruction_combined.png" \
  "${PLOTS_DIR}/ellipses_all_methods_5x2_axes.png" \
  "results/static/perturbed_plots/ellipses_all_methods_5x2_axes.png" \
  "results/static/ellipse_reconstruction_combined.png" \
  "results/static/linear_ellipse_reconstruction_combined.png" || true

copy_first_match \
  "${PAPER_DIR}/square_reconstruction_area.png" \
  "${PLOTS_DIR}/squares_all_methods_2x2.png" \
  "results/static/perturbed_plots/squares_all_methods_2x2.png" \
  "results/static/linear_square_reconstruction_area.png" \
  "results/static/square_reconstruction_area.png" || true

copy_first_match \
  "${PAPER_DIR}/zalesak_reconstruction_combined.png" \
  "${PLOTS_DIR}/zalesak_all_methods_2x2.png" \
  "results/static/perturbed_plots/zalesak_all_methods_2x2.png" \
  "results/static/linear_zalesak_reconstruction_combined.png" \
  "results/static/zalesak_reconstruction_combined.png" || true

copy_first_match \
  "${PAPER_DIR}/zalesak_reconstruction_area.png" \
  "${PLOTS_DIR}/zalesak_all_methods_2x2.png" \
  "results/static/perturbed_plots/zalesak_all_methods_2x2.png" \
  "results/static/linear_zalesak_reconstruction_combined.png" \
  "results/static/zalesak_reconstruction_combined.png" || true

copy_first_match \
  "${PAPER_DIR}/square_reconstruction_perturbed_all_methods_2x2.png" \
  "${PLOTS_DIR}/squares_all_methods_2x2.png" \
  "results/static/perturbed_plots/squares_all_methods_2x2.png" || true

copy_first_match \
  "${PAPER_DIR}/zalesak_reconstruction_perturbed_all_methods_2x2.png" \
  "${PLOTS_DIR}/zalesak_all_methods_2x2.png" \
  "results/static/perturbed_plots/zalesak_all_methods_2x2.png" || true

copy_if_exists "results/static/linear_square_reconstruction_results.txt" "${LOGS_DIR}/linear_square_reconstruction_results.txt" || true
copy_if_exists "results/static/linear_zalesak_reconstruction_results.txt" "${LOGS_DIR}/linear_zalesak_reconstruction_results.txt" || true

MANIFEST_PATH="${RELEASE_DIR}/manifest.md"
{
  echo "# Static Camera-Ready Release"
  echo ""
  echo "- Release ID: \`${RELEASE_ID}\`"
  echo "- Created (local): \`$(date '+%Y-%m-%d %H:%M:%S %Z')\`"
  echo "- Repo root: \`${REPO_ROOT}\`"
  echo "- Summary source: \`${SUMMARY_DIR}\`"
  echo "- Perturbed CSV hint: \`${PERTURBED_CSV:-<none>}\`"
  echo ""
  echo "## Bundled CSV files"
  find "$CSV_DIR" -maxdepth 1 -type f -print | sed "s#^#- \`#;s#\$#\`#"
  echo ""
  echo "## Bundled paper figures"
  find "$PAPER_DIR" -maxdepth 1 -type f -print | sed "s#^#- \`#;s#\$#\`#"
  echo ""
  echo "## Bundled summary plots"
  find "$PLOTS_DIR" -maxdepth 1 -type f -print | sed "s#^#- \`#;s#\$#\`#"
} > "$MANIFEST_PATH"

mkdir -p "$RELEASE_ROOT"
ln -sfn "$RELEASE_ID" "${RELEASE_ROOT}/latest"

if [[ "$SYNC_OVERLEAF" == "1" ]]; then
  mkdir -p "$OVERLEAF_DIR"
  find "$PAPER_DIR" -maxdepth 1 -type f -name "*.png" -exec cp -f {} "$OVERLEAF_DIR/" \;
  echo "Synced camera-ready figures to: $OVERLEAF_DIR"
fi

if [[ "$NOTIFY" == "1" ]]; then
  python3 -m experiments.static.notify_cameraready_release \
    --release_dir "$RELEASE_DIR" \
    --message "Static camera-ready bundle ready: ${RELEASE_ID}"
fi

echo ""
echo "=========================================="
echo "Static camera-ready bundle complete"
echo "=========================================="
echo "Release dir: $RELEASE_DIR"
echo "Manifest:    $MANIFEST_PATH"
