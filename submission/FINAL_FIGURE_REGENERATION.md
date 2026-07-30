# Final Figure Regeneration

Updated: 2026-07-30

For the audited mapping of all 26 active manuscript PDF includes, including the
`N=16,32,64` companion-run decision and paired-variant tooling gaps, follow
`submission/POST_SWEEP_FIGURE_MANIFEST.md`.

## Policy And Scope

- Reviewed July 2026 figures are the oldest eligible submission sources.
- Prefer figures regenerated from the completed final sweep whenever its data,
  algorithm, metrics, or method comparison supersedes the July source.
- March and May assets are layout references only.
- Submission assets are PDF. They must contain no raster image objects and every
  reported font must be embedded. PNG exports are 300-DPI review previews only;
  never install them in the manuscript.
- Generate both qualitative variants until author selection: open facet-endpoint
  circles in the annotated version, and clean main panels in the clean version.
  Both retain endpoint/cell-crossing markers in spyglasses and semantic corner
  diamonds everywhere.
- Do not regenerate from a partial release. At the time this workflow was written,
  `submission_static_20260730_201510_df31a8d5f9b3` was still `running` and was
  therefore intentionally not used.

This workflow does not change the reconstruction algorithm or the experiment map.
It builds a read-only canonical-name view of the immutable raw bundles because the
release names include a collision-proof namespace and profile suffixes.

## Environment

Run every command from the repository root after replacing `FINAL_ROOT` with the
completed release directory.

```bash
cd /path/to/interface-reconstruction
export FINAL_ROOT="$PWD/results/static/submission_static_<timestamp>_<commit>"
export RUN_NAMESPACE="$(basename "$FINAL_ROOT")"
export SOURCE_COMMIT="$(python -c 'import json,os; print(json.load(open(os.environ["FINAL_ROOT"] + "/submission_config.resolved.json"))["source"]["target_commit"])')"
export FIGURE_ROOT="$PWD/results/submission/final_figures_${SOURCE_COMMIT:0:12}"
export PLOTS_VIEW="$FIGURE_ROOT/final_plots_view"
mkdir -p "$FIGURE_ROOT"
```

Poppler's `pdfimages` and `pdffonts` must be on `PATH` for final QA.

## Release Gate

This preflight fails unless the release completed all `970` runs and `24,250`
cases with no failures, all raw bundles are present, and the resolved source commit
matches the captured source state.

```bash
python - "$FINAL_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "sweep_manifest.json").read_text())
config = json.loads((root / "submission_config.resolved.json").read_text())
state = json.loads((root / "diagnostics/source_state.json").read_text())

assert manifest["status"] == "completed", manifest["status"]
assert manifest["planned_run_count"] == 970
assert manifest["planned_case_count"] == 24250
assert manifest["successful_run_count"] == 970
assert manifest["failure_count"] == 0
assert config["status"] == "frozen"
assert config["source"]["target_commit"] == state["source_commit"]
assert state["source_dirty"] is False

with (root / "diagnostics/run_inventory.csv").open(newline="") as handle:
    assert sum(1 for _ in csv.DictReader(handle)) == 970
with (root / "diagnostics/case_metrics.csv").open(newline="") as handle:
    assert sum(1 for _ in csv.DictReader(handle)) == 24250
assert len(list((root / "raw_runs").iterdir())) == 970
assert (root / "perturbed_sweep.csv").stat().st_size > 0
print(f"READY: {root}")
PY
```

Do not continue after any failed assertion.

## Canonical Raw-Bundle View

The plotting driver expects canonical names such as
`perturb_sweep_squares_linearpluscorner_r0p5_w0p1_s0`. The immutable release
uses namespaced, profile-qualified names. Build a symlink-only view from the
`save_name` field in the final CSV:

```bash
python - "$FINAL_ROOT" "$PLOTS_VIEW" <<'PY'
import csv
import os
import sys
from pathlib import Path

from experiments.static.run_perturbed_sweeps import _make_save_name

release = Path(sys.argv[1]).resolve()
view = Path(sys.argv[2]).resolve()
if view.exists():
    raise SystemExit(f"Refusing to replace existing view: {view}")
view.mkdir(parents=True)

settings = {}
with (release / "perturbed_sweep.csv").open(newline="") as handle:
    for row in csv.DictReader(handle):
        key = (
            row["experiment"], row["algo"], float(row["resolution"]),
            float(row["wiggle"]), int(row["seed"]),
        )
        prior = settings.setdefault(key, row["save_name"])
        if prior != row["save_name"]:
            raise RuntimeError(f"multiple bundles for {key}: {prior}, {row['save_name']}")

for (experiment, algo, resolution, wiggle, seed), save_name in sorted(settings.items()):
    source = release / "raw_runs" / save_name
    if not source.is_dir():
        raise FileNotFoundError(source)
    canonical = _make_save_name(experiment, algo, resolution, wiggle, seed)
    destination = view / canonical
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"canonical-name collision: {destination}")
    destination.symlink_to(source, target_is_directory=True)

print(f"Linked {len(settings)} immutable run bundles into {view}")
PY
```

## Regeneration Commands

### Main Metrics

Source: `FINAL_ROOT/perturbed_sweep.csv`. Driver:
`experiments/static/generate_section6_maintext_figures.py`.

```bash
python -m experiments.static.generate_section6_maintext_figures \
  --csv "$FINAL_ROOT/perturbed_sweep.csv" \
  --plots_root "$PLOTS_VIEW" \
  --out_dir "$FIGURE_ROOT/section6" \
  --figure_groups quantitative \
  --experiments all
```

Authoritative outputs are
`section6/summary_plots/{lines,squares,circles,ellipses,zalesak}_maintext_metrics.pdf`.
The same stems ending in `.png` are review previews.

### Representatives

Source: selected VTK and facet metadata inside `PLOTS_VIEW`; the final CSV supplies
the exact immutable bundle mapping. The current proposed cases are lines `6`,
squares `24`, circles `12`, ellipses `12`, and Zalesak `12`.

```bash
python -m experiments.static.generate_section6_maintext_figures \
  --csv "$FINAL_ROOT/perturbed_sweep.csv" \
  --plots_root "$PLOTS_VIEW" \
  --out_dir "$FIGURE_ROOT/section6" \
  --figure_groups representative \
  --endpoint_variants paired \
  --case_overrides lines=6,squares=24,circles=12,ellipses=12,zalesak=12
```

Outputs are
`section6/representative_cases/<experiment>_maintext_representative_{with_endpoints,clean}.pdf`.
Do not choose or rename a variant until author review.

### Resolution Studies

These are dedicated final-commit companion runs at `N=16,32,64`, not rows in
the primary sweep. Current proposed cases are lines `0`, squares `22`, circles
`12`, ellipses `12`, and Zalesak `20`.

```bash
python -m experiments.static.run_appendix_resolution_visuals --only lines \
  --case_index 0 --resolutions 0.16,0.32,0.64 --wiggles 0,0.1 \
  --endpoint_variants paired --out_dir "$FIGURE_ROOT/resolution/lines"
python -m experiments.static.run_appendix_resolution_visuals --only squares \
  --case_index 22 --resolutions 0.16,0.32,0.64 --wiggles 0,0.1 \
  --endpoint_variants paired --out_dir "$FIGURE_ROOT/resolution/squares"
python -m experiments.static.run_appendix_resolution_visuals --only circles \
  --case_index 12 --resolutions 0.16,0.32,0.64 --wiggles 0,0.1 \
  --endpoint_variants paired --out_dir "$FIGURE_ROOT/resolution/circles"
python -m experiments.static.run_appendix_resolution_visuals --only ellipses \
  --case_index 12 --resolutions 0.16,0.32,0.64 --wiggles 0,0.1 \
  --endpoint_variants paired --out_dir "$FIGURE_ROOT/resolution/ellipses"
python -m experiments.static.run_appendix_resolution_visuals --only zalesak \
  --case_index 20 --resolutions 0.16,0.32,0.64 --wiggles 0,0.1 \
  --endpoint_variants paired --out_dir "$FIGURE_ROOT/resolution/zalesak"
```

Outputs are
`resolution/<experiment>/summary_plots/<experiment>_resolution_cartesian_vs_perturbed_{with_endpoints,clean}.pdf`.
Preserve each companion `manifest.json`, log set, and raw geometry directory.

### All-Method Panels

Source: `FINAL_ROOT/perturbed_sweep.csv`. Driver:
`experiments/static/run_perturbed_sweeps.py` in plot-only mode.

```bash
python -m experiments.static.run_perturbed_sweeps \
  --plot_from_csv "$FINAL_ROOT/perturbed_sweep.csv" \
  --summary_dir "$FIGURE_ROOT/all_method_summary_plots" \
  --no-notify
```

The five paper candidates are:

- `lines_all_methods_2x2.pdf`
- `squares_all_methods_2x2.pdf`
- `circles_all_methods_5x2_axes.pdf`
- `ellipses_all_methods_5x2_axes.pdf`
- `zalesak_all_methods_2x2.pdf`

### Guarded C0 Panels

The primary final sweep has C0 disabled and cannot source these four panels. Do
not use the March/May or pre-guard C0 bundles. Run the dedicated study from the
same frozen source commit after the primary release passes its audit:

```bash
export C0_ROOT="$PWD/results/static/final_guarded_c0_${SOURCE_COMMIT:0:12}"
test ! -e "$C0_ROOT"
python -m experiments.static.run_appendix_c0_study \
  --out_dir "$C0_ROOT" \
  --out_csv "$C0_ROOT/csv/appendix_c0_sweep.csv" \
  --log_dir "$C0_ROOT/logs" \
  --save_prefix "final_guarded_c0_${SOURCE_COMMIT:0:12}" \
  --seeds 0 \
  --ellipses 25 \
  --zalesak 25 \
  --endpoint_variants paired
```

Before accepting the plots, verify the complete `165`-setting study and all raw
run directories:

```bash
python - "$C0_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
with (root / "csv/appendix_c0_sweep.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
save_names = {row["save_name"] for row in rows}
assert len(save_names) == 165, len(save_names)
missing = [name for name in save_names if not (Path("plots") / name).is_dir()]
assert not missing, missing[:5]
print("READY: guarded C0 study, 165/165 settings")
PY
```

Authoritative outputs are
`C0_ROOT/summary_plots/{ellipses,zalesak}_appendix_c0_2x2.pdf` and
`C0_ROOT/representative_cases/{ellipses,zalesak}_appendix_c0_representative_{with_endpoints,clean}.pdf`.
The captions must describe C0 continuity as conditional because infeasible
endpoint refits retain the conservative pre-C0 facet.

### PLIC Stencil

Source: deterministic line case `4`, center cell `(14,13)`, `N=32`, perturbation
magnitude `0.3`, seed `0`. Driver:
`experiments/static/generate_plic_baseline_stencil_figure.py`.

```bash
mkdir -p "$FIGURE_ROOT/deterministic"
python -m experiments.static.generate_plic_baseline_stencil_figure \
  --out "$FIGURE_ROOT/deterministic/perfect_reconstruction_plic_stencil" \
  --case-index 4 --cell-x 14 --cell-y 13 \
  --resolution 0.32 --wiggle 0.3 --seed 0
```

The PDF is authoritative; the SVG is an auxiliary vector export and the 300-DPI
PNG is a review preview.

### Staged Reconstruction

Source: deterministic Zalesak case `22`, `N=100`, perturbation magnitude `0.1`,
seed `0`, with the final frozen defaults. Driver:
`experiments/static/generate_staged_reconstruction_figure.py`.

```bash
python -m experiments.static.generate_staged_reconstruction_figure \
  --output-dir "$FIGURE_ROOT/deterministic" \
  --prefix staged_reconstruction_zalesak \
  --case-index 22 --resolution 1.0 --wiggle 0.1 --seed 0
```

The PDF is authoritative. The 300-DPI PNG is only for review.

## Manuscript Asset Map

| Generated PDF | Manuscript asset |
| --- | --- |
| `deterministic/perfect_reconstruction_plic_stencil.pdf` | `perfect_reconstruction_plic_stencil.pdf` |
| `deterministic/staged_reconstruction_zalesak.pdf` | `staged_reconstruction_zalesak.pdf` |
| `section6/summary_plots/lines_maintext_metrics.pdf` | `line_reconstruction_maintext_metrics.pdf` |
| `section6/summary_plots/squares_maintext_metrics.pdf` | `square_reconstruction_maintext_metrics.pdf` |
| `section6/summary_plots/circles_maintext_metrics.pdf` | `circle_reconstruction_maintext_metrics.pdf` |
| `section6/summary_plots/ellipses_maintext_metrics.pdf` | `ellipse_reconstruction_maintext_metrics.pdf` |
| `section6/summary_plots/zalesak_maintext_metrics.pdf` | `zalesak_reconstruction_maintext_metrics.pdf` |
| `section6/representative_cases/<experiment>_maintext_representative_<approved>.pdf` | `<singular>_reconstruction_maintext_representative.pdf` |
| `resolution/<experiment>/summary_plots/<experiment>_resolution_cartesian_vs_perturbed_<approved>.pdf` | `<experiment>_resolution_cartesian_vs_perturbed.pdf` |
| `all_method_summary_plots/lines_all_methods_2x2.pdf` | `line_reconstruction_perturbed_all_methods_2x2.pdf` |
| `all_method_summary_plots/squares_all_methods_2x2.pdf` | `square_reconstruction_perturbed_all_methods_2x2.pdf` |
| `all_method_summary_plots/circles_all_methods_5x2_axes.pdf` | `circle_reconstruction_perturbed_all_methods_5x2_axes.pdf` |
| `all_method_summary_plots/ellipses_all_methods_5x2_axes.pdf` | `ellipse_reconstruction_perturbed_all_methods_5x2_axes.pdf` |
| `all_method_summary_plots/zalesak_all_methods_2x2.pdf` | `zalesak_reconstruction_perturbed_all_methods_2x2.pdf` |
| `C0_ROOT/summary_plots/<experiment>_appendix_c0_2x2.pdf` | same basename |
| `C0_ROOT/representative_cases/<experiment>_appendix_c0_representative_<approved>.pdf` | unsuffixed representative basename |

`<approved>` is either `with_endpoints` or `clean`. Record that choice and the
final release path in `submission/figure_provenance.csv` before installation.

## Vector And Visual QA

Audit only candidate submission assets, not the indexed review packet. The latter
may rasterize pages for convenient review and is not a manuscript asset.

```bash
python submission/pdf_vector_qa.py \
  "$FIGURE_ROOT/section6/summary_plots" \
  "$FIGURE_ROOT/section6/representative_cases" \
  "$FIGURE_ROOT/resolution" \
  "$FIGURE_ROOT/all_method_summary_plots" \
  "$FIGURE_ROOT/deterministic" \
  --json "$FIGURE_ROOT/pdf_vector_qa.json"

python submission/pdf_vector_qa.py \
  "$C0_ROOT/summary_plots" \
  "$C0_ROOT/representative_cases" \
  --json "$C0_ROOT/pdf_vector_qa.json"
```

Both commands must report `PDF QA: N/N passed`. A pass means zero image XObjects,
at least one font resource, and every reported font embedded. Type 1, Type 3, and
CID TrueType fonts are acceptable when Poppler reports `emb=yes`; CID TrueType is
preferred for newly generated Matplotlib text.

For visual inspection, rasterize copies from the authoritative PDFs:

```bash
mkdir -p "$FIGURE_ROOT/review_previews"
find "$FIGURE_ROOT" -type f -name '*.pdf' ! -path '*/review_previews/*' -print0 \
  | while IFS= read -r -d '' pdf; do
      stem="$(basename "${pdf%.pdf}")"
      pdftoppm -singlefile -png -r 300 "$pdf" \
        "$FIGURE_ROOT/review_previews/$stem"
    done
```

Inspect every preview for clipping, unreadable labels, inconsistent method colors,
overlapping spyglasses, missing corner diamonds, and endpoint-marker policy. Zoom
the authoritative PDF itself to at least `800%` to confirm paths and text remain
sharp. After author approval, copy only selected PDFs into the paper asset folder,
compile the manuscript and supplement, rerun `pdf_vector_qa.py` on every installed
asset, and update `submission/figure_provenance.csv`.
