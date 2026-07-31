# Post-Sweep Figure Regeneration Manifest

Updated: 2026-07-31

This checklist audits the active Overleaf figure paths and labels against
`docs/PAPER_EXPERIMENT_MAP.md` and
`submission/FINAL_FIGURE_REGENERATION.md`. It covers all 26 live
`\includegraphics` PDF assets, all seven native TikZ schematics, and all three
active manuscript tables. At each review, record the then-current paper commit
in the generated review/approval ledger with
`git -C <paper-worktree> rev-parse HEAD`; do not pin a paper commit in this
durable checklist.

Do not install any candidate in the manuscript until the final release, the
candidate figure packet, and the selected endpoint variants have been approved.

## Decisions From The Audit

1. **Resolution panels use `N=16,32,64`.** This is the contract in the active
   captions and `docs/PAPER_EXPERIMENT_MAP.md`. These rows are absent from the
   970-run release and require deterministic companion runs from the same frozen
   source commit.
2. **The completed final release is the numerical source of truth.** Once it
   passes the release audit, use its CSV and raw bundles for metrics, all-method
   panels, and main representatives. July 2026 remains comparison/recovery
   provenance only; March/May assets remain layout references only.
3. **Dedicated post-sweep runs inherit the final source commit.** The low-
   resolution, guarded-`C0`, PLIC-stencil, and staged-reconstruction figures are
   generated from that exact commit and preserve their own manifests/data JSON.
4. **Paired variants are required for qualitative review.** Main
   representatives, resolution panels, and guarded-`C0` representatives need
   both `with_endpoints` and `clean` PDFs. Clean main panels retain endpoint and
   cell-crossing markers in spyglasses and semantic corner diamonds everywhere.
   Quantitative and explanatory figures are not paired.
5. **PDF is authoritative.** Every installed asset must have zero image XObjects
   and all reported fonts embedded. PNG siblings are 300-DPI review previews,
   never manuscript inputs.
6. **Review and installation have different counts.** The complete review set is
   38 PDFs: 14 unpaired figures plus both variants for 12 qualitative slots.
   Author selection reduces those 24 paired candidates to 12, leaving exactly
   26 unsuffixed PDFs for manuscript installation.

## Integrated Tooling And Remaining Review Work

- `submission/FINAL_FIGURE_REGENERATION.md` now uses dedicated companion runs for
  the active `N=16,32,64` captions. Do not substitute primary-sweep resolution
  strips, whose grids differ by benchmark.
- The correct low-resolution entry point is
  `experiments.static.run_appendix_resolution_visuals`. Its default cases are
  stale for lines, squares, and Zalesak, so invoke it once per benchmark with
  the explicit case indices in Command R below.
- `run_appendix_resolution_visuals` and `run_appendix_c0_study` now accept
  `--endpoint_variants paired`. The required qualitative stems are
  `<experiment>_resolution_cartesian_vs_perturbed_{with_endpoints,clean}.pdf`.
  Guarded-`C0` metrics remain unpaired.
- Low-resolution and guarded-`C0` manifests, plus both deterministic figure data
  JSON files, now record the source commit and frozen reconstruction profile.
- The active `C0` prose and captions contain historical pre-guard numerical
  summaries. Reconcile them against the new guarded study and make any
  manuscript changes in blue.
- `submission_static_20260730_201510_df31a8d5f9b3` was the aborted
  storage-layout attempt. Use only the completed release selected by the release
  audit.
- The July-baseline alias path in `docs/PAPER_EXPERIMENT_MAP.md` is unnecessary
  for a complete final release. Prefer the final-only symlink view in
  `submission/FINAL_FIGURE_REGENERATION.md`; it prevents historical geometry
  from silently filling a missing final bundle.

## Shared Setup And Gates

Run from the repository root after the final sweep has completed:

```bash
export FINAL_ROOT="$PWD/results/static/submission_static_20260731_012430_505aefa45432"
export SOURCE_COMMIT="$(python -c 'import json,os; print(json.load(open(os.environ["FINAL_ROOT"] + "/submission_config.resolved.json"))["source"]["target_commit"])')"
export FIGURE_ROOT="$PWD/results/submission/final_figures_${SOURCE_COMMIT:0:12}"
export PLOTS_VIEW="$FIGURE_ROOT/final_plots_view"
export C0_ROOT="$PWD/results/static/final_guarded_c0_${SOURCE_COMMIT:0:12}"
```

Before any figure command:

- require `FINAL_ROOT/SHA256SUMS` and run
  `python submission/audit_final_release.py "$FINAL_ROOT" --verify-sha256-manifest`;
- require `completed`, `970/970` runs, `24,250/24,250` cases, zero failures,
  a clean captured source, and all 970 replayable raw bundles;
- build the final-only canonical symlink view using the snippet in
  `submission/FINAL_FIGURE_REGENERATION.md`;
- run the following exact gate immediately before every dedicated command R, C,
  D1, and D2:

  ```bash
  test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
  python submission/check_submission_freeze.py --source-only
  ```

## Command Catalog

### M: Main Metrics

```bash
python -m experiments.static.generate_section6_maintext_figures \
  --csv "$FINAL_ROOT/perturbed_sweep.csv" \
  --plots_root "$PLOTS_VIEW" \
  --out_dir "$FIGURE_ROOT/section6" \
  --figure_groups quantitative \
  --experiments all
```

### P: Paired Main Representatives

```bash
python -m experiments.static.generate_section6_maintext_figures \
  --csv "$FINAL_ROOT/perturbed_sweep.csv" \
  --plots_root "$PLOTS_VIEW" \
  --out_dir "$FIGURE_ROOT/section6" \
  --figure_groups representative \
  --endpoint_variants paired \
  --case_overrides lines=6,squares=24,circles=12,ellipses=12,zalesak=12
```

### A: All-Method Panels

```bash
python -m experiments.static.run_perturbed_sweeps \
  --plot_from_csv "$FINAL_ROOT/perturbed_sweep.csv" \
  --summary_dir "$FIGURE_ROOT/all_method_summary_plots" \
  --no-notify
```

### R: Paired `N=16,32,64` Resolution Companions

Run benchmarks separately so each manifest retains its explicit case selection:

```bash
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
python submission/check_submission_freeze.py --source-only

python -m experiments.static.run_appendix_resolution_visuals --only lines \
  --case_index 0 --resolutions 0.16,0.32,0.64 --wiggles 0,0.1 \
  --save_prefix "final_resolution_${SOURCE_COMMIT:0:12}_lines" \
  --endpoint_variants paired --out_dir "$FIGURE_ROOT/resolution/lines"
python -m experiments.static.run_appendix_resolution_visuals --only squares \
  --case_index 22 --resolutions 0.16,0.32,0.64 --wiggles 0,0.1 \
  --save_prefix "final_resolution_${SOURCE_COMMIT:0:12}_squares" \
  --endpoint_variants paired --out_dir "$FIGURE_ROOT/resolution/squares"
python -m experiments.static.run_appendix_resolution_visuals --only circles \
  --case_index 12 --resolutions 0.16,0.32,0.64 --wiggles 0,0.1 \
  --save_prefix "final_resolution_${SOURCE_COMMIT:0:12}_circles" \
  --endpoint_variants paired --out_dir "$FIGURE_ROOT/resolution/circles"
python -m experiments.static.run_appendix_resolution_visuals --only ellipses \
  --case_index 12 --resolutions 0.16,0.32,0.64 --wiggles 0,0.1 \
  --save_prefix "final_resolution_${SOURCE_COMMIT:0:12}_ellipses" \
  --endpoint_variants paired --out_dir "$FIGURE_ROOT/resolution/ellipses"
python -m experiments.static.run_appendix_resolution_visuals --only zalesak \
  --case_index 20 --resolutions 0.16,0.32,0.64 --wiggles 0,0.1 \
  --save_prefix "final_resolution_${SOURCE_COMMIT:0:12}_zalesak" \
  --endpoint_variants paired --out_dir "$FIGURE_ROOT/resolution/zalesak"
```

Each benchmark requires six one-case raw runs: three resolutions times
Cartesian/perturbed (`w=0,0.1`). Preserve `manifest.json`, logs, and raw geometry.

### C: Guarded-`C0` Study And Paired Representatives

Run the paired representative export with the guarded study:

```bash
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
python submission/check_submission_freeze.py --source-only

test ! -e "$C0_ROOT"
python -m experiments.static.run_appendix_c0_study \
  --out_dir "$C0_ROOT" \
  --out_csv "$C0_ROOT/csv/appendix_c0_sweep.csv" \
  --log_dir "$C0_ROOT/logs" \
  --save_prefix "final_guarded_c0_${SOURCE_COMMIT:0:12}" \
  --seeds 0 --ellipses 25 --zalesak 25 \
  --endpoint_variants paired
```

Require all 165 settings and their raw bundles. Confirm the guarded behavior in
the run provenance: an infeasible endpoint refit retains the conservative
pre-`C0` facet.

### D1: Deterministic PLIC Stencil

```bash
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
python submission/check_submission_freeze.py --source-only

mkdir -p "$FIGURE_ROOT/deterministic"
python -m experiments.static.generate_plic_baseline_stencil_figure \
  --out "$FIGURE_ROOT/deterministic/perfect_reconstruction_plic_stencil" \
  --case-index 4 --cell-x 14 --cell-y 13 \
  --resolution 0.32 --wiggle 0.3 --seed 0
```

Preserve `perfect_reconstruction_plic_stencil_data.json` and the SVG sibling.

### D2: Deterministic Staged Reconstruction

```bash
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
python submission/check_submission_freeze.py --source-only

python -m experiments.static.generate_staged_reconstruction_figure \
  --output-dir "$FIGURE_ROOT/deterministic" \
  --prefix staged_reconstruction_zalesak \
  --case-index 22 --resolution 1.0 --wiggle 0.1 --seed 0
```

Preserve `staged_reconstruction_zalesak_data.json` and the SVG sibling. This
candidate must replace the currently installed PDF, whose color key is raster.

## Main-Text Asset Manifest

`Pair` means both candidates are generated and only the approved one is copied
to the unsuffixed manuscript filename.

| Manuscript include | Source command | Data prerequisite | Candidate output(s) | Pair | Figure-specific QA |
| --- | --- | --- | --- | --- | --- |
| `perfect_reconstruction_plic_stencil.pdf` | D1 | Frozen source commit; deterministic line case 4, center cell `(14,13)`, `N=32`, `w=0.3`, seed 0 | `deterministic/perfect_reconstruction_plic_stencil.pdf` | No | Three methods and normalized errors agree with data JSON; zero raster objects |
| `staged_reconstruction_zalesak.pdf` | D2 | Frozen source commit; Zalesak case 22, `N=100`, `w=0.1`, seed 0 | `deterministic/staged_reconstruction_zalesak.pdf` | No | Six stages, merge outline, primitive colors, and counts agree with data JSON; zero raster objects |
| `line_reconstruction_maintext_metrics.pdf` | M | Final `perturbed_sweep.csv`, complete lines grid | `section6/summary_plots/lines_maintext_metrics.pdf` | No | Four named methods, Hausdorff/gap axes, medians and IQR |
| `line_reconstruction_maintext_representative.pdf` | P | Final raw bundles; case 6, `N=32`, `w=0.3`, seed 0 | `section6/representative_cases/lines_maintext_representative_{with_endpoints,clean}.pdf` | Yes | Inset outside geometry; clean panel hides main endpoints but inset retains them |
| `square_reconstruction_maintext_metrics.pdf` | M | Final `perturbed_sweep.csv`, complete squares grid | `section6/summary_plots/squares_maintext_metrics.pdf` | No | Four named methods, Hausdorff/gap axes, corner method present |
| `square_reconstruction_maintext_representative.pdf` | P | Final raw bundles; case 24, `N=50`, `w=0.1`, seed 0 | `section6/representative_cases/squares_maintext_representative_{with_endpoints,clean}.pdf` | Yes | Exactly one diamond per reconstructed corner; cell-boundary crossings are circles; spyglass does not overlap geometry |
| `circle_reconstruction_maintext_metrics.pdf` | M | Final `perturbed_sweep.csv`, complete circles grid | `section6/summary_plots/circles_maintext_metrics.pdf` | No | Four named methods, Hausdorff/tangent axes, solver-floor guide correct |
| `circle_reconstruction_maintext_representative.pdf` | P | Final raw bundles; case 12, `N=32`, `w=0.1`, seed 0 | `section6/representative_cases/circles_maintext_representative_{with_endpoints,clean}.pdf` | Yes | Arc geometry and endpoint policy are legible at 300-DPI review scale |
| `ellipse_reconstruction_maintext_metrics.pdf` | M | Final `perturbed_sweep.csv`, complete ellipses grid | `section6/summary_plots/ellipses_maintext_metrics.pdf` | No | Four named methods, Hausdorff/tangent axes, no stale third-order claim |
| `ellipse_reconstruction_maintext_representative.pdf` | P | Final raw bundles; case 12, `N=32`, `w=0.1`, seed 0 | `section6/representative_cases/ellipses_maintext_representative_{with_endpoints,clean}.pdf` | Yes | Coarse facets remain distinguishable; endpoint policy matches selected caption |
| `zalesak_reconstruction_maintext_metrics.pdf` | M | Final `perturbed_sweep.csv`, complete Zalesak grid | `section6/summary_plots/zalesak_maintext_metrics.pdf` | No | Four named methods, Hausdorff/gap axes, coarse-tail wording matches final data |
| `zalesak_reconstruction_maintext_representative.pdf` | P | Final raw bundles; case 12, `N=100`, `w=0.1`, seed 0 | `section6/representative_cases/zalesak_maintext_representative_{with_endpoints,clean}.pdf` | Yes | One diamond per corner, crossing circles on corner branches, nonoverlapping spyglass, no missing facets |

## Appendix Asset Manifest

| Manuscript include | Source command | Data prerequisite | Candidate output(s) | Pair | Figure-specific QA |
| --- | --- | --- | --- | --- | --- |
| `line_reconstruction_perturbed_all_methods_2x2.pdf` | A | Final `perturbed_sweep.csv`, complete lines grid | `all_method_summary_plots/lines_all_methods_2x2.pdf` | No | Complete method set, consistent colors/limits, Hausdorff/gap only |
| `square_reconstruction_perturbed_all_methods_2x2.pdf` | A | Final `perturbed_sweep.csv`, complete squares grid | `all_method_summary_plots/squares_all_methods_2x2.pdf` | No | Complete method set, consistent colors/limits, corner methods present |
| `circle_reconstruction_perturbed_all_methods_5x2_axes.pdf` | A | Final `perturbed_sweep.csv`, complete circles grid | `all_method_summary_plots/circles_all_methods_5x2_axes.pdf` | No | Use four metric rows only: Hausdorff, gap, curvature, tangent |
| `ellipse_reconstruction_perturbed_all_methods_5x2_axes.pdf` | A | Final `perturbed_sweep.csv`, complete ellipses grid | `all_method_summary_plots/ellipses_all_methods_5x2_axes.pdf` | No | Use four metric rows only: Hausdorff, gap, curvature, tangent |
| `zalesak_reconstruction_perturbed_all_methods_2x2.pdf` | A | Final `perturbed_sweep.csv`, complete Zalesak grid | `all_method_summary_plots/zalesak_all_methods_2x2.pdf` | No | Complete method set, consistent colors/limits, no hidden failed cases |
| `lines_resolution_cartesian_vs_perturbed.pdf` | R | Final-commit companion runs; case 0, `N=16,32,64`, `w=0,0.1` | `resolution/lines/summary_plots/lines_resolution_cartesian_vs_perturbed_{with_endpoints,clean}.pdf` | Yes | Six panels and caption N values agree; no historical geometry |
| `squares_resolution_cartesian_vs_perturbed.pdf` | R | Final-commit companion runs; case 22, `N=16,32,64`, `w=0,0.1` | `resolution/squares/summary_plots/squares_resolution_cartesian_vs_perturbed_{with_endpoints,clean}.pdf` | Yes | Six panels, one diamond per corner, crossings visible, no missed selected corner |
| `circles_resolution_cartesian_vs_perturbed.pdf` | R | Final-commit companion runs; case 12, `N=16,32,64`, `w=0,0.1` | `resolution/circles/summary_plots/circles_resolution_cartesian_vs_perturbed_{with_endpoints,clean}.pdf` | Yes | Six panels, stable framing, endpoint policy consistent |
| `ellipses_resolution_cartesian_vs_perturbed.pdf` | R | Final-commit companion runs; case 12, `N=16,32,64`, `w=0,0.1` | `resolution/ellipses/summary_plots/ellipses_resolution_cartesian_vs_perturbed_{with_endpoints,clean}.pdf` | Yes | Six panels, stable framing, endpoint policy consistent |
| `zalesak_resolution_cartesian_vs_perturbed.pdf` | R | Final-commit companion runs; case 20, `N=16,32,64`, `w=0,0.1` | `resolution/zalesak/summary_plots/zalesak_resolution_cartesian_vs_perturbed_{with_endpoints,clean}.pdf` | Yes | Six panels; inspect under-resolution honestly; one corner diamond per tip; spyglasses outside geometry |
| `ellipses_appendix_c0_2x2.pdf` | C | Complete guarded-`C0` CSV and raw bundles | `C0_ROOT/summary_plots/ellipses_appendix_c0_2x2.pdf` | No | New medians replace historical prose values; guarded failures are not presented as exact closure |
| `ellipses_appendix_c0_representative.pdf` | C | Guarded-`C0` case 12, `N=32`, `w=0.1`, seed 0 | `C0_ROOT/representative_cases/ellipses_appendix_c0_representative_{with_endpoints,clean}.pdf` | Yes | Endpoint policy and selected caption agree; conservative retained facets visible if present |
| `zalesak_appendix_c0_2x2.pdf` | C | Complete guarded-`C0` CSV and raw bundles | `C0_ROOT/summary_plots/zalesak_appendix_c0_2x2.pdf` | No | New medians and conservation audit replace historical claims; continuity described as conditional |
| `zalesak_appendix_c0_representative.pdf` | C | Guarded-`C0` case 12, `N=100`, `w=0.1`, seed 0 | `C0_ROOT/representative_cases/zalesak_appendix_c0_representative_{with_endpoints,clean}.pdf` | Yes | Corner diamonds/crossings correct, spyglass clear, selected caption matches markers |

## Native TikZ Figures

These seven included figures are already vector source and are not regenerated
from sweep data:

- main text: `topology_regular_cases.tex`, `topology_merging_cases.tex`, and
  `topology_orientation_dependencies.tex`;
- appendix: `appendix_linear_facet_fitting.tex`,
  `appendix_circular_facet_fitting.tex`, `appendix_circle_intersect_area.tex`,
  and `appendix_corner_facet_fitting.tex`.

Their gate is a clean manuscript compile and visual inspection at 800% zoom.

## Active Manuscript Tables

| Table label | Manuscript source | Reproducibility source | Gate |
| --- | --- | --- | --- |
| `tab:methods_compare` | `new_sections/problem_setup.tex` | Literature classification plus the frozen method/fallback contract in `submission/submission_config.json` | Author checks citations and conditional checkmarks; no sweep-generated values |
| `tab:reconstruction_parameters` | `new_sections/appendix/algorithms.tex` | `config/static/base.yaml`, `main/structs/polys/{base_polygon,neighbored_polygon}.py`, and `main/geoms/circular_facet.py` at the final source commit | Every displayed threshold matches the pinned source snapshot |
| `tab:static_benchmark_definitions` | `new_sections/appendix/static_benchmarks/overview.tex` | `experiments/static/{lines,squares,circles,ellipses,zalesak}.py`, `config/static/*.yaml`, final run manifests, and `submission/submission_config.json` | Geometry ranges, seed, and 25-case sampling match the sealed release |

## Author And Provenance Gates

- **Broad third-order claim:** unresolved. The mapped ellipse convergence
  evidence supports third-order facet-gap behavior, not a blanket third-order
  statement for Hausdorff and tangent error. The authors must narrow the broad
  abstract/introduction/method-overview wording or approve additional mapped
  evidence. All resulting paper changes stay blue until approved.
- **Guarded-`C0` prose:** unresolved. The active appendix numerical values and
  captions must be reconciled against the regenerated 165-setting guarded study
  and its conservation audit. All resulting paper changes stay blue until
  approved.
- The exact paper commit reviewed for the 26 PDFs, seven TikZ figures, three
  tables, and these two prose gates belongs in the review packet or approval
  ledger, not in this file.

## Final QA And Installation Checklist

- [ ] Final release audit passes before plotting.
- [x] Resolution and guarded-`C0` runners support paired PDF exports.
- [x] Dedicated companion commands write a manifest or data JSON identifying
      source commit, case, resolution, perturbation, seed, method, fallback, and
      profile.
- [ ] Run the authoritative allowlist, provenance, vector-PDF, 300-DPI preview,
      and indexed-review-packet workflow in
      `submission/FINAL_FIGURE_REGENERATION.md`. Its dedicated orchestrator owns
      the exact 38-candidate contract. Do not substitute recursive directory
      discovery or duplicate that acceptance logic here.
- [ ] Inspect every orchestrator-produced preview for clipping, unreadable text,
      inconsistent colors, legend/axis mismatch, overlapping spyglasses,
      missing corners, and endpoint-marker policy.
- [ ] Inspect authoritative PDFs at 800% zoom for path and text sharpness.
- [ ] Review the orchestrator-produced indexed packet with adjacent
      `with_endpoints` and `clean` candidates; it is never installed in the
      manuscript.
- [ ] Record author selections and exact source paths in
      `submission/figure_provenance.csv`.
- [ ] Select one PDF for each of the 12 paired qualitative slots, then copy the
      resulting 26 approved PDFs to the 26 unsuffixed manuscript filenames.
- [ ] If a clean variant is selected, revise any caption that says main-panel
      endpoints are visible; make every substantive manuscript diff blue.
- [ ] Reconcile all reported medians, convergence language, corner-tail
      language, and guarded-`C0` claims against final data in blue.
- [ ] Audit all three active tables against the sources listed above.
- [ ] Resolve the broad third-order and guarded-`C0` author gates above; retain
      blue diffs until approval.
- [ ] Compile the manuscript and appendix, inspect every figure page, then run
      `pdf_vector_qa.py` on the installed camera-ready PDF directory.
- [ ] Verify the sealed numerical release checksum manifest before figure
      generation. Figure acceptance and manuscript installation retain their
      separate provenance/checksum ledgers.

## Current Installed-Asset Baseline

A read-only audit of the active 26 PDFs produced `25/26` passes. Every asset but
`staged_reconstruction_zalesak.pdf` contains zero raster image objects and has
embedded fonts. The active staged-reconstruction PDF contains one raster image
object and is explicitly ineligible for final installation; replace it with the
D2 output even if its visual content is otherwise approved. The corrected
generator on the final source commit uses vector patches for the formerly
rasterized colorbar.
