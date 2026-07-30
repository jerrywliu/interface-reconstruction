# Post-Sweep Figure Regeneration Manifest

Updated: 2026-07-30

This checklist audits the active Overleaf manuscript at commit `4431912`
against `docs/PAPER_EXPERIMENT_MAP.md` and
`submission/FINAL_FIGURE_REGENERATION.md`. It covers all 26 live
`\includegraphics` PDF assets. The seven native TikZ schematics are listed
separately because they are source-native vector figures and do not depend on
the sweep.

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

## Known Documentation And Tooling Gaps

- `submission/FINAL_FIGURE_REGENERATION.md` currently regenerates appendix
  resolution strips from the primary sweep through
  `generate_section6_maintext_figures --figure_groups appendix_resolutions`.
  The live constants produce `N=32,64,100,150` for lines/circles/ellipses and
  `N=50,64,100,150` for squares/Zalesak, so those figures do not match the active
  `N=16,32,64` captions.
- The correct low-resolution entry point is
  `experiments.static.run_appendix_resolution_visuals`. Its default cases are
  stale for lines, squares, and Zalesak, so invoke it once per benchmark with
  the explicit case indices in Command R below.
- `run_appendix_resolution_visuals` does not yet accept an endpoint-variant
  option. Add paired export support before the final low-resolution run. The
  required stems are
  `<experiment>_resolution_cartesian_vs_perturbed_{with_endpoints,clean}.pdf`.
- `run_appendix_c0_study` likewise emits one endpoint-labeled representative.
  Add paired export support before the guarded-`C0` study. Metrics remain
  unpaired.
- The low-resolution and guarded-`C0` top-level manifests, and the two
  deterministic figure data JSON files, do not all record the source commit and
  frozen reconstruction profile explicitly. Add those fields (or write one
  enclosing generation manifest that records them and the input/output paths)
  before treating the regenerated packet as archival provenance.
- The active `C0` prose and captions contain historical pre-guard numerical
  summaries. Reconcile them against the new guarded study and make any
  manuscript changes in blue.
- The final-figure document's note that
  `submission_static_20260730_201510_df31a8d5f9b3` is running is stale; that was
  the aborted storage-layout attempt. Use only the completed release selected by
  the release audit.
- The July-baseline alias path in `docs/PAPER_EXPERIMENT_MAP.md` is unnecessary
  for a complete final release. Prefer the final-only symlink view in
  `submission/FINAL_FIGURE_REGENERATION.md`; it prevents historical geometry
  from silently filling a missing final bundle.

## Shared Setup And Gates

Run from the repository root after the final sweep has completed:

```bash
export FINAL_ROOT="$PWD/results/static/submission_static_<timestamp>_<commit>"
export SOURCE_COMMIT="$(python -c 'import json,os; print(json.load(open(os.environ["FINAL_ROOT"] + "/submission_config.resolved.json"))["source"]["target_commit"])')"
export FIGURE_ROOT="$PWD/results/submission/final_figures_${SOURCE_COMMIT:0:12}"
export PLOTS_VIEW="$FIGURE_ROOT/final_plots_view"
export C0_ROOT="$PWD/results/static/final_guarded_c0_${SOURCE_COMMIT:0:12}"
```

Before any figure command:

- run `python submission/audit_final_release.py "$FINAL_ROOT"`;
- require `completed`, `970/970` runs, `24,250/24,250` cases, zero failures,
  a clean captured source, and all 970 replayable raw bundles;
- build the final-only canonical symlink view using the snippet in
  `submission/FINAL_FIGURE_REGENERATION.md`;
- verify `SOURCE_COMMIT` is the checked-out commit for every dedicated run.

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

This command requires the paired-export extension noted above. Run benchmarks
separately so each manifest retains its explicit case selection:

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

Each benchmark requires six one-case raw runs: three resolutions times
Cartesian/perturbed (`w=0,0.1`). Preserve `manifest.json`, logs, and raw geometry.

### C: Guarded-`C0` Study And Paired Representatives

This command requires the paired-export extension noted above:

```bash
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
mkdir -p "$FIGURE_ROOT/deterministic"
python -m experiments.static.generate_plic_baseline_stencil_figure \
  --out "$FIGURE_ROOT/deterministic/perfect_reconstruction_plic_stencil" \
  --case-index 4 --cell-x 14 --cell-y 13 \
  --resolution 0.32 --wiggle 0.3 --seed 0
```

Preserve `perfect_reconstruction_plic_stencil_data.json` and the SVG sibling.

### D2: Deterministic Staged Reconstruction

```bash
python -m experiments.static.generate_staged_reconstruction_figure \
  --output-dir "$FIGURE_ROOT/deterministic" \
  --prefix staged_reconstruction_zalesak \
  --case-index 22 --resolution 1.0 --wiggle 0.1 --seed 0
```

Preserve `staged_reconstruction_zalesak_data.json` and the SVG sibling.

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

## Final QA And Installation Checklist

- [ ] Final release audit passes before plotting.
- [ ] Resolution and guarded-`C0` runners support paired PDF exports.
- [ ] Every command writes a manifest or data JSON identifying source commit,
      case, resolution, perturbation, seed, method, fallback, and profile.
- [ ] All 26 candidate manuscript PDFs pass:

  ```bash
  python submission/pdf_vector_qa.py \
    "$FIGURE_ROOT/section6/summary_plots" \
    "$FIGURE_ROOT/section6/representative_cases" \
    "$FIGURE_ROOT/all_method_summary_plots" \
    "$FIGURE_ROOT/resolution" \
    "$FIGURE_ROOT/deterministic" \
    "$C0_ROOT/summary_plots" \
    "$C0_ROOT/representative_cases" \
    --json "$FIGURE_ROOT/pdf_vector_qa.json"
  ```

- [ ] Every candidate has zero image XObjects, at least one font resource, and
      all reported fonts embedded.
- [ ] Rasterize review copies at 300 DPI and inspect every page for clipping,
      tiny text, inconsistent colors, legend/axis mismatch, overlapping
      spyglasses, missing corners, and endpoint-marker policy.
- [ ] Inspect authoritative PDFs at 800% zoom for path and text sharpness.
- [ ] Build an indexed review PDF with adjacent `with_endpoints` and `clean`
      candidates; the review packet itself may be rasterized but is never
      installed in the manuscript.
- [ ] Record author selections and exact source paths in
      `submission/figure_provenance.csv`.
- [ ] Copy only approved PDFs to the 26 unsuffixed manuscript filenames.
- [ ] If a clean variant is selected, revise any caption that says main-panel
      endpoints are visible; make every substantive manuscript diff blue.
- [ ] Reconcile all reported medians, convergence language, corner-tail
      language, and guarded-`C0` claims against final data in blue.
- [ ] Compile the manuscript and appendix, inspect every figure page, then run
      `pdf_vector_qa.py` on the installed camera-ready PDF directory.
- [ ] Generate and verify the final release checksum manifest only after the
      approved figure set is installed.

## Current Installed-Asset Baseline

A read-only audit of the active 26 PDFs produced `25/26` passes. Every asset but
`staged_reconstruction_zalesak.pdf` contains zero raster image objects and has
embedded fonts. The active staged-reconstruction PDF contains one raster image
object; deterministic regeneration through D2 is therefore mandatory even if
its visual content is otherwise approved. The corrected generator on the final
source commit uses vector patches for the formerly rasterized colorbar.
