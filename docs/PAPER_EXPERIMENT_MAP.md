# Paper Experiment and Figure Map

Updated: 2026-08-03

This is the human-readable, section-by-section map from the current manuscript
to the code, configuration, result records, plotting commands, and expected
vector outputs. Run commands from the repository root. This document is the
detailed authority; `README.md` and `experiments/static/README.md` only point
here.

## Start Here: Regenerate Or Rerun?

There are three distinct operations. Do not use them interchangeably.

1. **Reuse the approved paper figures.** Copy the selected vector PDFs from the
   atomic figure publication. This is the fastest and most reproducible route
   when neither data nor plotting code changed.
2. **Regenerate plots from saved paper-facing inputs.** Use the aggregate CSV,
   selected geometry, manifests, and command records in the atomic figure
   publication. This reruns plotting only; it does not recompute the 24,250
   reconstruction cases.
3. **Rerun experiments.** Use the frozen scientific source commit and the final
   sweep launcher. This creates a new scientific release and requires a new
   figure publication and author approval. It is not a plot refresh.

The submitted scientific configuration is frozen as:

```text
corner behavior:       pre_f8_corner
rescue behavior:       exact_linear_support_only
unresolved PLIC:       LVIRA
scientific commit:     505aefa454328d4ba34ade5e7247050a0acfc793
figure-generator commit: 699987d978536e9f72748ead9abfc1bf7d00d27b
```

The first setting restores the validated pre-f8 corner acceptance/orientation
behavior. The second retains only propagation of an already accepted exact
linear support and disables the other corner-rescue heuristics. `LVIRA` is the
fallback for cells whose orientation remains unresolved.

Set the local artifact roots once:

```bash
export FINAL_ROOT="$PWD/results/static/submission_static_20260731_012430_505aefa45432.sealed"
export FIGURE_PUBLICATION="$PWD/results/submission/final_figures_699987d97853_20260801_final"
export PAPER_ROOT=/path/to/active-paper/interface-reconstruction-paper
export REGEN_ROOT="$PWD/results/reproduced-paper-figures"
```

`FINAL_ROOT` is the immutable 970-run scientific release. Its complete
`SHA256SUMS` has SHA-256
`9b5cda54e469ee01bc6a9078bbe22b568ed1e211d16080be806dfe9458ff0b1e`.
`FIGURE_PUBLICATION` is the immutable 41-candidate direct-vector publication.
The complete publication inventory and checksums are in
`review/figure_candidate_source_map.csv` and
`provenance/published_tree_sha256.json` beneath that root.

## Scientific Result Set

The exact machine-readable experiment authority is
`$FINAL_ROOT/submission_config.resolved.json`; the tracked
`submission/submission_config.json` is only a launcher template. The completed
command, counts, and output roots are recorded in
`$FINAL_ROOT/sweep_manifest.json`.

The final sweep covers five mesh perturbations (`0, 0.05, 0.1, 0.2, 0.3`),
mesh seed `0`, 25 deterministic geometries per setting, 970 runs, and 24,250
cases. Geometry streams are independent of the mesh seed: lines, squares, and
ellipses use seed `42`; circles use `41`; Zalesak uses `43`.

| Manuscript experiment | Driver and config | Resolution grid | Methods | Primary result records |
| --- | --- | --- | --- | --- |
| Lines: representable straight interfaces | `experiments.static.lines`, `config/static/line.yaml` | `N=32,50,64,100,128,150` | Youngs, ELVIRA, LVIRA, `safe_linear`, `linear` | aggregate rows plus case Hausdorff and facet-gap metrics |
| Squares: line-line corners | `experiments.static.squares`, `config/static/square.yaml` | `N=50,64,100,128,150` | Youngs, ELVIRA, LVIRA, `safe_linear`, `linear`, `linear+corner`, `safe_circle`, `circular` | aggregate rows plus case Hausdorff, facet gap, and area error |
| Circles: constant curvature | `experiments.static.circles`, `config/static/circle.yaml` | `N=32,50,64,100,128,150` | Youngs, ELVIRA, LVIRA, `safe_linear`, `linear`, `safe_circle`, `circular` | aggregate rows plus case Hausdorff, tangent, gap, and curvature metrics |
| Ellipses: varying curvature | `experiments.static.ellipses`, `config/static/ellipse.yaml` | `N=32,50,64,100,128,150` | same as circles | aggregate rows, case metrics, and fitted convergence orders |
| Zalesak: arcs, slot edges, and corners | `experiments.static.zalesak`, `config/static/zalesak.yaml` | `N=50,64,100,128,150` | Youngs, ELVIRA, LVIRA, `safe_linear`, `linear`, `safe_circle`, `circular`, `circular+corner` | aggregate rows plus case Hausdorff, facet gap, and area error |

The final release contract is:

```text
$FINAL_ROOT/
  SHA256SUMS
  submission_config.resolved.json
  sweep_manifest.json
  failures.csv
  perturbed_sweep.csv
  raw_runs/<save_name>/run_manifest.json
  raw_runs/<save_name>/metrics/{case_geometry.jsonl,case_metrics.csv,cell_metrics.csv,merge_events.csv}
  raw_runs/<save_name>/vtk/...
  diagnostics/{run_inventory.csv,run_manifests.jsonl,case_geometry.jsonl,case_metrics.csv,cell_metrics.csv,merge_events.csv,unresolved_plic_fallbacks.csv}
```

`perturbed_sweep.csv` is the aggregate plotting input. Use
`diagnostics/case_metrics.csv` for case-indexed convergence and failures, and
`diagnostics/run_inventory.csv` to resolve a setting to its release-relative
raw bundle. The cell, merge, and fallback tables are private diagnostic depth,
not required inputs for the paper's aggregate plots.

### Rerun The Full Scientific Sweep

Create a clean checkout at the scientific commit, verify it, and launch the
single final controller:

```bash
git worktree add --detach /tmp/interface-scientific-505aefa \
  505aefa454328d4ba34ade5e7247050a0acfc793
cd /tmp/interface-scientific-505aefa
python submission/check_submission_freeze.py --source-only
bash submission/run_final_static_sweep.sh
```

The launcher creates a new collision-proof result root. Expected completion is
970 successful runs, 24,250 cases, zero failures, a resolved configuration,
one sweep manifest, the aggregate CSV, raw run bundles, consolidated
diagnostics, and a complete checksum ledger. Never overwrite or append to the
sealed release named above.

For one targeted setting, use the corresponding benchmark module. For example:

```bash
python -m experiments.static.zalesak \
  --config static/zalesak \
  --resolution 1.0 \
  --facet_algo 'circular+corner' \
  --mesh_type perturbed_quads \
  --perturb_wiggle 0.1 \
  --perturb_seed 0 \
  --perturb_fix_boundary 1 \
  --num_cases 25 \
  --plic_fallback LVIRA \
  --corner_behavior_profile pre_f8_corner \
  --rescue_profile exact_linear_support_only \
  --save_name <new-unique-name>
```

The expected targeted output is `plots/<new-unique-name>/`, including
`run_manifest.json`, `metrics/`, and `vtk/`.

## Plotting Commands

The atomic publication records the exact commands actually executed under
`provenance/*/command.json`. The commands here are portable equivalents. Their
outputs are local review artifacts until a fresh fail-closed figure publication
and author approval are completed.

### P1: Main Quantitative And Representative Panels

```bash
python -m experiments.static.generate_section6_maintext_figures \
  --csv "$FIGURE_PUBLICATION/provenance/release_input_snapshot/perturbed_sweep.csv" \
  --plots_root "$FIGURE_PUBLICATION/provenance/release_input_snapshot/plots" \
  --out_dir "$REGEN_ROOT/maintext" \
  --experiments all \
  --figure_groups quantitative,representative \
  --case_overrides lines=6,squares=24,circles=12,ellipses=12,zalesak=12 \
  --endpoint_variants paired
```

Inputs and exact selections are recorded in
`$FIGURE_PUBLICATION/provenance/maintext/maintext_manifest.json` and its
`release_run_manifests/` directory. Expected outputs are five quantitative PDFs,
ten endpoint-variant representative PDFs, 300-DPI review PNGs, and a new
`maintext_manifest.json` beneath `$REGEN_ROOT/maintext`.

The representative run bundles consumed by P1 are exact, compact copies under
`$FIGURE_PUBLICATION/provenance/release_input_snapshot/plots/`:

- lines: `perturb_sweep_lines_{youngs,elvira,lvira,linear}_r0p32_w0p3_s0`, case 6;
- squares: `perturb_sweep_squares_{elvira,lvira,linear,linearpluscorner}_r0p5_w0p1_s0`, case 24;
- circles: `perturb_sweep_circles_{elvira,lvira,linear,circular}_r0p32_w0p1_s0`, case 12;
- ellipses: `perturb_sweep_ellipses_{elvira,lvira,linear,circular}_r0p32_w0p1_s0`, case 12;
- Zalesak: `perturb_sweep_zalesak_{elvira,lvira,circular,circularpluscorner}_r1p0_w0p1_s0`, case 12.

Each copied bundle contains the run manifest, case metrics and geometry, mesh,
truth geometry, reconstructed facet VTP, and facet metadata used by the panel.

### P2: Appendix All-Method Panels

```bash
python -m experiments.static.run_perturbed_sweeps \
  --plot_from_csv "$FIGURE_PUBLICATION/provenance/release_input_snapshot/perturbed_sweep.csv" \
  --summary_dir "$REGEN_ROOT/all-methods" \
  --no-notify
```

The command writes the five allowlisted all-method vector PDFs plus auxiliary
summary plots. The exact executed command is
`$FIGURE_PUBLICATION/provenance/all_methods/command.json`.

### P3: Appendix Resolution Studies

These are 30 companion runs, not rows in the 970-run release. The five exact
commands are recorded in
`$FIGURE_PUBLICATION/provenance/resolution/<benchmark>/command.json`; each
family also has `manifest.json`, six `run_manifests/`, and six compact `inputs/`
bundles. To rerun one family from the scientific source commit:

```bash
python -m experiments.static.run_appendix_resolution_visuals \
  --only <benchmark> \
  --case_index <case> \
  --resolutions 0.16,0.32,0.64 \
  --wiggles 0,0.1 \
  --save_prefix final_resolution_<benchmark> \
  --endpoint_variants <paired-or-paired_with_hybrid_endpoints_n16_n32> \
  --out_dir "$REGEN_ROOT/resolution/<benchmark>"
```

Use cases 0, 22, 12, 12, and 6 for lines, squares, circles, ellipses, and
Zalesak, respectively. Lines, ellipses, and Zalesak use the hybrid endpoint
mode; squares and circles use paired mode. Add `--plot_only --plots_root
<materialized-input-view>` to redraw from saved companion bundles without
rerunning reconstruction. Expected outputs are the selected vector panel,
alternative endpoint variants, review PNGs, and `manifest.json`.

### P4: Guarded C0 Aggregate Study

The aggregate study is separate from the primary pre-C0 release. Its 165 fresh
runs compare linear, guarded linear C0, and circular fits for ellipses, and
circular, guarded circular C0, and circular-plus-corner fits for Zalesak.

To regenerate the aggregate panels from the saved aggregate tables:

```bash
python -m experiments.static.run_appendix_c0_study \
  --only ellipses \
  --plot_from_csv "$FIGURE_PUBLICATION/provenance/guarded_c0/ellipses/metrics.csv" \
  --out_dir "$REGEN_ROOT/c0/ellipses" \
  --endpoint_variants paired

python -m experiments.static.run_appendix_c0_study \
  --only zalesak \
  --plot_from_csv "$FIGURE_PUBLICATION/provenance/guarded_c0/zalesak/metrics.csv" \
  --out_dir "$REGEN_ROOT/c0/zalesak" \
  --endpoint_variants paired
```

The exact 90-run ellipse and 75-run Zalesak contracts are in the adjacent
`manifest.json` files. Recreating representative geometry requires either the
selected compact run bundles described under "Compact Public Release" or a
fresh P4 rerun; aggregate CSVs alone are insufficient. In plot-from-CSV mode,
the current script also attempts the representative panels. Without matching
`--plots_root` bundles it can emit missing-VTK warnings and placeholder
representative PDFs. Only the aggregate file under `summary_plots/` is eligible
from the two commands above.

| Figure slot | Generated vector candidate |
| --- | --- |
| `ellipses_appendix_c0_metrics` | `$FIGURE_PUBLICATION/candidates/c0_root/summary_plots/ellipses_appendix_c0_2x2.pdf` |
| `ellipses_appendix_c0_representative` | Connected-component case 9 candidates under `$FIGURE_PUBLICATION/candidates/c0_root/representative_cases/ellipses_appendix_c0_representative_{with_endpoints,clean}.pdf` |
| `zalesak_appendix_c0_metrics` | `$FIGURE_PUBLICATION/candidates/c0_root/summary_plots/zalesak_appendix_c0_2x2.pdf` |
| `zalesak_appendix_c0_representative` | Globally continuous guarded case 22 candidates under `$FIGURE_PUBLICATION/candidates/c0_root/representative_cases/zalesak_appendix_c0_representative_{with_endpoints,clean}.pdf` |

The author selected the clean connected-component ellipse case 9 and clean
globally continuous Zalesak case 22 representatives on 2026-08-03. The immutable
publication is `final_figures_87c40309d16c_20260803_final`; exact candidate IDs,
paths, and checksums for all 26 selected slots are recorded in
`submission/approved_final_figures_20260803.csv`. The two paper files still
contain those exact published bytes: ellipse was promoted to Overleaf at
`1ea4d3e`, Zalesak at `b593f5d`, and both are in standalone paper commit
`d2d9548`. Historical pre-guard `C0` assets are ineligible.

### P5: Deterministic Explanatory Figures

```bash
python -m experiments.static.generate_plic_baseline_stencil_figure \
  --out "$REGEN_ROOT/deterministic/perfect_reconstruction_plic_stencil" \
  --case-index 4 --cell-x 14 --cell-y 13 \
  --resolution 0.32 --wiggle 0.3 --seed 0

python -m experiments.static.generate_staged_reconstruction_figure \
  --case-index 22 --resolution 1.0 --wiggle 0.1 --seed 0 \
  --radius 15 --slot-width 5 --slot-top-rel 10 \
  --output-dir "$REGEN_ROOT/deterministic/staged" \
  --prefix staged_reconstruction_zalesak
```

Both commands write a direct-vector PDF, a review PNG, and JSON/SVG provenance
siblings. The exact executed commands and JSON inputs are in
`$FIGURE_PUBLICATION/provenance/deterministic/`.

### P6: Connected-Component C0 Extension And Approved Representatives

The manuscript's connected-component ellipse results are diagnostic
postprocessing; they do not change the frozen production reconstruction. The
two paper-facing sweeps use guarded linear C0 inputs at `N=32` and `N=64`,
perturbation `0.1`, mesh seed `0`, and all 25 cases:

```bash
python -m experiments.submission.ellipse_joint_c0_posthoc \
  --baseline-root plots/audit_c0_full_ellipse_linear_n32_w010_20260801 \
  --run-name ellipse_joint_c0_posthoc_n32_w010_20260801 \
  --output-dir results/submission/ellipse_joint_c0_posthoc_n32_w010_20260801 \
  --resolution 0.32 --perturb-wiggle 0.1 --perturb-seed 0

python -m experiments.submission.ellipse_joint_c0_posthoc \
  --baseline-root plots/audit_c0_full_ellipse_linear_n64_w010_20260801 \
  --run-name ellipse_joint_c0_posthoc_n64_w010_20260801 \
  --output-dir results/submission/ellipse_joint_c0_posthoc_n64_w010_20260801 \
  --resolution 0.64 --perturb-wiggle 0.1 --perturb-seed 0
```

Each output directory contains `case_summary.csv`, `component_summary.csv`, a
vector summary, a representative packet, and a README. The processed facet
bundles are written to `plots/<run-name>/`.

The approved globally continuous representative pair is audited by:

```bash
python -m experiments.submission.generate_c0_replacement_representatives \
  --ellipse-case 9 --zalesak-case 22 \
  --output-dir "$REGEN_ROOT/c0-approved-representatives"
```

The expected result is an ellipse case-9 direct-vector panel plus a manifest
that binds it and the retained Zalesak case-22 panel to exact run manifests,
facet files, SHA-256 digests, continuity checks, and conservation checks. These
two representatives require inclusion in the next atomic publication before
the final 26-row submission approval ledger can be sealed.

The `699987d97853` candidates
`ellipses_appendix_c0_representative_clean` and
`zalesak_appendix_c0_representative_clean` are superseded guarded-C0 choices.
Do not select them for the final approval ledger; use ellipse case 9 and
Zalesak case 22 from the replacement manifest above.

## Current Manuscript PDF Asset Map

The table contains every active generated PDF include exactly once. `Selected
source` is a candidate ID in the atomic publication except for the two approved
C0 replacements, which are bound by the P6 manifest. `Input / manifest` and
`command` identify the shortest reproducible path.

### Main Text

| Paper role | Paper asset | Selected source | Input / manifest | Command and expected output |
| --- | --- | --- | --- | --- |
| PLIC stencil comparison | `perfect_reconstruction_plic_stencil.pdf` | `perfect_reconstruction_plic_stencil` | deterministic JSON and command record | P5; deterministic vector panel |
| Staged Zalesak reconstruction | `staged_reconstruction_zalesak.pdf` | `staged_reconstruction_zalesak` | deterministic JSON and command record | P5; staged vector panel |
| Lines quantitative | `line_reconstruction_maintext_metrics.pdf` | `lines_maintext_metrics` | aggregate CSV | P1; `summary_plots/lines_maintext_metrics.pdf` |
| Lines representative | `line_reconstruction_maintext_representative.pdf` | `lines_maintext_representative_clean` | P1 line bundles, case 6 | P1; clean representative PDF |
| Squares quantitative | `square_reconstruction_maintext_metrics.pdf` | `squares_maintext_metrics` | aggregate CSV | P1; `summary_plots/squares_maintext_metrics.pdf` |
| Squares representative | `square_reconstruction_maintext_representative.pdf` | `squares_maintext_representative_clean` | P1 square bundles, case 24 | P1; clean representative PDF |
| Circles quantitative | `circle_reconstruction_maintext_metrics.pdf` | `circles_maintext_metrics` | aggregate CSV | P1; `summary_plots/circles_maintext_metrics.pdf` |
| Circles representative | `circle_reconstruction_maintext_representative.pdf` | `circles_maintext_representative_clean` | P1 circle bundles, case 12 | P1; clean representative PDF |
| Ellipses quantitative | `ellipse_reconstruction_maintext_metrics.pdf` | `ellipses_maintext_metrics` | aggregate CSV | P1; `summary_plots/ellipses_maintext_metrics.pdf` |
| Ellipses representative | `ellipse_reconstruction_maintext_representative.pdf` | `ellipses_maintext_representative_clean` | P1 ellipse bundles, case 12 | P1; clean representative PDF |
| Zalesak quantitative | `zalesak_reconstruction_maintext_metrics.pdf` | `zalesak_maintext_metrics` | aggregate CSV | P1; `summary_plots/zalesak_maintext_metrics.pdf` |
| Zalesak representative | `zalesak_reconstruction_maintext_representative.pdf` | `zalesak_maintext_representative_clean` | P1 Zalesak bundles, case 12 | P1; clean representative PDF |

### Appendix Resolution Studies

| Paper role | Paper asset | Selected source | Input / manifest | Command and expected output |
| --- | --- | --- | --- | --- |
| Lines, case 0 | `lines_resolution_cartesian_vs_perturbed.pdf` | `lines_resolution_hybrid_endpoints_n16_n32` | resolution lines manifest plus six inputs | P3; hybrid vector panel |
| Squares, case 22 | `squares_resolution_cartesian_vs_perturbed.pdf` | `squares_resolution_with_endpoints` | resolution squares manifest plus six inputs | P3; endpoint vector panel |
| Circles, case 12 | `circles_resolution_cartesian_vs_perturbed.pdf` | `circles_resolution_with_endpoints` | resolution circles manifest plus six inputs | P3; endpoint vector panel |
| Ellipses, case 12 | `ellipses_resolution_cartesian_vs_perturbed.pdf` | `ellipses_resolution_hybrid_endpoints_n16_n32` | resolution ellipses manifest plus six inputs | P3; hybrid vector panel |
| Zalesak, case 6 | `zalesak_resolution_cartesian_vs_perturbed.pdf` | `zalesak_resolution_hybrid_endpoints_n16_n32` | resolution Zalesak manifest plus six inputs | P3; hybrid vector panel |

### Appendix Full Panels And C0 Study

| Paper role | Paper asset | Selected source | Input / manifest | Command and expected output |
| --- | --- | --- | --- | --- |
| Lines all methods | `line_reconstruction_perturbed_all_methods_2x2.pdf` | `lines_all_methods` | aggregate CSV and all-method command record | P2; lines all-method vector panel |
| Squares all methods | `square_reconstruction_perturbed_all_methods_2x2.pdf` | `squares_all_methods` | aggregate CSV and all-method command record | P2; squares all-method vector panel |
| Circles all methods | `circle_reconstruction_perturbed_all_methods_5x2_axes.pdf` | `circles_all_methods` | aggregate CSV and all-method command record | P2; circles all-method vector panel |
| Ellipses all methods | `ellipse_reconstruction_perturbed_all_methods_5x2_axes.pdf` | `ellipses_all_methods` | aggregate CSV and all-method command record | P2; ellipses all-method vector panel |
| Zalesak all methods | `zalesak_reconstruction_perturbed_all_methods_2x2.pdf` | `zalesak_all_methods` | aggregate CSV and all-method command record | P2; Zalesak all-method vector panel |
| Ellipse guarded-C0 metrics | `ellipses_appendix_c0_2x2.pdf` | `ellipses_appendix_c0_metrics` | guarded-C0 ellipse metrics and manifest | P4; ellipse C0 vector metrics panel |
| Ellipse connected-C0 representative | `ellipses_appendix_c0_representative.pdf` | approved case 9 from P6 | replacement manifest plus three selected facet bundles | P6; clean connected-C0 representative |
| Zalesak guarded-C0 metrics | `zalesak_appendix_c0_2x2.pdf` | `zalesak_appendix_c0_metrics` | guarded-C0 Zalesak metrics and manifest | P4; Zalesak C0 vector metrics panel |
| Zalesak globally continuous representative | `zalesak_appendix_c0_representative.pdf` | approved case 22 from P6 | replacement manifest plus selected facet bundles | P6; clean globally continuous representative |

## Numerical Claims Without Dedicated Paper Figures

Some manuscript results are prose or table values rather than independent
figure slots. They still have executable provenance.

| Manuscript claim | Entry point | Exact input | Expected records |
| --- | --- | --- | --- |
| Ellipse fitted convergence orders | `python -m experiments.submission.analyze_ellipse_convergence` | `$FINAL_ROOT/diagnostics/case_metrics.csv` | points CSV, fits CSV, report JSON, vector diagnostic PDF |
| Connected-component ellipse C0 extension at N=32 and N=64 | P6 | guarded-C0 baseline bundles named in P6 | two case summaries, two component summaries, representative/summary PDFs |
| Final line/circle/square/Zalesak perfect-reconstruction audit | `submission/audits/final_perfect_reconstruction_2026-07-31/README.md` and its CSVs | sealed aggregate and case metrics | method/setting tables and vector audit panel |
| Five-benchmark conservation smoke | `experiments.submission.materialize_final_conservation_selection`, then `experiments.submission.conservation_analyzer` | sealed release plus checksum ledger | verified selection JSON, global/merged/base-cell residual tables |
| Shared-vertex and fallback incidence used for algorithm sanity checks | `experiments.submission.topology_consistency_diagnostics` | sealed cell metrics and run inventory | incidence tables, conflict cases, manifest |

These diagnostics may consume the private full release. Only the compact tables
that support statements retained in the manuscript need to accompany a public
paper archive.

## Author-Drawn TikZ Figure Map

These seven active figures are compiled directly by the manuscript and are not
outputs of a numerical experiment. Release-data provenance does not apply. All
seven depend on `interface-reconstruction.tex`, which loads `graphicx`, `tikz`,
and the `arrows.meta`, `calc`, and `positioning` TikZ libraries.

| Active figure | Manuscript include | Author-drawn source | Shared style dependency | Status |
| --- | --- | --- | --- | --- |
| Regular topology cases (`fig:regular_cases`) | `new_sections/topology_identification.tex` | `figs/tikz/topology_regular_cases.tex` | `figs/tikz/topology_styles.tex` | **Open:** author visual and caption review. |
| Merging cases (`fig:merging_ambiguous_cases`) | `new_sections/topology_identification.tex` | `figs/tikz/topology_merging_cases.tex` | `figs/tikz/topology_styles.tex` | **Open:** author visual and caption review. |
| Orientation graph (`fig:orientation_dependencies`) | `new_sections/topology_identification.tex` | `figs/tikz/topology_orientation_dependencies.tex` | `figs/tikz/topology_styles.tex` | **Open:** author visual and caption review. |
| Linear fitting (`fig:linear_facet_fitting`) | `new_sections/appendix/algorithms/linear_facets.tex` | `figs/tikz/appendix_linear_facet_fitting.tex` | `figs/tikz/algorithm_styles.tex` | **Open:** blue-caption author review. |
| Circular fitting (`fig:circular_facet_fitting`) | `new_sections/appendix/algorithms/circular_facets.tex` | `figs/tikz/appendix_circular_facet_fitting.tex` | `figs/tikz/algorithm_styles.tex` | **Open:** blue-caption author review. |
| Polygon-circle area (`fig:circle_quad_intersect`) | `new_sections/appendix/algorithms/circular_facets.tex` | `figs/tikz/appendix_circle_intersect_area.tex` | `figs/tikz/algorithm_styles.tex` | **Open:** blue-caption author review. |
| Corner fitting (`fig:corner_facet_fitting`) | `new_sections/appendix/algorithms/corner_facets.tex` | `figs/tikz/appendix_corner_facet_fitting.tex` | `figs/tikz/algorithm_styles.tex` | **Open:** blue-caption author review. |

## Manuscript Table Map

`FINAL_ROOT` is the sealed release defined above; it is the configuration
authority for every executable table check.

| Active table | Manuscript source | Code/data provenance | Verification |
| --- | --- | --- | --- |
| Methods comparison (`tab:methods_compare`) | `new_sections/problem_setup.tex` | `$FINAL_ROOT/submission_config.resolved.json` plus the cited literature; author curated | `PAPER_ROOT="$PAPER_ROOT" FINAL_ROOT="$FINAL_ROOT" python -m pytest -q test/submission/test_paper_experiment_map.py::test_methods_table_gate` |
| Numerical parameters (`tab:reconstruction_parameters`) | `new_sections/appendix/algorithms.tex` | `$FINAL_ROOT/submission_config.resolved.json`, `config/static/base.yaml`, `main/structs/polys/base_polygon.py`, `main/structs/polys/neighbored_polygon.py`, `main/geoms/circular_facet.py`, and `main/geoms/linear_facet.py`; LVIRA stops when `step < 1e-6` | `PAPER_ROOT="$PAPER_ROOT" FINAL_ROOT="$FINAL_ROOT" python -m pytest -q test/submission/test_paper_experiment_map.py::test_numerical_parameters_table_gate` |
| Benchmark geometry and sampling (`tab:static_benchmark_definitions`) | `new_sections/appendix/static_benchmarks/overview.tex` | resolved config, five benchmark drivers, and per-run manifests | `PAPER_ROOT="$PAPER_ROOT" FINAL_ROOT="$FINAL_ROOT" python -m pytest -q test/submission/test_paper_experiment_map.py::test_benchmark_table_gate` |

## Compact Public Release

The public paper archive does **not** need the complete 970-run `raw_runs/`
tree or the multi-million-row cell/merge diagnostics. Keep the full sealed
release privately and bind any later full-data deposit with its `SHA256SUMS`.

The compact archive should include:

- the scientific-source and figure-generator source snapshots;
- this paper map, environment files, and exact frozen commits;
- the resolved configuration, sweep manifest, aggregate CSV, run inventory,
  and checksum ledgers;
- selected direct-vector paper PDFs and the 26-row approval ledger;
- candidate source maps and executed plotting command records;
- the compact main-representative geometry bundles already under
  `provenance/release_input_snapshot/plots/`;
- all 30 resolution run manifests and their selected compact geometry inputs;
- guarded-C0 aggregate metrics/manifests, the two approved representative
  manifests and selected facet inputs, and the four connected-C0 case/component
  summary CSVs.

It may omit:

- unselected raw run bundles;
- full cell-level, merge-event, and fallback-event diagnostics;
- intermediate VTK/PNG review output not consumed by a paper figure;
- historical March/May/July comparison bundles.

The current `submission/package_submission.py` deliberately stages only the
aggregate result table and provenance bindings, while the complete raw release
is expected at an external deposit. If the team chooses not to deposit that
full release, add the compact figure-input items above to the final archive and
state that complete cell-level diagnostics are available on reasonable request.

## Verification

Check the scientific release and publication before using any bytes:

```bash
python submission/audit_final_release.py "$FINAL_ROOT" --verify-sha256-manifest
python - "$FIGURE_PUBLICATION" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
ledger = json.loads((root / "provenance/published_tree_sha256.json").read_text())
rows = ledger["files"] if isinstance(ledger, dict) else ledger
for row in rows:
    path = root / row["path"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == row["sha256"], path
print(f"verified {len(rows)} publication files")
PY
```

Run the repository map audit against the active paper:

```bash
PAPER_ROOT="$PAPER_ROOT" FINAL_ROOT="$FINAL_ROOT" \
  python -m pytest -q test/submission/test_paper_experiment_map.py
```

The audit verifies all 26 active PDF includes, all seven TikZ figures and style
dependencies, the methods table, the numerical-parameter table, and the
benchmark-definition table. Before submission, also verify that every promoted
PDF has zero raster image objects and no unembedded fonts.

## Remaining Mapping Gates

- **Broad third-order wording is unresolved.** The current mapped evidence is the
  ellipse convergence analysis. It supports robust third-order behavior for
  facet-gap error, while prior final-data analysis found lower empirical orders
  for Hausdorff and tangent errors. Before submission, the authors must either
  narrow the abstract/introduction/method-overview wording to the demonstrated
  metric and benchmark or add separately mapped evidence for a broader claim.
  Any manuscript change remains blue until approved.
- **Guarded-`C0` representative publication and promotion passed.** The author
  chose connected-component ellipse case 9 and guarded Zalesak case 22. Both
  checksum-pinned clean candidates are installed in Overleaf and the standalone
  paper; `4/4` promoted copies passed vector QA. Any later manuscript change
  remains blue until approved.
- **The reproducibility seed sentence is ambiguous.** It currently says only
  that the perturbed-mesh experiments use seed `0`. A blue revision should state
  that `0` is the mesh-perturbation seed and separately report geometry-stream
  seeds `42` (lines/squares/ellipses), `41` (circles), and `43` (Zalesak).
- **Author-drawn figure approval is open.** Explicitly approve the captions and
  visual content of the seven TikZ figures inventoried above. The four appendix
  captions are already blue; any needed changes to the three topology captions
  must also remain blue until approved.
- Record the exact paper commit audited for figures, tables, and prose in the
  review packet or approval ledger at review time using
  `git -C <paper-worktree> rev-parse HEAD`. Durable documentation must not pin a
  paper commit that becomes stale on the next writing edit.

- Publish the approved ellipse case-9 and Zalesak case-22 C0 representatives in
  a new immutable figure publication, then update the 26-row approval ledger.
- Decide whether the public artifact is the complete sealed release or the
  compact archive defined above. The current author preference is compact data
  plus an "available on reasonable request" statement for large cell-level
  diagnostics.
- Pin the final paper commit only after blue manuscript review is complete.
