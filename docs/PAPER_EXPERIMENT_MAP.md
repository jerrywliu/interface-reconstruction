# Paper Experiment and Figure Map

Updated: 2026-07-30

This is the paper-to-code index for the static reconstruction results in
Section 6 and its appendix. Run commands from the repository root. The target
submission outputs are vector PDFs; PNG files are review previews only.

## Result Sources

| Tier | Location | Use |
| --- | --- | --- |
| Historical paper assets | March/May 2026 camera-ready bundles | Layout comparison only. Do not use as final numerical provenance. |
| July candidate | `results/static/static_paper_simplified_default_20260717_212413/` | Reviewed candidate: 300 affected-method runs, 7,500 cases. Its source snapshot is reproducible but dirty, and its historical square/Zalesak `area_error` columns must not be promoted. |
| Final release | `results/static/submission_static_<UTC>_<12-char-commit>/` | Pending canonical source: 970 all-method runs, 24,250 cases, clean source commit, read-only raw bundles, and consolidated diagnostics. This supersedes July wherever data or figures are regenerated. |

The machine-readable specification is `submission/submission_config.json`.
Figure promotion status and the paper filename contract are in
`submission/figure_provenance.csv` and `submission/FIGURE_PROMOTION_PLAN.md`.

Set these shell variables when a final release exists:

```bash
RELEASE=results/static/submission_static_<UTC>_<12-char-commit>
JULY=results/static/static_paper_simplified_default_20260717_212413
FIGURES="$RELEASE/figures"
```

## Final Sweep

The approved launcher checks for committed source, records the exact commit,
creates a collision-proof namespace, and archives every raw run bundle inside
the release:

```bash
python submission/check_submission_freeze.py --source-only
bash submission/run_final_static_sweep.sh
```

The equivalent controller dry run was verified at exactly 970 runs and 24,250
cases. The launcher fixes:

- perturbation magnitudes `0, 0.05, 0.1, 0.2, 0.3`;
- seed `0` and 25 deterministic cases per setting;
- `pre_f8_corner + exact_linear_support_only + LVIRA`;
- five workers and pre-`C0` primary benchmark output.

Each final release has this contract:

```text
$RELEASE/
  submission_config.resolved.json
  sweep_manifest.json
  failures.csv
  perturbed_sweep.csv
  summary_plots/*.pdf
  logs/
  raw_runs/<save_name>/
  diagnostics/
    source_state.json
    source_snapshot.tar.gz
    run_inventory.csv
    run_manifests.jsonl
    case_geometry.jsonl
    case_metrics.csv
    cell_metrics.csv
    merge_events.csv
    unresolved_plic_fallbacks.csv
```

`case_metrics.csv` is the case-indexed source for tails and convergence;
`cell_metrics.csv` and `merge_events.csv` contain cell, facet, merge, rescue,
and final-primitive provenance. `run_inventory.csv` maps each row to a
release-relative raw bundle.

Compare a completed final release with the July affected-method baseline using
`submission/compare_release_results.py`. The utility matches exact case keys,
reports the expected candidate-only methods separately, and writes CSV, JSON,
and Markdown summaries without modifying either release. See
`submission/COMPARE_RELEASE_RESULTS.md` for the command and interpretation.

## Benchmark Map

The CLI config name in commands is `static/<name>`; its file is
`config/static/<name>.yaml`. Resolution `0.32` means `N=32` cells per side.

| Benchmark and paper role | Driver / config | Final grid and methods | Main metrics |
| --- | --- | --- | --- |
| Lines: linear perfect reconstruction | `experiments.static.lines` / `static/line` | `N=32,50,64,100,128,150`; Youngs, ELVIRA, LVIRA, `safe_linear`, `linear` | Hausdorff, facet gap |
| Squares: line-line corners | `experiments.static.squares` / `static/square` | `N=50,64,100,128,150`; Youngs, ELVIRA, LVIRA, `safe_linear`, `linear`, `linear+corner`, `safe_circle`, `circular` | Hausdorff, facet gap; area error is secondary |
| Circles: constant curvature | `experiments.static.circles` / `static/circle`; radius 10 | `N=32,50,64,100,128,150`; Youngs, ELVIRA, LVIRA, `safe_linear`, `linear`, `safe_circle`, `circular` | Hausdorff, tangent error; gap/curvature in full panel |
| Ellipses: varying curvature and convergence | `experiments.static.ellipses` / `static/ellipse`; major axis 30 | Same grid/methods as circles | Hausdorff, tangent error; gap/curvature and fitted orders |
| Zalesak: arcs, slot edges, corners, and merging | `experiments.static.zalesak` / `static/zalesak`; radius 15, slot width 5 | `N=50,64,100,128,150`; Youngs, ELVIRA, LVIRA, `safe_linear`, `linear`, `safe_circle`, `circular`, `circular+corner` | Hausdorff, facet gap; area error is secondary |

To rerun one setting, use the corresponding driver. For example:

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
  --save_name <unique-save-name>
```

The raw result is `plots/<unique-save-name>/`; final sweeps additionally copy
it to `$RELEASE/raw_runs/<unique-save-name>/`.

## Figure Regeneration

### Prepare final geometry aliases

Paper figure code uses stable, profile-free run names. Build a review view that
maps those names to the namespaced final raw bundles. The July merged CSV is a
key/name contract here, not the promoted numerical source:

```bash
python -m experiments.static.prepare_figure_review \
  --run_root "$RELEASE" \
  --baseline_csv "$JULY/figure_review/current_run_section6_merged.csv" \
  --current_plots_root "$RELEASE/raw_runs" \
  --archive_plots_root "$JULY/figure_review/plots_union"
```

Inspect `$RELEASE/figure_review/review_data_manifest.json`. Final rows should
replace every paper-grid key that is present in the final sweep. The resulting
geometry root is `$RELEASE/figure_review/plots_union`.

### Main-text panels

```bash
python -m experiments.static.generate_section6_maintext_figures \
  --csv "$RELEASE/perturbed_sweep.csv" \
  --plots_root "$RELEASE/figure_review/plots_union" \
  --out_dir "$FIGURES/section6" \
  --figure_groups quantitative,representative \
  --endpoint_variants paired
```

| Family | Generated vector PDF | Paper asset |
| --- | --- | --- |
| Lines metrics | `section6/summary_plots/lines_maintext_metrics.pdf` | `line_reconstruction_maintext_metrics.pdf` |
| Lines representative | `section6/representative_cases/lines_maintext_representative_{with_endpoints,clean}.pdf` | `line_reconstruction_maintext_representative.pdf` |
| Squares metrics | `section6/summary_plots/squares_maintext_metrics.pdf` | `square_reconstruction_maintext_metrics.pdf` |
| Squares representative | `section6/representative_cases/squares_maintext_representative_{with_endpoints,clean}.pdf` | `square_reconstruction_maintext_representative.pdf` |
| Circles metrics | `section6/summary_plots/circles_maintext_metrics.pdf` | `circle_reconstruction_maintext_metrics.pdf` |
| Circles representative | `section6/representative_cases/circles_maintext_representative_{with_endpoints,clean}.pdf` | `circle_reconstruction_maintext_representative.pdf` |
| Ellipses metrics | `section6/summary_plots/ellipses_maintext_metrics.pdf` | `ellipse_reconstruction_maintext_metrics.pdf` |
| Ellipses representative | `section6/representative_cases/ellipses_maintext_representative_{with_endpoints,clean}.pdf` | `ellipse_reconstruction_maintext_representative.pdf` |
| Zalesak metrics | `section6/summary_plots/zalesak_maintext_metrics.pdf` | `zalesak_reconstruction_maintext_metrics.pdf` |
| Zalesak representative | `section6/representative_cases/zalesak_maintext_representative_{with_endpoints,clean}.pdf` | `zalesak_reconstruction_maintext_representative.pdf` |

The generator writes `$FIGURES/section6/maintext_manifest.json`, including the
case and method selections. Current representative candidates are lines 6,
squares 24, circles 12, ellipses 12, and Zalesak 12.

### Full all-method panels

```bash
python -m experiments.static.run_perturbed_sweeps \
  --plot_from_csv "$RELEASE/perturbed_sweep.csv" \
  --summary_dir "$FIGURES/all_methods" \
  --no-notify
```

| Generated vector PDF | Paper asset |
| --- | --- |
| `all_methods/lines_all_methods_2x2.pdf` | `line_reconstruction_perturbed_all_methods_2x2.pdf` |
| `all_methods/squares_all_methods_2x2.pdf` | `square_reconstruction_perturbed_all_methods_2x2.pdf` |
| `all_methods/circles_all_methods_5x2_axes.pdf` | `circle_reconstruction_perturbed_all_methods_5x2_axes.pdf` |
| `all_methods/ellipses_all_methods_5x2_axes.pdf` | `ellipse_reconstruction_perturbed_all_methods_5x2_axes.pdf` |
| `all_methods/zalesak_all_methods_2x2.pdf` | `zalesak_reconstruction_perturbed_all_methods_2x2.pdf` |

### Resolution studies

These are deterministic companion runs at `N=16,32,64`, not rows in the
970-run release. Run each benchmark separately so its manifest remains intact:

```bash
python -m experiments.static.run_appendix_resolution_visuals \
  --only <benchmark> \
  --case_index <case> \
  --out_dir "$FIGURES/resolution/<benchmark>"
```

| Benchmark / candidate case | Generated and paper asset |
| --- | --- |
| Lines / 0 | `lines_resolution_cartesian_vs_perturbed.pdf` |
| Squares / 22 | `squares_resolution_cartesian_vs_perturbed.pdf` |
| Circles / 12 | `circles_resolution_cartesian_vs_perturbed.pdf` |
| Ellipses / 12 | `ellipses_resolution_cartesian_vs_perturbed.pdf` |
| Zalesak / 20 | `zalesak_resolution_cartesian_vs_perturbed.pdf` |

These case choices remain an author-approval gate. Preserve each generated
`manifest.json`, logs, and the associated `plots/<save-prefix>_*` geometry.

### Deterministic explanatory figures

```bash
python -m experiments.static.generate_plic_baseline_stencil_figure \
  --out "$FIGURES/deterministic/perfect_reconstruction_plic_stencil"

python -m experiments.static.generate_staged_reconstruction_figure \
  --output-dir "$FIGURES/deterministic" \
  --prefix staged_reconstruction_zalesak
```

The paper assets are `perfect_reconstruction_plic_stencil.pdf` and
`staged_reconstruction_zalesak.pdf`. Their sibling JSON/SVG outputs are the
important provenance files.

## Representative Variant Contract

`--endpoint_variants paired` produces two direct vector PDFs:

- `with_endpoints`: open circles mark every reconstructed facet endpoint;
- `clean`: main-panel endpoint/cell-crossing circles are hidden, but endpoints
  remain labeled in spyglass zooms and semantic reconstructed corners remain
  diamonds everywhere.

Only the author-selected variant is renamed to the unsuffixed paper asset.
Record the choice in the figure approval CSV and `submission/figure_provenance.csv`.

## Supporting Validation Map

| Claim / experiment | Entry point and command | Data and provenance |
| --- | --- | --- |
| Ellipse empirical orders | `python -m experiments.submission.analyze_ellipse_convergence --case-metrics "$RELEASE/diagnostics/case_metrics.csv" --output-dir "$RELEASE/validation/convergence"` | `ellipse_convergence_{points,fits}.csv`, report JSON, vector PDF |
| Shared-vertex incidence | `python -m experiments.submission.topology_consistency_diagnostics --full --cell-metrics "$RELEASE/diagnostics/cell_metrics.csv" --run-inventory "$RELEASE/diagnostics/run_inventory.csv" --output-dir "$RELEASE/validation/topology"` | paper table, tolerance table, case/conflict CSVs, compressed vertex rows, manifest |
| Exact conflict diagnosis | `python -m experiments.submission.diagnose_topology_conflicts --source-dir "$RELEASE/validation/topology" --output-dir "$RELEASE/validation/topology_diagnosis"` | taxonomy, incident facets, case counts, manifest, vector examples |
| Independent cells vs topology+merging | `python -m experiments.submission.run_topology_merging_ablation --output-dir "$RELEASE/validation/topology_merging"` | 10 jobs / 250 Zalesak cases; case/summary CSVs, manifest, vector PDF |
| Guarded optional `C0` | `python -m experiments.submission.c0_conservation_validation --num-cases 25 --output "$RELEASE/validation/c0_conservation"` | matched 100-case off/on data, eligible joins, regressions, report, manifest |
| Five-benchmark conservation smoke | `python -m experiments.submission.conservation_analyzer --selection <final-selection.json> --output "$RELEASE/validation/conservation"` | per-case global, merged-zone, and constituent-cell residuals. The checked-in selection JSON points to July and must be rewritten to final raw bundles. |
| Zalesak remote-corner tail | `python -m experiments.submission.zalesak_failure.diagnose_zalesak_failure --run-root <release-or-July-root> --artifact-root "$RELEASE/validation/zalesak_failure"` | archive-only case-23 diagnostic PDF/SVG, entities/comparison CSVs, provenance JSON, README |

The blue review manuscript intentionally reports guarded `C0` as text only.
The four older assets `ellipses_appendix_c0_2x2.pdf`,
`ellipses_appendix_c0_representative.pdf`, `zalesak_appendix_c0_2x2.pdf`, and
`zalesak_appendix_c0_representative.pdf` are historical and must not return
unless regenerated from the guarded implementation.

## Verification Notes

- CLI help was checked for every command above.
- Controller dry-run: 970 runs / 24,250 cases.
- Topology/merging and resolution-study dry runs emitted the expected commands.
- `python submission/check_submission_freeze.py --source-only` reported
  `SOURCE CLEAN` before this documentation commit.
- Matplotlib paper generators emit direct vector PDF siblings with embedded
  TrueType fonts. Promote those PDFs, not raster previews.
- `experiments.static.build_figure_review_pdf` is optional review tooling and
  requires `reportlab`, which is not currently declared in `requirements.txt`.
