# Visualization and Inspection Workflow

Updated: 2026-03-28

## Summary

We no longer need a separate VisIT-specific workflow for normal reconstruction inspection.
The current experiment scripts already generate the artifacts we need for day-to-day debugging and paper figure work:

- PNG plots under `plots/<save_name>/plt/`
- raw metrics under `plots/<save_name>/metrics/`
- VTK geometry under `plots/<save_name>/vtk/`
- paper-facing summary panels from merged CSVs
- representative reconstruction comparisons for static benchmarks

External VTK viewers such as ParaView or VisIT are still optional if you want them, but they are not required for the standard workflow.

## Output Layout

Most static runs write a directory of the form:

```text
plots/<save_name>/
  metrics/
  plt/
  vtk/
```

Typical contents:

- `metrics/`
  - scalar diagnostics such as `hausdorff.txt`, `facet_gap.txt`, `area_error.txt`
- `plt/`
  - Matplotlib renderings of area fractions, mixed-cell views, or experiment-specific comparison figures
- `vtk/`
  - mesh and interface geometry exports (`.vtk`, `.vtp`) for optional external inspection

## Common Workflows

### 1. Inspect a single static run

Run one case with a named output folder:

```bash
python -m experiments.static.circles \
  --config static/circle \
  --facet_algo circular \
  --save_name debug_circle_vis
```

Then inspect:

- plots:
  - `plots/debug_circle_vis/plt/`
- metrics:
  - `plots/debug_circle_vis/metrics/`
- optional VTK:
  - `plots/debug_circle_vis/vtk/`

The same pattern works for `lines`, `ellipses`, `squares`, and `zalesak`.

### 2. Re-plot saved sweep results without rerunning reconstruction

These commands rebuild the standard summary plots from existing results text files:

```bash
python -m experiments.static.lines --plot_only --results_file results/static/line_reconstruction_results.txt
python -m experiments.static.circles --plot_only --results_file results/static/circle_reconstruction_results.txt
python -m experiments.static.ellipses --plot_only --results_file results/static/ellipse_reconstruction_results.txt
python -m experiments.static.squares --plot_only --results_file results/static/square_reconstruction_results.txt
python -m experiments.static.zalesak --plot_only --results_file results/static/zalesak_reconstruction_results.txt
```

Use this path when the underlying runs already exist and you only want refreshed metric plots.

### 3. Regenerate all-method Section 6 panels from a merged CSV

Use the shared perturbed-sweep plotter in CSV-only mode:

```bash
python -m experiments.static.run_perturbed_sweeps \
  --plot_from_csv results/static/camera_ready/<bundle>/csv/section6_plotrefresh_merged.csv \
  --summary_dir results/static/debug/section6_summary_plots
```

This is the standard way to regenerate:

- `*_all_methods_2x2.png`
- `*_all_methods_5x2_axes.png`
- per-method summary panels

without rerunning the static experiments themselves.

### 4. Regenerate representative / main-text static figures

Use the Section 6 main-text figure generator:

```bash
python -m experiments.static.generate_section6_maintext_figures \
  --csv results/static/camera_ready/<bundle>/csv/section6_plotrefresh_merged.csv \
  --out_dir results/static/debug/section6_maintext
```

This produces:

- `summary_plots/*_maintext_metrics.png`
- `representative_cases/*_maintext_representative.png`
- `appendix_cases/*_best_by_resolution.png`
- `appendix_cases/*_cartesian_representative.png`

This is the normal path for paper-facing static figures.

### 5. Replay and visualize a deterministic Zalesak failure case

For targeted debugging, use the outlier replay helper:

```bash
python -m experiments.static.replay_zalesak_outlier \
  --outlier-csv results/static/camera_ready/<bundle>/outliers/zalesak_circularpluscorner_outliers.csv \
  --sort-by hausdorff \
  --row-index 0 \
  --debug-root results/static/debug/zalesak_replay
```

Or replay a specific deterministic case directly:

```bash
python -m experiments.static.replay_zalesak_outlier \
  --facet_algo "circular+corner" \
  --resolution 1.5 \
  --wiggle 0.3 \
  --seed 0 \
  --case_index 12 \
  --debug-root results/static/debug/zalesak_replay_manual
```

This is the preferred workflow for:

- before/after debugging images
- local corner/arc overlays
- inspecting the exact geometry that produced a bad metric outlier

## When To Use What

- Use `plots/<save_name>/plt/` when you want a fast local visual check of one run.
- Use `--plot_only` when the result text file already exists and you only want refreshed sweep plots.
- Use `run_perturbed_sweeps --plot_from_csv` when you want all-method summary panels from a merged CSV.
- Use `generate_section6_maintext_figures` when you want paper-style representative figures and compact main-text panels.
- Use `replay_zalesak_outlier` when you are debugging a specific mixed-feature failure.

## Relation To Camera-Ready Workflow

For release-style static figure bundling and Overleaf synchronization, see:

- `docs/STATIC_CAMERAREADY_WORKFLOW.md`

That document is about release packaging.
This document is about day-to-day local inspection and figure regeneration.

