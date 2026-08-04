# Static Reconstruction Experiments

This package contains the five benchmark drivers and shared plotting tools used
for the paper's static reconstruction study.

## Supported Drivers

| Benchmark | Module | Config | Paper role |
|---|---|---|---|
| Lines | `experiments.static.lines` | `static/line` | Straight-interface exactness |
| Squares | `experiments.static.squares` | `static/square` | Line-line corners |
| Circles | `experiments.static.circles` | `static/circle` | Constant curvature |
| Ellipses | `experiments.static.ellipses` | `static/ellipse` | Varying curvature and convergence |
| Zalesak | `experiments.static.zalesak` | `static/zalesak` | Arcs, slot edges, corners, and merging |

Paper-facing oriented reconstruction uses `LVIRA` as the unresolved PLIC
fallback. The frozen corner method uses
`--corner_behavior_profile pre_f8_corner` and
`--rescue_profile exact_linear_support_only`.

## Targeted Run

Run from the repository root and choose a unique output name:

```bash
python -m experiments.static.zalesak \
  --config static/zalesak \
  --resolution 1.0 \
  --facet_algo 'circular+corner' \
  --mesh_type perturbed_quads \
  --perturb_wiggle 0.1 \
  --perturb_seed 0 \
  --perturb_fix_boundary 1 \
  --case_indices 0 \
  --plic_fallback LVIRA \
  --corner_behavior_profile pre_f8_corner \
  --rescue_profile exact_linear_support_only \
  --save_name targeted_zalesak
```

The local output is `plots/targeted_zalesak/`. It contains metric, VTK, plot,
case-geometry, cell, merge, fallback, and reconstruction provenance according to
the requested diagnostics.

## Sweep And Figure Entry Points

| Task | Entry point |
|---|---|
| Development Cartesian sweep | `python -m experiments.static.run_linear_sweeps` |
| Shared perturbed sweep controller | `python -m experiments.static.run_perturbed_sweeps` |
| Frozen full submission sweep | `bash submission/run_final_static_sweep.sh` |
| Main-text static figures | `python -m experiments.static.generate_section6_maintext_figures` |
| Appendix resolution panels | `python -m experiments.static.run_appendix_resolution_visuals` |
| Guarded-continuity appendix study | `python -m experiments.static.run_appendix_c0_study` |
| Final allowlisted figure publication | `submission/run_final_figure_orchestrator` |

Manual figure commands are diagnostic. Only the final orchestrator binds
candidates to the sealed release, pinned source commit, allowlist, and external
approval record.

## Exact Paper Map

`docs/PAPER_EXPERIMENT_MAP.md` lists every active paper figure, benchmark row,
input artifact, producer command, expected output, and verification gate. Use it
instead of reconstructing a run recipe from filenames.

## Historical Workflows

Superseded per-shape wrappers, the March camera-ready pipeline, affected-row
launchers, and dated ablations were moved under `archive/`. They remain tracked
for provenance and must not be used to generate new paper results. The move map
is in `archive/README.md`.
