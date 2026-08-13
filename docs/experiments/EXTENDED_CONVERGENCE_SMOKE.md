# Extended Convergence Smoke Test

## Purpose

This targeted study extends the circle and ellipse resolution checks to
`N = 256, 300, 512` without changing the frozen submission release. It is an
exploratory smoke test, not a replacement for the sealed result set.

The matrix contains five deterministic geometry cases (`0` through `4`) at
perturbation magnitudes `w = 0` and `w = 0.2`:

- circles: Youngs, ELVIRA, LVIRA, per-cell linear, and graph-coordinated linear;
- ellipses: graph-coordinated circular facets.

These method names follow the existing static-sweep definition of the linear
method family. All unresolved graph-coordination cases use LVIRA as the PLIC
fallback. The runner reads the production corner-behavior default directly from
`MergeMesh`.

## Run

From the repository root:

```bash
python -m experiments.static.run_extended_convergence_smoke --dry-run
python -m experiments.static.run_extended_convergence_smoke --workers 2
```

The default output is a new directory named
`results/static/extended_convergence_smoke_<timestamp>/`. The runner refuses to
write elsewhere or overwrite an existing directory. Each benchmark driver first
writes to a unique temporary `plots/<save_name>/` path; after successful
completion, that entire run bundle is moved into the smoke-test directory.

Use one worker if memory pressure is a concern:

```bash
python -m experiments.static.run_extended_convergence_smoke --workers 1
```

For a smaller diagnostic before the full matrix:

```bash
python -m experiments.static.run_extended_convergence_smoke \
  --cells-per-side 256 --wiggles 0 --case-indices 0 --workers 1
```

## Output Contract

```text
extended_convergence_smoke_<timestamp>/
  manifest.json
  run_status.csv
  case_metrics.csv
  summary_metrics.csv
  failures.csv                 # present only when failures occur
  logs/
  raw_runs/<save_name>/
```

`manifest.json` records the exact commands, source state, Python/platform
environment, final algorithm settings, per-run status, and a measured estimate
for expanding the same settings from five to 25 cases. It is rewritten
atomically after every completed subprocess.

`case_metrics.csv` combines the driver-reported Hausdorff and facet-gap metrics
with post-hoc conservation diagnostics. The conservation analysis reconstructs
the exact seeded mesh from each run manifest and clips the saved final facets
back into their original cells. Reported fields include global relative phase
area error, signed global residual, maximum fitted-component residual, maximum
cell-area-relative residual, missing facets, and fallback counts.

The expected numerical fit floor is recorded as `1e-10 / resolution`, matching
the existing Section 6 plotting diagnostic. Metric-to-floor ratios are included
for diagnosis; they are not additional accuracy claims.

`run_status.csv` separately counts recoverable arc-fit and orientation messages
printed by the existing drivers. These counts are solver-path diagnostics, not
run failures. A nonzero process return code, missing output bundle, missing case
row, incomplete conservation analysis, or traceback is recorded as a failure.

## Decisions And Limitations

- `N` maps to the static driver resolution as `resolution = N / 100`.
- The same perturbation seed (`0`) and the same five geometry indices are used
  at every setting, matching the paired structure of the paper sweep.
- No C0 post-processing is requested. Circle linear methods and ellipse
  circular reconstruction use their standard driver defaults.
- The runner does not alter shared benchmark drivers or submission outputs.
- The 25-case time estimate scales observed five-case subprocess wall time by
  five. This is conservative because mesh construction occurs once per run.

## Completed 2026-08-12 Smoke Run

The first complete run is
`results/static/extended_convergence_smoke_20260812_182921/`. All 36 settings
and 180 cases completed. There were no process, traceback, missing-facet, or
conservation-analysis failures. The complete numerical report is in that
directory's `README.md`; `manifest.json` and `environment.json` hold the full
commands and environment.

The graph-coordinated linear circle method remains approximately second order
in both metrics over these resolutions. For graph-coordinated circular ellipse
reconstruction, median Hausdorff order is `1.99` at `w=0` and `2.04` at
`w=0.2`; median facet-gap order is `2.43` and `2.63`, respectively. Eight of
the ten per-case facet-gap fits lie between `2.82` and `3.05`; the two lower
fits are case 2 (`1.83` at `w=0`, `2.63` at `w=0.2`). Thus this small
high-resolution smoke sample does not by itself reproduce the earlier aggregate
`N^{-2.94}` facet-gap fit.

The run began at runner commit `6c56dac`. Concurrent commits recorded by later
per-run manifests changed only revision-layout prototype files. Git tree IDs for
the circle/ellipse drivers, `main/`, `util/reconstruction.py`, and static configs
are identical across all three recorded commit IDs; the scientific source did
not change during the run.
