# QUASI Five-Benchmark Cartesian Smoke

## Scope

This is the first matched project-benchmark run of the frozen Cartesian QUASI
port. It uses the canonical project cases `0--4` for lines, squares, circles,
ellipses, and Zalesak at `N = 32, 64, 128`. No policy was changed after looking
at these results.

Run:

```bash
MPLBACKEND=Agg PYTHONPATH=. python -m \
  experiments.baselines.run_quasi_project_smoke \
  --benchmarks lines,squares,circles,ellipses,zalesak \
  --resolutions 32,64,128 --cases 0,1,2,3,4 \
  --output experiments/baselines/results/quasi_project_smoke_20260813_final
```

The runner records native quadratic coefficients, the exact owning-cell
geometry and target area, per-cell and per-case errors, QUASI's frozen policy,
C1 sweep counts and misses, Section 2.5 curvature updates, vertex jumps, and
every conservative fallback event.

## Completion and conservation

- Completed `75/75` benchmark/case/resolution settings in `107.6 s` summed
  reconstruction time.
- Reconstructed all `7,375` mixed cells as native quadratics. There were no
  unsupported cells and no cells without geometry.
- `159` cells (`2.16%`) were explicitly classified as the paper-port's
  conservative local-quadratic fallback, arising from `131` unresolved
  correction events.
- The maximum absolute owning-cell area residual was `8.88e-16`.
- The reconstructed interfaces had zero measured shared-edge gap. `37`
  unmatched crossing diagnostics remain and are retained case by case.

## Numerical signal

Median unsigned midpoint-curvature error over the five cases:

| Benchmark | N=32 | N=64 | N=128 | Orders |
|---|---:|---:|---:|---:|
| Lines | `9.65e-5` | `5.03e-5` | `1.15e-4` | `0.94, -1.20` |
| Squares | `1.64e-1` | `6.84e-2` | `1.11e-2` | `1.26, 2.63` |
| Circles | `1.24e-3` | `3.25e-4` | `8.06e-5` | `1.93, 2.01` |
| Ellipses | `5.11e-4` | `2.87e-4` | `1.42e-4` | `0.83, 1.02` |
| Zalesak | `1.57e-2` | `1.72e-3` | `4.15e-4` | `3.18, 2.05` |

The clean circle signal is second order over both intervals. The ellipse signal
is approximately first order over this mini study. That contrast should be
treated as a real result of this frozen port, not tuned away. Curvature error
at sharp square and Zalesak corners is only a local diagnostic against the
nearest regular truth branch, not a smooth-interface convergence claim.

The common sampled reconstruction-to-truth diagnostic is dominated by open
physical-boundary endpoints for lines and by mesh-scale segment ownership for
closed curves; it is therefore retained as smoke-test provenance rather than a
paper-quality Hausdorff convergence estimate. Circle values are nonmonotone
over these five translations. Native geometry and case-level CSVs should be
used for any follow-up diagnosis.

## Sweep behavior

No case met the frozen `1e-11` maximum endpoint-displacement stopping criterion
within ten Gauss--Seidel sweeps. The implementation still returned conservative,
continuous geometry, but the convergence flag is `false` for all `75` cases.
This is the main caution before calling the port paper-ready. It must be
reported rather than interpreted as a crash or silently relaxed.

Across the matrix the corrector applied `71,553` C1 updates and reported
`1,577` missing C1 roots. Section 2.5 was used sparingly: six curvature updates
and four vertex jumps. Conservative fallback was concentrated in under-resolved
corners, particularly Zalesak (`94` fallback cells) and squares (`42`).

## Artifacts

- `case_results.csv`: one row per setting.
- `cells/*.csv`: one row per mixed cell, including native bulge, chord length,
  exact conservation, unsigned curvature error, and fallback attribution.
- `geometry/*.json`: exact cell polygons and native quadratic primitives.
- `summary.csv` and `summary.json`: five-case resolution aggregates.
- `quasi_all_benchmarks_summary.pdf`: vector-only all-benchmark summary.
- `run_manifest.json`: source commit, implementation checksums, frozen policy,
  cases, resolutions, and metric definitions.

## Recommendation

The port passes the smoke gate for execution, conservation, geometry retention,
and its intended second-order circle-curvature signal. It does not yet pass a
strict method-fidelity gate because every case exhausts the ten-sweep limit and
ellipse curvature is only approximately first order here. Keep QUASI as an
explicitly qualified baseline until those observations are reconciled with the
source paper or an author implementation; do not tune the frozen policies on
this result set.
