# PLVIRA Cartesian Project Smoke

## Scope

This study runs the operational PLVIRA implementation, with Cartesian
generalized-height-function (GHF) curvature, on cases 0--4 of each project
benchmark at `N = 32, 64, 128`. All meshes are uniform, axis-aligned Cartesian
meshes. The case geometry, random seeds, and draw order match the canonical
drivers in `experiments/static/`.

Command:

```bash
PYTHONPATH=. python experiments/baselines/run_plvira_project_smoke.py \
  --output experiments/baselines/results/plvira_project_smoke_20260813_final \
  --benchmarks lines,squares,circles,ellipses,zalesak \
  --resolutions 32,64,128 --cases 0,1,2,3,4
```

Wall time was 92.04 seconds. Timed reconstruction calls account for 36.51
seconds; fixture initialization, metric sampling, serialization, and plotting
account for the remainder.

## Result contract

- `run_manifest.json` records the method, Cartesian scope, selected cases,
  resolutions, curvature source, metrics, and boundary policy.
- `case_results.csv` contains one row for each of the 75 cases, including exact
  parameters/seeds, runtime, status counts, conservation, gap, geometry, and
  curvature-estimator diagnostics.
- `cells/*.csv` contains one row per mixed cell. It records status, exact local
  area residual, fitted objective, optimizer status, GHF path, estimated and
  analytic curvature, and the serialized diagnostic payload.
- `geometry/*.json` retains each native parabolic reconstruction through the
  shared external-baseline schema.
- `summary.csv`, `summary.json`, and `plvira_all_benchmarks_summary.pdf`
  aggregate the five cases without discarding failures.

The sampled symmetric Hausdorff diagnostic is completeness-sensitive: it
penalizes missing truth segments when the paper method has no boundary halo.
The reconstruction-to-truth diagnostic instead measures only returned
geometry. Both use fixed point spacing no larger than `2.5e-3`, and therefore
have an approximately `1.25e-3` nearest-sample floor. They are smoke-test
diagnostics, not replacements for the adaptive native-geometry paper metric.

Curvature-estimator error is separate from reconstruction error. It compares
the GHF curvature supplied to PLVIRA against analytic truth near each fitted
facet. It does not measure interface position, facet continuity, or volume
conservation.

## Completion and provenance

| Quantity | Count |
| --- | ---: |
| Cases | 75 |
| Mixed cells | 7,380 |
| Reconstructed cells | 7,328 |
| Unsupported cells | 52 |
| Unresolved cells | 0 |
| Optimizer failures | 0 |

All 52 unsupported cells occur on line interfaces at the domain boundary.
PLVIRA needs a complete 3x3 fitting stencil, while some GHF fallback paths need
an additional halo. The source method specifies no one-sided boundary rule, so
the runner records these cells as unsupported rather than inserting a project
fallback. Unsupported fractions fall from 8.4% at `N=32` to 1.7% at `N=128`.
The other four benchmarks lie away from the domain boundary and have complete
coverage.

GHF path counts over reconstructed cells are:

| GHF path | Cells |
| --- | ---: |
| Complete height function | 6,547 |
| Mixed-height parabola | 335 |
| PLIC-centroid parabola | 144 |
| Degenerate zero | 302 |

The largest exact local conservation residual is `1.12e-9`. This confirms the
per-cell volume constraint, but does not imply an accurate or continuous
interface.

## Trends

### Smooth benchmarks

For circles, the median absolute GHF curvature error is `4.45e-3`, `1.19e-3`,
and `2.81e-4` at `N=32`, `64`, and `128`. The two pairwise rates are 1.90 and
2.08, providing the expected second-order signal once the coarse zero-curvature
fallbacks are resolved. Median facet gap decreases from `3.28e-2` to
`2.16e-5`.

For ellipses, median absolute GHF curvature error decreases from `2.69e-3` to
`2.88e-4` and `1.52e-4`. This is a clear decrease but not a steady second-order
window over these three resolutions: the pairwise rates are 3.22 and 0.92.
The five-case smoke therefore does not yet support a clean ellipse curvature
rate claim. Median facet gap decreases from `2.02e-2` to `1.16e-3`.

### Sharp benchmarks

PLVIRA is a smooth parabolic reconstruction and has no explicit corner model.
Squares and Zalesak therefore show large interface-position and continuity
errors around sharp features even though every interior cell conserves volume.
The median square reconstruction-to-truth diagnostic decreases from `1.21` to
`0.279`; the median Zalesak value decreases from `1.83` to `0.277`. Shared-edge
crossing mismatches remain present. These results are useful negative controls,
not evidence for a PLVIRA corner capability.

## Interpretation

The implementation is operational for the matched Cartesian study and needs no
oracle curvature. Its strongest validation signal is the near-second-order
circle curvature estimate. Before using an ellipse curvature-convergence panel
in the paper, extend the regular benchmark to finer resolutions and compare the
same curvature norm for PLVIRA, PCIC, QUASI, and our circular method. Keep the
curvature panel explicitly separate from facet-gap and interface-position
panels.
