# Graph-Coordinated Circular Ellipse Common-Metric Replay

## Scope

This replay evaluates the finalized graph-coordinated circular method on the
same Cartesian ellipse subset used by the external higher-order baseline
screen: canonical cases `0--4` at `N=32,64,128`. It uses the sealed submission
reconstruction at source commit `505aefa454328d4ba34ade5e7247050a0acfc793`.
No reconstruction was rerun.

The sealed result contains exact schema-v2 `writeFacets` records for every
active line and arc. A dedicated adapter preserves each saved center, radius,
endpoint pair, and signed angular span as an `ExternalArcPrimitive`. All 15
saved geometry records match the canonical seed-42 ellipse sequence to an
absolute tolerance of `1e-12`.

## Common observables

- **Geometry:** partition-invariant native symmetric Hausdorff distance. Both
  directed suprema use native point-to-curve distance, projected target
  endpoints, and bounded optimization on each source interval.
- **Curvature:** arc-length-weighted mean absolute geometric-curvature error.
  The fitted arc curvature is compared at quadrature points with the nearest
  point on the analytic rotated ellipse.

Every reconstructed primitive in the selected cases is an arc. There are no
line or LVIRA fallback primitives in this 15-case subset.

## Results

| Cells per side | Median native Hausdorff | Median curvature error |
|---:|---:|---:|
| 32 | `1.03850e-2` | `1.38278e-3` |
| 64 | `1.23373e-3` | `6.86915e-4` |
| 128 | `1.83212e-4` | `3.52560e-4` |

The three-resolution median fits give:

- native symmetric Hausdorff order: **`2.912`**;
- geometric-curvature order: **`0.986`**.

The result is not driven by one case. Per-case Hausdorff orders range from
`2.726` to `3.048`; per-case curvature orders range from `0.962` to `1.031`.
The common observable therefore strengthens the near-third-order geometry
signal while confirming that this method's fitted constant curvature is only
approximately first-order on these ellipses. A circular representation alone
does not imply a second-order pointwise curvature estimator.

## Reproduction

```bash
PYTHONPATH=. python -m \
  experiments.baselines.run_graph_circular_ellipse_common_metrics \
  --workers 6
```

Artifacts:

- `experiments/baselines/results/graph_circular_ellipse_common_metrics_20260814/case_results.csv`
- `experiments/baselines/results/graph_circular_ellipse_common_metrics_20260814/summary.csv`
- `experiments/baselines/results/graph_circular_ellipse_common_metrics_20260814/case_orders.csv`
- `experiments/baselines/results/graph_circular_ellipse_common_metrics_20260814/manifest.json`
