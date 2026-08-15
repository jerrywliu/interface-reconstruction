# Ellipse Circular-Variant Common-Metric Study

This fresh matched Cartesian study compares the finalized `per-cell circular`, `graph-coordinated circular`, and `graph-coordinated circular + joint C0` variants on canonical ellipse cases 0--4 at `N=32,64,128`. The joint C0 variant is the production default: connected rejected-join components are refined over shared endpoints and conservative per-facet curvatures.

- reconstruction source commit(s): `3054f9163e8ba73bd24decfde92270e8d92c2ca7`
- native geometry: exact schema-v2 line/arc metadata
- curvature observable: arc-length-weighted mean absolute error in unsigned geometric curvature against the nearest analytic ellipse branch
- concavity diagnostic: negative project radius, reported by primitive count and native arc length

## Results

| Variant | N | Hausdorff median | Curvature MAE median | Facet-gap median | Joint components solved | Bad joins after joint | Max area residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| per-cell circular | 32 | 1.038496e-02 | 1.382777e-03 | 4.464034e-03 | 0 / 0 | 0 | n/a |
| per-cell circular | 64 | 1.233734e-03 | 6.869149e-04 | 5.622001e-04 | 0 / 0 | 0 | n/a |
| per-cell circular | 128 | 1.832119e-04 | 3.525604e-04 | 6.945111e-05 | 0 / 0 | 0 | n/a |
| graph-coordinated circular | 32 | 1.038496e-02 | 1.382777e-03 | 4.464034e-03 | 0 / 0 | 0 | n/a |
| graph-coordinated circular | 64 | 1.233734e-03 | 6.869149e-04 | 5.622001e-04 | 0 / 0 | 0 | n/a |
| graph-coordinated circular | 128 | 1.832119e-04 | 3.525604e-04 | 6.945111e-05 | 0 / 0 | 0 | n/a |
| graph-coordinated circular + joint C0 | 32 | 3.569651e-03 | 1.517707e-03 | 0.000000e+00 | 13 / 13 | 0 | 9.508531e-11 |
| graph-coordinated circular + joint C0 | 64 | 2.850972e-04 | 6.999664e-04 | 0.000000e+00 | 29 / 29 | 0 | 9.946999e-11 |
| graph-coordinated circular + joint C0 | 128 | 3.949769e-05 | 3.494550e-04 | 0.000000e+00 | 78 / 78 | 0 | 9.951045e-11 |

## Observed Orders

| Variant | Geometry | Curvature | Facet gap |
|---|---:|---:|---:|
| per-cell circular | 2.912 | 0.986 | 3.003 |
| graph-coordinated circular | 2.912 | 0.986 | 3.003 |
| graph-coordinated circular + joint C0 | 3.249 | 1.059 | n/a |

## Equivalence Check

The per-cell and graph-coordinated native geometries are byte-for-byte numerically identical in `15/15` matched cases and agree within `1e-12` in `15/15`. The largest native parameter difference is `0.000e+00`. Their combined merged-cell count is `0`.

## Interpretation

Joint C0 does not materially improve the common unsigned-curvature observable in this five-case study. Relative to graph-coordinated circular, its median curvature error changes by `+9.8%`, `+1.9%`, and `-0.9%` at `N=32,64,128`, respectively. The fitted curvature order changes only from `0.986` to `1.059`. Its clear benefits are instead geometric: lower Hausdorff error and much smaller facet gaps.

The optimizer solves `120/120` connected components, with `0` failures and `0` remaining eligible bad joins. All solved components reach the exact-C1 branch, and the maximum relative cell-area residual is `9.951e-11`.

Negative-radius (locally concave) arcs are reported explicitly because joint conservative refinement does not impose a convexity constraint. The unsigned curvature metric therefore remains paired with the signed curvature and concave-arc diagnostics. The joint C0 runs contain `4` straight-limit line facets in total; the common metric assigns these zero curvature.

The C0 facet sidecars are post-refinement: `runReconstruction` invokes the joint `makeC0` pass before collecting the returned facet list and writing the exact schema-v2 metadata. Component outcomes above come from the same final run's diagnostics. The saved native geometry differs from its matched pre-C0 reconstruction in `15/15` cases.

## Reproduce

```bash
PYTHONPATH=. python -m experiments.baselines.run_ellipse_circular_variant_metrics --workers 3 --metric-workers 6
```
