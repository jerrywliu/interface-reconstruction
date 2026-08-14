# Ellipse Circular-Variant Common-Metric Study

This fresh matched Cartesian study compares the finalized `per-cell circular`, `graph-coordinated circular`, and `graph-coordinated circular + guarded C0` variants on canonical ellipse cases 0--4 at `N=32,64,128`. The guarded C0 variant is the production single pass: eligible endpoint pairs are averaged and each facet is conservatively refit. It is not the later representative joint optimizer.

- reconstruction source commit(s): `ad3f3cb9034e649cb79931d848e5e498ae41c34c`
- native geometry: exact schema-v2 line/arc metadata
- curvature observable: arc-length-weighted mean absolute error in unsigned geometric curvature against the nearest analytic ellipse branch
- concavity diagnostic: negative project radius, reported by primitive count and native arc length

## Results

| Variant | N | Hausdorff median | Curvature MAE median | Signed curvature mean | Facet-gap median | Concave arc length | C0 accepted / rejected |
|---|---:|---:|---:|---:|---:|---:|---:|
| per-cell circular | 32 | 1.038496e-02 | 1.382777e-03 | 4.072502e-02 | 4.464034e-03 | 0.000000e+00 | 0 / 0 |
| per-cell circular | 64 | 1.233734e-03 | 6.869149e-04 | 4.069979e-02 | 5.622001e-04 | 0.000000e+00 | 0 / 0 |
| per-cell circular | 128 | 1.832119e-04 | 3.525604e-04 | 4.069423e-02 | 6.945111e-05 | 0.000000e+00 | 0 / 0 |
| graph-coordinated circular | 32 | 1.038496e-02 | 1.382777e-03 | 4.072502e-02 | 4.464034e-03 | 0.000000e+00 | 0 / 0 |
| graph-coordinated circular | 64 | 1.233734e-03 | 6.869149e-04 | 4.069979e-02 | 5.622001e-04 | 0.000000e+00 | 0 / 0 |
| graph-coordinated circular | 128 | 1.832119e-04 | 3.525604e-04 | 4.069423e-02 | 6.945111e-05 | 0.000000e+00 | 0 / 0 |
| graph-coordinated circular + guarded C0 | 32 | 3.569651e-03 | 1.495697e-03 | 4.072466e-02 | 2.922714e-05 | 0.000000e+00 | 297 / 13 |
| graph-coordinated circular + guarded C0 | 64 | 2.850972e-04 | 6.962905e-04 | 4.069292e-02 | 4.982808e-06 | 0.000000e+00 | 591 / 33 |
| graph-coordinated circular + guarded C0 | 128 | 3.626514e-05 | 3.472734e-04 | 4.069002e-02 | 1.342635e-06 | 0.000000e+00 | 1155 / 95 |

## Observed Orders

| Variant | Geometry | Curvature | Facet gap |
|---|---:|---:|---:|
| per-cell circular | 2.912 | 0.986 | 3.003 |
| graph-coordinated circular | 2.912 | 0.986 | 3.003 |
| graph-coordinated circular + guarded C0 | 3.311 | 1.053 | 2.222 |

## Equivalence Check

The per-cell and graph-coordinated native geometries are byte-for-byte numerically identical in `15/15` matched cases and agree within `1e-12` in `15/15`. The largest native parameter difference is `0.000e+00`. Their combined merged-cell count is `0`.

## Interpretation

Guarded C0 does not materially improve the common unsigned-curvature observable in this five-case study. Relative to graph-coordinated circular, its median curvature error changes by `+8.2%`, `+1.4%`, and `-1.5%` at `N=32,64,128`, respectively. The fitted curvature order changes only from `0.986` to `1.053`. Its clear benefits are instead geometric: lower Hausdorff error and much smaller facet gaps.

No negative-radius (locally concave) arc occurs in any of the 45 matched case-variant-resolution reconstructions. The unsigned curvature metric is therefore not masking sign errors here. The guarded C0 runs contain `4` straight-limit line facets in total; the common metric assigns these zero curvature.

The C0 facet sidecars are post-refinement: `runReconstruction` invokes the guarded `makeC0` pass before collecting the returned facet list and writing the exact schema-v2 metadata. C0 adjustment/rejection counts above come from the same final run's provenance events. The saved native geometry differs from its matched pre-C0 reconstruction in `15/15` cases.

## Reproduce

```bash
PYTHONPATH=. python -m experiments.baselines.run_ellipse_circular_variant_metrics --workers 3 --metric-workers 6
```
