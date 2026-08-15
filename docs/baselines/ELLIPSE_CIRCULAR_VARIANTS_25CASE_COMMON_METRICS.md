# Ellipse Circular-Variant Common-Metric Study

This fresh matched Cartesian study compares the finalized `per-cell circular`, `graph-coordinated circular`, and `graph-coordinated circular + joint C0` variants on canonical ellipse cases 0--24 at `N=32,64,128`. The joint C0 variant is the production default: connected rejected-join components are refined over shared endpoints and conservative per-facet curvatures.

- reconstruction source commit(s): `11ba91996698ee87f6f9c6983e6981b1960a56b0, 6e336f791f049f63148026a7516edbe821fdb445`
- native geometry: exact schema-v2 line/arc metadata
- curvature observable: arc-length-weighted mean absolute error in unsigned geometric curvature against the nearest analytic ellipse branch
- concavity diagnostic: negative project radius, reported by primitive count and native arc length
- normalized conservation residual: maximum fitted-group area residual divided by the geometric area of that cell or merged group

## Results

| Variant | N | Hausdorff median | Curvature MAE median | Facet-gap median | Joint components solved | Bad joins after joint | Max area residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| per-cell circular | 32 | 3.566079e-02 | 3.592705e-03 | 1.137812e-02 | 0 / 0 | 0 | 4.405249e-10 |
| per-cell circular | 64 | 6.051366e-03 | 1.848889e-03 | 1.634048e-03 | 0 / 0 | 0 | 8.451579e-08 |
| per-cell circular | 128 | 7.880836e-04 | 8.792140e-04 | 1.808678e-04 | 0 / 0 | 0 | 6.313493e-10 |
| graph-coordinated circular | 32 | 3.554270e-02 | 3.507567e-03 | 1.094745e-02 | 0 / 0 | 0 | 4.405249e-10 |
| graph-coordinated circular | 64 | 6.051366e-03 | 1.848889e-03 | 1.634048e-03 | 0 / 0 | 0 | 8.451579e-08 |
| graph-coordinated circular | 128 | 7.880836e-04 | 8.792140e-04 | 1.808678e-04 | 0 / 0 | 0 | 6.313493e-10 |
| graph-coordinated circular + joint C0 | 32 | 1.235550e-02 | 4.376729e-03 | 0.000000e+00 | 91 / 91 | 0 | 9.971200e-11 |
| graph-coordinated circular + joint C0 | 64 | 1.651043e-03 | 1.908280e-03 | 0.000000e+00 | 169 / 169 | 0 | 9.946999e-11 |
| graph-coordinated circular + joint C0 | 128 | 1.620025e-04 | 9.112119e-04 | 0.000000e+00 | 438 / 438 | 0 | 1.181834e-10 |

## Observed Orders

| Variant | Geometry | Curvature | Facet gap |
|---|---:|---:|---:|
| per-cell circular | 2.750 | 1.015 | 2.988 |
| graph-coordinated circular | 2.748 | 0.998 | 2.960 |
| graph-coordinated circular + joint C0 | 3.126 | 1.132 | n/a |

## Equivalence Check

The per-cell and graph-coordinated native geometries are byte-for-byte numerically identical in `69/75` matched cases and agree within `1e-12` in `69/75`. Of the remaining `6`, `4` change primitive type or count; the largest finite native parameter difference is `1.525e+01`. Their combined merged-cell count is `0`.

## Interpretation

Joint C0 does not materially improve the common unsigned-curvature observable in this 25-case study. Relative to graph-coordinated circular, its median curvature error changes by `+24.8%`, `+3.2%`, and `+3.6%` at `N=32,64,128`, respectively. The fitted curvature order changes only from `0.998` to `1.132`. Its clear benefits are instead geometric: lower Hausdorff error and much smaller facet gaps.

The optimizer solves `698/698` connected components, with `0` failures and `0` remaining eligible bad joins. `697` solved components reach the exact-C1 branch and `1` uses the conservative fallback. The maximum relative cell-area residual is `1.182e-10`.

Negative-radius (locally concave) arcs are reported explicitly because joint conservative refinement does not impose a convexity constraint. The unsigned curvature metric therefore remains paired with the signed curvature and concave-arc diagnostics. The joint C0 runs contain `17` straight-limit line facets in total; the common metric assigns these zero curvature.

The C0 facet sidecars are post-refinement: `runReconstruction` invokes the joint `makeC0` pass before collecting the returned facet list and writing the exact schema-v2 metadata. Component outcomes above come from the same final run's diagnostics. The saved native geometry differs from its matched pre-C0 reconstruction in `75/75` cases.

## Reproduce

```bash
PYTHONPATH=. python -m experiments.baselines.run_ellipse_circular_variant_metrics --workers 3 --metric-workers 6
```
