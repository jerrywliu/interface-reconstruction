# Ellipse Circular-Variant Common-Metric Study

This fresh matched Cartesian study compares the finalized `per-cell circular`, `graph-coordinated circular`, and `graph-coordinated circular + joint C0` variants on canonical ellipse cases 0--24 at `N=32,50,64,100,128,150,256,300`. The joint C0 variant is the production default: connected rejected-join components are refined over shared endpoints and conservative per-facet curvatures.

- reconstruction source commit(s): `005ca9f01fb15722e0227abeea2ec4daafa23920, 11ba91996698ee87f6f9c6983e6981b1960a56b0, 3f6d7f2edaf2fd2daa03dc6829d9f0845f247f16, 6e336f791f049f63148026a7516edbe821fdb445, 974be6d3504433daf8992c16d9602ff3b79f8461, e0a10d5da441fcf3bf80377259481c2ac74247b8`
- native geometry: exact schema-v2 line/arc metadata
- curvature observable: arc-length-weighted mean absolute error in unsigned geometric curvature against the nearest analytic ellipse branch
- concavity diagnostic: negative project radius, reported by primitive count and native arc length
- normalized conservation residual: maximum fitted-group area residual divided by the geometric area of that cell or merged group

## Results

| Variant | N | Hausdorff median | Curvature MAE median | Facet-gap median | Joint components solved | Bad joins after joint | Max area residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| per-cell circular | 32 | 3.566079e-02 | 3.592705e-03 | 1.137812e-02 | 0 / 0 | 0 | 4.405249e-10 |
| per-cell circular | 50 | 1.376397e-02 | 2.288538e-03 | 3.370986e-03 | 0 / 0 | 0 | 2.101584e-10 |
| per-cell circular | 64 | 6.051366e-03 | 1.848889e-03 | 1.634048e-03 | 0 / 0 | 0 | 8.451579e-08 |
| per-cell circular | 100 | 1.656666e-03 | 1.133523e-03 | 3.897885e-04 | 0 / 0 | 0 | 7.339174e-10 |
| per-cell circular | 128 | 7.880836e-04 | 8.792140e-04 | 1.808678e-04 | 0 / 0 | 0 | 6.313493e-10 |
| per-cell circular | 150 | 4.952804e-04 | 7.607826e-04 | 1.139587e-04 | 0 / 0 | 0 | 2.009501e-10 |
| per-cell circular | 256 | 1.151531e-04 | 4.785056e-04 | 2.324428e-05 | 0 / 0 | 0 | 1.689315e-09 |
| per-cell circular | 300 | 8.614147e-05 | 4.143105e-04 | 1.652235e-05 | 0 / 0 | 0 | 1.064022e-08 |
| graph-coordinated circular | 32 | 3.554270e-02 | 3.507567e-03 | 1.094745e-02 | 0 / 0 | 0 | 4.405249e-10 |
| graph-coordinated circular | 50 | 1.102948e-02 | 2.288538e-03 | 2.784562e-03 | 0 / 0 | 0 | 2.101584e-10 |
| graph-coordinated circular | 64 | 6.051366e-03 | 1.848889e-03 | 1.634048e-03 | 0 / 0 | 0 | 8.451579e-08 |
| graph-coordinated circular | 100 | 1.656666e-03 | 1.133523e-03 | 3.897885e-04 | 0 / 0 | 0 | 7.339174e-10 |
| graph-coordinated circular | 128 | 7.880836e-04 | 8.792140e-04 | 1.808678e-04 | 0 / 0 | 0 | 6.313493e-10 |
| graph-coordinated circular | 150 | 4.952804e-04 | 7.607826e-04 | 1.139587e-04 | 0 / 0 | 0 | 2.009501e-10 |
| graph-coordinated circular | 256 | 1.151531e-04 | 4.782901e-04 | 2.324428e-05 | 0 / 0 | 0 | 1.689315e-09 |
| graph-coordinated circular | 300 | 8.399849e-05 | 3.979057e-04 | 1.486325e-05 | 0 / 0 | 0 | 1.064022e-08 |
| graph-coordinated circular + joint C0 | 32 | 1.235550e-02 | 4.376729e-03 | 0.000000e+00 | 91 / 91 | 0 | 9.971200e-11 |
| graph-coordinated circular + joint C0 | 50 | 3.550439e-03 | 2.649800e-03 | 0.000000e+00 | 120 / 120 | 0 | 1.118090e-10 |
| graph-coordinated circular + joint C0 | 64 | 1.651043e-03 | 1.908280e-03 | 0.000000e+00 | 169 / 169 | 0 | 9.946999e-11 |
| graph-coordinated circular + joint C0 | 100 | 3.876724e-04 | 1.175351e-03 | 0.000000e+00 | 323 / 323 | 0 | 1.118312e-10 |
| graph-coordinated circular + joint C0 | 128 | 1.620025e-04 | 9.112119e-04 | 0.000000e+00 | 438 / 438 | 0 | 1.181834e-10 |
| graph-coordinated circular + joint C0 | 150 | 1.074471e-04 | 7.697876e-04 | 0.000000e+00 | 537 / 537 | 0 | 1.733564e-10 |
| graph-coordinated circular + joint C0 | 256 | 2.479979e-05 | 4.757267e-04 | 1.176998e-11 | 1044 / 1045 | 2 | 1.658387e-10 |
| graph-coordinated circular + joint C0 | 300 | 1.798408e-05 | 3.976647e-04 | 3.076768e-11 | 1301 / 1303 | 3 | 1.528179e-10 |

## Observed Orders

| Variant | Geometry | Curvature | Facet gap |
|---|---:|---:|---:|
| per-cell circular | 2.788 | 0.972 | 2.981 |
| graph-coordinated circular | 2.755 | 0.975 | 2.963 |
| graph-coordinated circular + joint C0 | 2.990 | 1.065 | n/a |

## Equivalence Check

The per-cell and graph-coordinated native geometries are byte-for-byte numerically identical in `185/200` matched cases and agree within `1e-12` in `185/200`. Of the remaining `15`, `12` change primitive type or count; the largest finite native parameter difference is `1.525e+01`. Their combined merged-cell count is `2`.

## Interpretation

Joint C0 does not materially improve the common unsigned-curvature observable in this 25-case study. Relative to graph-coordinated circular, its median curvature error changes by `+24.8%`, `+3.2%`, and `+3.6%` at `N=32,64,128`, respectively. The fitted curvature order changes only from `0.975` to `1.065`. Its clear benefits are instead geometric: lower Hausdorff error and much smaller facet gaps.

The optimizer solves `4023/4026` connected components, with `3` failures and `5` remaining eligible bad joins. `4021` solved components reach the exact-C1 branch and `2` uses the conservative fallback. The maximum relative cell-area residual is `1.734e-10`.

Negative-radius (locally concave) arcs are reported explicitly because joint conservative refinement does not impose a convexity constraint. The unsigned curvature metric therefore remains paired with the signed curvature and concave-arc diagnostics. The joint C0 runs contain `118` straight-limit line facets in total; the common metric assigns these zero curvature.

The C0 facet sidecars are post-refinement: `runReconstruction` invokes the joint `makeC0` pass before collecting the returned facet list and writing the exact schema-v2 metadata. Component outcomes above come from the same final run's diagnostics. The saved native geometry differs from its matched pre-C0 reconstruction in `200/200` cases.

## Reproduce

```bash
PYTHONPATH=. python -m experiments.baselines.run_ellipse_circular_variant_metrics --workers 3 --metric-workers 6
```
