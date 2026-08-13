# Ellipse Curvature Metric-Sensitivity Diagnostic

## Question

The sealed Cartesian ellipse reconstruction gives an observed curvature-error
order well below two when each fitted constant curvature is compared with the
analytic ellipse curvature at the projected chord midpoint. This diagnostic
tests whether that signal is caused by the representative point or by equal
weighting of facets, without changing or rerunning the reconstruction.

## Fixed data

- Source: sealed result set
  `submission_static_20260731_012430_505aefa45432.sealed`.
- Reconstruction commit: `505aefa454328d4ba34ade5e7247050a0acfc793`.
- Method: graph-coordinated circular reconstruction with LVIRA fallback.
- Mesh: Cartesian (`w=0`).
- Resolutions: `N = 32, 50, 64, 100, 128, 150`.
- Cases: all 25 fixed ellipse cases at every resolution.
- Geometry: exact serialized line/arc endpoints, arc centers, radii, and signed
  angular spans from `writeFacets`.

No result was selected, removed, or regenerated for this diagnostic.

## Definitions

For ellipse parameter `t`, semiaxes `a,b`, and speed

```text
s(t) = sqrt(a^2 sin(t)^2 + b^2 cos(t)^2),
```

the analytic unsigned curvature is

```text
kappa(t) = a b / s(t)^3.
```

Physical points are assigned an ellipse parameter by undoing the ellipse
rotation, scaling the local coordinates by `(a,b)`, and applying `atan2`. This
is the normalized-radial parameter projection used by the existing audit.

For every active primitive, the diagnostic compares its unsigned reconstructed
curvature with three analytic references:

1. **Projected chord midpoint.** This exactly reproduces the current corrected
   metric: the midpoint of the two serialized endpoints is parameter-projected
   to the ellipse.
2. **Projected native-arc midpoint.** The representative point is evaluated on
   the fitted arc itself using its serialized center, radius, and half of its
   signed angular span, then parameter-projected to the ellipse. A non-arc uses
   its chord midpoint.
3. **True-interval mean curvature.** The endpoint projections define the short
   true ellipse interval `I_j`. Its reference curvature is

```text
kappa_bar_j = integral_Ij kappa(t) s(t) dt / integral_Ij s(t) dt.
```

The ellipse integrals are evaluated with adaptive quadrature at absolute and
relative tolerances of `1e-13`.

Each reference is aggregated within a case in two ways:

```text
equal facet:          mean_j |kappa_h,j - kappa_ref,j|
arc-length weighted: sum_j L_j |kappa_h,j - kappa_ref,j| / sum_j L_j,
```

where `L_j` is the true ellipse interval length. The reported resolution value
is the median over 25 cases. Orders are least-squares slopes of
`log(error)` against `log(h)`, first over all six resolutions and then over the
finest four (`N = 64, 100, 128, 150`).

## Results

| Analytic reference | Aggregation | All-resolution order | Finest-four order |
|---|---|---:|---:|
| Projected chord midpoint | Equal facet | `1.1769` | `1.2900` |
| Projected chord midpoint | Arc-length weighted | `1.1601` | `1.3108` |
| Projected native-arc midpoint | Equal facet | `1.1921` | `1.3262` |
| Projected native-arc midpoint | Arc-length weighted | `1.1749` | `1.3157` |
| True-interval mean curvature | Equal facet | `1.1288` | `1.2752` |
| True-interval mean curvature | Arc-length weighted | `1.1303` | `1.2527` |

Changing the midpoint convention moves the median error by at most about 3%
at any resolution. The interval-average target lowers the equal-facet median
by about 2-10%, but does not increase its order. Arc-length weighting likewise
changes constants rather than the convergence regime.

The population contains 23,826 active primitives. Exactly one is a non-arc
LVIRA fallback: ellipse case 23 at `N=128`. It is `0.0042%` of the complete
population and `0.0172%` of the `N=128` population. The affected case remains
above the resolution median, so this single fallback does not set the reported
median or explain its slope.

## Conclusion

The roughly first-order-to-`1.3` signal in the sealed resolution range is not
an artifact of evaluating analytic curvature at the chord midpoint, nor of
weighting every fitted facet equally. All three reference choices and both
aggregations give the same qualitative behavior.

A circular representation can approximate interface position at high order
without its fitted constant curvature necessarily being a second-order
pointwise curvature estimator. Second-order curvature is a reasonable target
when the stencil, normal estimate, and representative location have the
corresponding consistency properties, as in the Cartesian GHF used by PLVIRA;
it does not follow from using circular arcs alone. These sealed data therefore
support convergence of the curvature estimate, but not a second-order claim
for the present circular method over `N=32-150`. Higher resolutions could test
for a later asymptotic transition, but the metric convention is not the
current blocker.

## Reproduction

```bash
MPLBACKEND=Agg PYTHONPATH=. python -m \
  experiments.static.diagnose_ellipse_curvature_metric_sensitivity
```

Outputs:

- `results/submission/ellipse_curvature_metric_sensitivity_20260813/metric_sensitivity.csv`
- `results/submission/ellipse_curvature_metric_sensitivity_20260813/case_metrics.csv`
- `results/submission/ellipse_curvature_metric_sensitivity_20260813/ellipse_curvature_metric_sensitivity.pdf`
- `results/submission/ellipse_curvature_metric_sensitivity_20260813/provenance.json`

The comparison PDF is vector-only and uses embedded fonts.
