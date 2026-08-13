# Ellipse Curvature-Convergence Audit

## Paper inventory

The paper does not show curvature convergence in the main ellipse figure.  The
main panel reports Hausdorff distance and facet gap.  The appendix does include
curvature error as the third row of the ten-panel full ellipse comparison
(`fig:appendix_ellipses_all_methods`, sourced from
`ellipse_reconstruction_perturbed_all_methods_5x2_axes.pdf`).

That appendix row is not valid curvature-convergence evidence.  The archived
driver evaluated

```text
abs(reconstructed_facet.curvature - positive_analytic_ellipse_curvature)
```

even though the reconstructed arc curvature is signed by orientation.  Most
ellipse arcs have the opposite sign convention, leaving an apparent error near
twice the mean curvature.  This explains the nearly flat values around
`3.8e-2` for the circular method in the published candidate panel.

`curvature_proxy_error` is a different diagnostic: it compares one global mean
turning-curvature proxy with a sampled reference.  It is not a substitute for
local curvature-estimate convergence and is not used here.

## Corrected metric

For case `c`, the recomputed metric is

```text
e_kappa(c) = (1 / K_c) sum_j |kappa_h,j - kappa_Gamma(x_j)|,
```

where `x_j` is the chord midpoint of active primitive `j`.  Exact serialized
metadata supplies `kappa_h,j = 1 / |R_j|` for an arc and `kappa_h,j = 0` for a
line.  The analytic ellipse curvature is

```text
kappa(phi) = a b / (a^2 sin(phi)^2 + b^2 cos(phi)^2)^(3/2).
```

The parameter `phi` is obtained by translating, undoing the ellipse rotation,
and scaling by `(a,b)` before applying `atan2`.

All active serialized primitives are eligible.  The audit does not discard
fallback lines, merged-cell primitives, difficult cells, or outlying cases.
It fails rather than silently skipping missing metadata, empty cases,
unsupported primitive types, or invalid radii.

## Data and aggregation

- Frozen source: sealed submission result set
  `submission_static_20260731_012430_505aefa45432.sealed`.
- Reconstruction source commit: `505aefa454328d4ba34ade5e7247050a0acfc793`.
- Method: graph-coordinated circular reconstruction with LVIRA fallback.
- Mesh: Cartesian (`w=0`).
- Resolutions: `N = 32, 50, 64, 100, 128, 150`.
- Cases: all `25` fixed ellipse instances at every resolution (`150` case
  measurements total).
- Per-resolution summary: median and interquartile range across cases.
- Reported order: least-squares slope of `log(median e_kappa)` against
  `log(h)`, with `h=100/N`.

The exact saved arc parameters make the frozen geometry sufficient; no
reconstruction rerun is required.  The audit does not mutate the sealed runs.

## Result and interpretation

The corrected median errors are:

| N | Median mean absolute curvature error |
|---:|---:|
| 32 | `1.6631e-3` |
| 50 | `8.6857e-4` |
| 64 | `7.6838e-4` |
| 100 | `4.4413e-4` |
| 128 | `3.1563e-4` |
| 150 | `2.5647e-4` |

The fitted order is `1.1769` over all six resolutions.  A fit over only the
four finest points gives `1.2900`; the signal therefore remains substantially
below second order rather than being explained by the coarsest point alone.

This evidence supports convergence of the circular method's local curvature
estimate, but it does **not** support a second-order curvature-convergence
claim.  It also does not currently support a direct PLVIRA/PCIC/QUASI
comparison: no metric-compatible ellipse outputs from those ports were
available when this candidate was generated.  Those curves should be added
only after each native primitive is evaluated with the same local unsigned
curvature metric and the same cases.

## Artifacts

- `results/submission/ellipse_curvature_convergence_20260813/ellipse_curvature_convergence.pdf`
- `results/submission/ellipse_curvature_convergence_20260813/ellipse_curvature_convergence.svg`
- `results/submission/ellipse_curvature_convergence_20260813/ellipse_curvature_case_metrics.csv`
- `results/submission/ellipse_curvature_convergence_20260813/ellipse_curvature_summary.csv`
- `results/submission/ellipse_curvature_convergence_20260813/provenance.json`

The PDF and SVG are vector outputs.  The candidate includes a second-order
reference slope for context; it does not present that slope as observed data.
