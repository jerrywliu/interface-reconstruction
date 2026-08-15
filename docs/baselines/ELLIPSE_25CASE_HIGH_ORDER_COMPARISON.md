# Ellipse 25-Case Higher-Order Comparison

## Scope

This frozen Cartesian study compares 25 canonical ellipse cases at
`N=32,64,128`. The primary comparison contains the three project circular
variants, operational PLVIRA, PCIC with center translation, and the frozen
QUASI port. PCIC radius adjustment is retained only as a five-case sensitivity
study and is not included in the primary panel.

All methods use the same partition-insensitive native symmetric Hausdorff
distance and geometric-curvature observable. The curvature metric is the
arc-length-weighted mean absolute error between each reconstructed primitive's
native geometric curvature and the nearest analytic ellipse branch. Facet gap,
mixed-cell coverage, and method status come from the corresponding production
diagnostics.

## Observed Orders

| Method | Hausdorff | Curvature | Facet gap |
|---|---:|---:|---:|
| Ours, per-cell circular | `2.750` | `1.015` | `2.988` |
| Ours, graph-coordinated circular | `2.748` | `0.998` | `2.960` |
| Ours, graph-coordinated circular + joint C0 | `3.126` | `1.132` | exact zero |
| PLVIRA | `1.299` | `0.282` | `2.698` |
| PCIC, center translation | `0.897` | `0.785` | `2.800` |
| QUASI, frozen port | `0.968` | `-1.040` | exact zero |

The fits use the medians at all three resolutions. A fitted order summarizes
these three points; it does not erase nonmonotonic behavior such as the PCIC
Hausdorff increase from `N=64` to `N=128`.

## Fine-Grid Results

| Method | Hausdorff median | Curvature MAE median | Facet-gap median | Coverage |
|---|---:|---:|---:|---:|
| Ours, per-cell circular | `7.881e-4` | `8.792e-4` | `1.809e-4` | `100%` |
| Ours, graph-coordinated circular | `7.881e-4` | `8.792e-4` | `1.809e-4` | `100%` |
| Ours, graph-coordinated circular + joint C0 | `1.620e-4` | `9.112e-4` | exact zero | `100%` |
| PLVIRA | `3.804e-2` | `9.166e-3` | `1.038e-3` | `100%` |
| PCIC, center translation | `2.567e-2` | `1.651e-3` | `6.852e-4` | `98.06%` |
| QUASI, frozen port | `4.711e-1` | `2.107e-1` | exact zero | `99.86%` |

At `N=128`, joint C0 improves median Hausdorff error by `158.5x` relative to
the best external baseline. The unrefined circular variants improve median
curvature error by `1.88x`; joint C0 improves it by `1.81x`. Joint C0 therefore
primarily improves geometric continuity and Hausdorff accuracy, while the
curvature observable remains approximately first order.

## Coverage And Conservation

PLVIRA reconstructs all `10,164` mixed cells. PCIC reconstructs `10,016` and
leaves `146` unresolved; its coverage decreases from `99.38%` at `N=32` to
`98.06%` at `N=128`. QUASI reconstructs `10,143`, retains `10` conservative
fallbacks, and leaves `11` numerically near-pure cells unresolved.

The maximum fitted-group conservation residual, normalized by the geometric
area of the fitted cell or merged group, is `8.45e-8` for the unrefined project
variants, `1.18e-10` for joint C0, `4.58e-9` for PLVIRA, `9.97e-11` for PCIC,
and `9.38e-17` for QUASI. The unrefined maximum comes from one cell in `N=64`,
case 8; all other project case maxima are below `6.32e-10`. The replay values
include clipping and saved-geometry precision.

## Qualifications

The PLVIRA implementation is operational and has full coverage, although one
fine-grid optimizer reports a line-search failure while still returning a
conservative reconstructed cell. PCIC uses the paper-motivated center-
translation correction; its unresolved intervals are retained in the coverage
metric rather than hidden by an added policy.

The QUASI result is a frozen Cartesian port of underspecified paper details.
The port fixes a documented root-selection, neighbor-selection, update-order,
boundary, and fallback policy before this study. None of the 75 cases satisfies
the combined sweep-convergence condition within the frozen ten-sweep cap, and
the run records `2,502` C1 root misses. Its exact facet gap therefore does not
establish accurate interface recovery. These qualifications must accompany any
paper use of the QUASI curve.

The project ellipse cases trigger no cell merges, although graph-based
orientation propagation changes `6/75` case-resolution reconstructions. Joint
C0 solves all `698/698` connected rejected-join components and leaves no
eligible bad joins. Of these, 697 reach exact C1 and one uses the conservative
fallback. Joint C0 introduces 13 locally concave arcs and 17 straight-limit
facets across all 25 cases and three resolutions; these are explicitly retained
in the saved diagnostics. The per-cell variant independently contains six
locally concave fine-grid arcs, while the unrefined graph-coordinated variant
contains none.

## Artifacts

- `experiments/baselines/results/ellipse_all_method_25case_comparison_20260814/ellipse_all_methods_metrics.pdf`
- `experiments/baselines/results/ellipse_all_method_25case_comparison_20260814/summary.csv`
- `experiments/baselines/results/ellipse_all_method_25case_comparison_20260814/case_metrics.csv`
- `experiments/baselines/results/common_native_metric_ellipse_25case_20260814/case_results.csv`
- `experiments/baselines/results/ellipse_circular_variants_joint_c0_25case_20260814/case_results.csv`

Reproduce the final panel after generating the method-specific inputs with:

```bash
PYTHONPATH=. python -m experiments.baselines.build_ellipse_all_method_comparison \
  --native-cases experiments/baselines/results/common_native_metric_ellipse_25case_20260814/case_results.csv \
  --ours-cases experiments/baselines/results/ellipse_circular_variants_joint_c0_25case_20260814/case_results.csv \
  --baseline-cases \
    experiments/baselines/results/plvira_ellipse_25case_20260814/case_results.csv \
    experiments/baselines/results/pcic_center_ellipse_25case_20260814/case_results.csv \
    experiments/baselines/results/quasi_ellipse_25case_20260814/case_results.csv \
  --output experiments/baselines/results/ellipse_all_method_25case_comparison_20260814 \
  --method-ids ours_per_cell ours_graph ours_c0 plvira pcic_center quasi
```
