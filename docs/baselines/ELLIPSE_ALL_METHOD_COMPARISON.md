# Ellipse All-Method Comparison

## Scope

This matched Cartesian diagnostic compares canonical ellipse cases `0--4` at
`N=32,64,128`. Native symmetric Hausdorff distance and geometric-curvature
error use the common partition-insensitive observables. Facet gap and mixed-cell
coverage come from the corresponding reconstruction diagnostics.

The compared methods are the per-cell, graph-coordinated, and joint-C0
circular variants; PLVIRA; both frozen PCIC corrections; and the frozen QUASI
port. The PCIC and QUASI results retain their documented implementation
qualifications.

## Observed Orders

| Method | Hausdorff | Curvature | Facet gap |
|---|---:|---:|---:|
| Ours, per-cell circular | `2.912` | `0.986` | `3.003` |
| Ours, graph-coordinated circular | `2.912` | `0.986` | `3.003` |
| Ours, graph-coordinated circular + joint C0 | `3.249` | `1.059` | exact zero |
| PLVIRA | `0.569` | `-0.128` | `2.060` |
| PCIC, center translation | `-0.347` | `0.614` | `2.993` |
| PCIC, radius adjustment | `-0.122` | `0.618` | `2.977` |
| QUASI, frozen port | `0.954` | `-1.060` | exact zero |

## Fine-Grid Comparison

At `N=128`, the strongest non-project baseline differs by metric. Relative to
that baseline:

| Circular variant | Hausdorff improvement | Curvature improvement | Nonzero facet-gap improvement |
|---|---:|---:|---:|
| Graph-coordinated circular | `136.9x` | `2.13x` | `3.20x` |
| Graph-coordinated circular + joint C0 | `635.1x` | `2.15x` | exact zero |

The external methods' best fine-grid Hausdorff and curvature values are PCIC's
`2.5084e-2` and `7.5160e-4`, respectively. The joint-C0 circular values are
`3.9498e-5` and `3.4946e-4`.

## Interpretation

Within this five-case Cartesian screen, the circular method is the clear
overall winner. It has the lowest native Hausdorff and curvature errors at all
three resolutions, third-order geometry/facet-gap behavior before C0, and full
mixed-cell coverage. Joint C0 solves all `120/120` connected rejected-join
components, leaves no eligible bad joins, and reduces the facet gap to zero
while preserving relative cell area below `1e-10`. It further improves geometry
but does not change the approximately first-order curvature regime.

QUASI and joint C0 both have exactly coincident eligible endpoints, so their
recorded facet gaps are zero. That isolated QUASI result does not translate
into interface accuracy. QUASI has the largest Hausdorff and
curvature errors, its curvature regresses with refinement, and its frozen C1
iteration does not satisfy the combined convergence check.

PCIC curvature remains competitive at coarse resolution, but its native
Hausdorff error regresses at `N=128` as missing intervals appear. Its aggregate
ellipse mixed-cell coverage falls to approximately `98%`. PLVIRA preserves full
coverage but is less accurate for these ellipses.

These results support a strong ellipse claim for the current circular method,
but not a second-order curvature claim. The evidence is a matched five-case
diagnostic, not yet a full 25-case higher-order-baseline study.

## Artifacts

- `experiments/baselines/results/ellipse_all_method_comparison_20260814/ellipse_all_methods_metrics.pdf`
- `experiments/baselines/results/ellipse_all_method_comparison_20260814/summary.csv`
- `experiments/baselines/results/ellipse_all_method_comparison_20260814/case_metrics.csv`

Reproduce with:

```bash
PYTHONPATH=. python -m \
  experiments.baselines.build_ellipse_all_method_comparison
```
