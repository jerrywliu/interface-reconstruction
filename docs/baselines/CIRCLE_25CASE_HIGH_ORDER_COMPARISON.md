# Frozen Six-Method Cartesian Circle Comparison

This study compares six frozen methods on exactly 25 canonical radius-10 circle placements over uniform Cartesian meshes with `w=0` and `N={32,50,64,100,128,150,256,300}`. No external-method policy was changed after outcomes were inspected.

## Protocol

- Project methods: per-cell circular, graph-coordinated circular, and graph-coordinated circular plus the production joint C0 pass.
- External methods: operational PLVIRA, bare PCIC with center translation, and the frozen Cartesian QUASI reproduction.
- Common observables: partition-invariant native symmetric Hausdorff, arc-length-weighted unsigned geometric-curvature MAE, facet gap, mixed-cell coverage/status, and normalized conservation residual.
- Support QA: `1200/1200` unique finite case rows and `48/48` summary rows.

## Results At N=300

| Method | Hausdorff median | Curvature MAE median | Facet-gap median | Coverage | Unresolved |
|---|---:|---:|---:|---:|---:|
| Ours: circular (per-cell) | 3.460221e-09 | 1.391627e-09 | 1.966082e-10 | 100.000% | 0 |
| Ours: circular (graph-coordinated) | 3.460221e-09 | 1.391627e-09 | 1.966082e-10 | 100.000% | 0 |
| Ours: circular (graph-coordinated + joint C0) | 2.718556e-09 | 7.228977e-09 | 5.121586e-11 | 100.000% | 0 |
| PLVIRA | 1.541452e-06 | 3.960470e-05 | 6.816081e-07 | 100.000% | 0 |
| PCIC (center translation) | 3.334861e-02 | 1.463267e-03 | 3.335419e-05 | 96.400% | 216 |
| QUASI (frozen port) | 1.988651e-01 | 4.356181e-01 | 0.000000e+00 | 99.950% | 3 |

## Fitted Median Orders

| Method | Hausdorff | Curvature | Facet gap |
|---|---:|---:|---:|
| Ours: circular (per-cell) | 0.771 | -0.634 | 1.283 |
| Ours: circular (graph-coordinated) | 0.312 | -0.904 | 0.951 |
| Ours: circular (graph-coordinated + joint C0) | 0.169 | -0.834 | 0.487 |
| PLVIRA | 4.861 | 2.534 | 4.583 |
| PCIC (center translation) | -0.869 | 0.084 | 2.188 |
| QUASI (frozen port) | 0.898 | -1.083 | n/a |

## Status And Qualifications

The production joint C0 pass solves `38/38` recorded components, with `0` failed components and `0` remaining eligible bad joins. Its maximum recorded relative C0 area residual is `1.778e-10`.

PLVIRA retains its frozen Cartesian boundary-halo and GHF policies. PCIC uses only the preselected center-translation conservation correction; radius adjustment is excluded. QUASI retains the frozen ten-sweep stopping and fallback rules; `0/200` circle cases report sweep convergence.

The maximum fitted-group normalized conservation residual over all methods and cases is `4.325e-08`. Exact-zero facet gaps are retained in CSV and shown at a labeled plotting floor only in the log-scale figure.

## Artifacts

- `circle_all_methods_metrics_paper.pdf` and `.png`: primary paper-ready all-method panel, rendered with the exact ellipse paper figure style and no figure-level title.
- `circle_all_methods_metrics.pdf` and `.png`: generic diagnostic comparison retained for internal review.
- `summary.csv` and `case_metrics.csv`: plotted aggregate values and the complete provenance-bearing 1,200-row case table.
- `circle_all_methods_metrics_paper.manifest.json`: frozen-input, display, hash, and vector-PDF provenance for the paper panel.

## Reproduce

```bash
PYTHONPATH=. python -m experiments.baselines.build_circle_all_method_comparison
PYTHONPATH=. python -m experiments.baselines.plot_circle_all_method_paper_comparison
```
