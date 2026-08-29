# Manuscript Convergence-Triangle Reaudit

## Scope

This audit covers every convergence-order triangle in the currently included manuscript figures.
The six triangle-bearing PDF assets contain 15 markers in total.
All assets were inspected in their rendered manuscript pages and again from the source PDFs at high resolution.

| Camera-ready asset | Current source PDF | Generator | Markers | Result |
| --- | --- | --- | ---: | --- |
| `ellipse_reconstruction_maintext_resolution_metrics.pdf` | `results/submission/maintext_style_refresh_20260829/summary_plots/ellipses_maintext_resolution_metrics.pdf` | `experiments/static/generate_section6_maintext_figures.py` | 1 | Clear; unchanged |
| `ellipse_higher_order_comparison.pdf` | `results/submission/maintext_style_refresh_20260829/additional/ellipse_all_methods_metrics_paper.pdf` | `experiments/baselines/plot_ellipse_all_method_paper_comparison.py` | 3 | Curvature marker moved; other two clear |
| `circle_reconstruction_perturbed_all_methods_5x2_axes.pdf` | `results/submission/triangle_audit_20260829/pooled/circle_reconstruction_perturbed_all_methods_5x2_axes.pdf` | `experiments/static/generate_pooled_perturbed_panels.py` | 2 | Both markers moved above the broad circular per-cell IQR band |
| `line_circle_extended_convergence.pdf` | `results/static/extended_convergence_smoke_line_circle_20260822/line_circle_extended_convergence.pdf` | `experiments/static/analyze_extended_line_circle.py` | 4 | Clear; unchanged |
| `circle_higher_order_comparison.pdf` | `results/submission/triangle_audit_20260829/higher_order_circle/circle_all_methods_metrics_paper.pdf` | `experiments/baselines/plot_circle_all_method_paper_comparison.py` and the shared ellipse plotter | 3 | Hausdorff and curvature markers moved; facet-gap marker clear |
| `ellipse_reconstruction_perturbed_all_methods_5x2_axes.pdf` | `results/submission/triangle_audit_20260829/pooled/ellipse_reconstruction_perturbed_all_methods_5x2_axes.pdf` | `experiments/static/generate_pooled_perturbed_panels.py` | 2 | Facet-gap marker moved into the gap between method families; curvature marker clear |

No other included non-TikZ manuscript figure contains a convergence-order triangle.
The final placements do not intersect data curves, uncertainty bands, labels, axes, or annotations.

## Presentation-Only Changes

Only triangle anchor coordinates and the higher-order circle anchor override were changed.
The data, aggregation, methods, colors, line styles, markers, uncertainty bands, axis limits, convergence orders, and all other scientific content are unchanged.

Changed source paths:

- `experiments/static/generate_pooled_perturbed_panels.py`
- `experiments/baselines/plot_ellipse_all_method_paper_comparison.py`
- `experiments/baselines/plot_circle_all_method_paper_comparison.py`
- `test/experiments/test_generate_pooled_perturbed_panels.py`
- `test/experiments/test_circle_all_method_paper_comparison.py`

## Final Candidates

- `final/pooled/circle_reconstruction_perturbed_all_methods_5x2_axes.pdf`
- `final/pooled/ellipse_reconstruction_perturbed_all_methods_5x2_axes.pdf`
- `final/higher_order_circle/circle_all_methods_metrics_paper.pdf`
- `final/higher_order_ellipse/ellipse_all_methods_metrics_paper.pdf`

## Validation

- Focused tests: `16 passed`.
- Vector PDF QA: `4/4 passed`.
- Every candidate is a one-page vector PDF with zero raster image objects and embedded DejaVu Serif fonts.
