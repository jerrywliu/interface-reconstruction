# Extended Cartesian Ellipse Paper Figure

## Scope

This export is a presentation-only rendering of the frozen six-method,
25-case Cartesian ellipse study at
`N=32,50,64,100,128,150,256,300`. It does not rerun reconstruction or alter
method policies. The authoritative input package is:

`experiments/baselines/results/ellipse_all_method_25case_extended_comparison_20260814/`

Before plotting, the generator verifies every input and packaged-artifact hash
in the frozen `manifest.json`, checks the complete `6 x 8 x 25 = 1,200` case
grid, and recomputes the plotted summary from `case_metrics.csv` against the
frozen `summary.csv`.

## Figure Convention

- Full manuscript-width `2 x 2` panel with no embedded figure-level title.
- The approved Appendix B serif typography with normal-weight panel titles and
  labels, plus the manuscript palette, circular markers, and current
  line-weight/linestyle conventions.
- Approved labels: `Per-cell`, `Graph-coordinated`, and
  `Graph-coordinated + joint C0`.
- Error panels show the median and interquartile range over 25 matched cases.
- Coverage is the aggregate reconstructed/mixed-cell ratio over each 25-case
  resolution group, matching the frozen comparison summary.
- Small reference triangles show orders three, one, and three for native
  Hausdorff, geometric-curvature MAE, and facet gap, respectively.
- Exact zero facet gaps are displayed at `1e-12`; the value is labeled in the
  panel and recorded in the manifest.

## Artifacts

- `ellipse_all_methods_metrics_paper.pdf`: one-page vector manuscript figure.
- `ellipse_all_methods_metrics_paper.png`: 300-DPI visual-review render.
- `ellipse_all_methods_paired_win_counts.csv`: all 15 method pairs for four
  metrics at each resolution and over all 200 matched case-resolution pairs.
- `ellipse_all_methods_metrics_paper.sha256`: hashes for the PDF, PNG, and
  paired-win CSV.
- `ellipse_all_methods_metrics_paper.manifest.json`: frozen-input provenance,
  generator hashes, study grid, display semantics, artifact hashes, and PDF QA.

Paired wins use lower-is-better for the three error metrics and
higher-is-better for coverage. Ties use relative tolerance `1e-12` and absolute
tolerance `1e-15`; the tolerances are recorded in both code and the manifest.

## Reproduction And QA

```bash
PYTHONPATH=. python \
  experiments/baselines/plot_ellipse_all_method_paper_comparison.py

PYTHONPATH=. pytest -q \
  test/experiments/test_ellipse_all_method_paper_comparison.py \
  test/experiments/test_ellipse_all_method_comparison.py \
  test/experiments/test_convergence_plotting.py
```

Generation fails if a frozen data hash or output hash changes, if the PDF is
not one page, if Poppler reports raster objects or unembedded fonts, or if the
approved labels are missing. No manuscript or memory files are modified.
