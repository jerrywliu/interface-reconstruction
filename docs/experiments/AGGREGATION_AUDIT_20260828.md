# Section 6 Aggregation Audit

## Question

The historical resolution panels first computed a median over 25 configurations at
each perturbation magnitude, then show the median and interquartile range (IQR)
of the five perturbation-level medians. This audit compares that statistic with
the median and IQR of all `5 x 25 = 125` case/perturbation observations at each
resolution.

## Result

- The pooled center lines differ by at most `9-16%` at substantive-error points,
  depending on the benchmark.
- The best-performing method does not change at any of the 56 plotted
  benchmark/metric/resolution settings.
- Observed convergence rates change by at most about `0.11`; the ellipse
  circular-facet gap rate changes from `2.94` to `2.92` and remains third order.
- The pooled IQR is wider at essentially every non-floor point. The median
  multiplicative increase in band width is `1.10x-1.54x` across benchmarks.

The pooled statistic is the clearer paper-facing choice. It summarizes the
full empirical distribution, while still weighting each perturbation magnitude
equally because every magnitude contains the same 25 configurations. The
current band instead measures only variation among five already-compressed
medians and omits within-perturbation case variability.

## Paper-wide audit and promotion

The same two-stage aggregation was also used by the five perturbed-mesh
appendix panels and by the compact joint-C0 resolution study. These panels were
regenerated from case-level rows and promoted together with the five Section 6
panels. For plots versus perturbation magnitude, pooling is over all plotted
resolutions and cases: `6 x 25 = 150` observations for lines, circles, and
ellipses, and `5 x 25 = 125` for squares and Zalesak.

The Cartesian higher-order circle and ellipse comparisons already compute each
median and IQR directly over 25 configurations at fixed resolution. The
fine-grid line-only circle study likewise uses the five case rows directly.
Neither contains a hidden perturbation stratum, so neither required a change.
Count/incidence tables and qualitative reconstruction panels are not affected
by this aggregation choice.

The pooled assets and matching blue manuscript updates are live on Overleaf at
commit `efe16d5`. The 43-page paper build remains vector-only and passed visual
review on all affected main-text and appendix pages.

## Artifacts

- Pooled paper-style panels:
  `results/submission/section6_pooled_20260828/summary_plots/`
- Pooled perturbed appendix panels:
  `results/submission/perturbed_pooled_panels_20260828/`
- Pooled compact joint-C0 study:
  `results/submission/c0_pooled_panels_20260828/`
- Side-by-side five-page audit:
  `output/pdf/aggregation_audit_20260828/aggregation_comparison_all_benchmarks.pdf`
- Exact current and pooled quantiles:
  `output/pdf/aggregation_audit_20260828/aggregation_quantiles.csv`
- Source case ledger:
  `results/static/submission_static_20260731_012430_505aefa45432.sealed/diagnostics/case_metrics.csv`

## Reproduction

```bash
python experiments/static/generate_section6_maintext_figures.py \
  --csv results/static/submission_static_20260731_012430_505aefa45432.sealed/perturbed_sweep.csv \
  --case_metrics_csv results/static/submission_static_20260731_012430_505aefa45432.sealed/diagnostics/case_metrics.csv \
  --resolution_aggregation pooled_cases \
  --figure_groups quantitative_resolution \
  --experiments all \
  --out_dir results/submission/section6_pooled_20260828

python experiments/static/generate_pooled_perturbed_panels.py
python experiments/static/generate_pooled_c0_panels.py
```
