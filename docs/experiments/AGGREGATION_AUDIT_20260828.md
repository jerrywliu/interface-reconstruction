# Section 6 Aggregation Audit

## Question

The current resolution panels first compute a median over 25 configurations at
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

## Artifacts

- Pooled paper-style panels:
  `results/submission/section6_pooled_20260828/summary_plots/`
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
```
