# Compare the Final Release with July 2026

`compare_release_results.py` compares the shared deterministic cases in two
completed static-sweep releases. It is intended for the final 970-run release
and the authoritative July 2026 affected-method release.

The 970-run release contains 670 method runs that July does not. Those rows are
reported as candidate-only coverage and are not an error. Every July run should
exist in the final release, and every case index inside a matched run should be
identical.

## Run

Use explicit roots for the submission audit:

```bash
python submission/compare_release_results.py \
  --baseline-root results/static/static_paper_simplified_default_20260717_212413 \
  --candidate-root results/static/submission_static_<UTC>_<commit> \
  --output-dir results/submission/release_comparisons/final_vs_july
```

Omit either root to discover the newest completed, zero-failure directory under
`results/static/` matching `static_paper_simplified_default_*` or
`submission_static_*`. Running, failed, incomplete, and malformed candidates are
not eligible. The output directory must be empty and must be outside both input
release roots.

## Outputs

- `REPORT.md`: concise human-readable coverage and scientific summary.
- `comparison.json`: machine-readable summary, thresholds, provenance, and issues.
- `run_coverage.csv`: union of run keys and matched case counts.
- `method_metric_comparison.csv`: benchmark/method medians, p95 values, maxima,
  and material outcome counts.
- `setting_metric_comparison.csv`: the same statistics for each
  benchmark/method/resolution/perturbation/seed key.
- `case_metric_comparison.csv`: long-form paired case values and deltas.
- `tail_cases.csv`: Hausdorff tails, threshold crossings, and material regressions.
- `perfect_reconstruction.csv`: square `linear+corner` and Zalesak
  `circular+corner` threshold checks.

The default perfect-reconstruction threshold is `1e-6` for both the setting
median Hausdorff and setting median facet gap. The output also records
Hausdorff-only and all-case joint checks. Hausdorff tails are summarized by the
paired 95th percentile, maximum, and the default threshold `1.0`.

All comparisons use `diagnostics/case_metrics.csv`; aggregate CSV rows are not
trusted as a substitute for matched cases. Numeric run keys use decimal
normalization, so values such as `0.1` and `0.10` match without binary-float key
ambiguity.

The July square and Zalesak `area_error` columns predate the geometry-faithful
metric repair. They are exported for audit, but they are not semantically
equivalent submission evidence. Final area-error claims should use the final
release alone.
