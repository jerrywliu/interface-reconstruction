# Submission release comparison

Status: **pass**

- Baseline: `$INTERFACE_RESULTS_ROOT/static_paper_simplified_default_20260717_212413`
- Candidate: `$INTERFACE_RESULTS_ROOT/submission_static_20260731_012430_505aefa45432`
- Matched coverage: `300` runs / `7500` cases; `670` candidate-only runs and `0` baseline-only runs.
- Exact case grids: `300/300` matched runs.

Candidate-only runs are expected here because the final sweep contains all paper methods while July contains the affected-method subset.

## Matched Hausdorff summary

| Benchmark / method | Cases | Median baseline/candidate | p95 baseline/candidate | Max baseline/candidate | > tail baseline/candidate | I/W |
|---|---:|---:|---:|---:|---:|---:|
| `circles/circular` | 750 | 6.388e-10 / 6.388e-10 | 4.759e-09 / 4.759e-09 | 8.650e+00 / 8.650e+00 | 7 / 7 | 0 / 0 |
| `circles/linear` | 750 | 3.577e-02 / 3.577e-02 | 1.880e-01 / 1.880e-01 | 8.659e+00 / 8.659e+00 | 7 / 7 | 0 / 0 |
| `ellipses/circular` | 750 | 2.375e-03 / 2.375e-03 | 1.660e-02 / 1.660e-02 | 1.836e+01 / 1.836e+01 | 11 / 11 | 0 / 0 |
| `ellipses/linear` | 750 | 1.545e-02 / 1.545e-02 | 8.267e-02 / 8.267e-02 | 1.837e+01 / 1.837e+01 | 11 / 11 | 0 / 0 |
| `lines/linear` | 750 | 3.733e-10 / 3.733e-10 | 2.132e-09 / 2.132e-09 | 8.527e-01 / 8.527e-01 | 0 / 0 | 0 / 0 |
| `squares/circular` | 625 | 2.915e-01 / 2.908e-01 | 5.875e-01 / 5.868e-01 | 1.825e+00 / 1.825e+00 | 5 / 4 | 3 / 3 |
| `squares/linear` | 625 | 3.296e-01 / 3.296e-01 | 7.325e-01 / 7.325e-01 | 1.825e+00 / 1.825e+00 | 3 / 3 | 0 / 0 |
| `squares/linear+corner` | 625 | 1.280e-09 / 1.280e-09 | 9.715e-03 / 9.715e-03 | 6.460e-01 / 6.460e-01 | 0 / 0 | 0 / 0 |
| `zalesak/circular` | 625 | 3.219e-01 / 3.226e-01 | 6.842e-01 / 6.842e-01 | 1.526e+00 / 1.526e+00 | 2 / 2 | 1 / 3 |
| `zalesak/circular+corner` | 625 | 1.287e-08 / 1.287e-08 | 5.560e-01 / 5.560e-01 | 5.108e+00 / 5.108e+00 | 5 / 5 | 0 / 1 |
| `zalesak/linear` | 625 | 3.511e-01 / 3.511e-01 | 8.173e-01 / 8.173e-01 | 1.278e+00 / 1.278e+00 | 5 / 5 | 0 / 0 |

## Perfect-reconstruction checks

Threshold: both setting-median Hausdorff and facet gap below `1.0e-06`. The CSV also records Hausdorff-only and all-case joint checks.

| Benchmark / method | Settings | H-median floor baseline/candidate | Joint-median floor baseline/candidate | All-case joint floor baseline/candidate | Lost/gained |
|---|---:|---:|---:|---:|---:|
| `squares/linear+corner` | 25 | 25 / 25 | 25 / 25 | 3 / 3 | 0 / 0 |
| `zalesak/circular+corner` | 25 | 17 / 17 | 17 / 17 | 4 / 4 | 0 / 0 |
| `zalesak/circular+corner (N>=100)` | 15 | n/a | 15 / 15 | n/a | n/a |

## Largest matched tails and regressions

| Benchmark / method | N | w | Case | H baseline/candidate | Delta | Reasons |
|---|---:|---:|---:|---:|---:|---|
| `ellipses/linear` | 50 | 0.3 | 20 | 1.837e+01 / 1.837e+01 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/circular` | 50 | 0.3 | 20 | 1.836e+01 / 1.836e+01 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/linear` | 50 | 0.2 | 20 | 1.832e+01 / 1.832e+01 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/circular` | 50 | 0.2 | 20 | 1.831e+01 / 1.831e+01 | 3.553e-15 | `baseline_tail;candidate_tail` |
| `ellipses/linear` | 50 | 0.3 | 15 | 1.416e+01 / 1.416e+01 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/circular` | 50 | 0.3 | 15 | 1.415e+01 / 1.415e+01 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/linear` | 50 | 0.3 | 10 | 1.409e+01 / 1.409e+01 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/circular` | 50 | 0.3 | 10 | 1.408e+01 / 1.408e+01 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/linear` | 50 | 0.2 | 10 | 1.321e+01 / 1.321e+01 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/circular` | 50 | 0.2 | 10 | 1.318e+01 / 1.318e+01 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/linear` | 64 | 0.3 | 22 | 1.138e+01 / 1.138e+01 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/circular` | 64 | 0.3 | 22 | 1.136e+01 / 1.136e+01 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/linear` | 150 | 0.1 | 7 | 1.092e+01 / 1.092e+01 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/circular` | 150 | 0.1 | 7 | 1.091e+01 / 1.091e+01 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/linear` | 32 | 0.2 | 5 | 9.626e+00 / 9.626e+00 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/circular` | 32 | 0.2 | 5 | 9.590e+00 / 9.590e+00 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `circles/linear` | 100 | 0.05 | 0 | 8.659e+00 / 8.659e+00 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `circles/circular` | 100 | 0.05 | 0 | 8.650e+00 / 8.650e+00 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/circular` | 50 | 0.1 | 24 | 8.558e+00 / 8.558e+00 | -2.132e-14 | `baseline_tail;candidate_tail` |
| `ellipses/linear` | 50 | 0.1 | 24 | 8.546e+00 / 8.546e+00 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `circles/linear` | 32 | 0.3 | 8 | 5.492e+00 / 5.492e+00 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `circles/circular` | 32 | 0.3 | 8 | 5.409e+00 / 5.409e+00 | -8.882e-16 | `baseline_tail;candidate_tail` |
| `zalesak/circular+corner` | 150 | 0.2 | 23 | 5.108e+00 / 5.108e+00 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `circles/linear` | 150 | 0.1 | 18 | 4.866e+00 / 4.866e+00 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `circles/circular` | 150 | 0.1 | 18 | 4.861e+00 / 4.861e+00 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `zalesak/circular+corner` | 100 | 0.2 | 21 | 4.445e+00 / 4.445e+00 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `circles/linear` | 128 | 0.1 | 16 | 4.355e+00 / 4.355e+00 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `circles/circular` | 128 | 0.1 | 16 | 4.349e+00 / 4.349e+00 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/linear` | 128 | 0.2 | 24 | 3.892e+00 / 3.892e+00 | 0.000e+00 | `baseline_tail;candidate_tail` |
| `ellipses/circular` | 128 | 0.2 | 24 | 3.889e+00 / 3.889e+00 | 0.000e+00 | `baseline_tail;candidate_tail` |

## Interpretation

- Candidate-only run keys are expected when comparing the 970-run all-method release with the 300-run July affected-method subset.
- July square and Zalesak `area_error` values predate the geometry-faithful repair; `area_error` pairs are exported for audit but are not submission-equivalent evidence.
- All error metrics are interpreted as smaller-is-better; tails use the paired 95th percentile, maximum, and Hausdorff threshold counts.

## Artifacts

- `comparison.json`: summary, thresholds, coverage, and issues
- `run_coverage.csv`: matched and unmatched run keys
- `method_metric_comparison.csv`: benchmark/method summaries
- `setting_metric_comparison.csv`: resolution/wiggle/seed summaries
- `case_metric_comparison.csv`: paired case-level metric values
- `tail_cases.csv`: Hausdorff tails and material regressions
- `perfect_reconstruction.csv`: square and Zalesak threshold checks
