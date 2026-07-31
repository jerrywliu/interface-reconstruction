# Final scientific results audit

## Verdict

The authoritative release passes the matched comparison against the July 17
baseline. All `300` July run keys and all `7,500` July case keys occur in the
authoritative release with the same `25` case indices per run. There are no
July-only runs, no case-grid mismatches, no duplicate keys, no metric
availability mismatches, and no missing required Hausdorff or facet-gap values.

The final sweep contains `970` runs and `24,250` cases. Its additional `670`
runs (`16,750` cases) are expanded method coverage and are not treated as
matched improvements or regressions. The exact matched profiles agree on all
`300` runs: LVIRA unresolved-cell fallback, `pre_f8_corner`, and the same
run-level rescue-profile field.

## Headline findings

- **The matched distribution is stable.** Across all `7,500` paired cases, the
  pooled Hausdorff median, p95, and maximum are unchanged at `1.133e-2`,
  `6.142e-1`, and `1.837e1`. Hausdorff tails above `1` decrease from `56` to
  `55`, with one fixed tail and no introduced tail.
- **Square perfect reconstruction is preserved exactly.** For
  `squares/linear+corner`, all `625` Hausdorff values and all `625` facet-gap
  values are exactly equal to July. All `25/25` settings retain both
  Hausdorff-median and joint Hausdorff/facet-gap median below `1e-6`; `3/25`
  settings have every case jointly below `1e-6` in both releases.
- **Zalesak's setting-level claim is preserved.** For
  `zalesak/circular+corner`, `17/25` settings retain joint median below `1e-6`,
  including `15/15` settings at `N>=100`; `4/25` settings have every case
  jointly below `1e-6` in both releases. The pooled Hausdorff median
  (`1.287e-8`), p95 (`5.560e-1`), maximum (`5.108`), and five tails above `1`
  are unchanged to the reported precision.
- **One coarse Zalesak corner case materially worsens without changing a
  threshold claim.** At `N=50`, perturbation magnitude `0.05`, case `6`,
  Hausdorff changes from `0.6131` to `0.6374` and facet gap from `0.02088` to
  `0.03177`. There are no material Hausdorff or facet-gap changes for this
  method at `N>=100`.
- **Smooth-interface metrics are materially unchanged.** Circle and ellipse
  curvature, tangent, and curvature-proxy errors have zero material changes;
  the line method also has zero material changes. Across the non-area metrics,
  `23,967/24,000` values are stable, `15` improve, and `18` worsen under the
  stated tolerance. Most changes belong to non-corner circular square/Zalesak
  rows; they do not alter the paper-facing corner-method median claims.
- **Area error is a measurement change, not an apples-to-apples algorithm
  ablation.** July predates the geometry-faithful active-partition area repair.
  The final values supersede it: `2,532/3,750` paired area values materially
  decrease and none increase, but those deltas should not be cited as method
  improvements. The Hausdorff and facet-gap checks above remain the appropriate
  reconstruction comparison.

## Coverage differences

July contains only the affected-method subset:

| Benchmark | July methods | Final-only methods |
|---|---|---|
| lines | linear | Youngs, ELVIRA, LVIRA, safe_linear |
| circles | linear, circular | Youngs, ELVIRA, LVIRA, safe_linear, safe_circle |
| ellipses | linear, circular | Youngs, ELVIRA, LVIRA, safe_linear, safe_circle |
| squares | linear, circular, linear+corner | Youngs, ELVIRA, LVIRA, safe_linear, safe_circle |
| Zalesak | linear, circular, circular+corner | Youngs, ELVIRA, LVIRA, safe_linear, safe_circle |

`coverage_profile.csv` records exact run/case counts, resolutions,
perturbation magnitudes, seeds, and profiles for every benchmark/method in both
releases. Final-only rows are intentionally excluded from matched statistics.

The July source state records a dirty worktree at commit `d02dd47`, but its
exact source snapshot is frozen and checksum-bound in the release. The final
release comes from clean commit `505aefa`. This audit therefore compares the
saved release artifacts and their frozen source snapshots, not commit labels
alone.

## Definitions

- Exact run key: `(benchmark, method, resolution, perturbation magnitude, seed)`.
- Exact case key: the run key plus `case_index`.
- Key metrics: Hausdorff, facet gap, area error, curvature error, tangent error,
  and curvature-proxy error wherever each benchmark defines it.
- p95: linearly interpolated 95th percentile of the matched case values.
- Hausdorff tail: value greater than `1`.
- Material case change: absolute delta greater than the larger of `1e-10` and
  `1%` of the larger paired magnitude. Smaller differences are numerical
  stability noise and remain available in the case-level CSV.
- Perfect-reconstruction setting: both median Hausdorff and median facet gap are
  below `1e-6`. The detailed table also reports Hausdorff-only and all-case
  checks.

All stored aggregate medians were independently recomputed from case rows:
`4,710/4,710` checks agree exactly across the two releases. The sorted July and
matched-final case-key SHA-256 digests agree, as do the aggregate-key digests.

## Artifacts

- `REPORT.md`: generated method-level Hausdorff table, perfect-reconstruction
  table, and largest matched tails.
- `setting_metric_comparison.csv`: primary machine-readable result, with
  medians, p95, maxima, tail counts, and improvement/regression counts by
  benchmark, method, resolution, perturbation magnitude, seed, and metric.
- `method_metric_comparison.csv`: the same statistics pooled by
  benchmark/method/metric.
- `case_metric_comparison.csv`: every paired case value and material outcome.
- `material_changes.csv`: compact ledger of all material improvements and
  regressions across every metric.
- `tail_cases.csv`: Hausdorff tails, fixed/introduced tails, and fallback-cell
  counts.
- `perfect_reconstruction.csv`: setting-level square and Zalesak threshold
  checks.
- `run_coverage.csv`, `coverage_profile.csv`, and `metric_coverage.csv`: exact
  key, method/profile, and metric-availability coverage.
- `integrity_audit.csv`: pass/fail checks and exact sorted-key digests.
- `input_checksums.csv`: SHA-256 checksums for the immutable input manifests,
  scientific tables, source states, and source snapshots.
- `comparison.json`: machine-readable summary and thresholds.
- `verify_and_summarize.py`: reproducible integrity and supplemental-table
  generator.
- `SHA256SUMS`: checksum ledger for the report, code, JSON, and CSV artifacts.

## Reproduction

From the repository root, point `INTERFACE_RESULTS_ROOT` at the directory that
contains the two release directories. It defaults to this checkout's
`results/static` directory. Choose an empty temporary output directory and
regenerate the core comparison with:

```bash
export INTERFACE_RESULTS_ROOT="${INTERFACE_RESULTS_ROOT:-$PWD/results/static}"
export BASELINE_ROOT="$INTERFACE_RESULTS_ROOT/static_paper_simplified_default_20260717_212413"
export CANDIDATE_ROOT="$INTERFACE_RESULTS_ROOT/submission_static_20260731_012430_505aefa45432"
export AUDIT_OUT="$(mktemp -d "${TMPDIR:-/tmp}/final-scientific-results.XXXXXX")"

python submission/compare_release_results.py \
  --baseline-root "$BASELINE_ROOT" \
  --candidate-root "$CANDIDATE_ROOT" \
  --output-dir "$AUDIT_OUT" \
  --report-tail-limit 30
```

Then run the additional integrity checks and supplemental tables:

```bash
python submission/audits/final_scientific_results_2026-07-31/verify_and_summarize.py \
  --baseline-root "$BASELINE_ROOT" \
  --candidate-root "$CANDIDATE_ROOT" \
  --output-dir "$AUDIT_OUT"
```
