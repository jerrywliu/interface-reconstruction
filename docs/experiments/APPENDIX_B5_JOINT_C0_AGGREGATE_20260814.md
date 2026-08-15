# Appendix B.5 joint-C0 aggregate study (2026-08-14)

## Scope

This run replaces the historical guarded endpoint correction in Appendix B.5
with the production joint-C0 refinement. It covers 25 cases for every paper
resolution and perturbation magnitude:

- Ellipses: `N = 32, 50, 64, 100, 128, 150` and
  `w = 0, 0.05, 0.10, 0.20, 0.30`.
- Zalesak: `N = 50, 64, 100, 128, 150` and the same five values of `w`.
- Generated C0 variants: graph-coordinated linear + joint C0 for ellipses and
  graph-coordinated circular + joint C0 for Zalesak.
- Unchanged no-C0 variants were imported from the sealed 2026-08-03 package.

The complete result bundle is
`results/static/camera_ready/appendix_b5_joint_c0_20260814/`.

## Command and runtime

```bash
PYTHONPATH=. python -m experiments.static.run_appendix_c0_study \
  --only ellipses,zalesak \
  --algos linear,linear+C0,circular,circular+C0,circular+corner \
  --ellipses 25 --zalesak 25 \
  --c0_mode joint --workers 6 --reuse_existing \
  --endpoint_variants paired \
  --save_prefix appendix_b5_joint_c0_20260814 \
  --out_dir results/static/camera_ready/appendix_b5_joint_c0_20260814
```

The full generated sweep plus the two-setting retry occupied about 82 minutes
of wall time (22:21:51--23:44:04 PDT). Once those 55 generated settings are
present, validation, joining, QA, and figure generation take about 29 seconds.
The retry was needed because two deterministic initial guesses supplied to
SciPy were outside their bounds. The solver now skips only such infeasible
guesses and continues through its existing alternate seeds; all feasible solve
paths are unchanged.

## Reused no-C0 provenance

No-C0 rows come from
`results/submission/final_figures_87c40309d16c_20260803_final/provenance/guarded_c0`
at source commit `87c40309d16c9be8b56393a7326c0e2f5e498291`.

| Benchmark | Input | SHA256 |
| --- | --- | --- |
| Ellipses | `ellipses/metrics.csv` | `5fa5562efddc06645324be93c9b9a9232be756bb4d69bb2b8e967416d431c220` |
| Ellipses | `ellipses/manifest.json` | `7bea4892d0e45fcbbad68bfb2353bdf4515a3eaff847d1bb8744d3aee360f976` |
| Zalesak | `zalesak/metrics.csv` | `2a3a190cfb7f7d1ca8ec20cdba1f3e3a61630a4867a1da43815b83d3bf1816b6` |
| Zalesak | `zalesak/manifest.json` | `875e9746bd31aedcda6a2ab734ddea73b881041314f8088f96bc5be3e51c8561` |

The aggregate CSV has 2,700 unique metric rows: 1,800 hash-pinned reused
no-C0 rows and 900 newly generated joint-C0 rows. The case QA table has all
1,375 expected joint-C0 cases.

## Joint versus guarded C0

The values below are medians over the per-setting case medians. A setting is a
fixed `(N, w, seed)` combination.

| Benchmark and metric | Guarded | Joint | Joint/guarded | Settings better/equal/worse |
| --- | ---: | ---: | ---: | ---: |
| Ellipse Hausdorff | `1.0251e-2` | `8.1291e-3` | `0.793` | `30 / 0 / 0` |
| Ellipse facet gap | `2.5886e-3` | `0` | `0` | `30 / 0 / 0` |
| Zalesak Hausdorff | `3.0866e-1` | `2.8787e-1` | `0.933` | `24 / 1 / 0` |
| Zalesak facet gap | `3.1944e-3` | `8.7899e-11` | `2.75e-8` | `25 / 0 / 0` |

Thus joint C0 is never worse in the matched aggregate comparisons. The main
effect is continuity: ellipse median facet gap reaches zero, and the Zalesak
median falls by roughly eight orders of magnitude.

## Conservation, continuity, and failure QA

| QA quantity | Ellipses | Zalesak |
| --- | ---: | ---: |
| Cases | 750 | 625 |
| Joint components solved / attempted | `7551 / 7555` (99.95%) | `1534 / 1660` (92.41%) |
| Cases with a retained failed component | 4 | 114 |
| Bad joins before / after | `19418 / 22` | `4095 / 418` |
| Bad-join reduction | 99.89% | 89.79% |
| Median facet gap | `0` | `8.30e-11` |
| Maximum component-relative conservation residual | `2.07e-9` | `7.50e-5` |
| Maximum global mixed-cell conservation residual | `2.17e-11` | `9.50e-7` |

There are no missing final facets and no non-finite Hausdorff or facet-gap
values. Failed joint components retain the conservative pre-refinement facets;
they are counted explicitly rather than dropped. The largest independent
conservation residual is Zalesak `N=100`, `w=0.20`, case 0, where both joint
components were rejected and the original merged circular facet was retained.

The independent conservation replay is partition-invariant for circular
facets. It integrates disk-polygon intersections analytically and uses 60-digit
evaluation only for nearly linear, very-large-radius arcs. This avoids two
known failure modes of the legacy area routine: non-additivity across merged
cell partitions and cancellation for radii near `1e8`. The evaluator used for
every QA row is recorded as `partition_invariant_analytic`.

## Paper integration files

Aggregate panels:

- `summary_plots/ellipses_appendix_c0_2x2.pdf`
- `summary_plots/zalesak_appendix_c0_2x2.pdf`

Representative panels:

- `representative_cases/ellipses_appendix_c0_representative_clean.pdf`
- `representative_cases/ellipses_appendix_c0_representative_with_endpoints.pdf`
- `representative_cases/zalesak_appendix_c0_representative_clean.pdf`
- `representative_cases/zalesak_appendix_c0_representative_with_endpoints.pdf`

All six PDFs are one-page vector outputs with embedded fonts and no raster
objects. The paired representative outputs use ellipse case 9 at `N=32`,
`w=0.10` and Zalesak case 22 at `N=100`, `w=0.10`.

Supporting artifacts:

- `csv/appendix_c0_sweep.csv`: all aggregate methods and metrics.
- `csv/joint_c0_case_qa.csv`: case-level continuity, solver, and conservation QA.
- `csv/joint_vs_guarded.csv`: setting-level matched comparison.
- `csv/joint_vs_guarded_summary.csv`: concise guarded-versus-joint table.
- `manifest.json`: commands, generated-run inventory, input provenance, and QA.
- `SHA256SUMS`: hashes for all 96 bundle files other than the manifest and hash
  list themselves.

## Validation

```bash
PYTHONPATH=. pytest -q \
  test/algos/test_c0_refinement.py \
  test/experiments/test_appendix_c0_joint_study.py \
  test/experiments/test_appendix_figure_exports.py \
  test/experiments/test_ellipse_circular_variant_metrics.py \
  test/experiments/test_circle_circular_variant_metrics.py
```

Result: 34 tests passed. Structural checks also verified 2,700 unique aggregate
keys, 1,375 unique case keys, finite primary metrics, exact source counts, and
all 96 SHA256 entries. PDF inspection verified one page per file, embedded
fonts, and zero raster objects.

The repository-wide suite reported 405 passed, 1 skipped, and 3 failures. The
three failures are outside this change: a square-driver monkeypatch assumption
in `test_static_area_metrics.py`, a paper-figure inventory mismatch in
`test_submission_freeze.py`, and the superseded `independent cells` label in
`test_sweep_profiles.py`. None exercises the Appendix B.5 runner or joint-C0
refinement.
