# Final perfect-reconstruction audit

## Verdict

The final release supports a qualified perfect-reconstruction claim for the
paper-facing corner methods:

- **Squares, `linear+corner`:** all `25/25` `(N,w)` setting medians are below
  both `1e-6` and `1e-8`. The overall Hausdorff median is `1.280e-9`.
- **Zalesak, `circular+corner`:** all `15/15` setting medians at `N >= 100`
  are below `1e-6`; `13/15` are below `1e-8`. The high-resolution bulk
  therefore reaches the numerical reconstruction floor, but the coarse
  `N=50,64` settings do not consistently do so.
- This is **not an all-case guarantee**. Squares retain small outlier sets, and
  Zalesak has five cases above Hausdorff `1`. The paper should say that the
  setting medians reach the numerical floor, not that every case is perfectly
  reconstructed.

The observed floor is roughly `1e-9` for squares and `1e-8` for Zalesak. It is
the practical solver/geometry floor under the frozen tolerances, not IEEE-754
machine epsilon: no setting median is below the stricter `1e-10` threshold.

## Final release results

All figures below use `25` cases per setting, five perturbation magnitudes
`w in {0, 0.05, 0.1, 0.2, 0.3}`, and `N in {50, 64, 100, 128, 150}`.

| Problem / method | Cases | H median | H p95 | H max | H <= `1e-6` | H <= `1e-8` | Setting medians <= `1e-6` | H > `1` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Squares / `linear+corner` | 625 | `1.280e-9` | `9.715e-3` | `0.6460` | 577 (92.3%) | 520 (83.2%) | 25/25 | 0 |
| Zalesak / `circular+corner` | 625 | `1.287e-8` | `0.5560` | `5.108` | 427 (68.3%) | 278 (44.5%) | 17/25 | 5 |

The corresponding joint Hausdorff-and-facet-gap counts at `1e-6` are
`573/625` for squares and `422/625` for Zalesak. All `1,250` critical-method
rows have finite Hausdorff, facet-gap, and area-error values, and all have zero
final missing cells.

### Squares by resolution

| N | H median | H p95 | H max | Cases H <= `1e-6` | Setting medians <= `1e-6` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | `6.844e-10` | `1.370e-2` | `0.6460` | 114/125 | 5/5 |
| 64 | `1.560e-9` | `1.069e-6` | `0.5414` | 118/125 | 5/5 |
| 100 | `9.019e-10` | `1.706e-7` | `0.2402` | 119/125 | 5/5 |
| 128 | `1.756e-9` | `8.440e-8` | `0.3499` | 119/125 | 5/5 |
| 150 | `1.460e-9` | `0.1637` | `0.4068` | 107/125 | 5/5 |

Every square setting has a floor-level median, but the case tail is not
monotone in resolution: `N=150` contains `18` cases above `1e-6` and has a
larger p95 than `N=100,128`. There are no cases above `1` at any resolution.

### Zalesak by resolution

| N | H median | H p95 | H max | Cases H <= `1e-6` | Setting medians <= `1e-6` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | `0.4899` | `0.6397` | `0.8677` | 14/125 | 0/5 |
| 64 | `0.2265` | `0.5141` | `1.526` | 62/125 | 2/5 |
| 100 | `7.370e-9` | `9.999e-2` | `4.445` | 115/125 | 5/5 |
| 128 | `6.401e-9` | `5.431e-4` | `2.179` | 118/125 | 5/5 |
| 150 | `5.969e-9` | `6.941e-5` | `5.108` | 118/125 | 5/5 |

The p95 drops sharply from `N=100` through `N=150`, so the bulk converges even
though the maximum is nonmonotone. The five Hausdorff-above-1 cases are:

| N | w | Case | July 14 H | Final H |
| ---: | ---: | ---: | ---: | ---: |
| 150 | 0.2 | 23 | `0.5608` | `5.108` |
| 100 | 0.2 | 21 | `0.5362` | `4.445` |
| 128 | 0.2 | 15 | `9.215e-3` | `2.179` |
| 128 | 0.3 | 6 | `1.271e-2` | `1.684` |
| 64 | 0.0 | 7 | `1.526` | `1.526` |

All five have zero unresolved-PLIC fallback cells. These are therefore not
LVIRA fallback failures; they arise later in the fully oriented reconstruction
path and should remain visible in any tail qualification.

## Older complex profile

The complete matched case-level comparator is the July 14 sweep
`static_paper_affected_diagnostics_20260714_102206`. It uses the post-f8
`current` corner behavior with the larger linear/branch rescue package and
LVIRA fallback. Curved rescue family #9 was already disabled in that sweep, so
this comparison measures the later corner-behavior and rescue simplification,
not #9 in isolation.

| Problem / method | H median old -> final | H p95 old -> final | H max old -> final | Floor cases old -> final | H > 1 old -> final | Material I/W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Squares / `linear+corner` | `0.2945 -> 1.280e-9` | `0.9598 -> 9.715e-3` | `3.216 -> 0.6460` | `91 -> 577` | `29 -> 0` | 555 / 5 |
| Zalesak / `circular+corner` | `1.944e-2 -> 1.287e-8` | `1.093 -> 0.5560` | `46.84 -> 5.108` | `295 -> 427` | `36 -> 5` | 186 / 11 |

Material changes require an absolute Hausdorff difference above `1e-10` and a
relative difference above `1%`. The final method is dramatically better in
the median and p95 for both problems. The qualification is that four of the
five final Zalesak cases above `1` are regressions relative to July 14, and the
high-resolution maxima at `N=100,128,150` are worse even as their medians,
p95 values, and floor counts improve strongly.

The separate July 15 keep-all-linear-rescues summary provides a cleaner check
of the heuristic cleanup after adopting `pre_f8_corner`. Across all `25`
Zalesak settings, the final setting medians and maxima match the keep-all
profile to at most `2.15e-13`. This supports removing the five inactive rescue
passes: the tested full-grid behavior is unchanged at reporting precision.

## Previously failing square settings

The invalid July 30 release failed exactly `24` square runs: three methods
(`linear`, `linear+corner`, and `circular`) at eight `(N,w)` settings. In the
corrected final release:

- all `24/24` settings completed;
- all `600/600` case rows have finite Hausdorff, facet-gap, and area-error
  values;
- all `600/600` rows have zero final missing cells;
- the cohort contains `66` recorded LVIRA fallback cells, all represented by
  finite final metrics.

The prior failures were caused by stale retired-parent bookkeeping in the
square area metric, not by missing reconstructed facets. The full cohort is in
`formerly_failed_square_settings.csv`.

## July 17 cross-check

The final square Hausdorff and facet-gap rows exactly reproduce July 17 in all
`625/625` cases. Zalesak has `46` non-identical Hausdorff rows (maximum absolute
difference `0.0243`), but its overall median, p95, maximum, floor count, and
five-above-1 count are unchanged at the reported precision. Historical area
errors are not used for scientific comparison because the final release uses
the repaired active-partition area metric.

## Recommended paper wording

A claim consistent with this evidence is:

> For square interfaces, the linear-corner method reaches the numerical
> reconstruction floor in the median at every tested resolution and
> perturbation magnitude. For the slotted-disk benchmark, the circular-corner
> method reaches the same regime for every tested perturbation magnitude from
> N=100 onward, while a small nonmonotone case-level tail remains.

Avoid unqualified statements that every case is reconstructed perfectly or
that the observed `1e-9` to `1e-8` floor equals binary64 machine epsilon.

## Reproduction and artifacts

Inputs are read only. Their portable paths and SHA-256 digests are recorded in
`analysis_manifest.json`. The recorded paths are relative to `results/`, so
the provenance is independent of the local checkout location. In an ordinary
checkout, `RESULTS_ROOT` defaults to the repository's ignored
`results/static/` directory. In an isolated worktree, set it to the matching
directory in the checkout that contains the frozen release roots.

Regenerate and verify this directory with:

```bash
REPO="$(git rev-parse --show-toplevel)"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO/results/static}"
FINAL_ROOT="$RESULTS_ROOT/submission_static_20260731_012430_505aefa45432"
AUDIT_DIR="$REPO/submission/audits/final_perfect_reconstruction_2026-07-31"

python "$AUDIT_DIR/analyze.py" \
  --final-release "$FINAL_ROOT" \
  --july-simplified "$RESULTS_ROOT/static_paper_simplified_default_20260717_212413" \
  --july-complex "$RESULTS_ROOT/static_paper_affected_diagnostics_20260714_102206" \
  --invalid-release "$RESULTS_ROOT/submission_static_20260730_202949_525d0cf5b4df" \
  --keep-all-summary "$RESULTS_ROOT/linear_rescue_cleanup_analysis_20260715/full_grid_comparison.csv" \
  --output-dir "$AUDIT_DIR"

(cd "$AUDIT_DIR" && shasum -a 256 -c SHA256SUMS)
python "$REPO/submission/pdf_vector_qa.py" \
  "$AUDIT_DIR/perfect_reconstruction_audit.pdf"
```

Primary artifacts:

- `final_setting_summary.csv`: all 50 critical `(problem,N,w)` settings.
- `final_resolution_summary.csv`: per-resolution medians, p95, maxima, and
  finite/floor/tail counts.
- `threshold_sensitivity.csv`: case and setting counts at `1e-6`, `1e-8`, and
  `1e-10`.
- `complex_profile_case_comparison.csv` and `complex_profile_summary.csv`:
  paired July 14 comparisons.
- `tail_inventory.csv`: every final critical-method case above `1e-6`.
- `formerly_failed_square_settings.csv`: all 24 repaired settings.
- `perfect_reconstruction_audit.pdf`: compact vector summary; it contains no
  raster image objects and all fonts are embedded.
- `pdf_qa.json`: canonical vector/no-raster/embedded-font QA record.
- `SHA256SUMS`: SHA-256 ledger for every other file in this audit directory.
