# First-six square failure audit

Audit date: 2026-07-30 PDT

Frozen source commit: `525d0cf5b4df8ad6526429316872acf9eb656517`

Audit branch/worktree: `codex/square-failure-audit-20260730` / `/Users/wei/Code/Interface/active/interface-reconstruction-square-failure-audit-20260730`

## Verdict

The first six square jobs recorded as failures are two deterministic bookkeeping mismatches repeated by `linear`, `linear+corner`, and `circular`:

| Grid | Wiggle | Zero-based case | Displayed case | Retained entries | Active polygon/facet pairs | Retired parent -> active LVIRA child |
|---:|---:|---:|---:|---:|---:|---|
| 50 | 0.2 | 3 | 4/25 | 32 | 30 | `17,20 -> 30,31` |
| 50 | 0.3 | 2 | 3/25 | 32 | 31 | `3 -> 31` |

**Primary cause: a preexisting stale-parent bookkeeping mismatch newly exposed by the repaired fail-closed area metric.** These are not six missing-active-facet failures. Every active component has a non-`None` facet. The unresolved parents are retained in the mesh dictionary for topology history, while active split children replace them in the reconstruction output and receive the configured LVIRA fallback.

The repaired metric is correctly refusing an unaligned `32`-polygon / `30`- or `31`-facet positional input, but the square driver supplied all retained dictionary entries instead of the active polygon list returned with the facets. The correct metric contract is one key-stable active polygon/facet pair per reconstruction component.

### Causal classification

- **Repaired fail-closed area metric:** immediate trigger and useful detector. It exposes the stale-parent mismatch rather than creating a reconstruction failure.
- **Dropping rescue family #9:** not causal. That family belongs to `circular+corner` cleanup used by Zalesak. These jobs are `linear`, `linear+corner`, and `circular`; the square sweep does not apply the rescue profile.
- **`pre_f8_corner`:** an enabling path, not a missing-facet regression. In these cases it splits unresolved parents into active single-cell children, then LVIRA reconstructs those children. The same active topology and fallback records were already present in the July 17 pre-F8 artifacts.
- **Preexisting missing facets:** rejected for the active reconstruction. Saved cell metrics contain 30/30 or 31/31 nonmissing active facets, including 2 or 1 successful LVIRA fallbacks. One older complex artifact labels a retired parent `missing`, but its active child for the same cell is present and fitted by LVIRA.
- **Preexisting stale-parent bookkeeping:** confirmed. July 17 contains the same active children and retained-parent mismatch; the legacy area path hid it.

## Exact release evidence

Release root:

`/Users/wei/Code/Interface/active/interface-reconstruction/results/static/submission_static_20260730_202949_525d0cf5b4df`

- `sweep_manifest.json`, failure entries `0..5`, identifies the six jobs.
- `logs/*squares_{linear,linearpluscorner,circular}_r0p5_w0p2_s0_corner_pre_f8_corner.log` ends with `squares case 3 ... got 30 facets for 32 polygons`.
- The analogous `w0p3` logs end with `squares case 2 ... got 31 facets for 32 polygons`.
- Failed-job diagnostics remain under `/Users/wei/Code/Interface/active/interface-reconstruction/plots/submission_static_20260730_202949_525d0cf5b4df_perturb_sweep_squares_*`.
- At `w=0.2`, case index `3`, each method has exactly 30 active `cell_metrics.csv` rows and no `missing` class. Merge IDs `30` and `31`, cells `26,22` and `27,22`, are nonmissing linear facets with `construction_path=plic_fallback`, `fallback_policy=LVIRA`, and matching entries in `unresolved_plic_fallbacks.csv`. Retired parents `17` and `20` do not appear in the active rows.
- At `w=0.3`, case index `2`, each method has exactly 31 active rows and no `missing` class. Merge ID `31`, cell `22,22`, is a nonmissing LVIRA facet. Retired parent `3` does not appear in the active rows.

The circular logs expose the parent state directly: IDs `17` and `20` only nominate each other at `w=0.2`, and ID `3` is unresolved at `w=0.3`. The saved diagnostics then show the replacement active children and successful LVIRA facets.

## Historical comparisons

### July 17 pre-F8 paper sweep

Aggregate root:

`/Users/wei/Code/Interface/active/interface-reconstruction/results/static/static_paper_simplified_default_20260717_212413`

Exact raw roots:

`/Users/wei/Code/Interface/active/interface-reconstruction/plots/perturb_sweep_squares_<method>_r0p5_w0p{2,3}_s0_corner_pre_f8_corner/metrics/`

- All six exact rows use source commit `d02dd479...`, LVIRA, and `pre_f8_corner`.
- The active counts, cell IDs, parent/child merge IDs, and fallback records match the release exactly.
- Sorted `linear` and `linear+corner` cell rows are byte-identical to the release. `circular` geometry differs only at floating-point solver scale; topology and fallback identities are unchanged.
- The July 15 shard `/Users/wei/Code/Interface/active/interface-reconstruction/results/static/corner_ablation_full_grid_missing_a_20260715/diagnostics/case_metrics.csv` independently contains the two exact `linear+corner` rows.

### July 14 older complex/default profile

Aggregate root:

`/Users/wei/Code/Interface/active/interface-reconstruction/results/static/static_paper_affected_diagnostics_20260714_102206`

Exact raw roots:

`/Users/wei/Code/Interface/active/interface-reconstruction/plots/perturb_sweep_squares_<method>_r0p5_w0p{2,3}_s0/metrics/`

- The nonmissing active cell-ID set is identical to the release for all six rows: 30 cells at `w=0.2` and 31 cells at `w=0.3`.
- At `w=0.2`, the older profile has 30 nonmissing rows and no fallback records. It reconstructed the same active cells without the pre-F8 split/fallback path.
- At `w=0.3`, it contains both a `missing` parent merge ID `3` and a nonmissing LVIRA child merge ID `31` for the same cell `22,22`. Counting active nonmissing cells gives 31, exactly matching the release. This is direct historical evidence that the `missing` row is a retired parent, not an unreconstructed physical cell.
- Normalized case geometry is identical across release, July pre-F8, and older complex artifacts. Geometry hashes are `9562f82ac2e1280f` (`w=0.2`) and `1f305dd8984aeff3` (`w=0.3`).

### Exact-support / rescue #9 validation

`/Users/wei/Code/Interface/active/interface-reconstruction/results/static/pre_f8_exact_support_validation_20260715/diagnostics/case_metrics.csv`

This root contains only 200 `zalesak/circular+corner` rows. It has no square rows because the exact-support and rescue #9 cleanup is structurally outside these square methods.

## Why July appeared to pass

The July square area path used truncating positional `zip(...)`, skipped a facet when an exception occurred, and added exact polygon-intersection area for every unpaired dictionary entry. It could therefore hide stale parents and, when parent removal changed ordering, mispair later polygons and facets. The historical area values in `first_six_comparison.csv` are preserved only as evidence of what was reported; they are not trustworthy reconstructions of these cases.

The new fail-closed check rejects the count mismatch before aggregation. Keep that behavior, but give it the returned active polygon list rather than all retained mesh entries.

## Integrated resolution

The reviewed submission-integration fix implements the audit recommendation without
changing orientation, merging, fallback, or facet fitting:

1. The square driver requests the active polygons returned with the facets and uses
   that same active partition for both exact truth area and reconstructed area. This
   removes the stale-parent double counting that a count-only fix would leave in the
   metric normalization.
2. Reconstruction now fails closed unless the returned polygons exactly cover the
   mesh's active merge IDs once each and every returned facet is non-`None`.
3. Unit tests cover the active-partition contract and invariant, while a real
   `N=50,w=0.2,case=3` regression exercises the unresolved LVIRA path.
4. Replays of both first-six geometries for all three methods produce facet metadata
   that is byte-identical to the frozen failed run, with unchanged Hausdorff and
   facet-gap values, successful LVIRA provenance, and zero missing active facets.

The historical conclusion is unchanged: July 17 has the same active topology,
fallback identities, and complete active facet coverage; the older complex profile
has the same nonmissing active-cell set, with the documented retired-parent row.
This does not claim that every older circular facet coordinate is byte-identical;
those differ only at floating-point solver scale where noted above.

Rescue family #9 remains structurally unrelated and is unchanged. It belongs to the
Zalesak `circular+corner` cleanup path and is not used by these square jobs.

The invalid release reached 24 square job failures over eight `(N,w)` settings. An
exact diagnostic retry is therefore 24 runs / 600 cases; the authoritative release
must still be rerun from one clean source commit before accepting any summary.

The row-level comparison is in `first_six_comparison.csv` beside this report. No live source, release artifact, or running process was modified during this audit.
