# Square active-partition confidence audit

Audit date: 2026-07-31 PDT

Corrected source commit: `505aefa454328d4ba34ade5e7247050a0acfc793`

## Verdict

**PASS.** The corrected confidence sweep contains `30/30` successful runs and `750/750` cases. Across `75,504` unique active mixed components, every component has one internally consistent non-null reconstructed facet, and each case's facet metadata contains exactly one facet index per active component. All required square metrics are finite.

The correction changes only the square area-metric partition supplied by the driver. It does not change reconstructed geometry: every available pre-fix facet JSON and VTP artifact is byte-identical to its corrected counterpart, and every saved pre-fix Hausdorff/facet-gap value is exactly unchanged.

## Coverage and invariants

| Check | Result |
|---|---:|
| Completed runs | `30/30` |
| Completed cases | `750/750` |
| Active mixed components | `75,504` |
| Active-component/facet invariant failures | `0` |
| Cases with finite Hausdorff, facet gap, and area error | `750/750` |
| Raw/consolidated case-metric matches | `750/750` |
| LVIRA fallback records | `66` in `39` cases |
| Raw/consolidated fallback-record matches | `66/66` |
| Non-LVIRA or inconsistent fallback records | `0` |

Each fallback was cross-checked in the run configuration, case summary, cell-level facet provenance, unresolved-fallback ledger, and merge-event ledger. Every record uses `LVIRA`, points to an active component, and ends in a non-null `LVIRA` line facet.

## Pre/post comparison

The invalid release supplied `6` complete control bundles. Its `24` failed settings retained partial run directories under `/Users/wei/Code/Interface/active/interface-reconstruction/plots`. Together they provide pre-fix facet artifacts for `282` cases:

- `150` cases from complete archived controls.
- `132` cases from retained failed-run prefixes, including each failing witness.
- `282/282` facet-metadata JSON hashes match exactly.
- `282/282` facet VTP hashes match exactly.
- `282/282` available fallback signatures match exactly.
- `258/258` saved Hausdorff values match exactly; maximum absolute difference `0.0e+00`.
- `258/258` saved facet-gap values match exactly; maximum absolute difference `0.0e+00`.

The failed jobs raised during area evaluation, before writing the failing case's case-metric row. Their witness facet files and fallback ledgers were already present and match the corrected run byte for byte; subsequent cases were never attempted in those partial bundles. The comparison CSV distinguishes these expected absences from mismatches.

## N=50 witness

At `N=50`, perturbation magnitude `0.2`, case index `3`, the old driver supplied 32 retained dictionary entries to a 30-facet active reconstruction. Retired parents `17` and `20` had already been replaced by active single-cell children `30` and `31`, both reconstructed by LVIRA. The corrected driver supplies the returned 30-element active polygon list, so polygon/facet pairing is `30/30`.

| Method | Active pairs | LVIRA cells | Hausdorff | Facet gap | Corrected area error |
|---|---:|---:|---:|---:|---:|
| linear | 30 | 2 | 4.941660e-01 | 1.283693e-01 | 4.515187e-12 |
| linear+corner | 30 | 2 | 3.079462e-01 | 1.926012e-11 | 1.100255e-11 |
| circular | 30 | 2 | 5.119878e-01 | 9.199533e-02 | 1.009631e-10 |

The [vector witness PDF](square_n50_w0p2_case3_before_after.pdf) and [300 DPI Slack preview](square_n50_w0p2_case3_before_after.png) show the stale parent overlays versus the corrected active-partition accounting. The reconstructed interface is identical in both panels.

## Artifacts and scope

- Case-level comparison: [`square_confidence_case_comparison.csv`](square_confidence_case_comparison.csv)
- Witness PDF: [`square_n50_w0p2_case3_before_after.pdf`](square_n50_w0p2_case3_before_after.pdf)
- Witness PNG: [`square_n50_w0p2_case3_before_after.png`](square_n50_w0p2_case3_before_after.png)
- Corrected confidence root: `/Users/wei/Code/Interface/active/interface-reconstruction/results/static/square_active_partition_retry_20260730_181530_505aefa45432`
- Invalid diagnostic root: `/Users/wei/Code/Interface/active/interface-reconstruction/results/static/submission_static_20260730_202949_525d0cf5b4df`
- Retained failed-run prefixes: `/Users/wei/Code/Interface/active/interface-reconstruction/plots`

This audit is scoped to the 30-run square confidence grid. It does not promote the invalid 970-run release and does not audit the still-running corrected authoritative release. Both source result roots were read only.
