# Final Topology, Merge, And Fallback Diagnostic Audit

Status: **PASS**

- Final release: `submission_static_20260731_012430_505aefa45432` (`970` runs, `24,250` cases).
- July comparison: `static_paper_simplified_default_20260717_212413` (`7,500` matched cases).
- Denominators: **3,091,429 mixed cells** and **3,088,697 final components**.
- Final missing facets: **0**.

## Headline Incidence

| Final facet | Mixed cells | Cell fraction | Components | Component fraction |
| --- | ---: | ---: | ---: | ---: |
| Linear | 2,501,448 | 80.9156% | 2,500,410 | 80.9536% |
| Circular | 572,983 | 18.5346% | 572,070 | 18.5214% |
| Linear corner | 10,739 | 0.3474% | 10,202 | 0.3303% |
| Curved corner | 6,259 | 0.2025% | 6,015 | 0.1947% |

| Path diagnostic | Mixed cells | Cell fraction | Components | Component fraction | Events |
| --- | ---: | ---: | ---: | ---: | ---: |
| Merged | 5,461 | 0.1766% | 2,729 | 0.0884% | - |
| Independent | 3,085,968 | 99.8234% | 3,085,968 | 99.9116% | - |
| LVIRA PLIC fallback (all reasons) | 9,838 | 0.3182% | 9,838 | 0.3185% | 9,838 |
| Exact-linear-support rescue | 379 | 0.0123% | 373 | 0.0121% | 373 |
| Local-line arc-fit fallback | 2,656 | 0.0859% | 2,576 | 0.0834% | 2,576 |

All unresolved PLIC fallback components are cross-checked across final cell state, merge-event provenance, and the dedicated fallback ledger. Direct Youngs, ELVIRA, and LVIRA method rows are counted as PLIC defaults, not as fallback events.
The fallback reasons are `9,836` unresolved orientations and `2` failed support-line fits. No Youngs or ELVIRA fallback policy occurs.
No corner--arc--corner, curved-loop, or curved-transition rescue event occurs. The final curved-corner facets above are direct reconstruction outcomes, not curved-rescue assignments.

| Direct PLIC method | Mixed cells | Cell fraction | Components | Component fraction |
| --- | ---: | ---: | ---: | ---: |
| `ELVIRA` | 443,698 | 14.3525% | 443,698 | 14.3652% |
| `LVIRA` | 443,698 | 14.3525% | 443,698 | 14.3652% |
| `Youngs` | 443,698 | 14.3525% | 443,698 | 14.3652% |

## Method Hotspots

The full benchmark/method and setting tables are in the CSV files. The largest fallback incidences are:

| Benchmark | Method | Mixed cells | LVIRA fallback | Fraction | Local-line fallback | Fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| squares | `safe_circle` | 62,581 | 2,713 | 4.3352% | 478 | 0.7638% |
| zalesak | `safe_circle` | 113,042 | 4,292 | 3.7968% | 708 | 0.6263% |
| circles | `safe_circle` | 52,855 | 877 | 1.6593% | 48 | 0.0908% |
| ellipses | `safe_circle` | 120,180 | 1,481 | 1.2323% | 59 | 0.0491% |
| lines | `linear` | 95,040 | 293 | 0.3083% | 0 | 0.0000% |
| squares | `circular` | 62,581 | 22 | 0.0352% | 529 | 0.8453% |
| squares | `linear` | 62,581 | 22 | 0.0352% | 0 | 0.0000% |
| squares | `linear+corner` | 62,581 | 22 | 0.0352% | 0 | 0.0000% |
| zalesak | `circular` | 113,042 | 28 | 0.0248% | 823 | 0.7280% |
| zalesak | `linear` | 113,042 | 28 | 0.0248% | 0 | 0.0000% |

## July Comparison

The comparison uses the exact `7,500`-case July coverage inside the final release; the other `16,750` final cases are excluded from these deltas.
Merged incidence is unchanged at `5,461` cells in `2,729` components. PLIC fallback incidence is unchanged at `475` cells/components/events, all using LVIRA.
The July event schema did not encode local-line fallback events or fallback reasons. The final local-line counts above are therefore standalone incidence, not a historical increase from zero.

Overall matched incidence changed only in the following categories:

| Category | July cells | Final cells | Delta | July components | Final components | Delta | July events | Final events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `final_facet:circular` | 323,879 | 323,882 | +3 | 322,966 | 322,969 | +3 | 0 | 0 |
| `final_facet:curved_corner` | 6,264 | 6,259 | -5 | 6,020 | 6,015 | -5 | 0 | 0 |
| `final_facet:linear` | 627,097 | 627,099 | +2 | 626,059 | 626,061 | +2 | 0 | 0 |
| `rescue:exact_linear_support` | 378 | 379 | +1 | 372 | 373 | +1 | 372 | 373 |

## Integrity

- Case metrics and cell diagnostics agree for all `24,250` final cases.
- PLIC provenance agrees for all `9,838` final fallback components; every policy is `LVIRA`.
- The hardened release auditor independently reports the same `3,091,429` cell rows, `3,544,985` merge-event rows, and `9,838` joined fallback rows.
- Final-facet classes are constant within each component, declared component sizes match member rows, and merged flags agree with component size.
- The four final facet classes partition both the mixed-cell and component denominators.
- Raw/consolidated checks use exact canonical scientific-row hashes within every run bundle:

| Table | Raw rows | Consolidated rows | Matching bundles | Status |
| --- | ---: | ---: | ---: | ---: |
| `run_manifests.jsonl` | 970 | 970 | 970/970 | **PASS** |
| `case_geometry.jsonl` | 24,250 | 24,250 | 970/970 | **PASS** |
| `case_metrics.csv` | 24,250 | 24,250 | 970/970 | **PASS** |
| `cell_metrics.csv` | 3,091,429 | 3,091,429 | 970/970 | **PASS** |
| `merge_events.csv` | 3,544,985 | 3,544,985 | 970/970 | **PASS** |
| `unresolved_plic_fallbacks.csv` | 9,838 | 9,838 | 970/970 | **PASS** |

## Files

- `final_incidence_long.csv`: complete overall/benchmark/method/setting incidence.
- `final_incidence_by_method.csv`: compact method table.
- `final_incidence_by_setting.csv`: compact `N`/perturbation table.
- `july_matched_incidence_comparison.csv`: matched July-to-final deltas.
- `raw_consolidated_reconciliation.csv`: exact raw-bundle reconciliation.
- `integrity_checks.csv`: release-level pass/fail ledger.
- `SHA256SUMS`: sorted SHA-256 manifest for every report artifact except the manifest itself.

Fractions are weighted by mixed cells or final components, as named. Rescue and local-fallback cell counts are unique member cells of affected components; event counts retain repeated assignments. Categories can overlap, so rescue/fallback fractions are not intended to sum to one.

## Reproduce

The report stores release names and relative artifact paths only. From a clean checkout, define the input roots for the local archive and run:

```bash
REPO=/path/to/interface-reconstruction
FINAL_ROOT=/path/to/submission_static_20260731_012430_505aefa45432
JULY_ROOT=/path/to/static_paper_simplified_default_20260717_212413
cd "$REPO"
python submission/audit_final_release.py "$FINAL_ROOT"
python submission/audit_topology_diagnostics.py \
  --final-root "$FINAL_ROOT" \
  --july-root "$JULY_ROOT" \
  --output-dir submission/audits/final_diagnostics_2026-07-31
(cd submission/audits/final_diagnostics_2026-07-31 && shasum -a 256 -c SHA256SUMS)
```

No PDF was generated for this audit, so raster-object and font-embedding QA are not applicable.
