# Zalesak Failure Diagnostic

This directory contains an archive-only diagnostic for the retained July 2026
Zalesak tail. It reads the frozen sweep artifacts and does not invoke or modify
the reconstruction algorithm.

## Canonical case

- run: `static_paper_simplified_default_20260717_212413`
- method: `circular+corner`
- resolution: `N=150` (`resolution=1.5`)
- perturbation: `w=0.2`, `seed=0`
- case: `case_index=23`

## Generate

From the repository root:

```bash
python experiments/submission/zalesak_failure/diagnose_zalesak_failure.py
```

The script writes the review bundle to
`results/submission/zalesak_failure_case23/` and mirrors the vector PDF to
`output/pdf/zalesak_failure_case23_diagnostic.pdf`.

## Evidence contract

The diagnosis uses the archived `case_metrics.csv`, `cell_metrics.csv`,
`merge_events.csv`, `case_geometry.jsonl`, exact facet metadata, truth metadata,
and structured mesh. The July provenance records the corner assignment but not
the support IDs or stage-time support geometry. The script therefore recovers
the two support owners by collinear containment of the accepted corner branch
attachment points in final per-cell geometry. The finite left support endpoint
is cross-checked against the archived same-case `circular` sibling, whose linear
stage is identical before corner augmentation.
