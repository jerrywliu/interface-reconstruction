# Ellipse case 13 metric-pairing diagnostic

## Scope

- Source run: `results/static/extended_convergence_smoke_full_cell_area_fix_20260813`
- Setting: ellipse case 13, `N=512`, `w=0.2`, graph-coordinated circular reconstruction
- Reported anomaly: mean Hausdorff distance `21.218465751`
- Reconstruction source commit: `f9a03660b669e2d703a14707721bcd476355f6ba`

## Root cause

The reconstruction geometry is sound. The large Hausdorff value is caused by
positional metric pairing after unresolved orientation fallback changes the
active merge-ID order.

The unresolved cells originally occupy merge IDs 35 and 36. `findOrientations`
keeps those stale polygons in `m.merged_polys`, while its active return list
removes them and appends their LVIRA replacements as IDs 926 and 927. The
ellipse metric loop then evaluates

```python
zip(m.merged_polys.values(), reconstructed_facets)
```

so every facet after list index 34 is compared with the wrong cell's analytic
ellipse segment. Reconstructing that stale positional sequence from the saved
provenance reproduces the reported mean to roundoff:

| Quantity | Value |
| --- | ---: |
| Reported Hausdorff | `21.218465751034433` |
| Reproduced stale positional pairing | `21.218465751034408` |
| Active-ID-keyed recomputation | `8.258790902056985e-5` |
| Keyed recomputation without fallback facets | `8.264004321630213e-5` |

The two LVIRA facets have individual keyed Hausdorff distances of
`4.6533e-5` and `7.0471e-5`. Removing them changes the corrected mean by only
`0.063%`. They are not geometric outliers.

## Other hypotheses

- **Fallback facet geometry:** rejected. Both fallback segments are short,
  lie on the reconstructed ellipse, and have ordinary local errors.
- **Phase or arc orientation:** rejected. Analytic-cell arcs and serialized
  reconstructed primitives agree under active-ID pairing.
- **Serialization:** rejected. Exact sidecar geometry and `cell_metrics.csv`
  agree in count, order, endpoints, and primitive type for all 926 active cells.
- **Metric pairing:** confirmed exactly as above.

## Facet gap and conservation

The reported facet gap is also affected by fallback topology bookkeeping, but
not by bad geometry. Ordering all saved active facets geometrically around the
ellipse gives mean endpoint gap `3.5768e-6`. Removing fallback facets from that
ordering gives `1.8789110204e-5`, exactly the reported production value. Thus
the production interface graph omits the two fallback bridges and measures the
gap across the resulting hole. Including them reduces the mean gap by `81.0%`.

Conservation is unaffected:

- global relative phase-area error: `1.609177e-11`
- conservation complete: yes
- conservation failures: 0
- maximum cell-relative area residual: `4.038442e-7`

## Recommended gate

1. Block publication of this full-sweep metric row until cell/facet pairing is
   keyed by active merge ID or uses the returned `(polygon, facet)` lists.
2. Strengthen `_validate_active_reconstruction` to verify order, not only equal
   object sets and lengths.
3. Attach replacement fallback cells to the final interface component graph so
   facet-gap metrics include their two joins.
4. Add one regression with an unresolved cell whose replacement ID is appended;
   require invariance under dictionary insertion order and metric recomputation
   from serialized active-ID geometry.
5. Rerun the affected ellipse setting. This is the only completed full-sweep
   case with a PLIC fallback, so the immediate numerical blast radius is one row.

## Artifacts

- `results/static/extended_convergence_case13_pairing_diagnostic_20260813/ellipse_case13_pairing_diagnostic.pdf`
- `results/static/extended_convergence_case13_pairing_diagnostic_20260813/ellipse_case13_pairing_diagnostic.png`
- `results/static/extended_convergence_case13_pairing_diagnostic_20260813/per_cell_hausdorff.csv`
- `results/static/extended_convergence_case13_pairing_diagnostic_20260813/summary.json`

Reproduce with:

```bash
PYTHONPATH=. python experiments/static/diagnose_ellipse_case13_pairing.py
```
