# Final Diagnostic Summary

After the final release passes its release audit, generate the reconstruction-path
incidence report with:

```bash
python submission/summarize_final_diagnostics.py \
  results/static/<completed-final-release>
```

The command writes:

- `diagnostic_summary/diagnostic_summary.csv`: long-form overall,
  per-experiment, and per-method rows;
- `diagnostic_summary/diagnostic_summary.json`: the same values with definitions
  and release metadata;
- `diagnostic_summary/README.md`: concise submission-facing tables.

Every fraction is weighted by mixed cells in the displayed scope. Final circular,
linear-corner, and curved-corner facets are read from `cell_metrics.csv`. Merge
incidence is the fraction of mixed cells in multi-cell components. Rescue,
orientation, and fallback incidence is reconstructed from `merge_events.csv` and
its archived member-cell lists. PLIC policies are cross-checked against both the
cell table and `unresolved_plic_fallbacks.csv`; case counts are cross-checked
against `case_metrics.csv` and `case_geometry.jsonl`.

The command is fail-closed. It refuses an incomplete or failed sweep, missing
required fields, inconsistent fallback ledgers, and rescue assignments whose
archived provenance cannot identify a unique rescue type. Existing output is not
replaced unless `--overwrite` is supplied.

The production profile currently retains only the `exact_linear_support` linear
corner rescue. The reader also recognizes uniquely recorded curved-corner loop and
transition rescues and corner-arc-corner triplets. Older broad rescue profiles can
contain assignments that the historical shared-stage schema cannot distinguish;
those releases require an explicit `rescue_type` provenance field before they can
be summarized without guessing.
