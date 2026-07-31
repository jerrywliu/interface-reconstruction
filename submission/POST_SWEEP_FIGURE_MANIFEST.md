# Post-Sweep Figure Manifest

The final review inventory is defined only by
`submission/final_figure_candidates.json`:

- 38 one-page vector PDFs;
- 14 unpaired figure slots;
- 12 paired slots with `with_endpoints` and `clean` variants; and
- five all-method summary PDFs, with auxiliary all-method exports excluded.

## Figure Families

| Family | Candidates | Scientific source |
|---|---:|---|
| Deterministic method figures | 2 | Exact PLIC and staged-Zalesak parameter contracts |
| Main-text quantitative and representative | 15 | Audited final release plus exact representative run bundles |
| All-method appendix summaries | 5 | Audited final release aggregate CSV |
| Resolution appendix | 10 | 30 fresh companion runs: six per benchmark |
| Guarded-C0 appendix | 6 | 165 fresh settings across ellipses and Zalesak |
| **Total** | **38** | Sealed orchestration snapshot |

## Promotion Rule

Historical March, May, and July assets remain layout or comparison references.
No existing figure directory may be passed into the submission gate. Run
`submission/final_figure_orchestrator.py`; it starts from nonexistent candidate
roots, invokes the exact generators, snapshots the outputs, and publishes only
after the full acceptance gate passes.

The publication root and exact command are documented in
`submission/FINAL_FIGURE_REGENERATION.md`. The machine-readable source map is
`review/figure_candidate_source_map.json`, and all producer/companion evidence
is under `provenance/` in the same atomic publication root.
