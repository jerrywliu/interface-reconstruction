# Post-Sweep Figure Manifest

The final review inventory is defined only by
`submission/final_figure_candidates.json`:

- 41 one-page vector PDFs;
- 14 unpaired figure slots;
- 12 paired slots with `with_endpoints` and `clean` variants; and
- three additional `hybrid_endpoints_n16_n32` candidates in the lines,
  ellipses, and Zalesak resolution slots; and
- five all-method summary PDFs, with auxiliary all-method exports excluded.

The three row-level maps serve different stages of the review:

- `submission/final_figure_candidates.json` defines every candidate ID, slot,
  endpoint variant, and generated path;
- `submission/figure_provenance.csv` maps the 26 manuscript slots to active TeX
  includes and their promotion targets; and
- `review/figure_candidate_source_map.csv` is generated with the packet and binds
  all 41 concrete PDFs to producer inputs and evidence.

After the author selects one PDF for each selectable slot, the approved-figures CSV
described in `submission/SUBMISSION_PACKAGING.md` pins the selected 26 manuscript
paths and file digests. Until then, no candidate is promoted automatically.

## Figure Families

| Family | Candidates | Scientific source |
|---|---:|---|
| Deterministic method figures | 2 | Exact PLIC and staged-Zalesak parameter contracts |
| Main-text quantitative and representative | 15 | Audited final release plus exact representative run bundles |
| All-method appendix summaries | 5 | Audited final release aggregate CSV |
| Resolution appendix | 13 | 30 fresh companion runs: six per benchmark |
| Guarded-C0 appendix | 6 | 165 fresh settings across ellipses and Zalesak |
| **Total** | **41** | Sealed orchestration snapshot |

## Promotion Rule

Historical March, May, and July assets remain layout or comparison references.
No existing figure directory may be passed into the submission gate. Run the
isolated `submission/run_final_figure_orchestrator` launcher documented below;
it starts from nonexistent candidate
roots, requires the external approval record for the exact integrated commit,
materializes the complete release before auditing it, derives all consumed
release inputs from that audit snapshot, reads config through the attested
source authority, invokes the exact generators, and publishes only after its
internal acceptance gate passes.
It uses only the private copy of this allowlist, records the attested
Poppler/Python/font runtime, snapshots the actual square/circle VTP truth inputs
used by the resolution plots, and seals a separately copied final tree before
the locked no-replace publication.

The publication root and exact command are documented in
`submission/FINAL_FIGURE_REGENERATION.md`. The machine-readable source map is
`review/figure_candidate_source_map.json`, and all producer/companion evidence
is under `provenance/` in the same atomic publication root.
