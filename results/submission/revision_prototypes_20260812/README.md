# August 12 Figure And Layout Prototypes

These are non-destructive revision prototypes. They do not replace any approved
paper figure, alter sealed experiment data, or change reconstruction logic.
Result-bearing panels read from the immutable `87c4030` final-figure release.

## Review Set

| Prototype | Vector PDF | Intended paper location | Status / finding |
| --- | --- | --- | --- |
| Three-stencil comparison | `conceptual_stencil_comparison.pdf` | New Section 5.3 figure | Clean one-row comparison; the selected three-cell support is much easier to distinguish from LVIRA after de-emphasizing inactive mesh cells. |
| Candidate graph on mesh | `candidate_graph_mesh_overlay.pdf` | Figure 4 replacement or companion | A two-stage mesh overlay makes the graph/mesh distinction explicit. This is a conceptual schematic, not a measured case. |
| Enlarged spyglasses | `enlarged_spyglasses/*.pdf` | Main Section 6 representatives | A `0.42` axes-fraction inset is readable without obscuring geometry. Circle and ellipse receive new rightmost/high-curvature windows. |
| Resolution panel, `5x2` | `benchmark_resolution_5x2.pdf` | Consolidated Appendix B.21--B.25 | Recommended. `N=32,64` remains readable at text width and avoids the known under-resolved `N=16` Zalesak example. |
| Resolution panel, `5x3` | `benchmark_resolution_5x3.pdf` | Consolidated Appendix B.21--B.25 | Useful as a completeness comparison, but each panel is smaller and the `N=16` Zalesak failure dominates the row. |
| Compact C0 page | `compact_c0_one_page.pdf` | Appendix B.5 | Fits both benchmarks on one page by retaining only resolution-dependent aggregate panels beside the approved representative reconstructions. |

PNG renders beside each PDF are visual-QA previews only. Matplotlib schematics
also have SVG exports. `provenance.json` records checksums and exact source
paths for every generated artifact.

## Sources

- Sealed publication and VTK snapshot:
  `results/submission/final_figures_87c40309d16c_20260803_final/`
- Current approved paper assets were inspected under:
  `../interface-reconstruction-paper/figs/cameraready/`
- Generator:
  `experiments/submission/generate_revision_layout_prototypes.py`

The consolidated resolution layouts crop only the perturbed-mesh panels from
the final vector PDFs. The compact C0 layout crops the cells-per-side columns
from the final aggregate PDFs. LaTeX includes the source PDFs directly, so the
assembled outputs remain vector rather than rasterizing the plots.

The sealed C0 metric PDFs still embed the historical `topology + merging`
legend text. Before promotion, regenerate those metric panels with the approved
`graph-coordinated` terminology; this prototype evaluates layout only.

## Reproduce

```bash
python -m experiments.submission.generate_revision_layout_prototypes
```

The default output is this directory. Pass `--output-dir` for an isolated
alternative location.

## Integration Notes

- Proposed stencil target: `figs/cameraready/conceptual_stencil_comparison.pdf`.
- Proposed graph-overlay target: a new cameraready PDF referenced from Section
  4.4.1; do not silently replace the current TikZ Figure 4 before author review.
- Proposed consolidated resolution target:
  `figs/cameraready/benchmark_resolution_5x2.pdf`.
- Proposed compact C0 target: `figs/cameraready/compact_c0_one_page.pdf`.
- Captions and manuscript prose remain Jerry-owned and were not edited.
