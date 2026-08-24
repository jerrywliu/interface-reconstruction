# Local and joint C0 refinement incidence

## Scope

This analysis replays the full production Appendix B.5 diagnostics for 750
ellipse cases and 625 Zalesak cases. The local stage is the simultaneous
endpoint-midpoint and conservative facet-refit pass. The joint stage acts only
on connected components containing endpoint gaps that remain after that pass.

## Join-level incidence

- The local pass closes `209,157/232,670` eligible joins (`89.89%`).
- It leaves `23,513/232,670` eligible joins (`10.11%`) for joint refinement.
- Joint refinement reduces that count to `440/232,670` (`0.189%`), repairing
  `98.13%` of the joins left by the local pass.
- Ellipses: residual joins are `16.161%` after the local pass and `0.0183%`
  after joint refinement.
- Zalesak: residual joins are `3.639%` after the local pass and `0.3715%`
  after joint refinement.

## Component incidence and size

- `1,354/1,375` cases (`98.47%`) contain at least one residual component.
  Thus the local pass is usually successful per join but rarely closes every
  join on an entire interface.
- `33,144/233,222` original mixed cells (`14.21%`) participate in a joint
  component: `22.45%` for ellipses and `5.45%` for Zalesak.
- Across 9,215 components, the mean size is `3.55` facets, the median is `3`,
  and 90% contain at most `5` facets.
- Combined component-size shares are: size 3, `69.85%`; size 4, `9.69%`;
  size 5, `11.04%`; and all sizes 6 or larger, `5.61%`.

## Solver effort

- Median recorded function evaluations are `20` for ellipses, `43.5` for
  Zalesak, and `22` combined.
- The 90th percentiles are `45`, `1,290`, and `139`, respectively.
- These are nonlinear-solver function evaluations, not outer refinement
  iterations. The heavy Zalesak tail reflects the current solver's additional
  tangent-matching search. The present evidence therefore supports describing
  the joint problems as small, but not claiming that they always take only a
  few iterations.

## Paper implication

The cleanest main claim is conservative C0 correction. A simple local pass
handles nearly 90% of joins. Small connected-component solves then raise the
eligible-join C0 rate to 99.81%. Tangent matching can be identified as an
optional stronger objective or extension without making it a principal result.

Artifacts are under
`results/submission/c0_refinement_incidence_20260823/`.
