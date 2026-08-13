# Cartesian external-baseline status

Date: 2026-08-13

## Summary

The three source-method tracks and the shared component-aware adapter have
been integrated. The focused baseline suite has 51 passing tests. This is a
substantial implementation checkpoint, but only PLVIRA has passed a primary-
source numerical trend check. PCIC and QUASI retain explicit scientific gates
instead of silently filling gaps in their articles.

| Method | Implemented state | Numerical evidence | Current eligibility |
| --- | --- | --- | --- |
| PLVIRA | Operational Cartesian GHF curvature plus restricted parabolic LVIRA; exact curvature is a separately named oracle | Deterministic Popinet circle subset reaches second-order GHF curvature convergence | Ready for component-aware Cartesian project smoke tests; source flower reconstruction remains an oracle-only comparison |
| bare PCIC | Parker--Young predictor, one-pass cited LLS, SWFL fit, both conservative corrections, all arc components | LLS angle error improves from `2.95e-2` to `6.08e-4` rad; both corrections conserve to about `1e-11` | Kernel checkpoint only; primary ellipse table remains blocked by unpublished phase/correction/root policies |
| QUASI | Sections 2.1--2.4 connected-cell kernel plus algebraic enumeration of distinct and repeated cubic roots | Quadratic conservation and connected-line fixtures pass | Partial checkpoint only; Section 2.5 neighbor selection and exact continuity-polynomial integration remain unresolved |

## PLVIRA

The operational entry point now obtains curvature exclusively from the
published Cartesian GHF hierarchy. The exact-curvature entry point is labeled
`PLVIRA (exact-curvature oracle)` and cannot be confused with the operational
row.

For circles with `R/h=12.8,25.6,51.2`, relative GHF curvature errors enter a
stable second-order regime. The `128 -> 256` orders are `2.005`, `2.006`, and
`2.023` in relative L1, L2, and Linf. One coarse `R/h=6.4` cell reaches the
published zero-curvature degeneracy fallback; it is retained in the report.

Artifacts:

- `experiments/baselines/PLVIRA_GHF_CIRCLE_REPRODUCTION.md`
- `experiments/baselines/results/plvira_ghf_circle/`

## Bare PCIC

The initial reconstruction is now a Cartesian Parker--Young predictor followed
by the cited one-pass LLS refinement. The two published conservative choices
remain distinct rows:

- `bare PCIC (center translation)`;
- `bare PCIC (radius adjustment)`.

Center translation now follows the fitted target-cell chord's perpendicular
bisector. Four-crossing cells retain two disconnected arc components. The
deterministic kernel check reports volume-fraction residuals `4.77e-11` and
`3.51e-11` for the two correction variants.

The primary randomized ellipse table is not yet reproducible without freezing
choices the article does not publish: phase orientation, correction
attribution, minimum-radius side, multiple translation roots, and the LLS
overcrowding scale. These must be fixed before seeing matched benchmark
outcomes, and both correction rows must be retained.

Artifacts:

- `experiments/baselines/PCIC_STATIC_CHECK.md`
- `experiments/baselines/run_pcic_static_check.py`

## QUASI

Sampling-based sign-change discovery has been replaced by an algebraic root
enumerator that retains even-multiplicity roots and explicitly reports an
identically-zero relation. It is intentionally not wired into the production
QUASI correction yet: the article's expanded continuity polynomial and
multiple-root policy must be frozen together.

Section 2.5 remains a source-level blocker. The paper does not define which
eligible neighboring mixed cell is selected when several satisfy its filter,
nor does it fully define root and update ordering. Consequently, the random-
circle reproduction is not claimed.

## Shared integration

The component-aware adapter provides native line, arc, parabola, and quadratic
geometry; exact-area callbacks; multiple disconnected components per cell;
all-crossing shared-edge gaps; native Hausdorff/tangent metrics; and lossless
JSON. Method-specific bridges preserve:

- every clipped PLVIRA parabola interval;
- every paired PCIC arc component;
- every completed QUASI quadratic facet;
- explicit unresolved and paper-fallback statuses.

No method passes through `runReconstruction`, `MergeMesh`, or an implicit
LVIRA fallback.

## Next gates

1. Run PLVIRA on the matched Cartesian five-case project smoke suite.
2. Freeze PCIC's ambiguous policies without looking at benchmark outcomes,
   then run both conservative variants through the same smoke suite.
3. Obtain collaborator approval or source clarification for QUASI Section 2.5
   and root/update policies before attempting its random-circle study.
4. Promote a method into paper figures only after its source check, matched
   smoke test, unsupported-cell accounting, and exact configuration are frozen.
