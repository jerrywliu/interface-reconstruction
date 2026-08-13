# Cartesian external-baseline execution contract

Date: 2026-08-13

## Objective

Complete source-faithful static implementations of PLVIRA, bare PCIC, and
QUASI on the Cartesian mesh class for which the primary articles define the
methods. The first acceptance target is reproduction of a primary-paper
static result. Matched project benchmarks follow only after that gate passes.

## Fixed rules

1. Do not tune a method against this project's benchmark outcomes.
2. Do not substitute this project's graph coordination, circular fit, or
   LVIRA fallback for an unspecified source-method step.
3. Preserve every in-cell interface component. Never truncate a multi-arc or
   multi-interval result to fit the current one-facet-per-cell API.
4. Report unsupported and unresolved cells explicitly.
5. Keep exact-geometry oracle inputs separate from operational baselines.
6. Keep materially different published variants as separate named rows.
7. Run source-paper reproduction before the matched project smoke tests.
8. Restrict unqualified method names to Cartesian results. Any future
   perturbed-mesh extension receives an `adapted` label and a separate report.

## Method gates

### PLVIRA

- Operational row: Cartesian GHF curvature plus the restricted fixed-curvature
  parabolic LVIRA optimization.
- Diagnostic row: `PLVIRA (exact-curvature oracle)` where target curvature is
  analytically available.
- Required source check: the paper's Cartesian flower study or a documented
  subset that establishes its reported symmetric-difference convergence
  scale.
- Ineligible shortcuts: curvature from this project's circular fit or exact
  curvature labeled simply `PLVIRA`.

### Bare PCIC

- Initial predictor: the cited LLS/Parker--Young construction, with every
  porting choice recorded.
- Published conservative corrections remain separate rows:
  `bare PCIC (center translation)` and `bare PCIC (radius adjustment)`.
- Required source check: the randomized Cartesian ellipse experiment or a
  documented subset establishing the reported error scale and order.
- Four-crossing cells must retain and metric all in-cell arc components.

### QUASI

- Implement Sections 2.1--2.5 without silently selecting an unstated neighbor,
  root, or update policy.
- Enumerate the published cubic roots algebraically before applying a recorded
  selection policy.
- Required source check: the Cartesian random-circle study or a documented
  subset establishing its reported L1 trend.
- If Section 2.5 remains ambiguous, report the affected population and keep
  the result labeled as a partial checkpoint rather than QUASI.

## Shared result contract

The external-baseline adapter should return one record per original Cartesian
mixed cell:

```text
ExternalCellReconstruction
  cell_index
  components: list[ExternalInterfaceComponent]
  exact_phase_area(polygon)
  source_method
  source_variant
  status: reconstructed | paper_fallback | unsupported | unresolved
  diagnostics

ExternalInterfaceComponent
  primitives: ordered list[Primitive]
  closed: bool
```

Metrics flatten all primitives for Hausdorff and tangent evaluation, pair
crossings on shared edges for continuity, and use the method's exact area
routine for conservation. Parabolic and quadratic primitives may be sampled
for VTK display, but metrics must not replace them by circular arcs or lines.

## Acceptance sequence

1. Focused kernel and ambiguity tests.
2. Primary-paper static reproduction with frozen policies.
3. Cartesian `N=32,64`, five-case project smoke tests on supported benchmarks.
4. Full Cartesian comparison only for methods passing steps 1--3.
5. Figure/table eligibility review with exact commit, configuration, failure
   counts, and source deviations frozen in provenance.
