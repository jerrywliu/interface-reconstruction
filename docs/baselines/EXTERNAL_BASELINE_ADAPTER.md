# Component-aware external-baseline adapter

Date: 2026-08-13

## Purpose

This adapter is the shared execution boundary for the Cartesian PLVIRA, bare
PCIC, and QUASI ports. It does not route results through `ArcFacet`,
`CompositeFacet`, `MergeMesh`, or `runReconstruction`. Those APIs assume one
connected endpoint sequence per active cell and cannot retain the two
disconnected arcs that PCIC may produce in a four-crossing cell.

The implementation lives in:

- `main/algos/baselines/external_geometry.py`: native primitive protocol,
  component/cell/run records, adapters, and JSON schema;
- `main/algos/baselines/external_metrics.py`: native Hausdorff and tangent
  evaluation, exact-area conservation, and shared-edge crossing matching;
- `experiments/baselines/external_runner.py`: Cartesian validation and the
  one-record-per-original-mixed-cell execution loop.

## Geometry and result contract

`ExternalCellReconstruction` owns zero or more
`ExternalInterfaceComponent` objects. Each component owns an ordered list of
native primitives and has an independent `closed` flag. Therefore a cell can
retain multiple disconnected interface pieces without inventing a connection.

The four serializable primitive implementations are:

| Primitive | Stored native parameters | Metric geometry |
| --- | --- | --- |
| line | two endpoints | exact segment distance and tangent |
| arc | center, positive radius, start angle, signed sweep | exact restricted-arc distance and tangent |
| PLVIRA parabola | frame center/angle, curvature, shift, retained tangent-coordinate interval | native parametric distance optimization and tangent |
| QUASI quadratic | endpoints and signed midpoint bulge | native parametric distance optimization and tangent |

`ExternalPrimitive` is a runtime protocol, so another published primitive can
be added without changing the component or metric records. The
`adapt_linear_facet`, `adapt_parabolic_interval`, and
`adapt_quadratic_facet` helpers copy parameters from current kernel objects
without retaining their one-facet ownership model.

Every original mixed cell has one explicit status:

- `reconstructed`: the published high-order path returned geometry;
- `paper_fallback`: a fallback specified by that source method returned
  geometry;
- `unsupported`: the source method does not define this cell/configuration;
- `unresolved`: the source algorithm reached an ambiguity or failed solve.

The last two statuses cannot carry geometry. The first two must carry at least
one component. This prevents a missing result from being silently counted as a
project-method fallback.

## Exact area and JSON

Each reconstructed cell carries its method's exact phase-area callback. The
conservation metric calls this callback directly on the owning polygon. It
does not infer area from sampled display geometry.

Python callables are intentionally not serialized. Before writing JSON, the
record evaluates the callback in its owning cell and stores that exact scalar.
Loading the JSON retains that value for audit and metric replay in the owning
cell. A method adapter must reattach its executable callback before evaluating
the same geometry in a different polygon.

All geometry parameters, component boundaries, statuses, source variants,
diagnostics, and run configuration are round-tripped by schema version 1.
Metadata is validated recursively; non-string keys, arbitrary Python objects,
and non-finite floats are rejected rather than stringified or emitted as
nonstandard JSON.

## Metric behavior

- Hausdorff evaluation flattens every component primitive. Source curves are
  sampled adaptively to estimate the directed supremum, while every
  sampled-point distance is evaluated against the target's native line, arc,
  parabola, or quadratic geometry.
- Tangent evaluation uses native primitive derivatives. It does not convert a
  parabola or quadratic to circular or linear facets.
- Conservation reports one exact residual per reconstructed or paper-fallback
  cell and counts unsupported/unresolved cells as unevaluated.
- Facet-gap evaluation finds every crossing on every shared Cartesian cell
  edge and performs a minimum-cost one-to-one match. Crossing-count mismatches
  remain explicit `unmatched_count` events. No first/last-endpoint shortcut is
  used.

Sampling is permitted for VTK/display output and adaptive supremum estimation.
It is not a replacement geometry in serialized results or conservation.

## Runner invariants

`run_external_static_baseline`:

1. rejects non-axis-aligned or nonrectangular cells before invoking a method;
2. visits original mesh cells only and performs no merging;
3. provides each callback its cell, fraction, target phase area, complete
   `3 x 3` stencil, and a `complete_3x3` boundary flag;
4. requires a returned record or an explicit `UnsupportedExternalCell` /
   `UnresolvedExternalCell` exception for every mixed cell;
5. verifies cell ownership, polygon identity, method/variant provenance, and
   target area before accepting the record;
6. records status counts and all frozen run configuration in JSON metadata.

The runner does not provide LVIRA, ELVIRA, Youngs, or this project's circular
fit as an implicit fallback.

## Method integration assumptions

### PLVIRA

- The method branch must expose every clipped parabola interval in the target
  cell and adapt each interval separately. Selecting only the outermost two
  crossings is not permitted.
- The exact-area callback should call PLVIRA's analytic
  `parabolic_polygon_area` with the fitted frame, curvature, and shift.
- Exact-curvature oracle and operational GHF variants must use distinct
  `source_variant` values.
- Boundary cells lacking the paper's required stencil must be marked with the
  source-specified behavior or an explicit unsupported/unresolved status.

### Bare PCIC

- The method branch must pair all target-cell circle crossings into
  phase-oriented in-cell arcs. Four crossings become two components, not one
  major arc and not a truncated pair.
- Arc geometry stores a positive radius and signed angular sweep. Disk versus
  complement phase remains in diagnostics and in the signed-radius exact-area
  callback.
- Center-translation and radius-adjustment corrections remain separate
  `source_variant` rows.
- A straight result is `paper_fallback` only when the PCIC source rule calls
  for that limit; it must not inherit a project fallback policy.

### QUASI

- Each completed `QuadraticFacet` adapts directly through its endpoints and
  bulge; its exact-area callback should call the analytic represented-area
  routine.
- Cells reaching the unresolved Section 2.5 path remain `unresolved` until
  the method branch implements the frozen source interpretation.
- Root-selection and update-order choices belong in run and cell diagnostics,
  not in this shared adapter.

## Tests

The focused suite covers:

- native distances and kernel-object adapters for all four primitive families;
- lossless JSON for a genuine two-component/four-crossing cell;
- status and JSON-domain validation;
- all-crossing shared-edge matching and explicit cardinality mismatches;
- native parabola Hausdorff/tangent evaluation and exact-area conservation;
- one-record-per-mixed-cell runner behavior, boundary outcomes, and rejection
  of perturbed Cartesian cells.

The adapter tests establish representation and metric behavior. They do not
establish source-paper fidelity of any baseline; that remains gated by each
method's kernel tests and primary-paper reproduction.
