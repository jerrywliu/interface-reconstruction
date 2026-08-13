# Cartesian Matched-Study QA

## Scope and verdict

This audit reviews the shared five-benchmark Cartesian fixture, smoke-reporting
layer, completed PLVIRA smoke, frozen bare-PCIC and QUASI ports, and the two
ellipse-curvature audits. It does not reinterpret or tune any reconstruction
kernel.

The geometry fixture is suitable for a matched Cartesian study. Its cases,
random-number streams, resolution convention, mesh indexing, and prescribed
volume fractions agree with the canonical static drivers. The external result
contract also retains method status and disconnected components correctly.

The current PLVIRA case and cell outputs are valid as a **mini smoke study**.
The newly completed PCIC and QUASI outputs are diagnostic checkpoints with the
qualifications below. They are not yet a paper-comparable all-method result
set. The inexpensive
point-cloud distance has a partition-dependent sampling floor, the conservation
residual is dimensional, and the PLVIRA curvature field is not the same native
curvature observable that PCIC and QUASI would expose. These quantities need a
common final-study definition before fitting cross-method rates.

Two shared aggregation defects were proven and corrected by this audit:

1. summaries now group by method and variant as well as benchmark and
   resolution, so two PCIC corrections or multiple methods cannot be pooled;
2. every summarized floating metric now reports its non-finite case count,
   rather than silently omitting failed cases from the median and maximum.

Neither defect changes the completed PLVIRA numbers: that output contains one
method/variant and all 75 case metrics are finite. The existing run remains
pinned to its launch code through its manifest.

The same audit found a QUASI-specific accounting defect: five canonical mixed
cells had no returned facet and were absent from the external result instead of
being recorded as unresolved. The runner now enumerates the mesh's mixed cells
and creates an explicit unresolved record for every missing facet. A corrected
replay accounts for all 7,380 canonical mixed cells and marks those five cells
as unresolved.

## Canonical benchmark fixture

### Random geometry and case identity: pass

`project_benchmarks.py` uses the same `numpy.random.default_rng` seeds and draw
order as the five static drivers:

| Benchmark | Seed | Case-varying parameters |
| --- | ---: | --- |
| Lines | 42 | two center-point draws; 25 deterministic orientations |
| Squares | 42 | center, rotation, and 25 deterministic side lengths |
| Circles | 41 | center; radius 10 |
| Ellipses | 42 | center, rotation, and 25 deterministic aspect ratios |
| Zalesak | 43 | center and rotation; fixed disk and slot dimensions |

Selecting a subset still advances the complete 25-case stream. Focused tests
independently replay draws through cases 0, 4, and 24. The generated parameters
also agree with the sealed submission geometry records inspected during this
audit.

### Resolution, mesh, and indexing: pass

- The physical domain is `[0,100]^2`.
- `cells_per_side=N` is implemented as repository resolution `N/100`, giving
  exactly `N x N` cells and cell width `h=100/N`.
- The fixture uses uniform axis-aligned Cartesian cells only.
- Repository storage is `[x][y]`. PLVIRA's fraction array is transposed to
  `[row=y][column=x]`, and its target index is `(y,x)`.
- The local `3 x 3` geometry and fraction arrays are similarly converted from
  the repository's column-major stencil into row-major order.
- The mixed-cell tolerance is `1e-10`, matching the project reconstruction
  threshold used for these studies.

The indexing regression reconstructs the canonical first circle with 24
interior cells and no unresolved result. No transposition or target-cell shift
was found.

### Truth geometry: pass with one smoke-only approximation

Lines, square edges, circles, and the Zalesak arc/slot boundary use exact native
line or arc primitives. Ellipse volume fractions are initialized analytically,
but the smoke-test distance target is a 720-segment polyline. That polyline is
adequate for coarse execution checks; it should be replaced by analytic ellipse
distance for a final convergence comparison.

## Phase, components, and status

The prescribed fraction is the area of the represented interior phase. The
runner passes the corresponding dimensional target area into every method.

- PLVIRA constrains its `q <= 0` phase to that area and preserves every clipped
  parabolic interval in the target cell.
- Bare PCIC freezes disk/complement orientation from the oriented central PLIC
  and preserves every paired in-cell arc, including four-crossing cells.
- QUASI retains one conservative quadratic per supported mixed cell, as defined
  by that method, and records its frozen policy set.

The external contract distinguishes `reconstructed`, `paper_fallback`,
`unsupported`, and `unresolved`. No project fallback is inserted. Case rows
retain all four populations, optimizer failures, and unmatched shared-edge
crossings.

The completed PLVIRA smoke contains 7,380 mixed cells: 7,328 reconstructed, 52
unsupported, zero unresolved, and zero optimizer failures. All unsupported
cells are domain-boundary line cells for which the source GHF/PLVIRA path has no
one-sided halo rule. This is a method limitation, not missing status data.

One limitation remains in the gap summary: shared edges are evaluated only when
both adjacent mixed-cell records are active. Unsupported/active edges are
excluded rather than assigned an artificial gap. Final tables must therefore
show reconstruction coverage and unmatched-crossing counts beside facet-gap
statistics.

## Metric comparability

### Point-cloud distances: smoke only

`sampled_symmetric_hausdorff` and `sampled_directed_hausdorff` independently
sample source and target primitives and compare the two point clouds. They are
not invariant to an equivalent curve being split into different primitives.
The focused QA fixture represents the same exact line once as one segment and
once as two segments; the reported distance is nonzero.

This effect is visible in the PLVIRA output. Reconstructed straight lines agree
with analytic truth to about `2e-13`, but the one-sided point-cloud diagnostic
is approximately `1.25e-3` at every resolution, exactly the documented
`spacing/2` floor. The symmetric line diagnostic additionally measures the
source method's unsupported boundary coverage and decreases in proportion to
`h`.

Consequences:

- do not fit convergence rates once a curve approaches the point-cloud floor;
- do not compare methods with different primitive partitioning using this
  diagnostic;
- use the native primitive-to-primitive adaptive metric, with analytic ellipse
  truth, for the final matched study.

### Shared-edge gaps and conservation

The component-aware gap routine matches all crossings on an active shared edge
and reports crossing-count mismatches explicitly. It does not truncate a
multi-component PCIC result to one endpoint pair.

Conservation is evaluated through each method's exact area callback. The
current summary stores dimensional absolute area residuals. Because cell area
changes as `h^2`, final cross-resolution tables should additionally report
volume-fraction residual, `|A_h-A|/A_cell`. Absolute residual remains useful as
a solver-accuracy diagnostic.

### Curvature: define a common observable before comparison

The completed PLVIRA smoke compares the scalar Cartesian-GHF curvature supplied
to PLVIRA with analytic truth at the first fitted primitive's parameter
midpoint. This is useful for a source-method sanity check, but it is not yet a
matched native-curvature metric:

- GHF curvature is estimated at the target-cell height/PLIC location;
- a PLVIRA parabola's geometric curvature varies along the primitive;
- PCIC supplies constant circle curvature `1/R`;
- QUASI supplies a varying quadratic curvature.

A final common panel should state one physical evaluation rule, for example the
native primitive midpoint projected to truth or a true-interval average, and
evaluate the geometric curvature of every method there. It should use unsigned
curvature for these convex smooth benchmarks, retain per-cell/case populations,
and exclude sharp-corner cells from a smooth-curvature convergence claim.

The separate graph-coordinated circular-method audit is internally sound. It
uses exact saved arc parameters, corrects the old signed-curvature comparison,
and includes all 23,826 active primitives. Chord midpoint, native arc midpoint,
true-interval mean, equal-facet weighting, and arc-length weighting all give
orders between about 1.13 and 1.33. The below-second-order trend is therefore
not a midpoint or weighting artifact. It should not be described as a
second-order result.

## Method readiness

### PLVIRA

The operational Cartesian GHF path has completed all 75 requested project
cases at `N=32,64,128`. Circle curvature errors give pairwise rates 1.90 and
2.08. The five-case ellipse signal is decreasing but not asymptotically steady
(pairwise rates 3.22 and 0.92). Sharp-feature cases are negative controls,
because PLVIRA has no corner primitive.

### Bare PCIC

The porting choices were frozen before project outcomes: PLIC-inferred phase,
LLS overcrowding multiplier 0.5, nearest conservative translation root, and
deterministic multi-arc chord selection. Center translation and radius
adjustment remain separate variants. The kernel and component adapter pass
focused tests. Both variants completed the requested 75 case settings without
post hoc variant selection, producing 150 rows and 14,760 mixed-cell records.
Across both variants there are 13,276 reconstructed cells, 578 published
straight-limit fallbacks, 270 unsupported boundary-halo cells, 636 unresolved
cells, and 26 multi-component cells.

Center translation has 116 unresolved cells; radius adjustment has 520. One
radius-adjustment square case (`case 1`, `N=32`) has all 13 mixed cells
unresolved, so its geometry and curvature metrics are non-finite. The original
summary silently excluded that case; the corrected aggregator will report its
non-finite count. The raw case row is intact.

On smooth cases, the variants are nearly indistinguishable. Circle median
curvature error is approximately `1.2e-3`, `1.0e-3`, and `1.1e-3` at
`N=32,64,128`; ellipse error is approximately `8.8e-4`, `4.4e-4`, and
`4.8e-4`. This is not a clean second-order curvature signal. Geometry error
reaches the point-cloud floor by the finest circle grid, so it cannot establish
a finer rate. A complete `7 x 7` halo is required, and coverage must accompany
every comparison.

### QUASI

The frozen port now includes the Section 2.5-style correction, algebraic root
enumeration, deterministic Gauss--Seidel order, ten-sweep cap, and conservative
fallback. These are documented best-judgment decisions where the article is
ambiguous. The focused kernel tests pass, and the corrected 75-row smoke
accounts for all 7,380 canonical mixed cells: 7,216 reconstructed, 159 retained
as conservative paper fallbacks, and five explicitly unresolved. The five
unresolved facets occur in line case 2 at `N=64`, square case 2 at `N=128`,
ellipse case 3 at `N=128`, Zalesak case 0 at `N=64`, and Zalesak case 3 at
`N=128`.

More importantly, none of the 75 cases satisfies the frozen `1e-11`
endpoint-displacement convergence criterion before the ten-sweep cap. The
corrected output contains 131 reported fallback events and marks 159 involved
cells as conservative fallbacks. Exact per-cell area residuals are small and
shared-edge matched gaps are zero, but straight-line geometry error remains
large and only first-order-like (`1.80`, `0.90`, `0.45`). These signals make
the current QUASI port unsuitable for a paper baseline without further
source-level diagnosis. Its status should remain a frozen-port diagnostic,
not an accuracy comparison.

## Provenance

The PLVIRA smoke manifest records launch commit
`a600825e0ab57346f3c330b6739df4a933ab981e`, hashes the direct fixture,
reporting, adapter, PLVIRA, and GHF files, and stores exact case parameters in
every case row. Native geometry and per-cell diagnostics are serialized. The
75 expected `(benchmark, case, resolution)` keys are unique and complete.

The ellipse-curvature audits pin the sealed source result set and reconstruction
commit `505aefa454328d4ba34ade5e7247050a0acfc793`; they do not mutate or rerun the
source geometry. Their reports and CSVs are diagnostic artifacts, not promoted
paper results.

Existing PLVIRA outputs should remain immutable. Their manifest hashes identify
the pre-audit reporting code. A later all-method run should use the corrected
aggregation code and a new output directory.

## Gates for the five-case matched run

1. Keep the canonical fixture and Cartesian indexing unchanged.
2. PCIC summaries have been regenerated under the method/variant and non-finite
   accounting fix; retain both corrections as separate diagnostic variants.
3. QUASI has been rerun with explicit missing-facet accounting; diagnose why
   all cases exhaust the ten-sweep cap before considering a full 25-case run.
4. Replace the point-cloud distance with a native, partition-invariant metric
   before making paper comparisons or fitting rates.
5. Add normalized conservation residual and active-edge coverage.
6. Freeze one common smooth-interface curvature location/weighting rule.
7. Run five cases at `N=32,64,128`; inspect failures and component populations
   before extending to all 25 cases.

## Verification

Focused QA covers the full RNG advance, Cartesian fixture creation, PLVIRA
index conversion, method/variant aggregation separation, explicit non-finite
counts, and the point-cloud partition floor. The complete baseline and audit
test suite is run before committing this report.
