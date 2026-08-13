# External-baseline integration fidelity review

Date: 2026-08-13

> **Implementation follow-up:** The original risk assessment below remains as
> the audit trail. Subsequent Cartesian work completed the PLVIRA GHF path,
> advanced the bare-PCIC predictor and component geometry, added algebraic
> QUASI root enumeration, and integrated a component-aware external geometry
> contract. Current readiness and numerical results are summarized in
> `docs/baselines/CARTESIAN_BASELINE_STATUS.md`.

## Scope

This is a read-only scientific-fidelity review of the static PLVIRA, PCIC, and
QUASI prototypes. It compares the modules, focused tests, implementation
reports, primary articles, the PLVIRA paper-linked implementation at commit
`4ee19e3f5b0fc946e3243c961aa2e6d7d35b9087`, and the repository's current
facet, metric, serialization, and static-driver APIs. It does not approve or
change any algorithmic choice.

Primary sources:

- R. A. Remmerswaal and A. E. P. Veldman, *Parabolic interface
  reconstruction for 2D volume of fluid methods*, JCP 469 (2022), 111473,
  [arXiv:2111.09627](https://arxiv.org/abs/2111.09627).
- R. K. Maity, T. Sundararajan, and K. Velusamy, *An accurate interface
  reconstruction method using piecewise circular arcs*, IJNMF 93 (2021),
  93--126, [doi:10.1002/fld.4876](https://doi.org/10.1002/fld.4876).
- S. V. Diwakar, S. K. Das, and T. Sundararajan, *A Quadratic Spline based
  Interface (QUASI) reconstruction algorithm for accurate tracking of
  two-phase flows*, JCP 228 (2009), 9107--9130,
  [doi:10.1016/j.jcp.2009.09.014](https://doi.org/10.1016/j.jcp.2009.09.014).

## Executive verdict

None of the three prototypes is ready to appear as an unqualified baseline in
the paper. The isolated kernels are useful and the focused tests pass, but the
scientifically defining end-to-end paths are not complete.

| Method | Faithful and testable now | Blocking issue before a reported baseline |
| --- | --- | --- |
| PLVIRA | Restricted fixed-curvature parabola, central-volume solve, `3 x 3` least-squares objective, analytic area and angle derivative | No paper-faithful GHF curvature provider; the paper defines the method on Cartesian grids, not perturbed polygons |
| bare PCIC | Six-point sampling, SWFL Riemann-sphere fit, guards, and two explicitly selected conservative-correction kernels | Initial LLS/Parker--Young predictor absent; correction choice unresolved; center-translation direction currently differs from the article; four-crossing cells lack a shared representation |
| QUASI Sections 2.1--2.4 | Cartesian Youngs initialization, local quadratic area constraint, common-edge predictor, and iterative continuity kernel for supported configurations | Section 2.5 absent; multiple-root and update-order policies are unstated by the paper; current root enumeration can miss repeated roots |

The current focused suite passes `15/15`. These tests establish local
mathematical consistency, not source-paper reproduction or matched-baseline
fidelity.

## Findings ordered by scientific risk

### 1. The intended perturbed-mesh comparison is outside all three source methods

PLVIRA's curvature construction is presented for a uniform rectilinear mesh.
Its local/generalized height functions use coordinate-aligned columns and
finite differences. The article notes other work for unstructured meshes but
does not make that method part of PLVIRA. PCIC is formulated on a 2D structured
grid and defines several rules using one Cartesian cell size `Delta x`. QUASI
is formulated on square Cartesian grids, and the prototype correctly rejects
perturbed cells.

Therefore an `N=32,64`, `w=0.2` comparison cannot be called a paper-faithful
implementation of any of these methods. Generalizing the geometry alone is
not enough: stencil construction, initial normals, curvature, boundary rules,
and several thresholds acquire new meanings.

**Author decision required:** choose one of the following and state it before
integration.

1. Compare these external methods only on Cartesian (`w=0`) benchmarks.
2. Add separately named and cited adapted variants; do not label them simply
   PLVIRA, PCIC, or QUASI.
3. Use exact target curvature for an explicitly named PLVIRA oracle study,
   not as the PLVIRA baseline.

### 2. PLVIRA's defining curvature input is not implemented

`reconstruct_plvira` requires caller-supplied curvature
(`main/algos/baselines/plvira.py:451`). This is scientifically honest, but the
paper's method is PLVIRA only when that curvature comes from its GHF path.

The paper-linked repository does not remove the ambiguity for our intended
experiments. Its flower benchmark evaluates a level-set-based local curvature
from the known target geometry, rather than supplying a standalone
volume-fraction-to-GHF implementation. Thus there are two distinct validation
questions:

- Does this Python kernel reproduce the restricted parabolic search when the
  curvature is prescribed?
- Does an end-to-end volume-fraction GHF + PLVIRA path reproduce the article?

Only the first is currently addressed. A perturbed-polygon curvature provider
would be a new hybrid, even if based on a separate unstructured-height-function
paper.

**Author decision required:** approve Cartesian GHF implementation as the only
unqualified PLVIRA path; separately decide whether an exact-curvature oracle
is useful. Do not silently use the repository's own circular fit to provide
curvature.

### 3. The PCIC center-translation correction currently changes the paper's geometry

The article moves the fitted circle center along the perpendicular bisector of
the fitted arc. The prototype moves it along the central PLIC normal
(`main/algos/baselines/pcic.py:232-235`). Those directions coincide only when
the fitted circle's target-cell chord remains parallel to the supplied PLIC
segment. A stencil circle fit does not guarantee this.

The implementation report currently treats “PLIC normal” and “perpendicular
bisector” as interchangeable. They are not. Results from
`correction="translate_center"` are therefore not yet attributable to the
published center-translation variant.

**Author decision required:** none about the method itself. Before this
variant is run, the implementation must use the fitted target-cell arc/chord
direction prescribed by the paper, or the row must be explicitly named as an
adaptation. This review does not prescribe the code change.

### 4. PCIC's reported bare method cannot yet be selected unambiguously

The article describes two conservative corrections: fixed-radius center
translation and fixed-center radius adjustment. It does not identify which
one generated each reported bare-PCIC table. The prototype correctly makes
the choice mandatory, but that means there is no single experiment-ready
`PCIC (bare)` row.

Additional unresolved behavior affects the result:

- The required LLS reconstruction “coupled with” Parker and Youngs is not
  implemented. The exact pass count, coupling, boundary handling, and point
  orientation are not stated in the PCIC article.
- When the fitted radius is below the cell diagonal, the paper relocates the
  center using the PLIC segment. The prototype chooses between the two chord
  centers by closest target volume (`pcic.py:356-387`); that selection rule is
  not in the article.
- The Riemann fit is unoriented. Choosing disk versus complement by whichever
  initial area is closer (`pcic.py:390-405`) is plausible but not specified.
- The center-translation bracket expands in both directions and chooses the
  closest sign-changing interval. If more than one conservative root exists,
  this is an unstated root-selection policy.
- Returning the PLIC facet for an exactly degenerate Riemann plane is
  consistent with the paper's large-radius straight-line limit, but the exact
  degeneracy rule is an implementation convention.

**Author decisions required:** (a) run the two conservative corrections as
separately named rows or seek author clarification; (b) approve an exact
LLS/Parker--Young construction from the cited LLS paper; (c) approve phase,
minimum-radius-center, and multiple-root policies before source reproduction.

### 5. Four-crossing PCIC cells are topologically unrepresentable in the current driver

The PCIC article explicitly includes a cell in which one circle intersects the
cell boundary four times. Restricting that circle to the cell may produce two
disconnected interface arcs. `ArcFacet` contains one center, one radius, and
two endpoints. `CompositeFacet` can hold several primitives but models one
connected sequence and its joints. Neither represents two disconnected
components owned by one cell.

`PCICCircle.intersections` correctly refuses silent truncation, but the raw
four points are not sufficient integration metadata: the adapter must pair
them into phase-oriented in-cell arc components. Flattening to one arc would
change area, topology, Hausdorff distance, and gap statistics.

**Author decision required:** approve a component-aware per-cell geometry
contract before integrating PCIC. Alternatively, limit a smoke study to cases
proved to have exactly two crossings and report the exclusion; that is not a
general PCIC baseline.

### 6. QUASI Section 2.5 is a material missing algorithmic path

The prototype stops when common-edge continuity cannot be formed. The article
then performs a curvature correction using a selected neighboring mixed cell
with `0.02 < F < 0.98`, including a possible vertex jump. It does not state how
to select the target neighbor when several qualify. This choice can alter both
topology and the quadratic in the target cell.

The current strict failure is preferable to inventing a policy. It also means
the paper's random-circle study and the project's circle/ellipse/Zalesak smoke
tests cannot yet be claimed.

**Author decision required:** seek clarification or approve a documented
selection rule. Until then, call the module “QUASI Sections 2.1--2.4
checkpoint,” not QUASI.

### 7. QUASI's current multiple-root handling invents behavior and can miss valid roots

The article derives a cubic continuity equation but does not state which
admissible root to select. The prototype chooses the root nearest the current
predictor (`main/algos/baselines/quasi.py:447-454`). That is an invented
least-displacement policy.

In addition, `_candidate_c1_roots` samples 65 points and applies Brent's method
only across sign changes (`quasi.py:403-439`). This is not equivalent to
solving the published cubic: it can miss an even-multiplicity root and can
miss closely spaced roots within one sampled interval. The issue is separate
from the ambiguous root-selection rule.

The deterministic, in-place mesh-index sweep is also unstated. A Jacobi update,
a different Gauss--Seidel order, or a different root can converge to a
different spline. Endpoint-index tie breaking is lower risk but likewise a
deterministic porting convention.

**Author decisions required:** approve the admissible-root selection and the
pair-update semantics/order. The implementation should enumerate the actual
published cubic roots before applying that approved selection; this is a
fidelity correction, not a method refinement.

### 8. PLVIRA's optimizer and stopping rule are transparent deviations

The article specifies L-BFGS with a More--Thuente line search and a
problem-scale stopping estimate. The prototype uses SciPy's unconstrained
`L-BFGS-B` call with `gtol=1e-8` by default (`plvira.py:490-496`). The article's
linked one-dimensional PLVIRA code, meanwhile, uses its own derivative-based
`brent_min` routine rather than the article's stated L-BFGS path.

The Python choice is defensible as an article-level port, but it has not been
shown to reproduce the same minima or stopping floor. It should remain a
recorded deviation. A non-Cartesian explicit initial angle would also be an
adaptation; the current API correctly requires the caller to supply it.

**Author decision required:** decide whether “paper specification” or
“paper-linked reference implementation” is authoritative for source
reproduction. Because the project requested paper-faithful baselines, the
article specification is the natural default, with optimizer sensitivity
reported rather than tuned.

### 9. Existing tests are necessary but not discriminating enough for publication use

The 15 tests verify exact primitive recovery, conservation, one derivative,
one connected QUASI line, cutoffs, and explicit failure paths. They do not
test:

- Cartesian GHF curvature or a PLVIRA source-paper curve;
- PLVIRA optimizer sensitivity, multiple parabola-cell components, or boundary
  stencils;
- PCIC noisy-data fits against a published value, the actual LLS predictor,
  either correction's paper geometry, phase orientation, minimum-radius side,
  or source-paper ellipse error;
- PCIC four-crossing arc pairing and downstream metrics;
- QUASI one-endpoint correction, multiple cubic roots, repeated roots,
  sweep-order sensitivity, Section 2.5, or the random-circle study.

Additional randomized checks performed during this review found the PLVIRA
analytic polygon area consistent with a dense independent geometric
approximation (maximum absolute discrepancy approximately `3.3e-8`, dominated
by reference discretization) and its analytic angle gradient consistent with
finite differences on valid perturbed convex stencils. This strengthens the
kernel assessment but does not resolve the method-level blockers above.

## Minimal shared representations and adapters

Do not force these methods through `ArcFacet` or the current
`runReconstruction` dispatch. The latter assumes one facet per active merged
polygon, exposes a scalar `.curvature`, and builds topology through one endpoint
pair.

The minimal geometry addition is a component-aware result used by external
baselines:

```text
CellInterfaceGeometry
  components: list[InterfaceComponent]
  exact_area(polygon) -> float
  source_method: str
  diagnostics: dict

InterfaceComponent
  primitives: ordered list[Primitive]
  closed: bool
```

Add primitive adapters, not approximating conversions:

- `ParabolicPrimitive` for a clipped PLVIRA interval;
- `QuadraticPrimitive` for a QUASI facet;
- existing `ArcPrimitive` for each paired in-cell PCIC arc;
- existing `LinePrimitive` for straight fallbacks.

Each new primitive needs `pLeft`, `pRight`, `sample`, tangent, normal,
`distance_to_point`, length/spacing sampling, and bounding points. PLVIRA and
PCIC may yield more than one component in a convex cell, so clipping must
return all intervals/components, not the first and last crossings.

The minimal driver adapter should be separate from merged reconstruction:

```text
run_external_static_baseline(mesh, method, config)
  -> mapping[cell_index, CellInterfaceGeometry] + diagnostics
```

It should:

1. operate on original Cartesian cells without graph merging;
2. make boundary-stencil exclusions explicit rather than falling back to one
   of this project's methods;
3. reject unsupported perturbed cells for the unqualified source methods;
4. require every mixed cell to return geometry or a paper-specified fallback;
5. preserve method parameters and unresolved events in diagnostics.

Metric/serialization adapters should:

- flatten all component primitives for global Hausdorff and tangent metrics;
- compute facet gaps by matching every crossing on each shared cell edge,
  rather than assuming one left/right endpoint per cell;
- evaluate conservation with each method's exact area routine;
- serialize all components and method parameters to metadata;
- sample parabolas/quadratics only for VTK display, never convert them to arcs
  for metric evaluation.

## Staged path to reproducible comparisons

### Stage 0: freeze author decisions

Record, without tuning against this paper's outcomes:

1. Cartesian-only versus separately named adapted variants.
2. PLVIRA curvature source and optimizer authority.
3. PCIC correction row(s), exact initial PLIC construction, phase convention,
   minimum-radius center side, and root selection.
4. QUASI Section 2.5 neighbor selection, cubic-root selection, and update
   order.

### Stage 1: complete discriminating kernel tests

- **PLVIRA:** test all intersection topologies; compare objective and gradients
  against the linked code on identical rectangular stencils; test the
  Cartesian GHF separately.
- **PCIC:** compare SWFL output on noisy points with an independent
  implementation; test the published perpendicular-bisector correction;
  validate both conservative alternatives; pair and conserve a genuine
  four-crossing cell.
- **QUASI:** enumerate the published cubic roots directly; add one/multiple/
  repeated-root fixtures; test one-endpoint and vertex-jump configurations;
  record order sensitivity.

### Stage 2: reproduce one primary-paper static result each

- **PLVIRA:** flower shape, equation (24), Cartesian grids, exact reference
  moments, and Figure 11 symmetric-difference order. Report prescribed/exact
  curvature separately from volume-fraction GHF curvature.
- **PCIC:** randomized ellipse, equation (24), 100 placements, Cartesian
  `N=10...640`, and Table 1's `E1` error. Run the two bare corrections as
  separately named variants unless clarified; do not choose the better one
  post hoc.
- **QUASI:** the paper's 1000 random circles with centers in
  `[0.4,0.6]^2`, radii in `[0.2,0.25]`, Cartesian `N=10...320`, and its `L1`
  error. This waits on Stage 0 and the Section 2.5 implementation.

Set acceptance criteria before running: matching convergence order and the
published error scale within a documented tolerance, not exact bitwise values.

### Stage 3: matched project smoke tests

Run `N=32,64`, five fixed cases per benchmark, on Cartesian meshes first.
Include line, circle, ellipse, square, and Zalesak only where the source method
defines a result. Record failures and paper-specified fallbacks; do not replace
them with LVIRA or this project's methods unless the source method says so.

Only after Cartesian source reproduction should separately named adaptations
be considered at `w=0.2`. Never use the perturbed results to select ambiguous
policies.

### Stage 4: manuscript eligibility gate

A method becomes eligible for a figure/table legend only when:

- its source-paper reproduction report passes;
- every material ambiguity and deviation is disclosed;
- the exact commit/configuration is frozen;
- unsupported cells and fallbacks are counted;
- geometry, conservation, metrics, and serialization preserve all interface
  components.

Until then use the names `PLVIRA kernel`, `bare-PCIC kernel`, and `QUASI
Sections 2.1--2.4 checkpoint` in internal reports only.
