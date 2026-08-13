# QUASI Cartesian Baseline

## Status

This is a frozen, static-only Cartesian port suitable for matched benchmark
execution. Sections 2.1--2.5 are represented. The article leaves several
material implementation choices open, so the choices below were declared
before examining project-benchmark results and are stored in every
`QuasiResult.policy` record. This is a best-judgment baseline port, not a claim
that the missing choices reproduce the authors' unpublished implementation.

Source:

- S. V. Diwakar, S. K. Das, and T. Sundararajan, "A Quadratic Spline based
  Interface (QUASI) reconstruction algorithm for accurate tracking of
  two-phase flows," *Journal of Computational Physics* 228 (2009), 9107-9130.
  DOI: `10.1016/j.jcp.2009.09.014`.

The specification below was extracted from the full primary article, not from
the abstract or a secondary implementation.

## Paper-to-code specification

1. **Initial PLIC reconstruction (Section 2.1).**
   QUASI initializes every mixed cell with Youngs' multidimensional-stencil
   normal and positions the line to satisfy the cell volume fraction. The
   prototype calls the repository's existing Youngs implementation for this
   exact role.

2. **C0 predictor (Section 2.2 and Fig. 3).**
   For adjacent mixed cells whose PLIC endpoints both lie on their common edge,
   the common endpoint is their coordinate average. If only one cell has an
   endpoint on the common edge, one endpoint in the neighboring cell is moved
   to that edge. Section 2.2.1 selects between the two possibilities first by
   consistency of the phase labels at the shared vertices and then by maximum
   dot product with the original PLIC normal. `_establish_c0` and
   `_select_movable_endpoint` implement these two cases and criteria.

3. **Piecewise parabolic representation (Eqs. 2-10).**
   The paper fits one parabola through two boundary endpoints and a central
   point, with the center abscissa halfway between endpoint abscissae. Simpson's
   identity in Eq. (7), combined with the volume constraint in Eqs. (8)-(10),
   determines the central ordinate. `QuadraticFacet` uses the equivalent
   chord-aligned form

   ```text
   p(s) = p_left + s d + 4 b s (1-s) n_left,
   b = 3 (A_chord - A_target) / (2 |d|).
   ```

   Thus the represented primary-fluid area is analytically
   `A_chord - 2 b |d| / 3 = A_target`. This is a coordinate rewrite of the
   paper's equations, not a new fitting criterion.

4. **C1 corrector (Section 2.4, Eqs. 11-16).**
   The shared endpoint is parameterized along the common cell edge. The
   endpoint is moved to a root at which the two area-preserving parabolas have
   matching tangents. Corrections are applied pairwise over ten sweeps, the
   paper's reported typical optimum.

5. **Curvature correction (Section 2.5, Eqs. 17-21).**
   When the C0 predictor cannot connect neighboring segments, the port applies
   the paper's midpoint-curvature correction against an eligible neighboring
   mixed cell with `0.02 < F < 0.98`. The moving endpoint remains on the shared
   Cartesian edge, and both cells are re-fit conservatively at the selected
   common point. A diagonal transition uses the shared cell vertex, matching
   the paper's vertex-jump construction. If no admissible correction exists,
   the conservative local quadratic is retained and the event is reported.

## Frozen ambiguous decisions

- **Target-neighbor selection.** Prefer the eligible mixed neighbor that
  exposed the discontinuity. If it is ineligible, use the nearest eligible
  mixed cell in the source cell's 8-neighborhood. Midpoint-tangent alignment,
  then lexicographic cell index, break exact ties.
- **Admissible and multiple roots.** Enumerate every algebraic root on the
  physical edge interval `[0,1]`, including repeated roots. Reject roots that
  fail the unsquared geometric relation. Among remaining roots, select the one
  with least displacement from the current predictor; the smaller parameter
  breaks exact ties.
- **Update order.** Apply pairwise corrections in lexicographic cell/endpoint
  order, in place (deterministic Gauss--Seidel).
- **Sweeps and convergence.** Perform at most ten sweeps, as reported by the
  article. Stop earlier only when the largest endpoint displacement is at most
  `1e-11` in mesh coordinates. Exhausting ten sweeps is reported separately
  from a missing root and does not trigger a different reconstruction.
- **Boundary handling.** Endpoints on the physical domain boundary remain open.
  No ghost volume fractions or contact-angle rule are invented.
- **Fallback.** If an eligible target or admissible correction is unavailable,
  retain the cell's area-preserving local quadratic and emit an unresolved
  diagnostic. `strict=True` converts that diagnostic into an exception.
- **One-endpoint C0 ties.** Apply the paper's phase-consistency and normal-
  alignment criteria. Endpoint index resolves an exact remaining tie.

## Deliberate porting deviations

- The source prints expanded frame-specific continuity equations. The port
  constructs the equivalent cross-multiplied polynomial from chord, area, and
  tangent polynomials, then enumerates its real roots algebraically. The same
  machinery is used for the squared midpoint-curvature relation, followed by an
  unsquared residual check to reject extraneous roots.
- The original method is formulated on square Cartesian grids. The repository
  supports perturbed quadrilateral meshes, but this prototype rejects them
  rather than extending QUASI beyond the paper.
- Dynamic advection, contact-angle boundary conditions, curvature-based surface
  tension, and the EMFPA coupling from Sections 4-5 are outside this static
  baseline.

## Validation

Focused tests cover:

- endpoint interpolation, tangent evaluation, and analytic area conservation
  of the quadratic facet;
- recovery of a connected straight interface initialized by Youngs PLIC;
- conservative diagonal vertex jumps;
- reported conservative fallback plus strict-mode failure;
- repeated algebraic continuity roots and interval filtering;
- deterministic sweep and policy metadata.

The source paper's circle study uses a unit square, `epsilon = 1e-6`, 1000
random circles with centers between `(0.4, 0.4)` and `(0.6, 0.6)`, radii in
`[0.2, 0.25]`, and grids from `10^2` through `320^2`. Reproducing that full
study remains a validation task; it is not claimed by the focused tests.

## Readiness decision

The kernel and adapter are ready for the five-benchmark Cartesian smoke suite.
Benchmark summaries must retain unresolved/fallback counts, convergence status,
and the frozen policy record. QUASI should not be extended to perturbed meshes,
and the policies above must not be tuned after inspecting benchmark outcomes.

## 2026-08-13 algebraic-root integration

`main/algos/baselines/quasi_roots.py` provides sampling-free enumeration of all
real roots in an admissible edge interval, including repeated roots and the
identically-zero relation. The enumerator is wired into both the C1 and
midpoint-curvature corrections, so repeated roots no longer depend on sampled
sign changes. Focused tests cover a repeated cubic root, interval filtering,
the non-unique zero polynomial, and integration through the QUASI verifier.
