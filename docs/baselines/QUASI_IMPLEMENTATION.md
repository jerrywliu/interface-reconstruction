# QUASI Static Prototype

## Status

This is a **partial, static-only checkpoint**, not yet a complete baseline for
paper experiments. The connected-cell core is implemented and tested. The
curvature correction for diagonal and boundary configurations is intentionally
not implemented because the primary paper leaves a material selection rule
unspecified.

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
   When C0 connection is impossible, the paper moves the discontinuous endpoint
   to match midpoint curvature with a "selected neighboring mixed cell" having
   `0.02 < F < 0.98`; out-of-range roots can trigger a vertex jump. The article
   does not define how that target neighbor is selected when several candidates
   exist. Because this choice changes the reconstructed geometry, the prototype
   reports these cases through `QuasiResult.unresolved` and raises
   `QuasiTopologyError` by default instead of inventing a policy.

## Ambiguous decisions

- **Multiple roots of Eq. (16).** The article derives a cubic but does not state
  which admissible root to use. The prototype uses the root in `[0, 1]` nearest
  the current C0 predictor, i.e. the least endpoint displacement. This decision
  needs author review before using the baseline in comparisons.
- **Ordering of pairwise corrections.** The paper says that the correction is
  applied to all interface segments iteratively, but does not prescribe an
  ordering. The prototype uses deterministic mesh-index order.
- **One-endpoint C0 ties.** The paper gives phase consistency as the primary
  criterion and normal alignment as the secondary criterion. Exact remaining
  ties are resolved by endpoint index solely for determinism.

## Deliberate porting deviations

- The source solves the expanded cubic in Eq. (16) directly. The prototype
  evaluates the mathematically equivalent tangent mismatch and brackets all
  roots on the shared edge with `scipy.optimize.brentq`. This avoids duplicating
  frame-dependent expanded coefficients but is a numerical-solver deviation.
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
- explicit failure/checkpoint behavior when the underspecified curvature path
  is required.

The source paper's circle study uses a unit square, `epsilon = 1e-6`, 1000
random circles with centers between `(0.4, 0.4)` and `(0.6, 0.6)`, radii in
`[0.2, 0.25]`, and grids from `10^2` through `320^2`. That reproduction is not
claimed here: circles can exercise Section 2.5, which must be resolved first.

## Completeness decision

Do not add QUASI to shared experiment drivers yet. Before continuing, verify
the intended target-neighbor rule for Section 2.5 and approve or replace the
least-displacement rule for multiple C1 roots. Once those decisions are fixed,
the next validation step is the paper's random-circle `L1` study, followed by
the project's matched static smoke suite.

## 2026-08-13 algebraic-root checkpoint

`main/algos/baselines/quasi_roots.py` now provides sampling-free enumeration
of all real roots in an admissible edge interval, including repeated roots and
the identically-zero relation. Focused tests cover a repeated cubic root,
interval filtering, and the non-unique zero polynomial. This removes the
numerical sign-change limitation identified in the fidelity audit, but it is
not yet wired into the reconstruction: doing so faithfully still requires the
paper's exact expanded continuity polynomial and an approved multiple-root
selection rule.

Section 2.5 remains unimplemented. The available source record does not settle
the target-neighbor selection when several mixed cells satisfy the published
volume-fraction filter, nor the associated root and update-order policies.
Accordingly, no random-circle reproduction is claimed at this checkpoint.
