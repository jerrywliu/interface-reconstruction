# PLVIRA static reconstruction prototype

## Scope and status

This module is a static-only prototype of the parabolic LVIRA (PLVIRA)
reconstruction of Remmerswaal and Veldman. It reconstructs one mixed cell from
a complete `3 x 3` stencil of convex polygonal cells and their volume
fractions. The curvature is a required input. The prototype does not compute
the generalized-height-function (GHF) curvature, advect the reconstruction,
or integrate with the repository's experiment drivers.

The implementation is intentionally isolated in
`main/algos/baselines/plvira.py`. It does not use the repository's circular or
three-cell reconstruction methods as substitutes for any PLVIRA step.

Primary sources:

- R. A. Remmerswaal and A. E. P. Veldman, *Parabolic interface
  reconstruction for 2D volume of fluid methods*, Journal of Computational
  Physics 469 (2022), 111473, DOI: 10.1016/j.jcp.2022.111473. Equations and
  sections below refer to the arXiv v2 text (arXiv:2111.09627).
- The paper-linked reference implementation,
  `ronaldremmerswaal/piecewise_parabolic_vof`, inspected at commit
  `4ee19e3f5b0fc946e3243c961aa2e6d7d35b9087` only where the article leaves a
  software-level choice unstated.

## Paper-to-code specification

For a center cell `c` with centroid `xc`, prescribed curvature `kappa`, unit
normal `eta = (cos(theta), sin(theta))`, and tangent
`tau = (-sin(theta), cos(theta))`, PLVIRA searches the restricted parabolic
space

```text
q(x) = eta . (x - xc) - phi
       + 0.5 * kappa * (tau . (x - xc))**2,
liquid = {x: q(x) <= 0}.
```

This is the paper's `Q_2^kappa` search space (Section 5.1, equation (22)). The
curvature is fixed, so the normal angle is the sole optimization variable.
The tangential origin is fixed at `xc`; no tangential-shift degree of freedom
is introduced.

For every trial angle, `phi` is chosen so the reconstructed liquid volume in
the center cell exactly equals its reference volume. This is equation (23) in
Section 5.5. The implementation brackets the monotone volume equation using
the exact extrema of `eta . (x - xc) + 0.5*kappa*(tau . (x - xc))**2` on the
center polygon and solves it with Brent's method.

The objective is the squared form of the LVIRA cost from Section 4.1,
equation (16):

```text
f_L2(theta)**2 = sum over the 3 x 3 vertex-sharing stencil of
    (reconstructed volume fraction - reference volume fraction)**2.
```

The center contribution is omitted in code because volume enforcement makes
it identically zero. The same reconstructed parabola is extended into every
neighboring cell.

Polygon/parabola intersection volumes are evaluated analytically. Polygon
edges are split at the roots of `q`, retained boundary pieces are combined
with exact line integrals along the parabolic boundary, and Green's theorem
gives the enclosed area. This implements the exact edge splitting and
polynomial correction described in Section 5.4 without numerical quadrature.

The derivative with respect to the normal angle uses Appendix C.2. The center
cell first determines `dphi/dtheta` from volume conservation (equation (C.9));
the same derivative is used when differentiating neighboring reconstructed
volumes and the squared LVIRA objective.

The objective is minimized with limited-memory BFGS from the LVIRA angle
guess. This follows Section 5.6. The paper-associated code supplies the
otherwise unstated initial guess: the angle of the centered volume-fraction
gradient. The helper in this prototype supports that initialization for a
rectilinear `3 x 3` stencil. A caller using another mesh must provide the
angle produced by its faithful LVIRA initialization explicitly.

## Decisions where the article is not fully explicit

1. **Curvature acquisition.** PLVIRA uses a GHF curvature (Sections 3.1 and
   5.2), but this paper does not fully specify the multidirectional GHF
   fallback and the repository's target experiments include perturbed
   polygons. The prototype therefore requires `curvature` from the caller.
   It does not estimate, repair, or replace it. End-to-end PLVIRA experiments
   remain blocked until a paper-faithful curvature provider is selected.

2. **Initial angle.** Section 5.6 does not state an initialization formula.
   The paper-linked implementation initializes PLVIRA with its
   `lvira_angle_guess`, a centered volume-fraction gradient. This prototype
   implements that formula for rectilinear stencils and accepts an explicit
   `initial_angle` otherwise.

3. **Optimizer details.** The article specifies limited-memory BFGS with a
   More-Thuente line search. SciPy 1.9.2, already pinned by this repository,
   exposes L-BFGS-B but does not expose the line-search selection through that
   interface. The prototype uses unconstrained `L-BFGS-B` with the exact
   analytic gradient. This is a porting deviation; no alternative search,
   restart, or multi-start heuristic is added.

4. **Stopping tolerance.** Section 5.6 gives a problem-scale-dependent error
   estimate `min(1e-2, (h/L)^2)`, but a standalone static cell call has no
   global length `L`. The API therefore exposes `gradient_tolerance` and uses
   `1e-8` by default, matching the paper-linked reconstruction code. No claim
   is made that this reproduces a particular paper figure's global tolerance.

5. **Volume-root bracket.** Appendix B writes the bracket for rectangular
   cells. The search equation is monotone for any polygon. For the accepted
   convex polygon input, the code computes exact boundary extrema of the same
   level-set expression, which is the direct polygonal bracket for equation
   (23), not a change to the search space or objective.

6. **Output representation.** The repository has no parabolic facet class and
   the allowed write scope excludes shared geometry types. The result is a
   method-local `ParabolicInterface` containing `center`, `angle`, `normal`,
   `tangent`, `curvature`, and `shift`, plus level-set and intersection-area
   evaluation. It is not silently converted to a circular arc.

## Validation

Focused tests cover:

- analytic line and parabola cuts of a unit square;
- exact center-cell volume enforcement;
- the Appendix C angle derivative against a centered finite difference;
- numerical recovery of an identifiable parabola from an exact uniform
  `3 x 3` volume-fraction stencil;
- input validation for the complete stencil and initialization contract.

Run with:

```bash
PYTHONPATH=. pytest -q test/algos/baselines/test_plvira.py
```

## Completeness checkpoint

The restricted-search PLVIRA reconstruction kernel is implemented and tested.
It is not yet an experiment-ready baseline because the following scientifically
material pieces are outside this isolated prototype:

- a faithful GHF curvature implementation for the intended mesh class;
- a shared parabolic facet/serialization representation;
- driver and metric integration;
- reproduction of the paper's flower reconstruction convergence study.

Those items should be reviewed before continuing. In particular, applying a
new curvature estimator to perturbed Cartesian cells would be an extension of
the published baseline and should not be done implicitly.
