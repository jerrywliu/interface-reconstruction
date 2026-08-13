# PCIC Cartesian baseline implementation report

## Status

This is a static-only Cartesian **bare-PCIC** implementation with a frozen,
predeclared policy for every source ambiguity needed by the project benchmark
runner. The implemented path contains:

1. the cited Parker--Young PLIC predictor;
2. the cited one-pass linear least-squares (LLS) refinement;
3. six PLIC-derived samples per eligible mixed cell;
4. the weighted Riemann-sphere (SWFL) circle fit;
5. the paper's large-radius PLIC retention and minimum-radius guard;
6. both published conservative corrections as separately named variants; and
7. component-aware output for every circle/cell crossing, including
   four-crossing and higher even-crossing cells.

It intentionally does not implement the underspecified smoothing or `c0`
passes. The two conservative bare-PCIC variants remain separately named. The
choices below were fixed before observing project benchmark outcomes and must
remain in the result provenance.

Primary sources:

- R. K. Maity, T. Sundararajan, and K. Velusamy, "An accurate interface
  reconstruction method using piecewise circular arcs," *International Journal
  for Numerical Methods in Fluids* 93 (2021), 93--126,
  [doi:10.1002/fld.4876](https://doi.org/10.1002/fld.4876).
- R. Scardovelli and S. Zaleski, "Interface reconstruction with least-square
  fit and split Eulerian--Lagrangian advection," *International Journal for
  Numerical Methods in Fluids* 41 (2003), 251--274,
  [doi:10.1002/fld.431](https://doi.org/10.1002/fld.431).
- J. E. Pilliod Jr. and E. G. Puckett, "Second-order accurate volume-of-fluid
  algorithms for tracking material interfaces," *Journal of Computational
  Physics* 199 (2004), 465--502,
  [doi:10.1016/j.jcp.2003.12.023](https://doi.org/10.1016/j.jcp.2003.12.023).

No implementation from the PCIC authors was located or used.

## Paper-to-code specification

### Cartesian input contract

`reconstruct_bare_pcic_cartesian_cell` accepts a complete, ordered `7 x 7`
block of uniform axis-aligned square cells. A full halo of that size is needed:
each of the nine final LLS lines uses preliminary Parker--Young lines from its
own `3 x 3` block, and each preliminary line itself needs a `3 x 3`
volume-fraction stencil. Boundary and perturbed-grid adaptation is rejected
explicitly.

### Parker--Young predictor

`parker_young_normal` implements Scardovelli and Zaleski Section 2.1: it
computes the volume-fraction gradient at the four corners of the central cell
and averages those values. This is algebraically the Parker--Young `a=2`
stencil written in Pilliod and Puckett Section 2.4. The line is then translated
at fixed normal until it conserves the target central-cell fraction.

### One-pass linear least-squares refinement

`reconstruct_lls_plic` follows Scardovelli and Zaleski Section 2.4. For every
preliminary cut-cell segment in the local `3 x 3` block it takes both endpoints
and the midpoint. It computes the radius of influence as the minimum distance
from the central preliminary-segment midpoint to the outer boundary of that
block, retains the points inside the circle, and solves the published ordinary
`2 x 2` least-squares normal equations. The direction is fitted as `y(x)` for a
more horizontal central segment and `x(y)` for a more vertical one. Only the
direction is retained; the final line is conservatively placed in the central
cell.

The cited LLS paper uses a centered-column line to initialize this procedure.
PCIC instead says LLS is "coupled with" Parker--Young. The implementation
therefore replaces only that preliminary line construction with Parker--Young
and otherwise leaves the cited one-pass LLS procedure unchanged. It does not
apply the optional two-to-four iterations mentioned in the LLS paper.

The LLS source says that when more than five cells are cut, the radius of
influence is multiplied by an unspecified number below one. This port fixes
that multiplier at `0.5`. It is the neutral midpoint of the stated interval,
has not been tuned against project outcomes, and is exposed as
`PCICConfig.lls_overcrowded_radius_scale` for provenance. Values outside
`(0,1)` are rejected.

### PLIC point sampling

`sample_plic_segment` implements PCIC Equations (8)--(10). For PLIC endpoints
`P` and `Q`, each fit point is

```text
p_k = alpha_k P + (1 - alpha_k) Q,
alpha = [0.2, 0.25, 0.3, 0.7, 0.75, 0.8].
```

`collect_stencil_samples` uses all supplied PLIC facets in the `3 x 3` stencil
whose volume fractions lie in `[0.01, 0.99]`, matching the curve-fit cutoff in
PCIC Section 2.2.1. The reconstruction cutoff remains `1e-6`.

### Riemann-sphere circle fit

`fit_riemann_sphere` implements PCIC Equations (12)--(20): stereographic
projection, weights `w_k = (1 + x_k^2 + y_k^2)^2`, the weighted scatter matrix,
the smallest-eigenvalue plane normal, and the inverse circle mapping. A plane
which maps to a line raises `PCICDegenerateFit`; the cell path then retains its
LLS PLIC facet, consistent with the paper's straight-line limit.

### Radius guards and phase

Section 2.2.1 retains PLIC when `R >= 1e6 Delta x`. A fitted radius below the
cell diagonal is reset to that diagonal and the center is placed on the
perpendicular bisector of the central PLIC chord.

The Riemann fit itself is unoriented, and the PCIC article does not specify how
to recover disk versus complement. The frozen default
`phase="infer_from_plic"` uses the oriented central LLS/Parker--Young segment:
the fitted circle is disk phase when its center lies on the reconstructed-fluid
side of the segment and complement phase otherwise. This reuses the source
predictor's material orientation rather than selecting the sign with the
smaller observed benchmark error. Explicit `disk` and `complement` inputs are
retained for deterministic checks.

### Separate conservative correction variants

The two article variants remain distinct:

- `bare PCIC (center translation)` holds radius fixed and translates the center
  along the perpendicular bisector of the **fitted target-cell chord**; and
- `bare PCIC (radius adjustment)` holds the fitted center fixed and adjusts the
  radius.

The first implementation checkpoint incorrectly used the central PLIC normal
for translation. That is now corrected. In a multi-arc target cell, the frozen
porting policy selects the in-cell arc chord whose midpoint is closest to the
central PLIC midpoint, breaking ties by tangent alignment and then coordinate
order. That chord determines only the translation direction. Every arc
component remains in the corrected output and in the metrics.

The article also does not state which conservative translation root to use if
several exist. The frozen `nearest_bracket` policy enumerates sign-changing
brackets symmetrically in both translation directions, beginning with 32
subintervals per cell width and expanding in bounded-work shells. It bisects
every bracket in the first shell containing a root and selects the root with
the smallest absolute displacement from the fitted center. A negative-offset
tie wins deterministically. Radius adjustment uses the monotone fixed-center
radius bracket.

Both corrections use the paper's `1e-6` volume-fraction tolerance and
normalized `1e-10` bisection criterion.

### Component-preserving geometry

`PCICCircle.intersections` retains every unique target-cell boundary crossing.
The crossings are sorted around the circle and adjacent pairs whose circular
interval lies inside the convex Cartesian cell become separate
`PCICArcComponent` objects. The orientation places the reconstructed phase on
the left: counterclockwise for disk phase and clockwise for complement phase.

`to_arc_facets()` returns every open component. `to_arc_facet()` succeeds only
for exactly one open component, so a four-crossing/two-component result cannot
be silently truncated to the repository's one-facet cell representation.
Closed circles and unresolved odd/tangent crossing populations remain explicit.

## Frozen porting decisions and incomplete parts

### LLS `y(x)` versus `x(y)` branch

The cited LLS source says both forms are checked but does not provide a precise
tie/selection rule. This port chooses the nonsingular form from the central
Parker--Young segment (`y(x)` when it is more horizontal, `x(y)` otherwise).
This convention is recorded and has not been tuned against project outcomes.

### Summary of source ambiguities

The operational Cartesian configuration freezes the following choices before
the matched project smoke test:

| Ambiguous source detail | Frozen policy |
| --- | --- |
| Disk versus complement | Infer from the fitted center's side of the oriented central PLIC |
| More than five LLS cut cells | Multiply the influence radius by `0.5` |
| Minimum-radius center side | Use the inferred PLIC phase |
| Multiple fixed-radius translation roots | Select the nearest enumerated conservative root |
| Multiple in-cell arc chords | Select the chord closest to and most aligned with the central PLIC; preserve all arcs |
| Conservative correction attribution | Report center translation and radius adjustment as separate variants |

### Smoothing pass

Section 2.2.2 specifies a 15-degree normal filter, refitting of neighboring arc
centroids, fixed-center radius correction, and rejection if the new radius
changes by more than 10%. It does not define the arc-centroid convention, the
normal evaluation point, or update ordering. Those choices affect the result,
so the pass is not implemented.

### `c0` correction

Section 2.2.3 specifies midpoint matching on shared edges, movement of a
perpendicular-edge endpoint to an endpoint on the common edge, and subsequent
center/radius adjustment at fixed endpoints. It exempts low-volume cells,
curvature sign changes, and "significant" radius changes, but gives no numeric
threshold for the last rule or a conflict/update order when a cell has two
neighbors. No deterministic global pass is implemented.

### Primary-paper ellipse reproduction

The paper's randomized Cartesian ellipse study remains a desirable source-scale
check, but an exact table replay is impossible because its random realizations
and correction attribution are unpublished. The ambiguities needed to execute
the method are now frozen above. Both correction variants must be run and
reported separately; the better result must not be selected after observation.

## Validation and numerical checkpoint

Focused tests cover:

- the published Parker--Young `a=2` gradient stencil;
- the source LLS endpoint/midpoint fit and conservative line placement;
- the frozen half-radius overcrowded LLS policy and invalid-scale rejection;
- the exact PCIC Equation (10) sample vector;
- recovery of a known circle by Equations (12)--(20);
- exclusion of the paper's low-fraction fit cell;
- volume conservation for both separately named corrections;
- the fitted-chord perpendicular-bisector direction;
- PLIC-based phase inference and the nearest-root policy;
- explicit rejection of incomplete Cartesian boundary halos;
- deterministic principal-chord selection for multi-arc translation; and
- pairing four crossings into two preserved arc components.

Run with:

```bash
PYTHONPATH=. python -m pytest -q test/algos/baselines/test_pcic.py
PYTHONPATH=. python -m experiments.baselines.run_pcic_static_check
```

For the frozen deterministic line fixture, the Parker--Young normal-angle error
is `2.9494863488e-2` rad and the one-pass LLS error is
`6.0824367739e-4` rad; the LLS target-fraction residual is
`6.63e-11`. On the frozen circular fixture, center translation and radius
adjustment have target-fraction residuals `4.77e-11` and `3.51e-11`,
respectively. These are kernel checks, not paper-table results.

## Remaining baseline gates

Before an unqualified paper comparison:

1. consume the shared five-benchmark Cartesian fixture and run both correction
   variants without policy changes;
2. report unsupported boundary-halo cells and every reconstructed, fallback,
   unresolved, and multi-component population;
3. compare the ellipse error scale/order with the primary paper while clearly
   labeling this as a matched project study rather than an exact table replay;
4. keep component-aware metrics and serialization for every crossing; and
5. label the comparison `bare PCIC`, because the source's smoothing and `c0`
   passes remain underspecified and are intentionally excluded.
