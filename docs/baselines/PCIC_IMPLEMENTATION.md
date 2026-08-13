# PCIC Baseline Implementation Report

## Status

This is a **paper-faithful, static-only bare-PCIC kernel**, not yet a complete
paper-comparison baseline. It implements the parts of Maity, Sundararajan, and
Velusamy that are fully specified and can be represented without changing the
repository's shared reconstruction dispatch:

1. six PLIC-derived samples per eligible mixed cell;
2. the weighted Riemann-sphere (SWFL) circle fit;
3. the paper's large-radius PLIC retention and minimum-radius guard;
4. both published conservative corrections, selected explicitly by the caller;
5. preservation of all circle/cell crossings, including four-crossing cells.

It intentionally stops before the smoothing and c0 passes. The source leaves
scientifically material choices unresolved, and the repository's `ArcFacet`
cannot represent the multi-arc cells shown in the PCIC paper. Those constraints
must be reviewed before this is wired into experiments or described simply as
"PCIC."

Primary source:

- R. K. Maity, T. Sundararajan, and K. Velusamy, "An accurate interface
  reconstruction method using piecewise circular arcs," *International Journal
  for Numerical Methods in Fluids* 93 (2021), 93-126.
  <https://doi.org/10.1002/fld.4876>

The paper was consulted in full through the publisher's article and PDF views.
No implementation from the authors was located or used.

## Paper-to-code specification

### Initial PLIC prediction

Section 2.2.1 says that bare PCIC begins with an LLS PLIC reconstruction
"coupled with" Parker and Youngs in a 3-by-3 cell block. The PCIC module does
not substitute this repository's Youngs, ELVIRA, LVIRA, or circular method.
Instead, `reconstruct_bare_pcic_cell` requires the complete PLIC stencil as an
input. This makes the dependency visible and lets the exact initial predictor
be supplied once its coupling is settled.

### PLIC point sampling

`sample_plic_segment` implements Equations (8)-(10). For PLIC endpoints `P`
and `Q`, each fit point is

```text
p_k = alpha_k P + (1 - alpha_k) Q,
alpha = [0.2, 0.25, 0.3, 0.7, 0.75, 0.8].
```

`collect_stencil_samples` uses all supplied PLIC facets in the 3-by-3 stencil
whose volume fractions lie in `[0.01, 0.99]`, exactly matching the curve-fit
cutoff in Section 2.2.1. The reconstruction cutoff remains `1e-6`.

### Riemann-sphere circle fit

`fit_riemann_sphere` implements Equations (12)-(20):

1. stereographically project `(x_k, y_k)` to `(x'_k, y'_k, z'_k)`;
2. use weights `w_k = (1 + x_k^2 + y_k^2)^2`;
3. form the weighted scatter matrix in Equation (17);
4. take the eigenvector of its smallest eigenvalue as `(alpha,beta,gamma)`;
5. recover `c` from Equation (19);
6. map the plane back to the circle center and radius using Equation (20).

A plane that maps to a line raises `PCICDegenerateFit`; the cell-level routine
then retains its supplied PLIC facet, consistent with the paper's straight-line
limit.

### Radius guards

Section 2.2.1 retains PLIC when `R >= 1e6 Delta x`. It also resets a fitted
radius below the cell diagonal to the diagonal and locates the center from the
PLIC segment. The source does not state which of the two possible PLIC-chord
centers is selected. This prototype tests both and chooses the one whose phase
area is closest to the target before conservative correction. That is a porting
decision, not an improvement to the fit.

### Volume correction

The paper presents two alternatives:

- `translate_center`: hold the radius fixed, translate the center along the
  PLIC-normal/perpendicular-bisector direction, bracket the target volume, and
  bisect;
- `adjust_radius`: hold the center fixed, bracket the target radius, and bisect.

The caller must select one. There is deliberately no default. Bisection follows
the paper's normalized `1e-10` convergence criterion and its stated `1e-6`
volume-fraction tolerance.

The Riemann fit is unoriented. The paper does not specify the phase-orientation
recovery after inverse projection. The prototype evaluates the disk and its
complement and carries forward the sign whose initial fraction is closer to the
target. Signed radii are only the repository's representation of this phase
choice.

## Material ambiguities and incomplete parts

### Which bare-PCIC correction produced the reported results

Section 2.2.1 describes center translation and radius adjustment, but does not
state a selection rule or identify one alternative as the source of all bare
PCIC tables. Treating either as the default would invent an experimental
decision. A paper comparison should either run both as separately named rows or
obtain clarification from the authors.

### Exact LLS/Parker-Young coupling

The PCIC paper names the initial predictor but does not provide its complete
algorithm. Scardovelli and Zaleski (2003) describe an LLS refinement of an
initial PLIC point cloud, while the PCIC wording says it is coupled to Parker
and Youngs. The number/order of LLS passes and exact boundary handling are not
specified in the PCIC article. This module therefore accepts those PLIC facets
instead of silently approximating the predictor.

### Smoothing pass

Section 2.2.2 specifies a 15-degree normal filter, refitting of neighboring arc
centroids, fixed-center radius correction, and rejection if the new radius
changes by more than 10%. It does not define the arc-centroid convention, the
normal evaluation point, or update ordering. Those choices affect the result,
so the pass is not implemented here.

### c0 correction

Section 2.2.3 specifies midpoint matching on shared edges, movement of a
perpendicular-edge endpoint to an endpoint on the common edge, and subsequent
center/radius adjustment at fixed endpoints. It exempts low-volume cells,
curvature sign changes, and "significant" radius changes, but gives no numeric
threshold for the last rule or a conflict/update order when a cell has two
neighbors. Implementing a deterministic global pass would require choices not
provided by the paper.

### Multi-arc cells

The paper explicitly illustrates cells with four circle-boundary crossings.
The repository's `ArcFacet` stores one center, one radius, and two endpoints,
so converting such a PCIC result would discard one interface component.
`PCICCircle` retains every crossing and refuses `to_arc_facet()` unless exactly
two crossings exist. Shared dispatch must not be added until the result model
can carry all required arcs.

## Validation

Focused tests cover:

- the exact Equation (10) sampling vector;
- recovery of a known circle by Equations (12)-(20);
- exclusion of the paper's `f=0.007`-style low-fraction fit cell;
- volume conservation for both published correction alternatives;
- refusal to truncate a four-crossing result into one `ArcFacet`.

Run with:

```bash
PYTHONPATH=. pytest -q test/algos/baselines/test_pcic.py
```

## Recommended checkpoint decision

Before continuing, decide:

1. whether to report the two bare corrections as separate baseline variants or
   seek the authors' intended selection rule;
2. the exact initial LLS/Parker-Young construction to use;
3. whether experiment plumbing will support multiple arcs per cell;
4. whether the baseline comparison should stop at bare PCIC or include the
   underspecified smoothing and c0 stages.

Until those decisions are resolved, this code should be called a
`bare-PCIC kernel`, not a complete PCIC baseline.
