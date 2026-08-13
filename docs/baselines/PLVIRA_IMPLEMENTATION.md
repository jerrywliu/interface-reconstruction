# PLVIRA Cartesian static baseline

## Scope and status

The operational entry point now implements the two-dimensional uniform-
Cartesian PLVIRA path specified by Remmerswaal and Veldman:

1. compute curvature from the Cartesian generalized-height-function (GHF)
   hierarchy;
2. hold that curvature fixed in the restricted parabolic search space;
3. impose the target-cell volume constraint for every normal-angle trial;
4. minimize the squared `3 x 3` LVIRA volume-fraction objective from the
   paper-associated LVIRA initialization.

`reconstruct_plvira(...)` accepts no curvature argument. Analytic target
curvature is available only through the explicitly named diagnostic entry
point `reconstruct_plvira_exact_curvature_oracle(...)`. Both entry points
reject non-square, nonuniform, or perturbed geometry. No shared driver or
other baseline is changed.

This is a static reconstruction baseline. It does not implement advection,
surface tension, adaptive grids, or a perturbed-grid extension.

## Primary sources

- R. A. Remmerswaal and A. E. P. Veldman, *Parabolic interface
  reconstruction for 2D volume of fluid methods*, Journal of Computational
  Physics 469 (2022), 111473, DOI `10.1016/j.jcp.2022.111473`. Equation and
  section references below use arXiv v2, `arXiv:2111.09627`.
- S. Popinet, *An accurate adaptive solver for surface-tension-driven
  interfacial flows*, Journal of Computational Physics 228 (2009),
  5838--5866, DOI `10.1016/j.jcp.2009.04.042`. Algorithms 4--7 define the GHF
  hierarchy cited by Remmerswaal and Veldman.
- The Remmerswaal--Veldman paper-linked implementation,
  `ronaldremmerswaal/piecewise_parabolic_vof`, inspected at immutable commit
  `4ee19e3f5b0fc946e3243c961aa2e6d7d35b9087`.
- Popinet's Gerris implementation is used only to resolve MYC algebra,
  coordinate ordering, and singular-fit behavior. Its later numerical-policy
  differences from the published algorithm are listed below and are not
  silently imported.

## GHF source mapping

`main/algos/baselines/plvira_ghf.py` takes a bounded volume-fraction grid
indexed `[row][column]`, with `row` increasing in `y` and `column` increasing
in `x`. The represented liquid is the phase with fraction one.

| Decision | Controlling source | Implemented rule |
| --- | --- | --- |
| Mesh class | Remmerswaal--Veldman Section 3.1; Popinet Algorithms 4--7 | Uniform square Cartesian cells only. |
| Normal | Popinet Algorithm 7 step 1 and Algorithm 3 | Two-dimensional Mixed-Youngs-Centered (MYC), pointing from liquid to empty phase. |
| Direction order | Popinet Algorithm 7 step 2 | Decreasing absolute MYC-normal component; coordinate order (`x`, then `y`) breaks exact ties. |
| Column construction | Popinet Algorithm 4 | Independently extend each column until an empty terminal cell and a full terminal cell are found; no periodic or boundary completion is invented. |
| Column validity | Remmerswaal--Veldman Section 3.1 | Full/empty terminal cells and monotonically decreasing fractions toward the empty side. |
| Standard curvature | Remmerswaal--Veldman equations in Section 3.1; Popinet Algorithm 5 | Three consecutive heights and centered second-order first/second differences. |
| Mixed-height fallback | Popinet Algorithms 5--7 | Collect consistent height positions from both Cartesian directions and fit the published unweighted quadratic. |
| Independent positions | Popinet Algorithm 6 | Greedy source ordering with the published separation threshold `distance >= h`. |
| Final fallback | Popinet Algorithm 7 step 4 | Replace mixed heights with MYC-PLIC fragment barycentres from the target `3 x 3`; return zero only when fewer than three independent points or a singular fit remains. |
| Sign convention | Remmerswaal--Veldman equation (22) and the linked `levelSet_curvature` routine | Heights are expressed in the inward coordinate, so a convex liquid domain supplies positive `kappa` to the PLVIRA level set. |

The volume-fraction grid is assumed already geometrically bounded and clipped.
Following the source algorithms, mixed/full/empty classification uses exact
comparisons with zero and one; no benchmark-tuned fraction threshold is
introduced.

### Material paper/code discrepancy

Popinet's published Algorithm 6 uses a one-cell independence distance and an
unweighted least-squares fit. The later Gerris/Basilisk implementation uses a
half-cell distance, adds the target PLIC-fragment center with a non-unit
weight, limits the column search, and caps fitted curvature. None of those
changes is stated in the Remmerswaal--Veldman article or present in its linked
repository. The published algorithm therefore controls this baseline.

This choice is visible numerically: the deterministic circle check has one
published-rule `degenerate_zero` fallback at `R/h = 6.4`; the asymptotic
`R/h >= 12.8` rows use complete height functions and recover the reported
second-order scale. An exact executable match to the unpublished GHF path used
inside the Remmerswaal--Veldman flow solver remains blocked by the absence of
that code and the policy conflict above.

## Restricted PLVIRA search

For target cell centroid `xc`, prescribed GHF curvature `kappa`, normal
`eta = (cos(theta), sin(theta))`, and tangent
`tau = (-sin(theta), cos(theta))`, PLVIRA searches

```text
q(x) = eta . (x - xc) - phi
       + 0.5 * kappa * (tau . (x - xc))**2,
liquid = {x: q(x) <= 0}.
```

This is `Q_2^kappa` from Section 5.1, equation (22). Curvature is fixed, the
normal angle is the only optimization variable, and the tangential origin is
fixed at `xc`. No tangential-shift degree of freedom is added.

For each angle, `phi` is selected so the target-cell liquid volume is exact
(equation (23)). The monotone equation is bracketed with exact extrema of the
same level-set expression on the square and solved with Brent's method.

The objective is the squared LVIRA cost from Section 4.1, equation (16):

```text
sum over the 3 x 3 vertex-sharing stencil of
    (reconstructed volume fraction - reference volume fraction)**2.
```

The target term is omitted because volume enforcement makes it zero. The same
parabola is extended into all eight neighbors. Polygon/parabola intersection
areas and the Appendix C.2 angle derivative are analytic.

Initialization uses `lvira_angle_guess` from the linked implementation: the
centered volume-fraction gradient. The article specifies limited-memory BFGS
with a More--Thuente line search. The Python port retains the checkpointed
SciPy `L-BFGS-B` solve with the analytic gradient because SciPy 1.9.2 does not
expose that line-search choice. No restart, multi-start, alternative minimum,
or result-driven tuning is added.

## Oracle separation

- `reconstruct_plvira(...)`: operational row, always obtains curvature from
  `cartesian_ghf_curvature(...)` and records `curvature_source="cartesian-ghf"`
  plus full GHF diagnostics.
- `reconstruct_plvira_exact_curvature_oracle(...)`: diagnostic only, requires
  the caller's analytic curvature and records
  `curvature_source="exact-curvature-oracle"`.

The lower-level objective and analytic area helpers accept fixed curvature
because both modes share the restricted-search kernel. They are not baseline
dispatch modes. Exact curvature must never be reported under the unqualified
PLVIRA name.

## Static source check

`experiments/baselines/run_plvira_ghf_circle_convergence.py` is a deterministic
subset of Popinet's Section 6.1/Figure 5 circle study. It uses one circle
radius, four fixed cell-relative translations, exact analytic circle-square
fractions, and Cartesian resolutions `32, 64, 128, 256`. It records relative
curvature norms, pairwise orders, sample counts, and GHF fallback counts in
JSON and CSV.

Run:

```bash
PYTHONPATH=. python experiments/baselines/run_plvira_ghf_circle_convergence.py
```

Tracked results and the numerical interpretation are in
`experiments/baselines/results/plvira_ghf_circle/` and
`experiments/baselines/PLVIRA_GHF_CIRCLE_REPRODUCTION.md`.

Focused validation:

```bash
PYTHONPATH=. python -m pytest -q \
  test/algos/baselines/test_plvira.py \
  test/algos/baselines/test_plvira_ghf.py
```

## Remaining blockers

1. The Remmerswaal--Veldman linked flower benchmark supplies level-set-based
   exact curvature, not GHF curvature. It is therefore an oracle check and
   cannot validate the operational baseline.
2. The exact GHF fallback policies used for the article's flow results are not
   published in its linked code. The published Popinet algorithm and later
   Gerris/Basilisk implementation materially disagree as described above.
3. The optimizer remains a transparent porting deviation: article-level
   L-BFGS/More--Thuente, linked-code one-dimensional `brent_min`, and this
   pinned SciPy `L-BFGS-B` path are not identical.
4. Shared parabolic serialization and component-aware project metrics remain
   outside this isolated write scope. No full project sweep should be launched
   until that adapter is approved.
5. No perturbed-grid result is supported. A future adaptation would require a
   separate method name and is not part of PLVIRA.
