# Whole-chain geometric-continuity smoke study

## Question

Can the conservative joint endpoint/curvature optimizer refine every maximal
smooth line/arc chain, instead of only chains containing an endpoint gap, and
thereby remove tangent discontinuities without degrading reconstruction quality?

This is a smoke study of the experimental `g1_chain` mode. The production
default remains `joint`.

## Design

- Benchmarks: project circles (cases 0, 1, 2) and ellipses (cases 0, 10, 24).
- Resolution: Cartesian-equivalent `N=32`.
- Meshes: Cartesian plus perturbation magnitude `w=0.1` with seeds 0, 1, 2.
- Matched modes: current joint endpoint refinement and whole-chain geometric
  continuity refinement.
- Common metrics: native symmetric Hausdorff error and arc-length-weighted
  curvature error recomputed from exact saved line/arc primitives.
- Guardrails: facet gap, cell-area residual, and concave-arc incidence.

The full matrix contains 48 matched case reconstructions (24 mode pairs).

## Results

### Ellipses

- Tangent-discontinuous joins after refinement: `620 -> 0` over 12 cases.
- Largest post-refinement tangent jump: `6.89e-2 -> 2.77e-13` radians.
- All 12 whole-chain components used the exact tangent-matched solution; none
  failed or used the conservative fallback.
- Median paired Hausdorff ratio (whole-chain/current): `0.790`, a 21% decrease.
  Nine of 12 cases improved by more than 5%; the largest increase was 3.84%.
- Median paired curvature-MAE ratio: `0.899`, a 10% decrease. All 12 cases
  improved, with ratios between `0.783` and `0.926`.
- Mean concave arc-length fraction: `0.00211 -> 0.00150`; concave arc count:
  `7 -> 6`.
- Largest relative cell-area residual: `9.49e-11 -> 5.62e-13`.

### Circles

- Tangent-discontinuous joins after refinement: `23 -> 0` over 12 cases.
- The nine circles requiring a whole-chain solve all used the exact
  tangent-matched solution; none failed or used the conservative fallback.
- Hausdorff and curvature errors remain at numerical-tolerance levels and do
  not regress. Several perturbed cases improve by orders of magnitude.
- No concave arcs occur under either mode.
- Largest relative cell-area residual remains below `9.45e-11`.

## Interpretation

This is a clear positive first signal. Whole-chain refinement achieves the
intended geometric continuity on every tested smooth chain while preserving
local volume and, for ellipses, generally improving both interface and
curvature accuracy. It also does not increase the small concavity incidence.

The important remaining caveat is cost. A three-case perturbed ellipse run took
roughly two to three minutes on this machine, materially longer than the current
localized pass. Before changing the production default, test at `N=64` and
`N=128`, record solver evaluations and wall time per component, and add a safe
size/cost policy if dense whole-chain solves scale poorly.

## Artifacts

- `results/submission/g1_chain_smoke_20260823/g1_chain_smoke_all_methods.pdf`
- `results/submission/g1_chain_smoke_20260823/case_results.csv`
- `results/submission/g1_chain_smoke_20260823/paired_results.csv`
- `results/submission/g1_chain_smoke_20260823/summary.csv`
- Reproduction: `python -m experiments.submission.run_g1_chain_smoke`
