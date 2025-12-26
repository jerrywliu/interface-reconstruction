# Interface Reconstruction Algorithms (Static Experiments)

This document summarizes the reconstruction algorithms used across the static experiments (lines, squares, circles, ellipses, Zalesak). For each method, we list the human-friendly name, the code identifier (string you pass via `--facet_algo` or set in configs), and a brief description focusing on merging and corner handling.

## Algorithm Catalog

- Friendly: Youngs
  - Code identifier: `Youngs`
  - Description: Baseline Youngs' PLIC interface reconstruction (planar/linear, no merging, no corner modeling).

- Friendly: LVIRA
  - Code identifier: `LVIRA`
  - Description: Least-squares VOF interface reconstruction (linear, no merging, no corner modeling).

- Friendly: Linear (without merging)
  - Code identifier: `safe_linear`
  - Description: Our linear reconstruction that skips cell merging. Faster but can be less robust near complex features; no explicit corner modeling.

- Friendly: Linear (with merging)
  - Code identifier: `linear`
  - Description: Our linear reconstruction with cell merging to improve fidelity near complex geometry; no explicit corner modeling.

- Friendly: Linear (with corners)
  - Code identifiers: `linear+corner`, `safe_linear_corner`
  - Description: Linear facets augmented with corner detection/modeling. Variants may differ in safety checks or merging behavior; used in some square/static sweeps.

- Friendly: Circular (without merging)
  - Code identifier: `safe_circle`
  - Description: Circular facet reconstruction that skips cell merging (falls back to Youngs when orientation/curvature is ambiguous by design).

- Friendly: Circular (with merging)
  - Code identifier: `circular`
  - Description: Circular facet reconstruction with cell merging for higher accuracy on curved interfaces.

- Friendly: Circular (with corners)
  - Code identifier: `circular+corner`
  - Description: Circular facets combined with corner detection/modeling (requires merging). Used in advection configs and now in Zalesak static sweeps.

## Notes on Merging and Corners

- Merging: Algorithms labeled “with merging” combine neighboring cells to stabilize/regularize the least-squares problems and produce more reliable facets in under-resolved or corner-rich regions. This often improves accuracy at extra compute cost.
- Corners: “+corner” variants aim to explicitly represent cusps/corners by combining multiple local facet hypotheses and/or extending facet construction to capture intersecting interfaces.

## Where to Use Which

- Linear vs Circular: Use linear variants for predominantly straight interfaces; circular variants for curved interfaces (e.g., circles, arcs). Mixed or cornered shapes (squares, Zalesak) benefit from merging and corner-aware variants.
- Safe vs merging: Prefer `safe_` (no merging) for quick runs or when speed matters; prefer merging variants (`linear`, `circular`, `circular+corner`) for accuracy and robustness in benchmarks.
