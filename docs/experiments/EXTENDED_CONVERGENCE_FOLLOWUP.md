# Extended Convergence Follow-Up

## Status

The shared circular-fitting defect has been corrected and regression-tested in
commit `d8e2d7b`. A fresh five-case smoke completed all `36` settings and `180`
cases without a false line-precheck candidate. The corrected evidence passes
the gate below, and the 25-case extension was launched on 2026-08-13.

No shared algorithm, paper source, sealed result, or prior smoke artifact was
changed during this diagnosis.

## Corrected Smoke Result

The corrected run is
`results/static/extended_convergence_smoke_cell_area_fix_20260813/`, with its
diagnostic in
`results/static/extended_convergence_diagnosis_cell_area_fix_20260813/`.

- false line-precheck candidates: `0`;
- ellipse median facet-gap order: `2.87` at `w=0` and `2.90` at `w=0.2`;
- ellipse case-2 facet-gap order: `2.87` at `w=0` and `2.90` at `w=0.2`;
- maximum global relative ellipse area error: `2.41e-11`, reduced from
  `3.40e-9` in the contaminated run;
- maximum accepted line volume-fraction residual: `8.43e-7`, within the
  configured `1e-6` linearity threshold.

The five per-case facet-gap orders are `2.84--3.04` at `w=0` and
`2.86--3.05` at `w=0.2`. This restores a stable near-third-order signal while
remaining above the metric floor.

## Nearly Full Cell Diagnosis

The largest conservation residual occurs for ellipse case 2 at `N=512`,
`w=0`, in Cartesian cell `(405, 266)`. The cell volume fraction is
`0.9998871334`. Circular reconstruction retains a straight facet whose enclosed
fraction is `0.9997752576`, giving:

- absolute volume residual: `4.2677e-6`;
- cell-area-relative residual: `1.1188e-4`;
- global relative phase-area error: `3.3961e-9`.

This is an algorithm failure, not a saved-grid, metric-sampling, or
near-full-cell conditioning artifact. The conservation analyzer regenerates the
exact seeded Cartesian grid, and its prescribed mixed-cell volume agrees with
the analytic ellipse integration within `3.33e-14`.

The straight-line precheck in
`main/structs/polys/neighbored_polygon.py::fitCircularFacet` computes

```text
line area / prescribed fluid area
```

where the intended line volume fraction is

```text
line area / cell area.
```

For the affected cell, the implemented expression is `0.9998881116`, only
`9.7820e-7` from the target and therefore just inside the `1e-6` linearity
threshold. The correct fraction differs from the target by `1.1188e-4` and
would reject the line. The same denominator pattern occurs in
`BasePolygon.runSafeCircle`. Four of the twelve straight facets retained by the
ellipse smoke satisfy this false-precheck signature; most have smaller absolute
effects, but they confirm that the event is not unique to one cell.

## Case-2 Convergence Tail

At `w=0`, the case-2 facet gaps at `N=256,300,512` are
`2.166e-5`, `1.354e-5`, and `5.762e-6`. The resulting three-point order is
`1.83`, with a `300 -> 512` window order of `1.60`.

The falsely retained `N=512` line has two adjacent endpoint gaps totaling
`2.537e-3`. Across 900 mixed-cell joins, those two gaps contribute approximately
`2.819e-6`, or 49% of the reported mean facet gap. Merely subtracting that
known contribution gives a lower-bound remainder of `2.942e-6`; this is not a
corrected reconstruction result, but it changes the diagnostic three-point and
last-window orders to approximately `2.87` and `2.86`.

At `w=0.2`, another false-precheck candidate contributes approximately
`5.45e-7` to the `N=512` case-2 mean. Removing only that known contribution
changes the diagnostic three-point and last-window orders from `2.63/2.58` to
approximately `2.90/2.90`.

Thus the smoke's low case-2 facet-gap tail is materially contaminated by the
same algorithm defect. The other eight per-case fits near third order remain
useful as a signal, but the current extended data cannot be used to confirm or
reject the paper's prior aggregate `N^{-2.94}` claim.

## Gate Before Full Run

1. Correct both circular straight-line prechecks to normalize by cell area.
2. Add focused regressions for nearly full and nearly empty cells, including
   the exact `(405, 266)` smoke geometry.
3. Rerun the five-case `N=256,300,512`, `w=0,0.2` ellipse smoke in a fresh
   result directory.
4. Confirm local volume residuals return to the configured tolerance scale and
   recompute per-case facet-gap windows.
5. If clean, launch the 25-case extension using the existing non-destructive
   runner.

All five conditions are complete. The full extension output is
`results/static/extended_convergence_smoke_full_cell_area_fix_20260813/`.

Three rows containing unresolved-orientation fallbacks required a metric-only
correction after the full run. Reconstruction geometry and conservation were
unchanged. Commit `e108128` fixes active polygon/facet pairing and replacement
topology; clean reruns patch those rows in the derived result set
`results/static/extended_convergence_corrected_e108128_20260813/`. The
corrected perturbed-ellipse all-case mean orders are `2.00` for Hausdorff and
`2.91` for facet gap, while the previously reported median orders remain
`2.02` and `2.93`.

## Reproduction

The read-only diagnostic is:

```bash
python -m experiments.static.diagnose_extended_convergence
```

The first diagnostic artifact is
`results/static/extended_convergence_diagnosis_20260813_002727/`. It contains:

- `linear_facet_diagnostics.csv`: every retained line, both denominator
  calculations, exact local residuals, and adjacent endpoint-gap estimates;
- `convergence_windows.csv`: aggregate inputs plus every per-case resolution
  window;
- `summary.json`: the gate status and anomaly counts.
