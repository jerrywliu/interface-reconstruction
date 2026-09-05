# Paper figure style guide

This file is the source of truth for manuscript-facing method names, metric labels, colors, markers, and layout rules.
The implementation lives in `experiments/plotting.py`, `experiments/static/run_perturbed_sweeps.py`, and the paper-facing figure generators.

## Method labels and styles

| Internal ID | Exact display label | Color | Marker | Line |
|---|---|---:|:---:|:---:|
| `Youngs` | Youngs | `#D94F9D` | circle | solid |
| `ELVIRA` | ELVIRA | `#00A6C8` | square | dashed |
| `LVIRA` | LVIRA | `#84B547` | diamond | dash-dot |
| `plvira` | PLVIRA | `#2D7D64` | circle | dashed |
| `pcic_center` | PCIC (center translation) | `#7C5AA6` | square | dash-dot |
| `quasi` | QUASI | `#4B5563` | diamond | dotted |
| `safe_linear` | Ours (linear, per-cell) | `#74A9CF` | up triangle | dashed |
| `linear` | Ours (linear, graph-coordinated) | `#2F6FA3` | down triangle | solid |
| `linear+corner` | Ours (linear + corners, graph-coordinated) | `#173F73` | X | solid |
| `safe_circle` | Ours (circular, per-cell) | `#E6AB02` | up triangle | dashed |
| `circular` | Ours (circular, graph-coordinated) | `#D55E00` | down triangle | solid |
| `circular+corner` | Ours (circular + corners, graph-coordinated) | `#B91C1C` | X | solid |
| `linear+C0` | Ours (linear, graph-coordinated + joint C^0) | `#2F6FA3` | filled plus | dotted |
| `circular+C0` | Ours (circular, graph-coordinated + joint C^0) | `#D55E00` | filled plus | dotted |
| `circular+corner+C0` | Ours (circular + corners, graph-coordinated + joint C^0) | `#B91C1C` | filled plus | dotted |

The higher-order comparison panels use the complete circular-family labels above rather than shortening them to only "per-cell" or "graph-coordinated."
Method order follows the table from top to bottom within each relevant subset.

## Metric language

| Internal metric | Axis label | Panel title with resolution |
|---|---|---|
| `hausdorff` | Hausdorff error | Hausdorff error vs cells per side |
| `facet_gap` | Facet-gap error | Facet-gap error vs cells per side |
| `curvature_error` | Curvature MAE | Curvature MAE vs cells per side |
| `tangent_error` | Tangent error | Tangent error vs cells per side |
| `area_error` | Area error | Area error vs cells per side |

Use `Cells per side, N` and `Perturbation magnitude, w` with the mathematical symbols typeset in math mode.
Use `Perturbation sweep` and `Resolution study` as the two-column headings in Appendix B.
Use `Mixed-cell coverage` rather than a shortened coverage label when the panel is not already unambiguous.

## Typography and spacing

- Tick labels, legend text, and convergence-order labels must be at least 7 pt at final manuscript size.
- Axis labels and panel titles must be at least 8 pt at final manuscript size.
- Shared vertical metric labels use a 5 pt visible gap from the left edge of the y-axis tick labels.
- Figure legends use a 7 pt visible gap above the topmost panel title.
- Figure 10 uses a 5 pt legend gap, Figure 12 uses an 11 pt gap, and Figure B.18 uses a 4 pt gap to account for their different legend row counts and panel density.
- Figure B.22 uses a 3 pt shared-label gap because its compact broken-axis layout otherwise leaves excessive white space.
- Dense appendix grids should use the available text width and should not be shrunk by a LaTeX height constraint.
- Legend rows may change by figure, but their order and label wording must not change.

## Broken axes

Omitted ranges are used only when they separate numerical-floor results from ordinary reconstruction errors.
The omitted interval and y-axis limits should match across adjacent Hausdorff and facet-gap panels when the data permit.
Lines and IQR ribbons stop at the break and resume in the other retained band.
When an IQR itself spans the omitted interval, its clipped portions appear in both retained bands.
This is expected and does not represent two independently computed IQRs.

## Statistical captions

For pooled perturbed-mesh resolution panels, use: "At each N, we report the median and interquartile range over all 125 combinations of perturbation magnitude and benchmark configuration."
Do not describe these summaries as medians of perturbation-level medians.

## Reference styles

- Reference interface: black dashed line.
- Numerical or geometric fitting floor: medium gray dotted line.
- Interquartile range: transparent fill in the corresponding method color.
- Full cells, empty cells, mixed cells, generating interfaces, and reconstructed interfaces follow the shared reconstruction palette used by the paper-facing plotting helpers.
