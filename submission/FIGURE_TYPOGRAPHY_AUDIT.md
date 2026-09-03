# Figure typography audit

## Approved minimum

- Tick labels, legends, and convergence-order labels: 7 pt at final manuscript size.
- Axis labels and panel titles: 8 pt at final manuscript size.

## Figures that currently meet the minimum

- Figures 2--5 and 7--9 use native TikZ/LaTeX text at manuscript size.
- Figure B.24, the qualitative resolution-study grid.
- Figure B.25, the joint-refinement component-size distribution.
- The regenerated Figure 14 and Figure B.20 color candidates meet the minimum, but they are not yet promoted to the manuscript.

Figures B.17 and B.22 are the closest misses. Their tick labels are approximately 7 pt at final size, but their legends remain slightly below 7 pt.

## Layout changes

### Figures B.18 and B.21

Each figure contains four metric rows with a perturbation-sweep column and a resolution-study column. This produces eight axes plus a shared legend on one page. Raising every label to the approved minimum would crowd the axes and convergence markers.

Proposed fix: split each figure into a primary 2x2 panel for Hausdorff and facet-gap error and a secondary 2x2 panel for curvature and tangent diagnostics.

### Figure B.26

The figure places two metric panels and three representative reconstructions for each of two benchmarks on one page. It is assembled from previously rendered child PDFs, so the child text is scaled once during panel generation and again during composition.

Proposed fix: render the layout directly and split it into one ellipse figure and one Zalesak figure. Each figure should use a 1x2 metric row and a 1x3 representative-reconstruction row.

