# Omitted-Range Figure Refresh

This package applies broken logarithmic axes only where the plotted methods
separate into ordinary-error and numerical-floor regimes.
Curves are split at the omitted interval, so no line segment is drawn across
compressed plot space.

## Applied treatment

- Main text: square, circle, and Zalesak resolution panels.
- Appendix B.17: square Hausdorff distance and facet gap.
- Appendix B.18: circle Hausdorff distance, facet gap, and curvature MAE;
  tangent error remains unbroken.
- Appendix B.22: Zalesak Hausdorff distance and facet gap.
- Main higher-order ellipse comparison: facet gap only.
- Appendix B.26--B.27: ellipse facet gap only, and both Zalesak error metrics.

Side-by-side metrics use the same retained ranges whenever both panels require
an omitted interval.
The B.26 continuity composite was split into benchmark-specific ellipse and
Zalesak figures, each with one metrics legend and its representative
reconstructions.

## Sources

- Perturbed benchmark metrics:
  `results/static/submission_static_20260731_012430_505aefa45432.sealed`
- Perturbed curvature metrics:
  `results/submission/perturbed_native_curvature_panels_20260822`
- Higher-order ellipse metrics:
  `experiments/baselines/results/ellipse_all_method_25case_extended_comparison_20260814`
- Joint-C0 runs:
  `plots/appendix_b5_joint_c0_20260814_perturb_sweep_*`

## Validation

- Focused tests: `19 passed`.
- Vector PDF QA: `9/9 passed`; no raster image objects were introduced.
- The full 43-page manuscript compiles without float-overflow, undefined-label,
  or multiply-defined-label warnings.
