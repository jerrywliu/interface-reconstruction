# Appendix B topology diagnostics

## Questions

This audit addresses two collaborator questions about the sealed Appendix B
panels:

1. Does the per-cell degradation with mesh perturbation in Figure B.16 come
   from unresolved topology, and does the same explanation hold broadly?
2. Does the persistent Zalesak tail in Figure B.20 come from cusp topology
   rather than perturbation magnitude?

The analysis is read-only with respect to the sealed sweep at
`results/static/submission_static_20260731_012430_505aefa45432.sealed/`
(source commit `505aefa454328d4ba34ade5e7247050a0acfc793`). It evaluates
13,125 saved cases and does not rerun a reconstruction.

## Definitions and limitations

- A **shared-vertex conflict** uses the manuscript's formal consistency test:
  saved facets incident to the same mesh vertex assign that vertex different
  phase labels. Vertices within `max(1e-12, 1e-10 * domain diagonal)` of an
  incident facet are excluded.
- A **high-error line case** has facet gap above `1e-7`. The line results are
  strongly bimodal between the numerical floor and errors well above this
  threshold, so the classification is insensitive to modest threshold changes.
- A **high-error Zalesak case** has facet gap above `1e-5`. At `N=50,64`, the
  numerical-floor and unresolved-corner groups are separated by several orders
  of magnitude.
- `num_final_linear_corner_cells` counts constituent mixed cells assigned a
  line--line corner representation. It does not count distinct physical cusps.
- A shared-vertex conflict is a direct inconsistency diagnostic, but its absence
  does not prove that the correct neighbor support or cusp primitive was found.
  Conversely, line facets approximating a genuinely curved or cusped interface
  can disagree at shared vertices even when their geometric error is reasonable.

## Figure B.16: lines

### Result

The meeting hypothesis is strongly supported for the exactly representable line
benchmark, with one qualification: the formal conflict test identifies a
substantial part, but not all, of the graph-coordination benefit.

Across all six resolutions (150 cases per perturbation magnitude), the per-cell
line variant has shared-vertex conflicts in:

| Perturbation | 0 | 0.05 | 0.1 | 0.2 | 0.3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| Per-cell cases with conflicts | 0% | 0% | 4.7% | 16.0% | 29.3% |
| Graph-coordinated cases with conflicts | 0% | 0% | 0% | 2.0% | 2.7% |
| Per-cell high-error cases | 0% | 0.7% | 7.3% | 26.7% | 45.3% |
| Graph-coordinated high-error cases | 0% | 0% | 0% | 2.0% | 3.3% |

Every per-cell case with a detected conflict is high-error. At `w=0.1,0.2,0.3`,
detected conflicts account for `63.6%`, `60.0%`, and `64.7%` of per-cell
high-error cases. The remaining high-error cases do not violate the saved
shared-vertex test, so they should be described as additional failures of the
independent support selection, not as proven phase-label conflicts.

The graph-coordinated setting medians remain at the numerical floor across the
perturbation sweep. The per-cell setting median rises sharply at `w=0.3` because
the affected-case fraction approaches one half. This matched behavior supports
the claim that graph coordination stabilizes the reconstruction under mesh
perturbation. It does not isolate merging from orientation propagation.

### Recommended factual takeaway

> On the linear benchmark, independent per-cell support selection becomes
> increasingly vulnerable to inconsistent neighbor orientations as mesh
> perturbation grows. Shared-vertex conflicts are observed only in high-error
> cases, while graph coordination suppresses both the conflict incidence and
> the associated error tail.

Avoid saying that every per-cell error is a formal topology conflict.

## How broadly does the B.16 explanation hold?

The answer depends on whether the fitted primitive can represent the target.

- **Circular facets on circles:** graph coordination reduces the fraction of
  cases with shared-vertex conflicts from `12--25%` to at most `0.7%`. At low
  perturbation both variants still have numerical-floor geometric errors, which
  shows that phase-label consistency and Hausdorff/facet-gap accuracy are
  related but distinct properties. At `w=0.2,0.3`, the per-cell geometric error
  also develops a large tail.
- **Circular facets on ellipses:** per-cell conflict incidence grows from
  `4.7%` to `26.7%`, while graph coordination remains between `0.7%` and `4%`.
  The median per-cell/graph-coordinated facet-gap ratio grows from `1.0` to
  `2.22`. This is consistent with topology/orientation coordination helping,
  although curvature approximation remains part of the error.
- **Line facets on circles and ellipses:** graph coordination improves facet
  gaps, especially as perturbation grows, but the shared-vertex test is not a
  clean attribution tool because a set of independent straight segments cannot
  exactly represent the curved interface.
- **Squares and Zalesak with line-only or circle-only primitives:** phase-label
  conflicts are common in both variants because the selected primitive family
  omits explicit cusps. Graph coordination gives a modest improvement, but this
  comparison cannot separate topology from model inadequacy.

Thus, the broad defensible statement is that graph coordination becomes more
valuable as perturbation stresses per-cell neighbor selection. The stronger
claim that all cross-benchmark degradation is caused by unresolved topology is
not supported.

## Figure B.20: Zalesak

### First clarify the plotted band

The perturbation panel in Figure B.20 does **not** show the quartiles of all
individual cases. For each resolution and perturbation, the sweep first stores
the median over 25 cases. The plotted line is the median of those five
resolution-level medians, and the shaded band is their interquartile range.

Consequently, the upper edge primarily exposes coarse-resolution behavior. It
is high at `w=0,0.1,0.2`, when the `N=64` setting median is high, but returns near
the numerical floor at `w=0.05,0.3`, when just over half of the `N=64` cases
succeed. The `N=50` median remains between `1.05e-2` and `1.15e-2` for every
perturbation magnitude but is the maximum rather than the 75th percentile of
the five resolution medians.

### Cusp-resolution diagnosis

At the two coarse resolutions, failure to accept line--line corner cells almost
perfectly identifies the high-error population:

| Resolution | High error, no line--line corner cells | High error, with line--line corner cells | Low error, no line--line corner cells | Share of high errors explained |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 109 | 2 | 0 | 98.2% |
| 64 | 62 | 1 | 0 | 98.4% |

At `N=64`, the high-error fraction stays between `48%` and `52%` across all five
perturbation magnitudes, and the fraction with no line--line corner cells also
stays between `48%` and `52%`. Every one of the 62 no-corner cases is high-error.
This is strong evidence that the persistent **case-level** upper quartile is an
under-resolved cusp-identification effect rather than a monotone response to
perturbation magnitude.

The formal shared-vertex conflict rate is lower (`16--32%` of `N=64` cases).
It detects a subset of the bad reconstructions, whereas the missing line--line
corner representation detects nearly all of them. The accurate phrasing is
therefore "unresolved cusp geometry" or "failure to identify the straight slot
corners," not simply "topological inconsistency."

At `N>=100`, every case contains line--line corner cells and setting medians are
at the numerical floor. A small residual high-error tail remains (`0--16%` by
setting, using the conservative `1e-5` threshold), so those cases have another
cause and should not be attributed to missing corner identification without a
separate local audit.

### Recommended factual takeaway

> The coarse-grid Zalesak tail is nearly entirely associated with cases in
> which the straight slot corners are not assigned line--line corner facets.
> Its incidence is approximately constant across perturbation magnitudes at
> `N=64` and disappears as a missing-corner failure mode by `N=100`, indicating
> an under-resolution threshold rather than perturbation-driven degradation.

For the current B.20 caption, also explain that the shaded band aggregates
resolution-level medians; do not call it the worst 25% of individual cases.

## Reproduction

Run from the repository root:

```bash
python experiments/submission/diagnose_appendix_b_topology.py
```

Outputs are written to
`results/submission/revision_diagnostics_20260813/`:

- `topology_by_setting.csv`: geometric errors and formal conflict rates by
  benchmark, method, resolution, and perturbation.
- `paired_variant_summary.csv`: matched per-cell/graph-coordinated summaries.
- `line_conflict_association.csv`: B.16 conflict/error contingency statistics.
- `zalesak_cusp_by_setting.csv`: B.20 setting-level errors and corner incidence.
- `zalesak_cusp_contingency.csv`: coarse-resolution corner/error contingency.
- `zalesak_b20_plot_semantics.csv`: exact across-resolution aggregation shown
  by the perturbation panel.
- `b16_lines_topology_diagnostic.{pdf,png}`: line error and conflict trends.
- `cross_benchmark_topology_diagnostic.{pdf,png}`: scope and limitations of the
  same explanation across benchmarks.
- `b20_zalesak_cusp_diagnostic.{pdf,png}`: resolution transition and `N=64`
  corner-incidence comparison.
- `manifest.json`: sealed input, source commit, thresholds, and row count.

