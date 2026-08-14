# Common native metrics and QUASI convergence diagnosis

Date: 2026-08-13

## Scope

This diagnostic closes two QA gaps in the five-case Cartesian baseline study:

1. replace independent point-cloud sampling with a partition-insensitive native
   geometry metric and analytic ellipse truth; and
2. determine whether QUASI's universal ten-sweep exhaustion represents slow
   convergence, oscillation, or an unresolved update policy.

No PLVIRA, PCIC, or QUASI reconstruction policy was tuned after inspecting the
five-case results. The QUASI study extends only the diagnostic sweep limit from
10 to 100.

## Common observables

### Native symmetric Hausdorff distance

Each directed distance maximizes exact point-to-native-curve distance over the
source primitives. Target endpoints are projected onto each source primitive to
partition intervals where the nearest target can change, and each interval is
optimized independently. Nearest-target queries use conservative native-curve
bounding boxes. Lines, circular arcs, PLVIRA parabolas, and QUASI quadratics are
never converted to point clouds. Ellipse truth is one analytic rotated ellipse,
not 720 line segments.

The metric is invariant to curve partition to optimization tolerance. Tests
cover identical whole/split lines, a nonzero parabola-to-chord distance before
and after splitting the parabola, and identical circles represented by two or
four arcs.

### Geometric-curvature observable

For circles and ellipses, the common curvature observable is

```text
integral_Gamma_h |kappa_h - kappa_truth(nearest point)| ds
---------------------------------------------------------- .
                   length(Gamma_h)
```

The integral uses the actual geometric curvature of each returned native
primitive and arc-length quadrature. It therefore does not compare PLVIRA's GHF
stencil estimate against PCIC's radius or QUASI's midpoint value. The additive
arc-length weighting is insensitive to splitting the same curve into more
facets. Sharp-interface benchmarks are omitted from this curvature comparison.

## Native replay results

The replay evaluated all 300 saved rows: PLVIRA, both frozen PCIC corrections,
and QUASI on five benchmarks, three resolutions, and five cases. There are 299
finite geometry rows. PCIC radius adjustment on square case 1 at `N=32` has no
active reconstructed geometry and remains explicitly unevaluated.

The old sampled and new native Hausdorff values have a median ratio of 1.00 over
the 299 finite rows, so ordinary cases are stable. The important change is the
removal of the point-cloud floor. For PLVIRA circles at `N=128`, the old sampled
median is about `1.25e-3`, whereas the native median is `4.4e-5`.

Fine-grid (`64 -> 128`) signals from the common smooth-interface observable are:

| Method | Circle Hausdorff order | Circle curvature order | Ellipse Hausdorff order | Ellipse curvature order |
| --- | ---: | ---: | ---: | ---: |
| PLVIRA | 3.82 | 2.11 | 0.92 | -0.74 |
| PCIC center translation | -2.20 | 0.33 | -2.94 | 0.38 |
| PCIC radius adjustment | -1.18 | 0.33 | -2.36 | 0.38 |
| QUASI frozen port | 0.65 | -1.37 | 0.97 | -0.76 |

These are five-case diagnostic orders, not paper-ready convergence estimates.
PLVIRA supplies the only clean fine-grid second-order geometric-curvature signal,
and only on circles. PCIC's symmetric Hausdorff regression at `N=128` is caused
by missing reconstructed intervals: reconstruction-to-truth errors remain much
smaller than truth-to-reconstruction gaps. QUASI's common geometric curvature
does not converge in this frozen port.

## QUASI extended-sweep diagnosis

The `N=64`, cases `0--4` screen traces 25 cases for up to 100 lexicographic
Gauss-Seidel sweeps. It records update and net endpoint displacement, two-sweep
motion, residual C1 tangent mismatch, root misses and branch changes, and exact
area residual.

At sweep 10, the median maximum update is `2.41e-4` and the maximum is `9.60e-1`.
The median residual C1 mismatch is `1.31e-1`. Ten sweeps are plainly insufficient
for the current port.

By sweep 100:

- `9/25` cases satisfy the frozen `1e-11` displacement stop;
- `12/25` have maximum C1 mismatch at or below the root verifier's `1e-8`
  residual scale; and
- only `7/25` satisfy both conditions.

Two cases stop by displacement while retaining C1 mismatches of `7.9e-2` and
`6.0e-3`. A missing root produces zero displacement, so displacement alone is
not a valid continuity convergence test for this port.

Root-branch switching is transient: all 88 observed branch changes occur by
sweep 16. Root misses persist instead. There are 4,873 misses across the study,
including 3,419 at sweep 30 or later. Exact local area conservation remains
intact throughout, with maximum residual `4.44e-16`.

Five representative high-residual cases were replayed at the final state. The
verified cubic has no admissible shared-edge root for `3/50` circle joins,
`8/122` ellipse joins, `11/123` line joins, `3/48` square joins, and `4/118`
Zalesak joins. A dense shared-edge check confirms that at least one missed join
in each case retains a nonzero attainable mismatch; the largest per-case minima
range from about `1.9e-3` to `2.9e-1`. This is not explained by too few sweeps or
late root-branch switching.

## Source fidelity and root policy

The paper's Section 2.4 parameterizes a shared endpoint by an edge coordinate
`alpha`. Re-fitting both neighboring parabolas to preserve their respective
cell volumes and equating their endpoint slopes gives the cubic in Eq. (16).
The paper says that the cubic is solved directly and that the correction is
iterated, but it does not specify which in-edge root to select when several are
admissible, what to do when no root lies in `0 <= alpha <= 1`, whether pair
updates are simultaneous or in-place, their ordering, or a numerical stopping
criterion.

The frozen port enumerates and verifies every admissible real root, chooses the
one nearest the current endpoint, and applies joins in lexicographic in-place
Gauss--Seidel order. A recorded root switch means only that the selected root's
ordinal in the sorted admissible-root list changed since that join's previous
successful update. It is therefore a policy diagnostic, not proof that a
mathematically continuous root branch crossed another. Multiple admissible
roots are common (`63,038/172,526` successful updates), but selected-root
switches are rare and early.

Section 2.5 is a separate curvature correction for endpoints that could not be
connected during the earlier continuity stage or that lie at a boundary. The
paper says that this correction is used iteratively in conjunction with Eq.
(16), but it does not prescribe the schedule or the selection among eligible
neighboring cells. The frozen port applies this correction once before the C1
sweeps. This is an additional fidelity uncertainty, distinct from the
persistent no-root events observed for the Section 2.4 cubic itself.

## Recommendation

Retain QUASI as a qualified diagnostic prototype. Increasing the sweep count or
relaxing `1e-11` alone is not enough: the one-parameter, area-preserving C1 update
can have no verified root, and the displacement stop can report convergence in
that state. At a frozen no-root state, moving only one shared endpoint while
preserving both cell volumes and holding the other endpoints fixed cannot
achieve exact tangent continuity. This is a limitation of that pairwise update,
but not yet evidence that every faithful QUASI implementation reaches the same
state. Before using QUASI in a manuscript comparison, reconcile the Section 2.4
multiple-root and no-root policies, update ordering, Section 2.5 schedule, and
convergence criterion with an author implementation or author guidance.

PLVIRA remains the strongest operational higher-order baseline in the current
ports. PCIC remains usable only with explicit missing-interval/status reporting.

## Artifacts

- `experiments/baselines/results/common_native_metric_replay_20260813/`
- `experiments/baselines/results/quasi_sweep_diagnostic_20260813/`
- `experiments/baselines/results/quasi_join_admissibility_20260813/`
