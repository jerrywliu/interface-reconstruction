# Square missing-facet recovery

## Diagnosis

The fail-closed area metric exposed a stale merge-object accounting error in the
square driver. When topology orientation cannot resolve a merged component,
`findOrientations()` replaces that component with one active polygon per member
cell. Each replacement has the original cell's 3x3 stencil. `fitFacets()` then
applies the configured unresolved-orientation PLIC policy, which is LVIRA in the
frozen submission profile.

The inactive pre-split merged polygon remains in `MergeMesh.merged_polys` because
other topology objects can still refer to it. The square driver incorrectly paired
the facets returned by `runReconstruction()` with every value in that dictionary.
The result was one or more extra polygons and a deliberate `AreaMetricError`.
This was not evidence that an active mixed cell lacked a facet.

The fix requests the active polygon list from `runReconstruction()` and uses that
same list for both reconstructed and exact truth area. The fail-closed area check is
unchanged, and a reconstruction-level invariant now additionally requires every
active polygon to have exactly one non-null facet. Any real missing facet still fails
the case.

## Fallback provenance

For each unresolved active cell with a 3x3 stencil, the merged reconstruction path:

1. creates the requested Youngs, ELVIRA, or LVIRA facet;
2. assigns it to the active polygon with reason `unresolved_orientation`;
3. appends one record to `MergeMesh.plic_fallback_records`;
4. emits a facet-provenance event with `event_kind=plic_fallback`; and
5. writes the case-indexed event to `metrics/unresolved_plic_fallbacks.csv` and the
   active cell geometry/path to `metrics/cell_metrics.csv`.

An unresolved polygon without a usable stencil is not silently accepted. A final
reconstruction invariant now checks that the returned polygons are exactly the
mesh's active merge IDs and that each has one non-null facet. A separate algorithmic
policy would be required before such a cell could be admitted to a result set.

## Observed affected settings

At the read-only audit point, the running release manifest contained 24 failed square
runs: `linear`, `linear+corner`, and `circular` at each of these resolution and
perturbation pairs:

| Resolution | Perturbation magnitude |
| ---: | ---: |
| 0.50 | 0.20 |
| 0.50 | 0.30 |
| 0.64 | 0.30 |
| 1.00 | 0.20 |
| 1.00 | 0.30 |
| 1.28 | 0.30 |
| 1.50 | 0.20 |
| 1.50 | 0.30 |

The first reported six failures were the first two rows of this table across the
three merged methods; additional failures appeared as the square stage continued.

## Recovery decision

Do not append post-fix retries to the existing release or rewrite their source
metadata. The final-release audit requires the release source snapshot, consolidated
diagnostics, and every raw run manifest to identify one exact source commit. A retry
overlay from the fix commit would therefore be a transparent but multi-source result
set and would fail the current submission audit.

Safe sequence:

1. Let the current sweep finish and retain it unchanged as incident evidence.
2. Integrate this bounded driver fix and its regression tests into the reviewed
   submission source.
3. Run the 24 failed settings in a separate diagnostic root as a quick confirmation;
   do not promote that overlay as the final release.
4. Launch the authoritative 970-run sweep from the clean fix commit.
5. Require 970/970 runs, 24,250/24,250 cases, zero final missing facets, one source
   commit, and a passing final-release audit before figure regeneration.

If avoiding the full rerun becomes a hard requirement, first define and review a
new multi-source recovery-manifest policy and extend the release auditor to verify
the exact old/new run partition. That is a submission-provenance decision, not a
metric workaround.
