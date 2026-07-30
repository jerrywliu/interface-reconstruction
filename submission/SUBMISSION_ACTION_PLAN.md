# Submission Action Plan

Updated: 2026-07-30

This is the concise execution view. Technical provenance lives in
`RESULT_SET_AUDIT.md`, `submission_config.json`, and `figure_provenance.csv`.
The human-readable replacement sequence is `FIGURE_PROMOTION_PLAN.md`.

## P0: Freeze One Submitted Result Set

- [x] Define the target configuration: 970 runs, 24,250 cases, five workers.
- [x] Lock the production method to `pre_f8_corner + exact_linear_support_only + LVIRA`.
- [x] Normalize the independent-cell circular fallback hierarchy to match production.
- [x] Repair square/Zalesak area-error measurement for lines, signed circles, and corners.
- [x] Review the combined implementation diff and run the broad regression suite.
  - focused release/diagnostic suite: `41 passed`
  - full repository suite: `182 passed`
  - final launcher dry plan: `970` runs / `24,250` cases
- [x] Make final releases collision-proof and self-contained.
  - reject tracked/untracked source changes and existing namespaces
  - archive read-only raw bundles under the release root with relative inventory paths
- [x] Commit the clean integrated source in reviewable checkpoints.
  - `ed3b785`: production reconstruction, metrics, diagnostics, launcher, and tests
  - `8b8f81c`: submission validation and figure-review tools
  - post-commit full suite: `182 passed`; source-only audit: `SOURCE CLEAN`
- [ ] Flip `launch_approved` only after author review.
- [ ] Launch `bash submission/run_final_static_sweep.sh`.
- [ ] Audit failures, missing facets, fallback rates, maxima, and case-level tails.
- [ ] Archive the resolved config, source state, source snapshot, manifests, CSVs, and logs.

## P0: Submission Validation

- [x] Conservation smoke over all five benchmarks.
  - Global relative error is `1.22e-13--5.26e-11`.
  - The guaranteed constraint is each independent-cell or merged-zone total, not
    every constituent base cell after merging.
- [x] Complete matched before/after C0 conservation and eligibility checks.
  - The pass closes eligible facet gaps but introduces a coarse-Zalesak conservation tail.
  - `6/25` `N=64` cases exceed `1e-6`; worst global relative error is `3.24e-3`.
- [x] Guard C0 endpoint adjustment/refitting by direct area evaluation and rerun the
  identical 100-case validation.
  - no case exceeds `1e-6`; worst global relative error is `1.44e-10`
  - infeasible adjustments retain the conservative pre-C0 facet, so exact C0 is conditional
- [x] Implement the exact shared-mesh-vertex conflict diagnostic.
- [x] Pass representative/fallback topology smoke with zero conflicts.
- [x] Finish the 500-case topology coverage/conflict table at `w=0.1`.
  - Coverage is `99.994%`, but 22 resolved-path conflicts occur in 18 cases.
  - Squares/circles are clean; ellipses have 5 conflicts and Zalesak has 17.
- [x] Diagnose all 22 conflicts before freezing the topology claim or source commit.
  - exact seeded meshes and independent classifiers confirm `22/22` genuine conflicts
  - incidence is `22/94,998` evaluated vertices (`0.0232%`) in `18/500` cases
  - none involves PLIC fallback, a corner facet at the vertex, rounded mesh data, or ambiguity
- [ ] Replace exact topological-consistency guarantees with a blue empirical
  near-consistency result and limitation; do not add a late acceptance heuristic.
- [x] Fit empirical ellipse convergence orders.
  - Hausdorff: approximately `1.60--1.66`.
  - Tangent error: approximately `1.48--1.61`.
  - Facet gap: approximately `2.93--3.13`.
- [x] Complete the matched topology/merging ablation on 250 Zalesak cases.
  - Topology+merging improves median Hausdorff and gap at every tested resolution.
  - It removes two independent-cell `H>1` cases and nearly eliminates LVIRA use.
- [x] Diagnose a high-perturbation Zalesak failure.
  - A locally admissible but nonlocal support intersection creates a false corner.

## P0: Manuscript Accuracy

All substantive additions or replacements remain blue until collaborator approval.

- [ ] Review the isolated blue journal-style build at `e57a8b6`, then rebase and
  integrate it over the collaborator's current Overleaf edits.
- [ ] State conservation as independent-cell or merged-zone volume matching.
- [ ] Replace the broad third-order claim with metric-specific empirical orders.
- [x] Distinguish coherent topology-stage orientation from facet-induced shared-vertex
  agreement; report the measured `22/94,998` conflict incidence and remove exact guarantees.
- [ ] Use one consolidated fallback/eligibility discussion instead of repeating it.
- [ ] Add the topology/merging ablation and the Zalesak limitation example compactly.
- [x] Qualify C0 as optional and limited to eligible oriented line/arc joins; explain
  that infeasible conservative refits are rejected, so exact continuity is conditional.
- [ ] State that the study is static and demonstrated on Cartesian/perturbed-Cartesian
  meshes; keep unstructured polygonal and three-dimensional validation as future work.
- [ ] Fix five rendered `Appendix Appendix` references and remove live source TODOs.

## P0: Figures

- [x] Inventory all 26 active manuscript PDFs.
- [x] Confirm the July main quantitative and paired qualitative candidates are vector-only.
- [ ] Regenerate and promote figures from the completed, audited final sweep.
  - use July only for comparison/recovery; do not promote March/May numerical assets
- [ ] Approve representative cases and clean versus endpoint variants.
- [ ] Regenerate all-method and reorganized resolution/supplement panels from the final
  result bundle rather than reusing historical PNG data.
- [ ] Regenerate C0 panels from the validated conservation-guarded implementation.
- [ ] Regenerate the staged and PLIC method figures from the clean final commit.
- [ ] Verify every submitted PDF has no raster image objects and no layout regressions.

March and May assets are historical comparison baselines, not the target submission set.

## P0: Bibliography And Declarations

- [ ] Replace four Semantic Scholar placeholder records with publisher/DOI metadata.
- [ ] Remove blanket `\nocite` lists or cite the intended works in context.
- [ ] Correct the unsupported thesis entry type, DOI URL fields, and identified typos.
- [ ] Add data/code availability text tied to the final clean tag and archived result data.
- [ ] Add competing-interest and corresponding-author information.
- [ ] Compile with zero undefined citations/references and clear the remaining material
  bibliography/layout warnings.

## P1: Condense The Primary Paper

- [ ] Review the writing worker's exact primary-paper/supplement split.
- [ ] Move the five full all-method panels, resolution grids, extended C0 evidence,
  endpoint diagnostics, and adverse-tail details to supplement.
- [ ] Keep the staged method, compact quantitative evidence, and strongest square/Zalesak
  reconstructions in the primary paper.
- [ ] Consider combining or moving repetitive line/circle/ellipse representative pages.
- [ ] Compile both artifacts and measure the actual page savings; target 6--10 pages.

## Final Approval Gates

1. Accept the algorithm and metric patches.
2. Accept the blue empirical near-consistency qualification for the 22 genuine shared-vertex conflicts.
3. Accept the blue conditional-C0 wording and regenerated guarded-C0 figures.
4. Accept the blue manuscript diff and supplement split.
5. Freeze and run one clean full sweep.
6. Approve the final-sweep figure packet.
7. Promote figures, remove accepted revision coloring, compile, and package submission files.
