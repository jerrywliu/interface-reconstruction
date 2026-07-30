# Submission Package Audit

Date: 2026-07-30

Scope: read-only audit of the active JCP manuscript at
`/Users/wei/Code/Interface/active/overleaf/interface-reconstruction-paper`.
No manuscript, core-code, or Overleaf files were changed.

## Prioritized Findings

### P0: Submission Statements Are Missing

The active manuscript has acknowledgments and funding text, but no explicit:

- data availability statement;
- code availability statement;
- declaration of competing interests; or
- corresponding-author email/marker in the title block.

Concrete locations:

- Add availability/declaration sections after `new_sections/acknowledgements.tex` and before `\appendix` at `interface-reconstruction.tex:87-91`, or use the exact Elsevier/JCP section placement required by the submission template.
- Add the corresponding-author marker and email near `interface-reconstruction.tex:54-59`.
- Do not claim a permanent archival release until the code commit, result bundle, and repository/DOI are frozen. The current public code remote is `https://github.com/jerrywliu/interface-reconstruction.git`, but the submission statement should cite a clean tagged release or archival DOI rather than the dirty working tree.

Suggested factual structure for the later blue manuscript edit:

1. Data: processed data and figure-generation inputs for the submitted result set, with permanent repository/DOI once deposited.
2. Code: reconstruction and experiment code at the exact clean commit/tag used for the final sweep.
3. Raw output: state whether the full per-cell/per-case artifacts are deposited or available on reasonable request if the permanent bundle is too large.

### P0: Five Rendered "Appendix Appendix" References

The compiled PDF contains five duplicated labels:

- `new_sections/topology_identification.tex:78`
- `new_sections/facet_fitting.tex:28`
- `new_sections/facet_fitting.tex:38`
- `new_sections/facet_fitting.tex:46`
- `new_sections/facet_fitting.tex:55`

Cause: the Elsevier appendix subsection number already contains the word `Appendix`, while `\Cref` adds the reference type a second time. Replace these five appendix `\Cref{...}` calls with `\ref{...}` (or centrally redefine the cleveref appendix label format). A plain `\ref` currently renders `Appendix A.x` correctly.

### P0: Numerical Claim Must Match The New Convergence Audit

`new_sections/introduction.tex:30` says the method achieves third-order spatial accuracy in the remaining mixed-feature regimes. The July ellipse audit instead finds approximately:

- Hausdorff order: `1.60-1.66` on finest-five fits;
- tangent-error order: `1.48-1.61` on finest-five fits;
- facet-gap order: `2.93-3.13` on finest-five fits.

Therefore the paper should reserve the third-order empirical statement for **facet gap**, unless another metric-specific derivation or result supports a broader claim. Also review `new_sections/abstract.tex:2` (currently says "targeting third-order spatial accuracy") and every other unqualified third-order statement after the final sweep.

### P1: Scope Claims Need Static/Structured Qualification

The implementation is formulated for polygonal cells, but all reported numerical evidence is static and uses Cartesian or perturbed-Cartesian meshes. The following statements need precise separation between algorithmic scope and demonstrated scope:

- `new_sections/topology_identification.tex:4`: "naturally extended to general polygonal meshes and to higher dimensions" is prospective and unsupported; qualify as future work or remove the higher-dimensional claim.
- `new_sections/problem_setup.tex:103`: the unconditional checkmark for "Unstructured polygonal grid" can read as an experimentally validated capability. Mark it as algorithmic support only, add a qualifier, or remove it until an unstructured benchmark is included.
- `new_sections/problem_setup.tex:124` and `new_sections/conclusion.tex:4`: "on polygonal meshes" is acceptable as a method formulation only if nearby text explicitly states that the present experiments are restricted to Cartesian and perturbed-Cartesian meshes.
- `new_sections/introduction.tex:9` and `:27-30`: calibrate the novelty and topology/continuity language to the resolved path, optional continuity pass, and unresolved-fallback exception already described elsewhere.
- `new_sections/method_overview.tex:24`: do not claim every module is independently ablated until the normalized independent-cell versus topology-and-merging experiment is complete and included.

Recommended high-level scope sentence: the method is formulated for two-dimensional polygonal cells, while the present numerical study is limited to static Cartesian and perturbed-Cartesian benchmarks; advection, fully unstructured validation, and three-dimensional extension remain future work.

### P1: Revision Markup Inventory

There are **38 active `\color{blue}` starts across 14 files**. These are intentional review diffs and must remain blue until collaborator approval; all blue markup should be removed or accepted before final submission.

| File | Blue starts | Lines |
|---|---:|---|
| `new_sections/abstract.tex` | 1 | 2 |
| `new_sections/introduction.tex` | 2 | 13, 23 |
| `new_sections/problem_setup.tex` | 4 | 47, 53, 88, 91 |
| `new_sections/method_overview.tex` | 3 | 18, 23, 35 |
| `new_sections/topology_identification.tex` | 3 | 77, 92, 99 |
| `new_sections/facet_fitting.tex` | 10 | 26, 35, 45, 52, 61, 66, 73, 75, 76, 78 |
| `new_sections/static_tests.tex` | 3 | 150, 159, 169 |
| `new_sections/conclusion.tex` | 1 | 4 |
| `new_sections/appendix/algorithms.tex` | 2 | 3, 25 |
| `new_sections/appendix/algorithms/plic_baselines.tex` | 1 | 4 |
| `new_sections/appendix/algorithms/linear_facets.tex` | 2 | 4, 19 |
| `new_sections/appendix/algorithms/circular_facets.tex` | 3 | 4, 19, 35 |
| `new_sections/appendix/algorithms/corner_facets.tex` | 2 | 4, 19 |
| `new_sections/appendix/static_benchmarks/qualitative_examples.tex` | 1 | 27 |

There are also four live source TODO comments that do not render but should be resolved or removed before packaging:

- `new_sections/problem_setup.tex:58`
- `new_sections/method_overview.tex:95-96`
- `new_sections/static_tests.tex:163`

### P1: Primary Paper Versus Supplement Split

Current build: **47 pages**.

- Main text through conclusion: pages 1-28.
- Algorithm appendix: approximately pages 28-33.
- Static benchmark appendix: pages 34-45.
- References: pages 46-47.
- Active figure environments: 15 in the main text and 16 in the appendices.
- Section 6 alone occupies roughly pages 15-27 and uses 10 figures: one quantitative and one representative figure for each of five benchmark families.
- The appendix contains five full all-method panels, three grouped resolution-study figures, and four continuity-study figures, in addition to four algorithm schematics.

Assessment: the current artifact is an all-in-one 47-page paper, not a true primary-paper/supplement package. The scientific content is useful, but the main paper repeats a benchmark-by-benchmark rhythm while the appendix repeats much of the quantitative story with denser all-method panels.

Recommended split for the writing/condensation pass:

1. Keep in the primary paper: the PLIC perfect-reconstruction figure, staged reconstruction figure, topology diagrams, one compact cross-benchmark quantitative summary, and the strongest square/Zalesak qualitative evidence.
2. Move to supplement: all five full all-method panels, all resolution strips, all $C^0$ studies, per-case endpoint diagnostics, adverse-tail diagnostics, and detailed algorithm area-construction schematics not needed to follow the method.
3. Consider combining or moving the line/circle/ellipse representative reconstructions; they currently consume three full pages while making closely related smooth/linear points.
4. Preserve at least one adverse or limitation-facing Zalesak example in the primary or supplement rather than showing only favorable cases.

A realistic first condensation target is 6-10 pages: 3-5 pages from Section 6 by consolidating representative/metric figures, 2-4 pages by moving detailed Appendix A schematics, and about 1 page by removing the uncited bibliography dump. Exact savings should be measured after the writing agent's proposed split is compiled.

### P2: $C^0$ Notation Is Textually Consistent, But Claims Need One Qualification

Visible prose consistently uses `$C^0$`. The `C0` strings in `\texorpdfstring{...}{C0}` are PDF-bookmark fallbacks and are appropriate; file names containing `c0` are not visible notation. No active `\mathcal{C}^0` usage was found.

One semantic inconsistency remains:

- `new_sections/method_overview.tex:22` says the final pass removes residual gaps without changing reconstructed volumes, which sounds unconditional.
- `new_sections/method_overview.tex:23` and `new_sections/facet_fitting.tex:73-78` correctly restrict the guarantee to eligible oriented line/arc joins and exclude corner/fallback cells.

Qualify line 22 to match the later precise wording. In the comparison table at `new_sections/problem_setup.tex:96-103`, annotate the method's $C^0$ checkmark as optional/conditional rather than relying only on the preceding paragraph.

### P2: Build And Layout Warnings

The audit build succeeds with no undefined citations or references. Remaining warnings:

- overfull boxes of approximately `1.9 pt`, `2.7 pt`, `4.8 pt`, and `0.65 pt`;
- six small-caps italic font substitutions in Algorithm 1;
- one `!h` float changed to `!ht`;
- one BibTeX warning for `Pilliod1992` (covered in the bibliography audit).

The most actionable layout items are the methods comparison table at `new_sections/problem_setup.tex:95-107` and the PLIC appendix paragraph at `new_sections/appendix/algorithms/plic_baselines.tex:17-18`. These are minor, but should be cleared in the final camera-ready build.

## Submission Checklist

- [ ] Freeze clean code commit/tag and exact result bundle.
- [ ] Promote reviewed July figures or newer validated submission-pass figures.
- [ ] Resolve all blue manuscript diffs with collaborator approval.
- [ ] Fix the five duplicated appendix references.
- [ ] Reconcile third-order language with metric-specific convergence results.
- [ ] Add data/code availability, competing-interest, and corresponding-author information.
- [ ] Replace Semantic Scholar bibliography placeholders and remove blanket `\nocite` lists.
- [ ] Decide primary-paper versus supplement split and compile both artifacts.
- [ ] Remove live TODO comments and final revision coloring.
- [ ] Run final clean compile with zero undefined references/citations and review remaining layout warnings.
