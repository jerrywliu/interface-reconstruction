# Submission Result-Set Audit

Status: **candidate, not frozen**

## Decision

The completed, audited final sweep is the primary source for submission metrics,
geometry, and all-method comparisons. July 2026 remains comparison/recovery
provenance only, while March and May assets remain layout references.

The current manuscript is not yet a single-version result set. It contains 26 active
PDF graphics: 24 vector conversions of historically approved March/May plots and two
July deterministic method figures. The numerical panels therefore must be replaced
or explicitly revalidated before the result set is called frozen.

The July review bundle contains vector-only comparison candidates for all five main
quantitative panels and paired qualitative views. Use them to diagnose visual or
numerical differences, not as the promotion source. Regenerate main, all-method,
resolution, and supplement panels from the final result bundle or final-commit
companion runs. Regenerate C0 panels from the repaired conservation-guarded
implementation.

## Comparison And Recovery Sources

- July affected-method sweep:
  `results/static/static_paper_simplified_default_20260717_212413/`
- July paired figure review:
  `results/static/static_paper_simplified_default_20260717_212413/figure_review/`
- July source snapshot SHA-256:
  `4975ad99db05ac6d606de4f6f17f74242df78b993d4070a0e51c7ce0a5c1e7fd`
- July result: 300/300 affected-method runs, 7,500 cases, zero controller failures.

The July sweep was generated from a dirty snapshot based on commit `d02dd47`; the
snapshot makes it reproducible, but it is not the desired clean-commit submission
freeze. It also contains only affected method rows. Its review CSV combines those
rows with older baseline rows.

## Target Final Sweep

The machine-readable target is `submission/submission_config.json`:

- 970 runs and 24,250 deterministic cases;
- all paper methods over five benchmarks;
- production default `pre_f8_corner + exact_linear_support_only + LVIRA`;
- normalized independent-cell circular comparison with the same fallback hierarchy;
- exact case geometry and case/cell/provenance diagnostics saved with the results.

The validation patches are integrated at `ed3b785` and `8b8f81c`; the post-commit
repository suite passes all `182` tests, and the source-only audit is clean. The
configured target commit remains intentionally blank until launch
approval. The final launcher records the approved clean `HEAD` in a resolved
configuration inside the result bundle; this avoids the impossible requirement that
a committed file contain its own commit hash.

## Promotion Gates

1. Review the new conservation, topology, convergence, merging, and Zalesak evidence.
   Exact replay confirms the 22 resolved-path conflicts are genuine facet-induced
   phase-label disagreements (`22/94,998` evaluated vertices in `18/500` cases).
   Replace the exact guarantee with an empirical near-consistency statement and
   explicit limitation; do not add a late algorithm heuristic for submission.
   The matched C0 validation found and repaired a coarse-Zalesak conservation regression.
   The guard retains a conservative pre-C0 facet when the adjusted endpoints cannot be
   refit within tolerance, so the optional continuity result must be stated conditionally.
2. Use the repaired geometry-faithful area metric; do not reuse historical square
   or Zalesak `area_error` columns in final tables or figures.
3. Review the integrated implementation checkpoints and approve the final launch commit.
4. Run the complete all-method sweep from that commit.
5. Regenerate July-style main-text, representative, resolution, and all-method figures
   from the final saved results; regenerate C0 from the validated guarded implementation.
6. Approve representative cases and the endpoint display variant.
7. Replace active manuscript PDFs using `submission/figure_provenance.csv`.
8. Compile the paper, verify every active PDF is vector-only, and archive the exact
   configuration, source state, run manifests, CSVs, and figure manifest.

Run `python submission/check_submission_freeze.py` for the current structural audit.
It is expected to report `NOT FROZEN` until the final commit and figure promotion are
complete.

After the accepted patches are committed, launch the complete sweep with:

```bash
bash submission/run_final_static_sweep.sh
```

The launcher refuses tracked or untracked source/config/test changes, resolves the
exact source commit, assigns a collision-proof namespace, requires the explicit
`launch_approved` gate, checks the configuration, and then starts the explicit
`970`-run controller sweep. Each scientific run bundle is copied read-only into the
release directory so the final result set is self-contained. Exact meshes, facets,
metrics, case geometry, and provenance are retained; disposable per-case raster
previews are excluded because final figures are replayed as vector PDFs. After the
archive is verified, only that run's temporary namespaced `plots/` directory is
removed.
