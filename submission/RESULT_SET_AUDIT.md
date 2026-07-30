# Submission Result-Set Audit

Status: **candidate, not frozen**

## Decision

The submission figures should be promoted from the July 2026 review bundle wherever
they pass scientific and visual review. July 2026 is the oldest eligible final
provenance date. New figures produced by the submission validation or final sweep
supersede July candidates whenever data, algorithms, or comparison sets changed.
March and May assets remain useful only as comparison baselines.

The current manuscript is not yet a single-version result set. It contains 26 active
PDF graphics: 24 vector conversions of historically approved March/May plots and two
July deterministic method figures. The numerical panels therefore must be replaced
or explicitly revalidated before the result set is called frozen.

The July review bundle already contains vector-only PDF candidates for all five main
quantitative panels and for the paired main/appendix qualitative views. Direct main-
text promotion is therefore available after case/style approval. The all-method and
  any reorganized resolution/supplement panels should be regenerated from the final
  result bundle instead of converting the July PNG exports. C0 panels should be
  regenerated from the repaired conservation-guarded implementation.

## Candidate Sources

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

The final source commit remains intentionally blank until the validation patches are
integrated, tests pass, and the working tree is committed cleanly. The final launcher
records that clean `HEAD` in a resolved configuration inside the result bundle; this
avoids the impossible requirement that a committed file contain its own commit hash.

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
3. Integrate the accepted implementation patches and freeze one clean code commit.
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
`970`-run controller sweep. Each raw run bundle is copied read-only into the release
directory so the final result set is self-contained.
