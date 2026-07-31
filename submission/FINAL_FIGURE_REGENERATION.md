# Final Figure Regeneration

Use this workflow only after the final static release reports `completed` and
passes `submission/audit_final_release.py` with its complete `SHA256SUMS` ledger.

## 1. Review And Pin The Generator Commit

Commit all reviewed figure-orchestration and plotting changes. The worktree must
be clean, and the approved generator commit must descend from the scientific
release commit stored in `FINAL_ROOT/submission_config.resolved.json`.

```bash
git status --short
GENERATOR_COMMIT="$(git rev-parse HEAD)"
git merge-base --is-ancestor \
  "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"]["target_commit"])' \
    "$FINAL_ROOT/submission_config.resolved.json")" \
  "$GENERATOR_COMMIT"
```

Do not set `assume-unchanged` or `skip-worktree` on tracked paths. The wrapper
checks the actual bytes of every tracked path, not only `git status`.

## 2. Run The Single Submission Wrapper

Choose a new output path. The wrapper rejects an existing destination.

```bash
FINAL_FIGURE_ROOT="results/submission/final_figures_$(date -u +%Y%m%d_%H%M%S)"

python submission/final_figure_orchestrator.py \
  --repository "$PWD" \
  --release-root "$FINAL_ROOT" \
  --approved-generator-commit "$GENERATOR_COMMIT" \
  --output-root "$FINAL_FIGURE_ROOT"
```

The wrapper runs, in order:

1. final-release audit and full checksum verification;
2. generator checkout byte attestation;
3. main-text regeneration from the final release;
4. all-method regeneration, staging only five allowlisted PDFs;
5. 30 dedicated resolution companion runs;
6. 165 dedicated guarded-C0 runs;
7. the two deterministic figures;
8. 38-PDF inventory, vector, page-count, preview, and page-map QA; and
9. final rehash plus atomic publication.

The dedicated studies are real fresh runs. This command is therefore much
longer than an ordinary plot refresh.

## 3. Review Outputs

```text
$FINAL_FIGURE_ROOT/
  candidates/figure_root/...
  candidates/c0_root/...
  provenance/final_figure_orchestration.json
  provenance/published_tree_sha256.json
  provenance/release/...
  provenance/resolution/.../run_manifests/       # 30
  provenance/guarded_c0/run_manifests/           # 165
  review/figure_candidate_review.pdf
  review/figure_candidate_source_map.json
  review/figure_candidate_source_map.csv
  review/figure_candidate_vector_qa.json
  review/previews/                                # 38 fresh 300-DPI PNGs
```

Open `review/figure_candidate_review.pdf` for the indexed vector review. The
candidate PDFs under `candidates/` are the only files eligible for manuscript
promotion.

## Compatibility

Commands such as the following remain ordinary plotting operations and do not
require submission-only flags:

```bash
python -m experiments.static.run_perturbed_sweeps \
  --plot_from_csv some_metrics.csv \
  --summary_dir some_plot_directory \
  --no-notify
```

The historical `merge_section6_with_lvira.py` and
`finalize_sharded_zalesak.py` callers continue to use this interface. Their
outputs are not final submission candidates unless regenerated and accepted by
the wrapper above.
