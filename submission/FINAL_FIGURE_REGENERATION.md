# Final Figure Regeneration

Use this workflow only after the final static release reports `completed` and
passes `submission/audit_final_release.py` with its complete `SHA256SUMS`.

## 1. Integrate First

Merge or cherry-pick all reviewed generator/orchestrator changes. Commit the
result and leave the checkout clean. Do not approve a pre-integration branch
SHA: integration changes the commit ID.

```bash
git status --short
GENERATOR_COMMIT="$(git rev-parse HEAD)"
GENERATOR_TREE="$(git rev-parse HEAD^{tree})"
```

## 2. Obtain External Approval

An independent reviewer reviews `GENERATOR_COMMIT`, then creates the approval
record described in `FINAL_FIGURE_PROVENANCE_CONTRACT.md` outside this
repository. The reviewer records the exact commit/tree, the source commit in
`$FINAL_ROOT/submission_config.resolved.json`, and the SHA-256 of
`submission/final_figure_candidates.json`.

The reviewer supplies two values through the submission sign-off channel:

```bash
FINAL_FIGURE_APPROVAL_RECORD=/private/review/final_figure_approval.json
FINAL_FIGURE_APPROVAL_SHA256=<separately-communicated-64-hex-digest>
```

Do not derive the expected digest from an unreviewed replacement record in the
same command. The wrapper requires the record to be outside the repository and
owned by the current user, with no group/world write permission.

## 3. Run The Single Wrapper

Choose a new destination. The wrapper reserves it exclusively and never
replaces an existing path.

```bash
FINAL_FIGURE_ROOT="results/submission/final_figures_$(date -u +%Y%m%d_%H%M%S)"
GENERATOR_COMMIT="$(git rev-parse HEAD)"

python submission/final_figure_orchestrator.py \
  --repository "$PWD" \
  --release-root "$FINAL_ROOT" \
  --approved-generator-commit "$GENERATOR_COMMIT" \
  --approval-record "$FINAL_FIGURE_APPROVAL_RECORD" \
  --approval-record-sha256 "$FINAL_FIGURE_APPROVAL_SHA256" \
  --output-root "$FINAL_FIGURE_ROOT"
```

The wrapper performs, in order:

1. full final-release audit and checksum verification;
2. exact commit, index, tracked-byte, and external-approval attestation;
3. read-only source materialization from approved Git blobs;
4. immutable checksum-verified release-input snapshot;
5. main-text and all-method generation from that snapshot;
6. 30 fresh resolution runs plus quantitative/geometry evidence capture;
7. 165 fresh guarded-C0 runs plus exact 2,700-row metrics validation;
8. deterministic PLIC and staged figures;
9. internal 38-PDF vector/preview/page-map acceptance; and
10. final rehash plus atomic no-replace publication.

The dedicated studies are real fresh runs, so this is substantially longer
than a plot refresh.

## 4. Review Outputs

```text
$FINAL_FIGURE_ROOT/
  candidates/figure_root/...
  candidates/c0_root/...
  provenance/final_figure_orchestration.json
  provenance/external_approval_record.json
  provenance/release_input_snapshot/...
  provenance/resolution/.../run_manifests/       # 30
  provenance/resolution/.../inputs/              # metrics + geometry
  provenance/guarded_c0/run_manifests/           # 165
  provenance/guarded_c0/{ellipses,zalesak}/metrics.csv
  provenance/published_tree_sha256.json
  review/figure_candidate_review.pdf
  review/figure_candidate_source_map.json
  review/figure_candidate_source_map.csv
  review/figure_candidate_vector_qa.json
  review/previews/                                # 38 fresh 300-DPI PNGs
```

Only PDFs under `candidates/` are eligible for manuscript promotion.

## Compatibility

Historical and ordinary plotting commands keep their existing interfaces. For
example:

```bash
python -m experiments.static.run_perturbed_sweeps \
  --plot_from_csv some_metrics.csv \
  --summary_dir some_plot_directory \
  --no-notify
```

Those outputs are not submission candidates unless regenerated and accepted
inside the final wrapper.
