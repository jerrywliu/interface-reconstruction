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
`$FINAL_ROOT/submission_config.resolved.json`, the SHA-256 of the exact
`$FINAL_ROOT/SHA256SUMS`, the SHA-256 of
`submission/final_figure_candidates.json`, the exact 41-candidate count
contract, and orchestration schema version 4. The record must use status
`approved`, `revoked=false`, a reviewer identity of 3--200 printable
characters, and a UTC timestamp formatted `YYYY-MM-DDTHH:MM:SSZ`; extra fields
are rejected.

The reviewer supplies two values through the submission sign-off channel:

```bash
FINAL_FIGURE_APPROVAL_RECORD=/private/review/final_figure_approval.json
FINAL_FIGURE_APPROVAL_SHA256=<separately-communicated-64-hex-digest>
```

Do not derive the expected digest from an unreviewed replacement record in the
same command. The wrapper requires the record to be outside the repository and
owned by the current user, with no group/world write permission.
Reviewer authentication and communication of this digest are intentionally
out of band; the wrapper proves exact content, not human identity.

## 3. Run The Single Wrapper

Choose a new destination. The wrapper reserves it exclusively and never
replaces an existing path.

```bash
FINAL_FIGURE_ROOT="results/submission/final_figures_$(date -u +%Y%m%d_%H%M%S)"
GENERATOR_COMMIT="$(git rev-parse HEAD)"
TRUSTED_PYTHON="/absolute/non-symlink/path/to/reviewed/python"

submission/run_final_figure_orchestrator \
  --python "$TRUSTED_PYTHON" \
  --repository "$PWD" \
  --release-root "$FINAL_ROOT" \
  --approved-generator-commit "$GENERATOR_COMMIT" \
  --approval-record "$FINAL_FIGURE_APPROVAL_RECORD" \
  --approval-record-sha256 "$FINAL_FIGURE_APPROVAL_SHA256" \
  --output-root "$FINAL_FIGURE_ROOT"
```

The wrapper performs, in order:

1. start a fresh isolated Python process with no caller Python path, user site,
   startup customization, or preloaded repository modules;
2. pin the live release identity and ledger, copy the complete ledger into a
   private read-only tree, and run the scientific audit plus checksum verification
   only on that immutable tree;
3. exact commit, index, tracked-byte, and external-approval attestation;
4. read-only source materialization from approved Git blobs, a private sealed
   detached Git view whose `HEAD` and index are fixed to the approved commit,
   per-read sealed config verification, and repeated source/config/Git-view
   re-attestation;
5. one private read-only allowlist copy and compact release-input view derived
   only from the audited complete snapshot;
6. main-text and all-method generation from that snapshot;
7. 30 fresh resolution runs plus quantitative/geometry evidence capture;
8. 165 fresh guarded-C0 runs plus exact 2,700-row metrics validation;
9. a fresh connected-component joint-C0 replay for ellipse case 9 and
   fail-closed representative generation for ellipse case 9 and Zalesak case 22;
10. deterministic PLIC and staged figures;
11. internal 41-PDF vector/preview/44-page-map acceptance with absolute attested Poppler tools; and
12. publication-root-relative path resolution for every wrapper-owned artifact,
    checksum-copy to a separate sealed publication tree, locked final rehash, and
    atomic no-replace publication.

The dedicated studies are real fresh runs, so this is substantially longer
than a plot refresh.

## 4. Review Outputs

```text
$FINAL_FIGURE_ROOT/
  candidates/figure_root/...
  candidates/c0_root/...
  provenance/final_figure_orchestration.json
  provenance/approved_candidate_allowlist.json
  provenance/external_approval_record.json
  provenance/trusted_runtime.json
  provenance/execution_config_authority.json
  provenance/release_input_snapshot/...
  provenance/resolution/.../run_manifests/       # 30
  provenance/resolution/.../inputs/              # metrics + geometry
  provenance/guarded_c0/run_manifests/           # 165
  provenance/guarded_c0/{ellipses,zalesak}/metrics.csv
  provenance/approved_c0_representatives/        # case 9/22 inputs and audits
  provenance/published_tree_sha256.json
  review/figure_candidate_review.pdf
  review/figure_candidate_source_map.json
  review/figure_candidate_source_map.csv
  review/figure_candidate_vector_qa.json
  review/previews/                                # 41 fresh 300-DPI PNGs
```

Only PDFs under `candidates/` are eligible for manuscript promotion.

The authoritative 2026-08-03 publication is
`final_figures_87c40309d16c_20260803_final`. Its completed 26-slot author
selection and exact published-candidate checksums are recorded in
`submission/approved_final_figures_20260803.csv`. Promotion completed at
Overleaf commits `1ea4d3e` (ellipse) and `b593f5d` (Zalesak), with both files in
standalone paper commit `d2d9548`; all four promoted copies passed vector QA and
match the published checksums.

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

Child provenance commands never read the live checkout's `HEAD`, refs, or
index. They use the private detached Git view over the immutable source, so
`source_commit` remains the approved commit even if a live branch advances
during a long run. `source_branch` is therefore intentionally empty, while
status and diff provenance remain available and must report a clean source.
