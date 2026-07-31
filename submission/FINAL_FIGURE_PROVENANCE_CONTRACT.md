# Final Figure Provenance Contract

The submission figure set is produced only by
`submission/final_figure_orchestrator.py`. The review builder in
`submission/accept_figure_candidates.py` is internal-only: it accepts a
process-local state created by the orchestrator and has no standalone
acceptance CLI. A JSON manifest that merely claims `status=completed` has no
authority.

## Two-Phase Approval

Generator/orchestrator changes are integrated and committed first. An
independent reviewer then approves that exact resulting commit, not a branch
SHA from before cherry-pick or merge. The external approval record must be
outside the repository, be a regular file that is not group/world writable,
be owned by the current user, and have a separately communicated SHA-256.

The record schema is:

```json
{
  "schema_version": 1,
  "record_type": "final_figure_generator_approval",
  "approved_generator_commit": "<full 40-hex commit>",
  "approved_generator_tree": "<Git tree object ID>",
  "scientific_release_commit": "<full 40-hex release commit>",
  "allowlist_sha256": "<SHA-256 of submission/final_figure_candidates.json>",
  "approved_by": "<reviewer identity>",
  "approved_at_utc": "<timestamp>"
}
```

The command-line commit, record fields, record digest, approved Git tree,
release source commit, and allowlist digest must all agree.

## Source Trust

Every Git command receives a scrubbed Git environment. Replacement objects are
disabled, caller `GIT_*` configuration is removed, and object facts are read
with `ls-tree` and `cat-file`. The wrapper verifies:

1. The approved and release commits exist and hash to their requested object IDs.
2. The approved commit descends from the scientific release commit.
3. `HEAD`, the index entries, executable modes, and every live tracked byte
   equal the approved commit.
4. The checkout is clean and has no `assume-unchanged`, `skip-worktree`, or
   nonstandard index flags.

Generation never imports the live checkout. The wrapper materializes a new,
read-only source tree from the approved commit blobs. Ignored files and bytecode
cannot enter that tree. Generator subprocesses use `-B`, disable bytecode and
user site packages, discard inherited Python/Git environment variables, and
use only this immutable tree as their module path. The execution configuration
is a physical copy from the same tree; no live-source symlink is used.

## Release Input Snapshot

The final release must first pass its complete audit and `SHA256SUMS` check. The
wrapper then snapshots every release byte the generators consume before any
figure generation:

- resolved configuration, completed sweep manifest, aggregate CSV, and checksum ledger;
- the complete raw run bundles required by all representative main-text panels.

Each file is opened without following a symlink, read once, checked for stable
inode/size/timestamps during the read, verified against `SHA256SUMS`, and copied
without replacement. The live ledger is checked again after the copy. The
snapshot is made read-only. Main-text and all-method generators consume only
the snapshotted CSV and physical raw-bundle copies, never a live CSV, symlink,
or release directory.

## Scientific Contracts

The fixed contracts are:

- Final release: 970 completed runs, 24,250 cases, exact method and grid sets,
  seed 0, 25 cases per setting, and the
  `LVIRA`/`pre_f8_corner`/`exact_linear_support_only` profile.
- Main text: all five experiments, exact methods and representative cases, and
  paired endpoint variants.
- Resolution appendix: exact cases `0/22/12/12/20`, designated methods,
  `N=16,32,64`, perturbations `0,0.1`, seed 0, and 30 newly completed runs. In
  addition to all run manifests, the exact per-case quantitative CSV,
  case-geometry JSONL, mesh, reconstructed VTP, facet metadata, and line truth
  geometry consumed by the panels are validated and snapshotted.
- Guarded C0 appendix: exact six/five resolution grids, five perturbations,
  three variants, seed 0, and 25 cases per setting. All 165 runs must be newly
  completed. The ellipse metrics CSV must contain exactly 90 settings x 20
  metric keys (1,800 rows); Zalesak must contain exactly 75 settings x 12 metric
  keys (900 rows). Missing, duplicate, extra, non-finite, partial, planned,
  collected, or plot-only evidence fails. Representative geometry inputs are
  also snapshotted.
- Deterministic PLIC and staged reconstruction parameters are pinned to the
  reviewed cases, cells, mesh settings, seed, and Zalesak dimensions.

Ordinary plotting and historical merge/finalize CLIs remain general-purpose
and do not establish submission provenance.

## Acceptance And Publication

The orchestrator's in-memory state covers exactly 38 allowlisted one-page PDFs.
Internal acceptance runs vector QA fail-closed, renders fresh 300-DPI previews,
verifies their dimensions, builds the indexed vector review PDF, measures its
41 pages, verifies the page map, and writes JSON/CSV source maps.

The requested destination is exclusively reserved before staging. Immediately
before publication, every candidate and provenance snapshot is rehashed. The
complete staged tree is checksummed and published with an atomic no-replace
directory rename. A concurrent destination wins untouched; the wrapper fails
and removes only its own staging and reservation.

## Invocation

After external approval of the exact integrated commit:

```bash
GENERATOR_COMMIT="$(git rev-parse HEAD)"

python submission/final_figure_orchestrator.py \
  --repository "$PWD" \
  --release-root "$FINAL_ROOT" \
  --approved-generator-commit "$GENERATOR_COMMIT" \
  --approval-record "$FINAL_FIGURE_APPROVAL_RECORD" \
  --approval-record-sha256 "$FINAL_FIGURE_APPROVAL_SHA256" \
  --output-root "$FINAL_FIGURE_ROOT"
```

`FINAL_FIGURE_ROOT` must not exist.
