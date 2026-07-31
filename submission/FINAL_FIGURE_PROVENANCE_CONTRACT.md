# Final Figure Provenance Contract

The submission figure set is produced only through
`submission/run_final_figure_orchestrator`, which starts the internal
`submission/final_figure_orchestrator.py` entry point in an isolated process.
That entry point refuses ordinary Python import before loading any project
module. Reservation, generator execution, operational acceptance, and atomic
publication exist only behind this script boundary. The importable
`submission/accept_figure_candidates.py` module contains non-publishing review
utilities only: it exposes no acceptance authority, operational acceptance
routine, or standalone acceptance CLI. A JSON manifest that merely claims
`status=completed` has no authority.

## Two-Phase Approval

Generator/orchestrator changes are integrated and committed first. An
independent reviewer then approves that exact resulting commit, not a branch
SHA from before cherry-pick or merge. The external approval record must be
outside the repository, be a regular file that is not group/world writable,
be owned by the current user, and have a separately communicated SHA-256.

The record schema is:

```json
{
  "schema_version": 2,
  "record_type": "final_figure_orchestration_approval",
  "approval_status": "approved",
  "revoked": false,
  "approved_generator_commit": "<full 40-hex commit>",
  "approved_generator_tree": "<Git tree object ID>",
  "scientific_release_commit": "<full 40-hex release commit>",
  "release_sha256sums_sha256": "<SHA-256 of final release SHA256SUMS>",
  "allowlist_sha256": "<SHA-256 of submission/final_figure_candidates.json>",
  "candidate_contract": {
    "candidate_pdfs": 38,
    "unpaired_candidates": 14,
    "paired_slots": 12,
    "paired_candidates": 24
  },
  "orchestrator_schema_version": 4,
  "approved_by": "<reviewer identity>",
  "approved_at_utc": "YYYY-MM-DDTHH:MM:SSZ"
}
```

The schema is closed: missing or unknown fields, any status other than
`approved`, `revoked=true`, a malformed reviewer identity or UTC timestamp, or
any mismatch fails. The command-line commit, record digest, approved Git tree,
release source commit, exact release-ledger digest, private allowlist digest,
candidate counts, and orchestration schema must all agree. File ownership and
the separately supplied digest protect local substitution; authenticating the
reviewer's identity and communicating that digest remain out-of-band review
responsibilities.

## Source Trust

The supported CLI starts only through
`submission/run_final_figure_orchestrator`. This fixed `/bin/sh` launcher
requires an explicit absolute, non-symlink Python executable, clears Python
startup and module-path variables, constructs a minimal environment, and
executes a fresh interpreter with `-I` while passing an inherited descriptor
pinned to the launcher file. The Python entry point refuses a missing or
replaced launcher descriptor, a
non-isolated process, user/site customization, or any preloaded `submission`
or `experiments` module, and it raises `ImportError` before exposing production
authority when loaded as a module. Direct
`python .../final_figure_orchestrator.py` execution is deliberately
unsupported. The boundary contains one orchestration entry point and no
injectable command, acceptance, publication, or race hooks. Importable helpers
can validate and freeze bytes but cannot reserve a destination, execute a
generator, accept candidates, or publish a tree. An isolated subprocess probe
checks that all four capabilities remain unreachable through ordinary import.

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
use only this immutable tree as their module path. Configuration YAML is read
directly from that attested tree. A private authority manifest records every
config path, size, and SHA-256; its own digest is fixed in every generator
environment. Every nested resolution/C0 process verifies the authority and
hashes each YAML before parsing those same bytes. The full config/source
inventory is also re-attested before and after every top-level generator
command. A mutation therefore cannot produce an accepted run.

The candidate allowlist is copied once from the approved materialized source
into private staging and made read-only. Approval, generation staging,
acceptance, source maps, and final rehashes use only that private copy.

## Trusted Figure Runtime

Poppler is never selected through caller `PATH`. The wrapper resolves each
required executable from fixed system/Homebrew locations, follows it to an
absolute regular executable, rejects unsafe ownership or write modes, and
records its resolved path, SHA-256, and version. The executable is rehashed
before every invocation. Poppler receives a minimal subprocess environment.

All generator subprocesses receive a fresh private `HOME`, XDG config/cache,
`MPLCONFIGDIR`, Fontconfig file/path, TeX state, and temporary directory. The
wrapper pins the noninteractive Agg backend, white figure/axes/save backgrounds,
embedded TrueType PDF fonts, no TeX rendering, UTC, C locale, deterministic
Python hashing, and `SOURCE_DATE_EPOCH`. It copies only the selected DejaVu/Vera
fonts into the private font directory and records font family/style/version and
SHA-256, Python/package versions, tool versions, and configuration digests in
`provenance/trusted_runtime.json`.

## Release Input Snapshot

Before any scientific audit, the wrapper pins the live release-root
device/inode, exact `SHA256SUMS` bytes and SHA-256, exact resolved-config
SHA-256, and scientific source commit. It then checksum-copies every file in
that ledger into a private complete release snapshot and makes the tree
read-only. Both `audit_final_release` and complete-inventory checksum
verification run only on this immutable snapshot, never on the live release.

After that audit passes, the wrapper derives the smaller plotting view solely
from the audited snapshot:

- resolved configuration, completed sweep manifest, aggregate CSV, and checksum ledger;
- the complete raw run bundles required by all representative main-text panels.

Each full-release and plotting-view file is opened without following a symlink,
checked for stable inode/size/timestamps during the read, verified against the
pinned ledger, and copied without replacement. The live root is re-attested
after the complete copy; later live replacement cannot affect the private audit
or any consumed bytes. The compact view is also read-only and must retain the
audited ledger bytes, resolved-config digest, and source commit exactly.
Main-text and all-method generators consume only the snapshotted CSV and
physical raw-bundle copies, never a live CSV, symlink, or release directory.

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
  case-geometry JSONL, mesh, reconstructed VTP, and facet metadata are validated
  and snapshotted. The square and circle panels consume the VTP truth geometry
  from their `N=16`, Cartesian base runs, so those exact two VTP files are also
  snapshotted and hashed. The line panel derives its truth segment analytically
  from the case definition and mesh bounds; `true_line*.vtp` is not consumed and
  is deliberately not claimed as figure input.
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

The orchestrator's boundary-local immutable state covers exactly 38 allowlisted
one-page PDFs; it uses no mutable authority flag. Operational acceptance calls
the fixed PDF inspector, page inspector, preview renderer, review builder, and
page-map verifier with the attested runtime. It runs vector QA fail-closed,
renders fresh 300-DPI previews, verifies their dimensions, builds the indexed
vector review PDF, measures its 41 pages, verifies the page map, and writes
JSON/CSV source maps. Every
wrapper-owned PDF, preview, QA, source-map, candidate, snapshot, and provenance
artifact path is a publication-root-relative POSIX path. Before publication,
each recorded path must resolve inside the frozen publication tree and exist;
temporary staging paths are never accepted as logical artifact paths.

The requested destination is exclusively reserved with an open advisory lock
before staging. After acceptance, the complete accepted inventory is captured
and rechecked, then checksum-copied into a separate private publication tree.
The wrapper writes a ledger over every final file except the ledger itself,
makes files `0400` and directories `0500`, and rehashes the ledger, exact
inventory, modes, and every artifact while still holding the lock immediately
before an atomic no-replace directory rename. Any late mutation fails and all
owned temporary trees are removed. A concurrent destination wins untouched.
The transaction-level adversarial test drives this fixed path with 38 real
embedded-font vector PDFs and no production hooks, covering acceptance,
reservation, freeze/rehash, successful no-replace rename, competing-destination
preservation, and owned cleanup.

## Invocation

After external approval of the exact integrated commit:

```bash
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

`FINAL_FIGURE_ROOT` must not exist.
