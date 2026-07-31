# Deterministic Submission Packaging

`package_submission.py` builds the compact archival package only after the final
scientific release and manuscript figures have passed their separate approval gates.
It does not modify the release, the paper source, or Overleaf.

## Required Inputs

1. A completed `970`-run / `24,250`-case release root.
2. A successful programmatic audit by `audit_final_release.py`.
3. A complete and valid release-wide `SHA256SUMS`.
4. A clean generator Git worktree pinned to the exact full commit that produced
   the approved figure packet.
5. An explicit documentation commit, descending from the generator commit, that
   contains the reviewed `docs/PAPER_EXPERIMENT_MAP.md`.
6. The sealed, read-only final-figure publication root. It must contain the
   canonical orchestration manifest, JSON/CSV candidate source maps, private
   allowlist and external-approval snapshots, 38 candidate PDFs, and full
   published-tree checksum ledger.
7. A clean paper Git worktree pinned to an explicit full commit. Its root must
   contain `interface-reconstruction-paper/`.
8. A 26-row approval manifest selecting exactly one published candidate for each
   manuscript figure slot.
9. A non-placeholder DOI or URL for the complete raw-data deposit.
10. A `sha256:<digest>` identifier for the exact `SHA256SUMS` deposited with the
   complete release.
11. Either a locally downloaded/fetched copy of the deposited `SHA256SUMS`, or an
   explicit acknowledgment that remote deposit contents remain a manual gate.
12. A pre-existing output parent owned by the current user, represented by a real
   directory, and not writable by group or other users.

The packager reruns the full release audit and verifies every entry in the release
checksum manifest. A stale audit claim, incomplete sweep, failed run, changed raw
file, missing checksum, dirty or mismatched paper worktree, failed manuscript
compile, rasterized figure, unembedded figure font, unapproved
`\includegraphics` target, or mismatched deposit-manifest digest stops packaging.
The packager also verifies the complete sealed final-figure tree and requires the
orchestration manifest, both source maps, allowlist, selected candidate PDFs, and
approval CSV to agree on candidate ID, slot, variant, path, and SHA-256. Those
records must independently agree with the pinned generator commit/tree, audited
scientific commit, and exact release-`SHA256SUMS` digest. Any mismatch fails closed.

Every release authority payload is checked again during staging using the digest
frozen in the release `SHA256SUMS`. Public presentation copies are then
deterministically privacy-sanitized as described below. This closes the interval
between planning and materialization; mutating either a payload, final-figure
publication, or release manifest after planning aborts before final publication.

Seal and verify a private complete-release snapshot after all final release
artifacts have been installed. The source release is never modified:

```sh
install -d -m 700 "$HOME/interface-release-seals"
SEALED_RELEASE="$HOME/interface-release-seals/$(basename "$RELEASE").sealed"
test ! -e "$SEALED_RELEASE"
python submission/audit_final_release.py "$RELEASE" \
  --write-sha256-manifest \
  --sealed-release-output "$SEALED_RELEASE" \
  --verify-sha256-manifest
```

Use `"$SEALED_RELEASE"` for the packager's `--release-root` argument.

Create a separate clean paper worktree at the exact commit selected for submission.
Do not point the packager at the active Overleaf checkout or at the inner manuscript
directory:

```sh
export PAPER_REPO=/path/to/overleaf
export PAPER_COMMIT="$(git -C "$PAPER_REPO" rev-parse HEAD)"
export PAPER_WORKTREE=/tmp/interface-reconstruction-paper-submission

git -C "$PAPER_REPO" worktree add --detach "$PAPER_WORKTREE" "$PAPER_COMMIT"
export PAPER_WORKTREE="$(cd "$PAPER_WORKTREE" && pwd -P)"
test "$(git -C "$PAPER_WORKTREE" rev-parse --show-toplevel)" = "$PAPER_WORKTREE"
test -z "$(git -C "$PAPER_WORKTREE" status --porcelain --untracked-files=all)"
test -f "$PAPER_WORKTREE/interface-reconstruction-paper/interface-reconstruction.tex"
```

The packager requires the full 40-character commit and confirms `HEAD` exactly.
It enumerates the committed tree with `git ls-tree` and materializes source and
figure bytes with `git cat-file` using pinned object IDs. Worktree cleanliness is
still required as an operator gate, but worktree bytes are never packaged. Thus
`assume-unchanged`, `skip-worktree`, and worktree races cannot substitute different
manuscript content. Approved figures must be tracked at that commit. Tracked files
under `.git/`, `.hg/`, `.svn/`, `__pycache__/`, `build/`, `dist/`, or `output/`
are excluded from source discovery even when they exist in the pinned commit.

Apply the same clean, detached-worktree discipline to the code repository at the
exact reviewed generator commit:

```sh
export GENERATOR_REPO=/path/to/interface-reconstruction
export GENERATOR_COMMIT=<reviewed-40-character-commit>
export GENERATOR_WORKTREE=/tmp/interface-reconstruction-generator-submission
git -C "$GENERATOR_REPO" worktree add --detach "$GENERATOR_WORKTREE" "$GENERATOR_COMMIT"
test -z "$(git -C "$GENERATOR_WORKTREE" status --porcelain --untracked-files=all)"
```

The packager materializes the generator archive from the approved generator
commit. It materializes the experiment map from a separate explicit documentation
commit so documentation-only repairs do not falsely appear to have produced the
older figure packet. The documentation commit must descend from the generator
commit; its commit, tree, map blob, original digest, and public digest are recorded.
Worktree bytes cannot substitute either plotting or documentation content.

```sh
export DOCUMENTATION_COMMIT=<reviewed-40-character-documentation-commit>
git -C "$GENERATOR_REPO" merge-base --is-ancestor \
  "$GENERATOR_COMMIT" "$DOCUMENTATION_COMMIT"
```

Point `--final-figure-root` at the exact output published by
`run_final_figure_orchestrator`. The packager uses fixed canonical paths beneath
that root; callers cannot substitute a different manifest or source map:

```text
provenance/final_figure_orchestration.json
provenance/external_approval_record.json
provenance/approved_candidate_allowlist.json
provenance/published_tree_sha256.json
review/figure_candidate_source_map.json
review/figure_candidate_source_map.csv
```

The root and every contained file/directory must remain sealed read-only. The
packager verifies every non-ledger file against the publication ledger during both
planning and materialization. It validates the complete external approval schema,
digest, generator/scientific/release authorities, reviewer fields, approval status,
and nonrevocation state, then requires the orchestration's embedded approval fields
to match the snapshot exactly.

## Figure Approval Manifest

The approval manifest is CSV with these required columns:

```csv
candidate_id,slot_id,variant,paper_path,source_path,sha256,approval_status,approval_reference
lines_maintext_metrics,lines_maintext_metrics,unpaired,interface-reconstruction-paper/figs/cameraready/line_reconstruction_maintext_metrics.pdf,interface-reconstruction-paper/figs/cameraready/line_reconstruction_maintext_metrics.pdf,<64-hex-sha256>,approved,author-review-2026-08-01
```

- `candidate_id`, `slot_id`, and `variant` must exactly match one row of the
  published candidate source map.
- `paper_path` is the path used by the manuscript, relative to the outer paper
  worktree root.
- `source_path` is optional; when blank, it defaults to `paper_path`.
- `sha256` pins the exact approved bytes.
- `approval_status` must be exactly `approved` (case-insensitive).
- `approval_reference` records the review packet, decision sheet, or approval date.

The CSV must contain exactly 26 rows: all 14 unpaired slots and one selected
`clean` or `with_endpoints` candidate for each of the 12 paired slots. Every
approved paper PDF must be byte-identical to its published candidate. Only those
PDFs are copied. All active `\includegraphics` targets must be PDFs and must appear
exactly once in the approval manifest. Poppler's `pdfimages` and `pdffonts` are
used to require zero raster image objects and embedded fonts whenever fonts are
present.

## Raw-Data Binding

The deposition URL alone is not accepted. Compute the SHA-256 of the final release
checksum manifest and pass it as an explicit manifest identifier:

```sh
export RAW_DATA_MANIFEST_ID="sha256:$(shasum -a 256 "$SEALED_RELEASE/SHA256SUMS" | awk '{print $1}')"
```

The packager recomputes this digest from the already audited local release and
fails on any mismatch. The normalized deposit record in the package binds together
the deposition location, release directory name, `SHA256SUMS` filename, and exact
manifest digest. The external deposit must expose the same `SHA256SUMS` bytes.

Preferred verification supplies a copy downloaded or fetched from the deposit:

```sh
export DEPOSITED_MANIFEST=/path/to/downloaded/SHA256SUMS
cmp "$SEALED_RELEASE/SHA256SUMS" "$DEPOSITED_MANIFEST"
```

Pass it with `--deposited-release-manifest "$DEPOSITED_MANIFEST"`. The packager
requires exact byte equality, rechecks the staged evidence digest, and includes it
as `provenance/deposit/SHA256SUMS.downloaded`.

When the deposited manifest cannot yet be fetched, omit that file and pass
`--acknowledge-unverified-remote-deposit`. The package then records
`manual_acknowledgment_remote_contents_unverified`. This is an explicit manual
submission gate, not a remote verification claim. The packager performs no network
request and never claims that a DOI/URL was reachable or that its contents matched.
Values such as `10.xxxx/record`, `10.0000/...`, `10.1234/xxxx`, `/record`, `pending`, or
`placeholder` are rejected rather than accepted as deposit identifiers.

## Public-Package Privacy

Release metadata and historical source-audit notes may contain absolute user-home
paths or the sweep workstation hostname. The compact public package does not copy
those presentation bytes verbatim. It deterministically replaces user-home
prefixes with `<HOME>` and the hostname captured in `environment.json` with
`<HOSTNAME>` in textual release metadata, the figure orchestration and external
approval records, and text members of both code archives.

This does not weaken the scientific authority. For every sanitized payload,
`provenance/privacy_redactions.json` records the exact original authority SHA-256,
the public-byte SHA-256, format, and replacement count. The scientific snapshot
also records its explicit Git commit/tree and original release-archive digest; the
generator snapshot records its Git commit/tree and original deterministic archive
digest. The complete original scientific release remains checksum-bound in the
external deposit. A final recursive privacy audit, including both tar archives,
rejects any remaining captured hostname or `/Users/<name>`, `/home/<name>`, or
Windows user-home marker before package checksums are written.

## Dry Run

Dry run performs the expensive scientific audit, full checksum verification,
paper Git-state check, manuscript import check, vector-PDF inspection, deposit
binding, and a disposable manuscript compile, but writes nothing:

```sh
export PACKAGE_PARENT=/path/to/private-submission-output
install -d -m 700 "$PACKAGE_PARENT"
export PACKAGE="$PACKAGE_PARENT/interface-reconstruction-submission"
: "${FINAL_FIGURE_ROOT:?set the sealed approved figure publication root}"
: "${RAW_DATA_DOI_OR_URL:?set the actual deposited-release DOI or URL}"
```

The exact parent must exist before planning; the packager never creates it.

```sh
python submission/package_submission.py \
  --release-root "$SEALED_RELEASE" \
  --final-figure-root "$FINAL_FIGURE_ROOT" \
  --generator-worktree-root "$GENERATOR_WORKTREE" \
  --generator-commit "$GENERATOR_COMMIT" \
  --documentation-commit "$DOCUMENTATION_COMMIT" \
  --paper-worktree-root "$PAPER_WORKTREE" \
  --paper-commit "$PAPER_COMMIT" \
  --approved-figures-manifest "$APPROVAL_CSV" \
  --review-bundle "$REVIEW_PDF" \
  --raw-data-deposition "$RAW_DATA_DOI_OR_URL" \
  --raw-data-manifest-id "$RAW_DATA_MANIFEST_ID" \
  --deposited-release-manifest "$DEPOSITED_MANIFEST" \
  --output-dir "$PACKAGE" \
  --dry-run
```

Remove `--dry-run` to create both `$PACKAGE/` and a deterministic
`$PACKAGE.tar.gz`. Existing destinations are never overwritten. Use `--no-archive`
only when a staged directory without the outer archive is required.

An existing package is identified by regular top-level `INVENTORY.json` and
`SHA256SUMS` markers. Neither the output directory nor archive destination may be
inside such a package or contain one. This namespace check runs during planning,
before staging in a real build, and immediately before publication. A plan therefore
fails safely if a conflicting package appears after the plan was created; attempts
to build `bundle/nested`, wrap an existing `bundle`, or use an archive path that
contains a package do not modify the existing package.

A real build atomically reserves the output directory and, when requested, archive
destination with exclusive sidecar lock files. A concurrent invocation targeting
either exact path fails before staging. Locks are held through archive verification.
Archive construction uses a unique sibling temporary and exclusive hard-link
publication, so concurrent builds cannot share or overwrite a temporary or archive.
The deterministic archive and its extracted manuscript are fully verified before
either final destination is created.

Portable Python does not provide atomic no-replace directory publication on every
supported platform. The security boundary is therefore the pre-existing output
parent: it must remain the same non-symlink directory, owned by the current user,
without group/other write permission, and must not be modified by hostile same-user
processes or permissive ACLs during the build. Planning records its device/inode and
the build rechecks ownership, mode, and identity before staging and publication.
Failure cleanup removes only unchanged, inode-owned staging/archive temporaries.
Once any final path has been created, failure cleanup never removes it; uncertain or
replaced paths and partial publication are left for manual inspection.

A process killed outside normal exception handling can leave a sidecar lock. Its
JSON payload records the PID, target, and random owner token. Inspect the named
process and target before manually removing a stale lock; the packager deliberately
fails closed rather than guessing that a lock is stale.

For a real archive, manuscript compilation is checked three times: from the exact
planned source during preflight, from a disposable copy of the staged package, and
from a disposable copy of the extracted archive. The compile uses `latexmk -norc -pdf`
with `interface-reconstruction-paper/interface-reconstruction.tex`. All TeX build
outputs go to temporary directories outside the package. Package checksums are
verified before and after the extracted-source compile, so generated files cannot
contaminate checksum-sealed content.

Compile gates discard inherited `TEXINPUTS`, `BIBINPUTS`, `BSTINPUTS`, all
`TEXMF*` variables, latexmk rc selectors, and Perl library/startup overrides. They
install exact source-only search prefixes with system defaults, use isolated
`HOME`, `XDG_CONFIG_HOME`, `TEXMFHOME`, `TEXMFCONFIG`, and cache directories, and
pass `-norc` so user and global latexmkrc files cannot affect acceptance. System
TeX packages remain available through the normal distribution defaults.

All Git inspection and blob materialization commands set `GIT_OPTIONAL_LOCKS=0`.
Dry-run writes only to temporary directories: it does not change paper index bytes
or mtime, create `.git/index.lock`, or create the output directory, archive,
reservation, staging sibling, TeX product in the paper checkout, or release artifact.
It only reads and validates the required pre-existing output parent.

## Package Layout

```text
README.md
INVENTORY.csv
INVENTORY.json
SHA256SUMS
code/scientific_source_snapshot.tar.gz
code/figure_generator_snapshot.tar.gz
docs/PAPER_EXPERIMENT_MAP.md
manuscript/source/interface-reconstruction-paper/...
manuscript/review/...
results/perturbed_sweep.csv
provenance/approved_figures.csv
provenance/code_snapshots.json
provenance/privacy_redactions.json
provenance/manuscript_build.json
provenance/vector_pdf_qa.json
provenance/figures/approved_figure_bindings.json
provenance/figures/final_figure_orchestration.json
provenance/figures/external_approval_record.json
provenance/figures/figure_candidate_source_map.json
provenance/figures/figure_candidate_source_map.csv
provenance/figures/approved_candidate_allowlist.json
provenance/figures/published_tree_sha256.json
provenance/RAW_DATA_DEPOSITION.md
provenance/deposit/SHA256SUMS.downloaded  # only when supplied
provenance/release/...
```

The scientific and generator archives are deterministic privacy-sanitized public
copies. Their exact pre-redaction archive hashes, commits, trees, Git object IDs,
and public hashes are recorded independently in `provenance/code_snapshots.json`.
The paper experiment map is also privacy-sanitized for presentation; its independent
documentation commit/tree, pinned Git object ID, and original blob digest remain
recorded beside the public-copy digest. The orchestration and external approval
records likewise retain their original sealed-publication authority hashes beside
the hashes of their sanitized public copies.

## Raw-Data Boundary

The compact package deliberately excludes `raw_runs/` and the large case-, cell-,
merge-, and fallback-indexed tables. It retains the aggregate result table, run
inventory, full-release checksum manifest, and deposition record. The external
deposit must contain the complete audited release and the exact checksum manifest
identified by `--raw-data-manifest-id`, so each paper result can be traced back to
its raw bundle. Without a supplied deposited manifest, remote-content verification
remains a separately recorded manual gate.

The package's own `SHA256SUMS` covers every staged file except itself. Verify it with:

```sh
sha256sum -c SHA256SUMS
# macOS
shasum -a 256 -c SHA256SUMS
```
