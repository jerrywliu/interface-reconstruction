# Deterministic Submission Packaging

`package_submission.py` builds the compact archival package only after the final
scientific release and manuscript figures have passed their separate approval gates.
It does not modify the release, the paper source, or Overleaf.

## Required Inputs

1. A completed `970`-run / `24,250`-case release root.
2. A successful programmatic audit by `audit_final_release.py`.
3. A complete and valid release-wide `SHA256SUMS`.
4. A clean paper Git worktree pinned to an explicit full commit. Its root must
   contain `interface-reconstruction-paper/`.
5. An explicit approval manifest for every imported figure.
6. A non-placeholder DOI or URL for the complete raw-data deposit.
7. A `sha256:<digest>` identifier for the exact `SHA256SUMS` deposited with the
   complete release.
8. Either a locally downloaded/fetched copy of the deposited `SHA256SUMS`, or an
   explicit acknowledgment that remote deposit contents remain a manual gate.
9. A pre-existing output parent owned by the current user, represented by a real
   directory, and not writable by group or other users.

The packager reruns the full release audit and verifies every entry in the release
checksum manifest. A stale audit claim, incomplete sweep, failed run, changed raw
file, missing checksum, dirty or mismatched paper worktree, failed manuscript
compile, rasterized figure, unembedded figure font, unapproved
`\includegraphics` target, or mismatched deposit-manifest digest stops packaging.
Every compact release payload is checked again after it is copied into staging,
using the digest frozen in the release `SHA256SUMS`. This closes the interval
between planning and materialization; mutating either a payload or the manifest
after planning aborts before final publication.

Generate and verify the complete-release checksum manifest after all final release
artifacts have been installed:

```sh
python submission/audit_final_release.py "$RELEASE" \
  --write-sha256-manifest \
  --verify-sha256-manifest
```

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

## Figure Approval Manifest

The approval manifest is CSV with these required columns:

```csv
paper_path,source_path,sha256,approval_status,approval_reference
interface-reconstruction-paper/figs/cameraready/line_reconstruction_maintext_metrics.pdf,interface-reconstruction-paper/figs/cameraready/line_reconstruction_maintext_metrics.pdf,<64-hex-sha256>,approved,author-review-2026-08-01
```

- `paper_path` is the path used by the manuscript, relative to the outer paper
  worktree root.
- `source_path` is optional; when blank, it defaults to `paper_path`.
- `sha256` pins the exact approved bytes.
- `approval_status` must be exactly `approved` (case-insensitive).
- `approval_reference` records the review packet, decision sheet, or approval date.

Only those PDFs are copied. All active `\includegraphics` targets must be PDFs and
must appear in the approval manifest. Poppler's `pdfimages` and `pdffonts` are used
to require zero raster image objects and embedded fonts whenever fonts are present.

## Raw-Data Binding

The deposition URL alone is not accepted. Compute the SHA-256 of the final release
checksum manifest and pass it as an explicit manifest identifier:

```sh
export RAW_DATA_MANIFEST_ID="sha256:$(shasum -a 256 "$RELEASE/SHA256SUMS" | awk '{print $1}')"
```

The packager recomputes this digest from the already audited local release and
fails on any mismatch. The normalized deposit record in the package binds together
the deposition location, release directory name, `SHA256SUMS` filename, and exact
manifest digest. The external deposit must expose the same `SHA256SUMS` bytes.

Preferred verification supplies a copy downloaded or fetched from the deposit:

```sh
export DEPOSITED_MANIFEST=/path/to/downloaded/SHA256SUMS
cmp "$RELEASE/SHA256SUMS" "$DEPOSITED_MANIFEST"
```

Pass it with `--deposited-release-manifest "$DEPOSITED_MANIFEST"`. The packager
requires exact byte equality, rechecks the staged evidence digest, and includes it
as `provenance/deposit/SHA256SUMS.downloaded`.

When the deposited manifest cannot yet be fetched, omit that file and pass
`--acknowledge-unverified-remote-deposit`. The package then records
`manual_acknowledgment_remote_contents_unverified`. This is an explicit manual
submission gate, not a remote verification claim. The packager performs no network
request and never claims that a DOI/URL was reachable or that its contents matched.

## Dry Run

Dry run performs the expensive scientific audit, full checksum verification,
paper Git-state check, manuscript import check, vector-PDF inspection, deposit
binding, and a disposable manuscript compile, but writes nothing:

```sh
export PACKAGE_PARENT=/path/to/private-submission-output
install -d -m 700 "$PACKAGE_PARENT"
export PACKAGE="$PACKAGE_PARENT/interface-reconstruction-submission"
```

The exact parent must exist before planning; the packager never creates it.

```sh
python submission/package_submission.py \
  --release-root "$RELEASE" \
  --paper-worktree-root "$PAPER_WORKTREE" \
  --paper-commit "$PAPER_COMMIT" \
  --approved-figures-manifest "$APPROVAL_CSV" \
  --review-bundle "$REVIEW_PDF" \
  --raw-data-deposition "https://doi.org/10.xxxx/record" \
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
code/source_snapshot.tar.gz
docs/PAPER_EXPERIMENT_MAP.md
manuscript/source/interface-reconstruction-paper/...
manuscript/review/...
results/perturbed_sweep.csv
provenance/approved_figures.csv
provenance/manuscript_build.json
provenance/vector_pdf_qa.json
provenance/RAW_DATA_DEPOSITION.md
provenance/deposit/SHA256SUMS.downloaded  # only when supplied
provenance/release/...
```

The code archive and paper experiment map come from the audited source snapshot,
not from the checkout running the packager. This keeps source and result provenance
aligned even if packaging occurs later.

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
