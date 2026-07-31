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

The packager reruns the full release audit and verifies every entry in the release
checksum manifest. A stale audit claim, incomplete sweep, failed run, changed raw
file, missing checksum, dirty or mismatched paper worktree, failed manuscript
compile, rasterized figure, unembedded figure font, unapproved
`\includegraphics` target, or mismatched deposit-manifest digest stops packaging.

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

The packager requires the full 40-character commit, confirms `HEAD` exactly, and
packages only tracked files under `interface-reconstruction-paper/`. Approved
figures must also be tracked at that commit.

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

## Dry Run

Dry run performs the expensive scientific audit, full checksum verification,
paper Git-state check, manuscript import check, vector-PDF inspection, deposit
binding, and a disposable manuscript compile, but writes nothing:

```sh
python submission/package_submission.py \
  --release-root "$RELEASE" \
  --paper-worktree-root "$PAPER_WORKTREE" \
  --paper-commit "$PAPER_COMMIT" \
  --approved-figures-manifest "$APPROVAL_CSV" \
  --review-bundle "$REVIEW_PDF" \
  --raw-data-deposition "https://doi.org/10.xxxx/record" \
  --raw-data-manifest-id "$RAW_DATA_MANIFEST_ID" \
  --output-dir "$PACKAGE" \
  --dry-run
```

Remove `--dry-run` to create both `$PACKAGE/` and a deterministic
`$PACKAGE.tar.gz`. Existing destinations are never overwritten. Use `--no-archive`
only when a staged directory without the outer archive is required.

For a real archive, manuscript compilation is checked three times: from the exact
planned source during preflight, from a disposable copy of the staged package, and
from a disposable copy of the extracted archive. The compile uses `latexmk -pdf`
with `interface-reconstruction-paper/interface-reconstruction.tex`. All TeX build
outputs go to temporary directories outside the package. Package checksums are
verified before and after the extracted-source compile, so generated files cannot
contaminate checksum-sealed content.

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
its raw bundle.

The package's own `SHA256SUMS` covers every staged file except itself. Verify it with:

```sh
sha256sum -c SHA256SUMS
# macOS
shasum -a 256 -c SHA256SUMS
```
