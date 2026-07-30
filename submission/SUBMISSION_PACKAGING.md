# Deterministic Submission Packaging

`package_submission.py` builds the compact archival package only after the final
scientific release and manuscript figures have passed their separate approval gates.
It does not modify the release, the paper source, or Overleaf.

## Required Inputs

1. A completed `970`-run / `24,250`-case release root.
2. A successful programmatic audit by `audit_final_release.py`.
3. A complete and valid release-wide `SHA256SUMS`.
4. A manuscript source root.
5. An explicit approval manifest for every imported figure.
6. A non-placeholder DOI or URL for the complete raw-data deposit.

The packager reruns the full release audit and verifies every entry in the release
checksum manifest. A stale audit claim, incomplete sweep, failed run, changed raw
file, missing checksum, rasterized figure, unembedded figure font, or unapproved
`\includegraphics` target stops packaging before the output directory is created.

Generate and verify the complete-release checksum manifest after all final release
artifacts have been installed:

```sh
python submission/audit_final_release.py "$RELEASE" \
  --write-sha256-manifest \
  --verify-sha256-manifest
```

## Figure Approval Manifest

The approval manifest is CSV with these required columns:

```csv
paper_path,source_path,sha256,approval_status,approval_reference
figs/camera_ready/line_reconstruction_maintext_metrics.pdf,figs/camera_ready/line_reconstruction_maintext_metrics.pdf,<64-hex-sha256>,approved,author-review-2026-08-01
```

- `paper_path` is the path used by the manuscript, relative to the paper source root.
- `source_path` is optional; when blank, it defaults to `paper_path`.
- `sha256` pins the exact approved bytes.
- `approval_status` must be exactly `approved` (case-insensitive).
- `approval_reference` records the review packet, decision sheet, or approval date.

Only those PDFs are copied. All active `\includegraphics` targets must be PDFs and
must appear in the approval manifest. Poppler's `pdfimages` and `pdffonts` are used
to require zero raster image objects and embedded fonts whenever fonts are present.

## Dry Run

Dry run performs the expensive scientific audit, full checksum verification,
manuscript import check, and vector-PDF inspection, but writes nothing:

```sh
python submission/package_submission.py \
  --release-root "$RELEASE" \
  --paper-source-root "$PAPER" \
  --approved-figures-manifest "$APPROVAL_CSV" \
  --review-bundle "$REVIEW_PDF" \
  --raw-data-deposition "https://doi.org/10.xxxx/record" \
  --output-dir "$PACKAGE" \
  --dry-run
```

Remove `--dry-run` to create both `$PACKAGE/` and a deterministic
`$PACKAGE.tar.gz`. Existing destinations are never overwritten. Use `--no-archive`
only when a staged directory without the outer archive is required.

## Package Layout

```text
README.md
INVENTORY.csv
INVENTORY.json
SHA256SUMS
code/source_snapshot.tar.gz
docs/PAPER_EXPERIMENT_MAP.md
manuscript/source/...
manuscript/review/...
results/perturbed_sweep.csv
provenance/approved_figures.csv
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
deposit must contain the complete audited release so each paper result can be traced
back to its raw bundle.

The package's own `SHA256SUMS` covers every staged file except itself. Verify it with:

```sh
sha256sum -c SHA256SUMS
# macOS
shasum -a 256 -c SHA256SUMS
```
