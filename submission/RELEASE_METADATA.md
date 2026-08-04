# Release Metadata and Approval Gates

## Proposed Public Metadata

- Source code: BSD 3-Clause (`LICENSE`).
- Deposited processed results, provenance manifests, checksum ledgers, and
  project-authored reproducibility documentation: CC BY 4.0 (`DATA_LICENSE`).
- Citation metadata: `CITATION.cff`, with Jerry Liu, Kenneth Weiss, and Jin Yao
  in manuscript order.
- Attribution and funding notice: `NOTICE`.
- Repository: `https://github.com/jerrywliu/interface-reconstruction`.

The article's publication license is separate from the code and data licenses.
Do not use this repository's BSD or CC BY notices to relicense the publisher's
formatted article or third-party material.

## Compact Deposit Boundary

The public archive contains processed paper-facing results and the material needed
to reproduce experiments and figures. It does not contain the complete 970 raw run
bundles, complete case/cell/merge/fallback diagnostics, or raw VTK/VTP files. Those
complete diagnostics are available from the corresponding author on request.

The complete local sealed release remains authenticated by its own full
`SHA256SUMS`. The compact deposit has a separate `SHA256SUMS` and must contain:

```text
provenance/COMPLETE_RELEASE_AUTHORITY.json
```

That JSON record uses schema version 1 and must contain these exact fields:

```json
{
  "schema_version": 1,
  "deposit_scope": "compact_processed_results_and_reproducibility_materials",
  "complete_release_name": "<sealed-release-directory-name>",
  "complete_release_sha256sums_sha256": "<64-hex-digest>",
  "scientific_commit": "<40-hex-commit>",
  "complete_diagnostics_policy": "available_from_corresponding_author_on_request"
}
```

The compact manifest must cover every downloaded deposit file except itself. The
packager requires byte equality between the approved compact manifest and the
downloaded manifest, verifies every downloaded file, rejects bulk diagnostics and
raw VTK/VTP content, and checks the authority record against the audited complete
release and scientific commit.

## Required Approvals Before Public Release

- Confirm the legal copyright holder and whether LLNL or Stanford ownership text
  must replace or supplement the author copyright line.
- Obtain LLNL release review, export-control review, and the final LLNL release
  number.
- Confirm BSD-3-Clause and CC BY 4.0 with all authors and the applicable
  institutional technology-transfer or legal offices.
- Review the repository's open dependency-security alerts and validate a
  patched public-install environment. Preserve the accepted scientific
  environment record as immutable provenance rather than silently rewriting
  the environment associated with the frozen results.
- Audit bundled and copied third-party material; retain its original license and
  attribution rather than applying the project licenses.
- Replace only the three explicit unknowns in `NOTICE`: archival DOI, final article
  citation, and LLNL release number.
- Add the final archival DOI and article bibliographic fields to `CITATION.cff`,
  then validate the file with a CFF 1.2 validator.
- Generate the compact deposit ledger, upload the exact compact tree, download it
  to a fresh directory, and pass the full checksum and authority-binding gate.
- Confirm the manuscript's CRediT statement, competing-interest declaration,
  generative-AI disclosure, corresponding-author details, and data/code
  availability statement separately; those are article metadata, not repository
  license terms.
