# Final Figure Provenance Contract

The 38 review candidates are accepted only when their bytes can be traced to an
audited final release and to the command that generated each figure family. A
caller cannot supply or override the scientific source commit. The acceptance
gate derives it from `FINAL_ROOT/submission_config.resolved.json` and verifies
the complete `FINAL_ROOT/SHA256SUMS` before reading any figure manifest.

## Generator Manifest

Every final figure command atomically writes a `figure_provenance.json` manifest
with this interface:

```json
{
  "schema_version": 1,
  "manifest_type": "final_figure_generation",
  "status": "completed",
  "generator": "section6_maintext",
  "generation_provenance": {
    "source_commit": "<full commit>",
    "source_dirty": false,
    "source_status": [],
    "reconstruction_profile": {
      "plic_fallback": "LVIRA",
      "corner_behavior_profile": "<frozen profile>",
      "rescue_profile": "<frozen profile>"
    }
  },
  "release": {
    "root": "<absolute FINAL_ROOT>",
    "name": "<release basename>",
    "source_commit": "<full commit>",
    "reconstruction_profile": {},
    "artifacts": {
      "submission_config.resolved.json": {"path": "...", "sha256": "..."},
      "sweep_manifest.json": {"path": "...", "sha256": "..."},
      "perturbed_sweep.csv": {"path": "...", "sha256": "..."},
      "SHA256SUMS": {"path": "...", "sha256": "..."}
    }
  },
  "inputs": [
    {
      "role": "producer_manifest",
      "path": "<absolute path>",
      "sha256": "<sha256>",
      "release_relative_path": null
    }
  ],
  "outputs": [
    {
      "candidate_id": "<ID from final_figure_candidates.json>",
      "path": "<absolute PDF path>",
      "sha256": "<sha256>"
    }
  ]
}
```

The generator and its producer manifest must report the same full commit, a
clean source tree, and the frozen reconstruction profile. That generator commit
may be a later tooling commit than the frozen scientific-release commit; both
are retained in the acceptance source map. The scientific data commit is never
accepted from the caller or generator: it is derived from the audited release.

Every input and output is rehashed during acceptance. Inputs inside `FINAL_ROOT`
must also match their release-relative entry in `SHA256SUMS`. Main-text geometry
and the aggregate CSV are required to be final-release checksum-ledger entries;
the all-method figures must use exactly `FINAL_ROOT/perturbed_sweep.csv`.

Dedicated resolution, guarded-C0, and deterministic commands preserve their
scientific `manifest.json` or data JSON as `producer_manifest`, and list the
underlying CSV/mesh/facet/metadata files as checksummed inputs. The common
manifest is the checksum source map for those dedicated artifacts.

## Required Manifests

The inventory is fixed and fail-closed:

```text
FIGURE_ROOT/section6/figure_provenance.json
FIGURE_ROOT/all_method_summary_plots/figure_provenance.json
FIGURE_ROOT/resolution/{lines,squares,circles,ellipses,zalesak}/figure_provenance.json
C0_ROOT/figure_provenance.json
FIGURE_ROOT/deterministic/perfect_reconstruction_plic_stencil_figure_provenance.json
FIGURE_ROOT/deterministic/staged_reconstruction_zalesak_figure_provenance.json
```

Missing or extra provenance manifests fail acceptance, as do missing or extra
candidate PDFs. The ten manifests must cover the explicit 38-candidate allowlist
exactly once.

## Acceptance Output

Acceptance renders each one-page PDF directly with Poppler at 300 DPI; candidate
PNG files supplied by a caller are ignored. It verifies the rendered dimensions
against the PDF page box, composes a vector review PDF, measures the merged page
count, and raster-compares every mapped review page to its accepted candidate.

All review artifacts are built in a sibling temporary directory. Only after the
PDF checks, preview checks, page-map check, JSON/CSV writes, and staged inventory
check pass is that directory atomically renamed to the requested output path.

```bash
python submission/accept_figure_candidates.py \
  --release-root "$FINAL_ROOT" \
  --figure-root "$FIGURE_ROOT" \
  --c0-root "$C0_ROOT" \
  --output-dir "$REVIEW_ROOT"
```

`REVIEW_ROOT` must not already exist. A failure leaves it absent.
