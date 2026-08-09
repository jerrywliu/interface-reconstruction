# Interface Reconstruction

Python implementation and reproducibility tooling for a volume-of-fluid
interface reconstruction method using line, circular-arc, and corner facets.

The paper-facing production configuration is frozen as:

```text
corner behavior: pre_f8_corner
rescue profile:  exact_linear_support_only
PLIC fallback:   LVIRA
```

The defaults are enforced in the implementation, benchmark CLIs, submission
configuration, and regression tests.

## Start Here

| Goal | Entry point |
|---|---|
| Understand the supported source tree | `docs/CODE_STRUCTURE.md` |
| Run one static benchmark | `python -m experiments.static.<benchmark>` |
| Reproduce the frozen full sweep | `submission/run_final_static_sweep.sh` |
| Map every paper result and figure to code | `docs/PAPER_EXPERIMENT_MAP.md` |
| Regenerate final vector figures | `submission/run_final_figure_orchestrator` |
| Build the submission package | `submission/package_submission.py` |
| Inspect supported entry points | `docs/ENTRY_POINTS.md` |

The five paper benchmarks are `lines`, `squares`, `circles`, `ellipses`, and
`zalesak`. Historical implementations, wrappers, camera-ready scripts, and
dated ablations were removed from this submission branch; they remain available
in Git history.

## Install And Test

The validated public environment uses CPython 3.9.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pytest -q test
```

Use `requirements-figures.txt` when regenerating paper figures. Dependency and
clean-environment details are in `docs/DEPENDENCIES.md` and
`submission/CLEAN_ENV_REPRODUCIBILITY_VALIDATION.md`.

## Run A Targeted Case

Run commands from the repository root and use a unique output name:

```bash
python -m experiments.static.circles \
  --config static/circle \
  --facet_algo circular \
  --resolution 1.0 \
  --case_indices 0 \
  --plic_fallback LVIRA \
  --save_name quickstart_circle
```

For the full corner method, the frozen profile can be written explicitly:

```bash
python -m experiments.static.zalesak \
  --config static/zalesak \
  --facet_algo 'circular+corner' \
  --resolution 1.0 \
  --case_indices 0 \
  --plic_fallback LVIRA \
  --corner_behavior_profile pre_f8_corner \
  --rescue_profile exact_linear_support_only \
  --save_name quickstart_zalesak
```

Outputs are written beneath `plots/<save_name>/`. The audited final result set
uses an immutable release layout rather than those mutable local directories;
see `docs/PAPER_EXPERIMENT_MAP.md` before reproducing manuscript results.

## Reproduce The Paper

The full submission sweep contains 970 runs and 24,250 cases. Inspect its
resolved configuration and dry-run gates before launching:

```bash
python submission/check_submission_freeze.py --source-only
bash submission/run_final_static_sweep.sh
```

Final figures are direct vector PDFs generated from the sealed numerical
release through the fail-closed orchestrator. Manual plot commands are useful
for diagnosis but are not authoritative submission candidates.

## Data And Provenance

Disposable local products such as `plots/`, logs, raster previews, and virtual
environments are ignored. Release CSV, JSON, PDF, SVG, VTK/VTP, checksum, and
manifest files remain visible for review. See `docs/GENERATED_FILES.md` for the
policy and `submission/CODE_REPRODUCIBILITY_AUDIT.md` for the trust model.

The repository preserves compact audit evidence and the code needed to
reproduce paper outputs. Large sealed releases and raw cell-level bundles are
managed as external immutable artifacts rather than committed source files.
The intended public archive contains code, aggregate results, paper-facing
figure inputs, manifests, and checksums; full raw run trees and large cell/merge
diagnostics remain available from the corresponding author on reasonable
request.

Source code is proposed for release under Apache-2.0; processed results and
project-authored reproducibility metadata in the compact deposit are proposed
for release under CC BY 4.0. See `LICENSE`, `DATA_LICENSE`, `CITATION.cff`,
`NOTICE`, and `submission/RELEASE_METADATA.md` for scope, attribution,
unresolved placeholders, and institutional approval gates. The compact deposit
has its own checksum ledger and a cryptographic binding to the separately
audited complete local release.
