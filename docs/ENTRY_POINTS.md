# Entry Point Status

This repository retains current paper workflows, exploratory utilities, and older
research drivers. The labels below distinguish supported public entry points from
historical replay paths without deleting code that may still be useful.

## Canonical Paper And Release Paths

| Task | Entry point | Status |
|---|---|---|
| Reproduce the frozen full static result set | `bash submission/run_final_static_sweep.sh` | Canonical submission launcher. It fixes the benchmark grid, method profiles, seeds, fallback policy, diagnostics, and release layout. Read the submission configuration before running; this is a large sweep. |
| Check a source or release freeze | `python submission/check_submission_freeze.py --source-only` or the full checker | Canonical preflight audit. The full checker also requires an approved frozen configuration and matching commit. |
| Capture the execution environment | `python submission/capture_environment.py --output <release>/environment.json` | Canonical release metadata utility. Run it with the same interpreter used for the sweep. |
| Run one static benchmark | `python -m experiments.static.{lines,circles,ellipses,squares,zalesak}` | Supported targeted interface. Set the method, resolution, case indices, mesh perturbation, and output name explicitly. |
| Map paper results and figures to code | `docs/PAPER_EXPERIMENT_MAP.md` | Canonical paper-to-code index, including exact final-data and figure commands. |
| Regenerate and inspect figures | `docs/VISUALIZATION_WORKFLOW.md` and `submission/FINAL_FIGURE_REGENERATION.md` | Supported plotting workflow. Regenerate from an immutable release bundle rather than mutable top-level run directories. |
| Validate vector PDFs | `python submission/pdf_vector_qa.py <pdf-or-directory>` | Canonical PDF QA. Requires Poppler command-line tools. |
| Run automated tests | `python -m pytest -q test` | Canonical local test command. The live Slack integration is skipped unless explicitly enabled. |

The full submission launcher delegates to
`experiments/static/run_perturbed_sweeps.py`. Direct controller use is supported
for development and plot-only replay, but it is not equivalent to the frozen
submission command unless every launcher parameter is reproduced.

## Supported Research Interfaces

| Path | Intended use |
|---|---|
| `run.py` | YAML-driven static/advection research runs. It is maintained as a research interface, but it is not the source of the paper's final static tables. Run it from the repository root because configuration paths are repository-relative. |
| `experiments/submission/*.py` | Focused conservation, convergence, topology, ablation, and failure-diagnosis studies used to audit manuscript claims. Consult the paper experiment map for the exact applicable command and source bundle. |
| `experiments/static/replay_zalesak_outlier.py` | Deterministic replay of a selected Zalesak case for diagnosis and visual comparison. |
| `experiments/static/run_linear_sweeps.py` | General Cartesian development sweep. Useful for exploratory comparison, but not the frozen submission grid. |
| Per-shape `run_*.sh` wrappers | Convenience wrappers around individual static drivers. Review their fixed arguments before use; prefer direct module commands for recorded work. |

## Legacy Or Superseded Paths

These files are retained for historical reproduction. They should not be used to
generate new paper results.

| Path | Reason |
|---|---|
| `run_old.py`, root-level compatibility modules such as `facet.py`, and the older `main/algos/{plic,local_reconstruction,static_interface_reconstruction}.py` stack | Pre-package implementation path with hard-coded settings and older imports. The current method uses `util/reconstruction.py`, `main/structs/`, and `main/algos/plic_normals.py`. |
| `run_static.sh` and `run_advection.sh` | Hard-coded convenience loops from earlier development. Use direct module/config commands for new work. |
| `main/algos/static_interface_reconstruction.py` | Older reconstruction path with process-global random behavior; it is outside the final submission launcher. |
| `util/initialize/initialize_areas_old.py` | Retained initialization implementation from the older code path. |
| `run_cameraready_static_*.sh`, `bundle_static_cameraready_release.sh`, and `retro_wire_static_cameraready_existing.sh` | March 2026 camera-ready workflow, superseded for final results by the submission launcher and immutable release bundle. |
| Scripts named for a dated ablation, shard finalization, or candidate analysis | Artifact-specific historical analysis. Use only with the result layout documented by that study. |

Legacy means "not a supported source of new submission results," not necessarily
"broken." Moving or deleting these paths would make historical replay harder and
is intentionally outside this cleanup.
