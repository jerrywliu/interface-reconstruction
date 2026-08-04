# Entry Point Status

This repository distinguishes supported paper workflows from current research
interfaces. See `docs/CODE_STRUCTURE.md` for the module-level map.

## Canonical Paper And Release Paths

| Task | Entry point | Status |
|---|---|---|
| Reproduce the frozen full static result set | `bash submission/run_final_static_sweep.sh` | Canonical submission launcher; fixes benchmark grid, profiles, seeds, fallback policy, diagnostics, and release layout. |
| Check a source or result freeze | `python submission/check_submission_freeze.py --source-only` or the full checker | Canonical preflight and release gate. |
| Capture the execution environment | `python submission/capture_environment.py --output <release>/environment.json` | Canonical environment record. |
| Run one static benchmark | `python -m experiments.static.{lines,circles,ellipses,squares,zalesak}` | Supported targeted interface. Set method, resolution, cases, mesh perturbation, and a unique output name explicitly. |
| Map paper results and figures to code | `docs/PAPER_EXPERIMENT_MAP.md` | Canonical paper-to-code index. |
| Regenerate final figures | `submission/run_final_figure_orchestrator` | Only supported producer of final allowlisted vector candidates. |
| Validate vector PDFs | `python submission/pdf_vector_qa.py <pdf-or-directory>` | Canonical vector, raster-object, and font check. |
| Build the submission package | `python submission/package_submission.py ...` | Deterministic, fail-closed final package assembly. |
| Run automated tests | `python -m pytest -q test` | Canonical local test command. Live Slack integration remains opt-in. |

The full sweep launcher delegates to
`experiments/static/run_perturbed_sweeps.py`. Direct controller use is supported
for development and plot-only replay, but is not equivalent to the frozen
launcher unless every recorded parameter is reproduced.

## Supported Research Interfaces

| Path | Intended use |
|---|---|
| `run.py` | YAML-driven static/advection research runs using the current reconstruction structures. Not a source for final paper tables. |
| `experiments/static/run_linear_sweeps.py` | General Cartesian development sweep. |
| `experiments/static/replay_zalesak_outlier.py` | Deterministic replay of a selected Zalesak case. |
| `experiments/submission/*.py` | Conservation, convergence, topology, continuity, ablation, and failure diagnostics used to validate manuscript claims. |
| `experiments/static/{build_figure_review_pdf,prepare_figure_review,generate_figure_review_diagnostics}.py` | Author-review tooling; does not publish final candidates. |

## Removed Historical Paths

The submission branch removes the legacy reconstruction stack, hard-coded shell
wrappers, superseded March camera-ready tooling, dated ablations, metric-repair
scripts, and stale experiment notes. Git history retains those materials. The
supported replacements are the current `main/` and `util/` implementations, the
benchmark modules and frozen sweep launcher, the final-figure orchestrator, and
`docs/PAPER_EXPERIMENT_MAP.md`.

Tracked `results/static/*_reconstruction_results.txt` files are small historical
summary artifacts. They remain in place because supported benchmark CLIs still
offer an explicit compatibility plot/replay mode, but they are not members of
the audited final release.

Historical commands may depend on layouts or artifacts that are absent from
this submission branch and are not supported for reproduction.
