# Historical Archive

This directory preserves source that is not part of the supported paper or
submission workflow. Nothing under `archive/` is imported by the frozen
reconstruction implementation, the canonical static benchmark drivers, the
final figure orchestrator, or the submission packager.

The archive remains tracked for scientific provenance and historical replay.
New experiments and fixes must not depend on it.

| Directory | Contents | Replacement |
|---|---|---|
| `legacy_v1/` | Pre-package implementation, examples, and one sample VTP artifact | `main/structs/`, `main/geoms/`, and `util/reconstruction.py` |
| `legacy_wrappers/` | Hard-coded convenience shell loops | Direct benchmark modules or `submission/run_final_static_sweep.sh` |
| `march_2026_camera_ready/` | Superseded mutable camera-ready bundling workflow | `submission/run_final_figure_orchestrator` |
| `ablations/` | Dated corner/fallback/profile analyses | Frozen production profile and sealed result release |
| `historical_sweeps/` | Superseded sweep launchers and artifact-specific repairs | Canonical static controller and submission launcher |
| `historical_experiment_docs/` | Superseded algorithm catalog and experiment narrative | `docs/PAPER_EXPERIMENT_MAP.md` and `experiments/static/README.md` |

Python utilities in `ablations/` and `historical_sweeps/` can be invoked as
modules from the repository root, for example
`python -m archive.ablations.analyze_drop9_resolution`. Their original input
layouts may no longer be present in a fresh checkout.

See `docs/CODE_STRUCTURE.md` for the supported tree and a complete cleanup
rationale.
