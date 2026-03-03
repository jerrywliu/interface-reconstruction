# Static Camera-Ready Workflow

Updated: 2026-03-03

## Current State (Snapshot)

- Raw static run outputs are mostly under `plots/` in a flat namespace (`perturb_sweep_*`).
- `plots/` currently holds hundreds of perturbed run folders and is large (~16 GB).
- Perturbed summary plots are centralized in `results/static/perturbed_plots/`.
- Aggregated CSVs are mixed in `results/static/` (generic + merged + legacy files).
- Overleaf camera-ready figures are copied manually into `../overleaf/interface-reconstruction-paper/figs/cameraready/`.
- This manual copy path can create stale-file mismatches because there is no release manifest tying paper figures back to exact CSV/run sources.

## Target Structure

Use release-scoped bundles under:

`results/static/camera_ready/<release_id>/`

with subfolders:

- `csv/`: sweep CSVs used for plotting
- `summary_plots/`: generated method/metric plots
- `paper_figs/`: canonical figure names for paper inclusion
- `logs/`: run logs and copied result summaries
- `manifest.md`: release provenance and file inventory

`results/static/camera_ready/latest` is a symlink to the newest release.

## Script Entry Points

All scripts are in `experiments/static/`.

- `run_cameraready_static_all.sh`
  - One-command pipeline: perturbed sweep(s) + Cartesian sweep(s) + release bundle.
- `run_cameraready_static_perturbed.sh`
  - Runs perturbed sweeps (default can be set via `ONLY`; supports all five static tests) and writes release-scoped CSV + summary plots.
  - By default sends Slack summary plots at completion (`NOTIFY=1`).
- `run_cameraready_static_cartesian.sh`
  - Runs Cartesian sweeps (default: `squares,zalesak`) for static placeholders/legacy figures.
  - By default sends Slack plots at completion (`NOTIFY=1`).
- `bundle_static_cameraready_release.sh`
  - Builds/updates a release bundle from existing outputs and optional CSV hints.
  - By default sends canonical release figures + manifest to Slack (`NOTIFY=1`).
- `retro_wire_static_cameraready_existing.sh`
  - Retroactive wiring for already-computed outputs (no reruns required).

## Cleanup Plan

1. Keep all new camera-ready work in release-scoped directories only.
2. Stop manual ad-hoc copying from `results/static/perturbed_plots/` to Overleaf.
3. Publish to Overleaf only through `bundle_static_cameraready_release.sh` (`SYNC_OVERLEAF=1`).
4. For each release, archive the exact CSVs and maintain `manifest.md`.
5. After paper lock, archive older raw run folders and keep only release bundles + required provenance.

## Typical Usage

Full pipeline:

```bash
./experiments/static/run_cameraready_static_all.sh
```

Perturbed-all static tests (Section 6 coverage):

```bash
ONLY=lines,circles,ellipses,squares,zalesak ./experiments/static/run_cameraready_static_perturbed.sh
```

Retro-wire current outputs into a structured release:

```bash
./experiments/static/retro_wire_static_cameraready_existing.sh
```

Sync bundled `paper_figs/` to Overleaf:

```bash
SYNC_OVERLEAF=1 ./experiments/static/bundle_static_cameraready_release.sh
```

Disable Slack notifications for a run:

```bash
NOTIFY=0 ./experiments/static/run_cameraready_static_all.sh
```
