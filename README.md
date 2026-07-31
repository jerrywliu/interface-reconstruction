# Interface Reconstruction

An example implementation in Python of an interface reconstruction method, using linear/circular elements and cusps.

## Quick Start

This repository includes current paper workflows alongside historical research
drivers. Start with the status index before choosing an entry point:

- `docs/ENTRY_POINTS.md`: canonical, supported research, and legacy commands
- `docs/DEPENDENCIES.md`: runtime, test, and figure environments
- `docs/GENERATED_FILES.md`: scratch-output and release-metadata policy
- `docs/PAPER_EXPERIMENT_MAP.md`: paper experiments and plots mapped to code

Install the frozen runtime and test tooling, then run the test suite:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pytest -q test
```

For a compact targeted reconstruction, run one deterministic circle case. Paper-
facing runs use `LVIRA` whenever the oriented reconstruction requires a PLIC
fallback:

```bash
python -m experiments.static.circles \
  --config static/circle \
  --facet_algo circular \
  --resolution 1.0 \
  --case_indices 0 \
  --plic_fallback LVIRA \
  --save_name quickstart_circle
```

The frozen full-paper sweep is launched through
`submission/run_final_static_sweep.sh`. It is intentionally not the quick-start
command because it plans 970 runs and 24,250 cases; consult the submission config
and paper experiment map first.

For a full description of the sweep experiments (linear + perturbed), see `docs/EXPERIMENTS.md`.
For local plotting, reconstruction inspection, and paper-figure regeneration, see `docs/VISUALIZATION_WORKFLOW.md`.
For the paper-to-code map of Section 6 experiments, final data, diagnostics, and figure assets, see `docs/PAPER_EXPERIMENT_MAP.md`.

For deterministic, fail-closed assembly of the final code/manuscript package, see `submission/SUBMISSION_PACKAGING.md`.

## Contributors

Jerry Liu, jwl50@stanford.edu

## Table of Contents

- [Algorithms](#algorithms)
- [Static Experiments](#static-experiments)
- [Advection Experiments](#advection-experiments)
- [Limitations](#limitations)

## Algorithms

Our algorithms consist of two main features:
- **Circular facets**
- **Corner facets** (either linear or circular), which requires merging cells

### Supported Algorithms

#### Baselines
- **Youngs**
- **ELVIRA**
- **LVIRA**

#### Our Algorithms

**Independent Cells**
- **safe_linear**: Linear reconstruction without shared topology or merging
- **safe_circle**: Circular reconstruction without shared topology or merging

**Coordinated Topology And Merging**
- **linear**: Linear reconstruction with coordinated topology and merging
- **circular**: Circular reconstruction with coordinated topology and merging

## Static Experiments

The supported targeted interface is the benchmark's Python module. Set the config,
method, resolution, cases, fallback, and unique output name explicitly. For example:

```bash
python -m experiments.static.ellipses \
  --config static/ellipse \
  --facet_algo circular \
  --resolution 1.0 \
  --case_indices 0 \
  --plic_fallback LVIRA \
  --save_name example_ellipse

python -m experiments.static.zalesak \
  --config static/zalesak \
  --facet_algo 'circular+corner' \
  --resolution 1.0 \
  --case_indices 0 \
  --plic_fallback LVIRA \
  --corner_behavior_profile pre_f8_corner \
  --rescue_profile exact_linear_support_only \
  --save_name example_zalesak
```

The corresponding modules and configs are:

| Benchmark | Module | Config |
|---|---|---|
| Lines | `experiments.static.lines` | `static/line` |
| Circles | `experiments.static.circles` | `static/circle` |
| Ellipses | `experiments.static.ellipses` | `static/ellipse` |
| Squares | `experiments.static.squares` | `static/square` |
| Zalesak | `experiments.static.zalesak` | `static/zalesak` |

Use `bash submission/run_final_static_sweep.sh` for the exact paper result set;
per-shape `experiments/static/run_*.sh` files are retained as legacy convenience
wrappers and are not equivalent to that launcher. The tracked
`results/static/*_reconstruction_results.txt` files and root `test.vtp` are small
historical artifacts, not the audited final release. See `docs/ENTRY_POINTS.md` for
the complete status classification and `docs/VISUALIZATION_WORKFLOW.md` for plotting
from an immutable release.

## Advection Experiments

These are supported research configurations, not sources for the paper's final
static result set. See `docs/ENTRY_POINTS.md` for that distinction.

### Zalesak's Disk
```bash
python3 run.py --config advection/zalesak/50/zalesak_50_ccorner
python3 run.py --config advection/zalesak/100/zalesak_100_ccorner
```

### x+o Problem
```bash
# Working configuration
python3 run.py --config advection/x+o/50/x+o_50_safecircle

# Other configurations
python3 run.py --config advection/x+o/50/x+o_50_circular
python3 run.py --config advection/x+o/50/x+o_50_ccorner  # TODO: Producing circular facets with inverted curvature and incorrect corners
python3 run.py --config advection/x+o/100/x+o_100_ccorner
python3 run.py --config advection/x+o/150/x+o_150_ccorner  # TODO: Mostly ok, but need to adjust corner threshold. Some corners failing and reforming
```

### Vortex Problem
**Algorithms**: safecirclec0, safecircle, safelinear  
**Resolutions**: 32, 64, 128

```bash
python3 run.py --config advection/vortex/32/vortex_32_safecirclec0
python3 run.py --config advection/vortex/50/vortex_50_safecircle
python3 run.py --config advection/vortex/100/vortex_100_safecircle
```

## Limitations

The manuscript and submission audit document the validated scope and known
limitations of the frozen method. Historical scripts and profiles may retain older
failure modes and are classified in `docs/ENTRY_POINTS.md`; they are not supported
sources of new paper results.

---

<div align="right"><a href="#table-of-contents">back to top</a></div>
