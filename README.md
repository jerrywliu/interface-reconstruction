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

For a compact targeted reconstruction, run one deterministic circle case:

```bash
python -m experiments.static.circles \
  --config static/circle \
  --facet_algo circular \
  --resolution 1.0 \
  --case_indices 0 \
  --save_name quickstart_circle
```

The frozen full-paper sweep is launched through
`submission/run_final_static_sweep.sh`. It is intentionally not the quick-start
command because it plans 970 runs and 24,250 cases; consult the submission config
and paper experiment map first.

For a full description of the sweep experiments (linear + perturbed), see `docs/EXPERIMENTS.md`.
For local plotting, reconstruction inspection, and paper-figure regeneration, see `docs/VISUALIZATION_WORKFLOW.md`.
For the paper-to-code map of Section 6 experiments, final data, diagnostics, and figure assets, see `docs/PAPER_EXPERIMENT_MAP.md`.

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

These experiments test interface reconstruction on various geometric shapes with different algorithms and mesh resolutions.
For the current local inspection workflow, including `--plot_only`, Section 6 figure regeneration, and Zalesak outlier replay, see `docs/VISUALIZATION_WORKFLOW.md`.

### Lines
```bash
./experiments/static/run_lines.sh
```
Tests reconstruction of straight lines with varying orientations (0 to 2π).

To plot:
```bash
python -m experiments.static.lines --plot_only --results_file results/static/line_reconstruction_results.txt
```

### Circles
```bash
./experiments/static/run_circles.sh
```
Tests reconstruction of circles with varying centers and fixed radius.

To plot:
```bash
python -m experiments.static.circles --plot_only --results_file results/static/circle_reconstruction_results.txt
```

### Ellipses
```bash
./experiments/static/run_ellipses.sh
```
Tests reconstruction of ellipses with varying aspect ratios (1.5 to 3.0).

To plot:
```bash
python -m experiments.static.ellipses --plot_only --results_file results/static/ellipse_reconstruction_results.txt
```

To run the unit tests of the ellipse helper functions, run:
```bash
python -m experiments.static.ellipses --test_plot_ellipse_arc
python -m experiments.static.ellipses --test_plot_hausdorff_case
python -m experiments.static.ellipses --test_ellipse_hausdorff
```

### Squares
```bash
./experiments/static/run_squares.sh
```
Tests reconstruction of squares with varying orientations.

To plot:
```bash
python -m experiments.static.squares --plot_only --results_file results/static/square_reconstruction_results.txt
```

To run the unit test for the square edge alignment metric:
```bash
python -m experiments.static.squares --test_edge_metric
```

### Zalesak (Static)
```bash
python -m experiments.static.zalesak --config static/zalesak --sweep --num_cases 15
```
Tests reconstruction of Zalesak's disk (circle with slot) with random centers and random rotations.

To plot:
```bash
python -m experiments.static.zalesak --plot_only --results_file results/static/zalesak_reconstruction_results.txt
```

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
