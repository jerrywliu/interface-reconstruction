# Static Shape Reconstruction Experiments

This directory contains experiments for evaluating interface reconstruction algorithms on static shapes (lines, squares, circles, ellipses, and Zalesak's disk). Each experiment tests different reconstruction algorithms across various resolutions and measures reconstruction quality using appropriate metrics for each shape type.

## Experimental Setup

All experiments use a 100x100 Cartesian grid with varying resolutions. The shapes are centered in the grid cell with corners (50, 50) and (51, 51) to avoid boundary effects. Each experiment uses a fixed random seed (42) for reproducibility.

### Reconstruction Algorithms Tested
- **Lines**: `Youngs`, `ELVIRA`, `LVIRA`, `safe_linear`, `linear`
- **Circles / Ellipses**: `Youngs`, `ELVIRA`, `LVIRA`, `safe_linear`, `linear`, `safe_circle`, `circular`
- **Squares**: `Youngs`, `ELVIRA`, `LVIRA`, `safe_linear`, `linear`, `linear+corner`, `safe_circle`, `circular`
- **Zalesak**: `Youngs`, `ELVIRA`, `LVIRA`, `safe_linear`, `linear`, `safe_circle`, `circular`, `circular+corner`

### Resolution Range
- **Lines / Circles / Ellipses**: [0.32, 0.50, 0.64, 1.00, 1.28, 1.50]
- **Squares / Zalesak**: [0.50, 0.64, 1.00, 1.28, 1.50]

## Running the Experiments

### Current Paper-Facing Rerun

Use the dedicated launcher for the affected paper rows. It fixes the method set,
LVIRA fallback, rescue profile, and canonical perturbation spectrum; the Python
runner selects the appropriate resolution list for each experiment.

```bash
# Validate the 300-run plan without starting reconstruction.
DRY_RUN=1 ./experiments/static/run_paper_affected_sweep.sh

# Run with five concurrent subprocesses and Slack summary notifications.
./experiments/static/run_paper_affected_sweep.sh

# Optional: choose a run id or disable notifications.
RUN_ID=my_run NOTIFY=0 ./experiments/static/run_paper_affected_sweep.sh
```

Each release contains the aggregate plotting CSV, all-method summary panels,
per-run logs, a sweep manifest/failure ledger, and consolidated case, cell,
merge-event, fallback-event, geometry, and run-manifest diagnostics.

The paper-facing corner-method default is
`--corner_behavior_profile pre_f8_corner --rescue_profile exact_linear_support_only`.
It retains exact linear-support propagation while disabling the five inactive
linear-corner cleanup passes and all curved loop/transition rescues. This is a
validated production default; see
`results/static/linear_rescue_cleanup_analysis_20260715/README.md` for evidence
and use the other named profiles only for historical ablations.

For the current local visualization workflow, including:

- inspecting a single `plots/<save_name>/` directory
- rebuilding plots with `--plot_only`
- regenerating Section 6 summary panels from CSVs
- replaying deterministic Zalesak failures

see:

- `docs/VISUALIZATION_WORKFLOW.md`

### Camera-Ready Pipelines

```bash
# One-command static camera-ready pipeline (perturbed + cartesian + bundle)
./experiments/static/run_cameraready_static_all.sh

# Perturbed-only camera-ready sweep (default: lines,circles,ellipses)
./experiments/static/run_cameraready_static_perturbed.sh

# Perturbed sweeps for all Section 6 static tests
ONLY=lines,circles,ellipses,squares,zalesak ./experiments/static/run_cameraready_static_perturbed.sh

# Cartesian-only camera-ready sweep (default: squares,zalesak)
./experiments/static/run_cameraready_static_cartesian.sh

# Bundle/sync existing outputs into a release folder
./experiments/static/bundle_static_cameraready_release.sh

# Retro-wire already finished outputs (no reruns)
./experiments/static/retro_wire_static_cameraready_existing.sh

# Optional: disable Slack notifications for a run
NOTIFY=0 ./experiments/static/run_cameraready_static_all.sh
```

### Line Reconstruction
```bash
# Run parameter sweep
python3 -m experiments.static.lines --config static/line --sweep

# Run single test
python3 -m experiments.static.lines --config static/line --facet_algo linear --save_name line_linear
```

### Square Reconstruction
```bash
# Run parameter sweep
python3 -m experiments.static.squares --config static/square --sweep

# Run single test
python3 -m experiments.static.squares --config static/square --facet_algo circular --save_name square_mergecircle
```

To plot only from saved results:
```bash
python -m experiments.static.squares --plot_only --results_file results/static/square_reconstruction_results.txt
```

### Circle Reconstruction
```bash
# Run parameter sweep
python3 -m experiments.static.circles --config static/circle --sweep

# Run single test
python3 -m experiments.static.circles --config static/circle --facet_algo circular --save_name circle_mergecircle
```

### Ellipse Reconstruction
```bash
# Run parameter sweep
python3 -m experiments.static.ellipses --config static/ellipse --sweep

# Run single test
python3 -m experiments.static.ellipses --config static/ellipse --facet_algo circular --save_name ellipse_mergecircle
```

Each legacy per-shape sweep tests all configured algorithms, generates metric
plots, and saves both structured diagnostics and compatibility text metrics.

## Line Reconstruction

The line reconstruction experiment tests the algorithms' ability to reconstruct straight interfaces. Each test case:
- Generates a random line with random orientation
- Places the line in the test domain
- Reconstructs the interface using each algorithm
- Measures reconstruction quality using Hausdorff distance

### Metrics
- **Hausdorff Distance**: Maximum distance between the true line and reconstructed interface, measuring the worst-case reconstruction error

This experiment serves as a baseline test, as lines have zero curvature and should be reconstructed exactly by all algorithms. The Hausdorff distance provides a strict measure of reconstruction accuracy.

## Square Reconstruction

The square reconstruction experiment tests the algorithms' ability to reconstruct piecewise linear interfaces with sharp corners. Each test case:
- Generates a square with side length varying from 10 to 30
- Places the square at a random center with random orientation in the test domain
- Reconstructs the interface using each algorithm
- Measures reconstruction quality using Hausdorff distance, facet gap, and area error

### Metrics
1. **Hausdorff Distance**: Geometric discrepancy between true and reconstructed interfaces
2. **Facet Gap**: Separation between neighboring reconstructed facets
3. **Area Error**: Difference between reconstructed and target mixed-cell areas

This experiment tests the algorithms' ability to handle sharp corners and piecewise linear interfaces, which are common in practical applications.

## Circle Reconstruction

The circle reconstruction experiment tests the algorithms' ability to reconstruct curved interfaces with constant curvature. Each test case:
- Generates a circle with fixed radius (10.0)
- Places the circle at a random center in the test domain
- Reconstructs the interface using each algorithm
- Measures reconstruction quality using curvature error and facet gaps

### Metrics
1. **Curvature Error**: Average absolute difference between reconstructed and true curvature (1/radius)
2. **Facet Gap**: Average minimum distance between adjacent facet endpoints

This experiment is particularly important for evaluating the circular facet algorithms, as circles represent the ideal case for these methods. The constant curvature allows for precise evaluation of the algorithms' ability to capture curvature.

## Ellipse Reconstruction

The ellipse reconstruction experiment tests the algorithms' ability to reconstruct interfaces with varying curvature. Each test case:
- Generates an ellipse with fixed major axis (30.0) and varying aspect ratios (1.5 to 3.0)
- Places the ellipse at a random center with random orientation in the test domain
- Reconstructs the interface using each algorithm
- Measures reconstruction quality using curvature error and facet gaps

### Metrics
1. **Curvature Error**: Average absolute difference between reconstructed and true curvature (varies along the boundary)
2. **Facet Gap**: Average minimum distance between adjacent facet endpoints

This experiment is the most challenging as it combines:
- Varying curvature along the interface
- Non-uniform sampling of the interface
- Orientation-dependent reconstruction quality

## Zalesak Reconstruction (Static)

The Zalesak experiment tests reconstruction of a circle with a vertical slot (asymmetric; one slot edge passes through the circle center). Each test case:
- Samples a random center and random rotation for the slot
- Reconstructs the interface using each algorithm
- Measures reconstruction quality using area error (vs analytic) and facet gaps

```bash
# Run parameter sweep
python3 -m experiments.static.zalesak --config static/zalesak --sweep --num_cases 15

# Run single test
python3 -m experiments.static.zalesak --config static/zalesak --facet_algo circular --save_name zalesak_mergecircle

# Plot only from saved results
python -m experiments.static.zalesak --plot_only --results_file results/static/zalesak_reconstruction_results.txt
```

## Results

Results are presented as log-log plots showing:
1. For lines: Hausdorff distance vs. resolution
2. For circles and ellipses:
   - Average curvature error vs. resolution
   - Average facet gap vs. resolution
3. For squares:
   - Area error vs. resolution
   - Edge alignment error vs. resolution

These plots allow for:
- Comparison of different algorithms' performance
- Analysis of convergence rates
- Identification of optimal resolution ranges for each algorithm
- Evaluation of algorithm robustness across different interface types 
