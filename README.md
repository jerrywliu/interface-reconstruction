# Interface Reconstruction

An example implementation in Python of an interface reconstruction method, using linear/circular elements and cusps.

## Contributors

Jerry Liu, jwl50@stanford.edu

## Table of Contents

- [Algorithms](#algorithms)
- [Static Experiments](#static-experiments)
- [Advection Experiments](#advection-experiments)
- [TODO](#todo)

## Algorithms

Our algorithms consist of two main features:
- **Circular facets**
- **Corner facets** (either linear or circular), which requires merging cells

### Supported Algorithms

#### Baselines
- **Youngs**
- **LVIRA**

#### Our Algorithms

**Without Cell Merging (Faster)**
- **safe_linear**: Linear reconstruction method without cell merging
- **safe_circle**: Circular reconstruction method without cell merging

**With Cell Merging (More Accurate)**
- **linear**: Linear reconstruction method with cell merging
- **circular**: Circular reconstruction method with cell merging

## Static Experiments

These experiments test interface reconstruction on various geometric shapes with different algorithms and mesh resolutions.

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

Note: There are currently no dedicated unit tests for square-specific helper functions.

### Zalesak (Static)
```bash
python -m experiments.static.zalesak --config static/circle --sweep --num_cases 15
```
Tests reconstruction of Zalesak's disk (circle with slot) with random centers and random rotations.

To plot:
```bash
python -m experiments.static.zalesak --plot_only --results_file results/static/zalesak_reconstruction_results.txt
```

### TODO
- Randomly generated polygons (corners)
- Pac-man (circular corners)

## Advection Experiments

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

## TODO

Mainly two lingering bugs with the full algorithm:

### Known Issues
1. **Corners**: Sometimes fail to generate, other times generate in completely wrong places
   - Possible causes: threshold used to generate corner, failing to find a proper facet to extend (resolution issue)
2. **Circles**: Very rarely (usually low resolution) will choose the wrong curvature solution
   - Likely because of heuristics used to choose the orientation

### C0 Issues
- C0 doesn't seem to work with merge-less algorithms (e.g., safe-circle)

### Fixes Needed
- Corner threshold adjustment
- `findOrientation()` in MergeMesh needs to be fixed

### Checkpointing Bug
- Bug relating to circular references within MergeMesh
- Likely because of neighbor storing

---

<div align="right"><a href="#table-of-contents">back to top</a></div>