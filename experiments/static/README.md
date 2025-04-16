# Static Shape Reconstruction Experiments

This directory contains experiments for evaluating interface reconstruction algorithms on static shapes (lines, squares, circles, and ellipses). Each experiment tests different reconstruction algorithms across various resolutions and measures reconstruction quality using appropriate metrics for each shape type.

## Experimental Setup

All experiments use a 100x100 Cartesian grid with varying resolutions. The shapes are centered in the grid cell with corners (50, 50) and (51, 51) to avoid boundary effects. Each experiment uses a fixed random seed (42) for reproducibility.

### Reconstruction Algorithms Tested
- **Youngs**: Standard Youngs' method for interface reconstruction
- **LVIRA**: Least Squares Volume-of-Fluid Interface Reconstruction Algorithm
- **Linear**: Linear facet reconstruction
- **Safe Circle**: Circular facet reconstruction without merging
- **Circular**: Circular facet reconstruction with merging

### Resolution Range
Experiments are conducted at resolutions: [0.32, 0.50, 0.64, 1.00, 1.28, 2.00]

## Running the Experiments

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
python3 -m experiments.static.squares --config static/circle --sweep

# Run single test
python3 -m experiments.static.squares --config static/circle --facet_algo circular --save_name square_mergecircle
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

Each sweep will:
1. Test all algorithms at each resolution
2. Generate plots showing performance metrics
3. Save results to text files for further analysis

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
- Measures reconstruction quality using area error and edge alignment error

### Metrics
1. **Area Error**: Relative difference between reconstructed and true area (side_length²)
2. **Edge Alignment Error**: Average distance between reconstructed facets and true edges

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