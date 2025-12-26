# Static Interface Reconstruction: Experimental Setup

## Abstract

This document describes the comprehensive experimental framework for evaluating interface reconstruction algorithms on static geometries. Our evaluation suite consists of five benchmark test cases designed to systematically assess algorithm performance across different interface characteristics: straight lines, piecewise linear geometries with sharp corners, constant-curvature curved interfaces, varying-curvature elliptical shapes, and complex geometries with both curvature and corners.

## Experimental Infrastructure

### Computational Domain

All experiments are conducted on a uniform Cartesian mesh with the following specifications:
- **Grid size**: 100 × 100 cells
- **Resolution range**: [0.32, 0.50, 0.64, 1.00, 1.28, 1.50, 2.00] (length units per grid cell)
- **Test region**: Central region with corners at (50, 50) and (51, 51), avoiding boundary effects
- **Mixed cell threshold**: 1.0 × 10⁻¹⁰ for volume fraction classification
- **Random seed**: Fixed seed for reproducible results across all test cases

### Test Geometry Sampling

Each experimental setup generates a randomized ensemble of test cases to ensure robust statistical evaluation:
- **Sample size**: 25 configurations per algorithm-resolution combination
- **Randomization**: Center position uniformly sampled in [50, 51] × [50, 51]
- **Orientation**: Random rotation angle for anisotropic geometries (squares, ellipses, Zalesak disk)

### Reconstruction Algorithms

We evaluate seven interface reconstruction algorithms representing different classes of methods:

#### Baseline Methods
1. **Youngs** (`Youngs`): Classical Youngs' PLIC (Piecewise Linear Interface Calculation) method with planar facets and no cell merging
2. **LVIRA** (`LVIRA`): Least Squares Volume-of-Fluid Interface Reconstruction Algorithm with linear facets and least-squares optimization

#### Linear Facet Methods
3. **Safe Linear** (`safe_linear`): Linear facet reconstruction without cell merging for computational efficiency
4. **Linear** (`linear`): Linear facet reconstruction with cell merging for improved accuracy near complex features

#### Circular Facet Methods
5. **Safe Circle** (`safe_circle`): Circular arc facet reconstruction without cell merging
6. **Circular** (`circular`): Circular arc facet reconstruction with cell merging for higher accuracy on curved interfaces
7. **Circular + Corner** (`circular+corner`): Circular facet reconstruction with explicit corner modeling and cell merging

#### Algorithm Selection by Test
- **Lines**: Youngs, LVIRA, safe_linear, linear
- **Squares**: Youngs, LVIRA, safe_linear, linear, linear+corner, safe_circle, circular
- **Circles**: Youngs, LVIRA, safe_linear, linear, safe_circle, circular
- **Ellipses**: Youngs, LVIRA, safe_linear, linear, safe_circle, circular
- **Zalesak**: Youngs, LVIRA, safe_linear, linear, safe_circle, circular, circular+corner

### Evaluation Metrics

We employ geometry-appropriate metrics for each test case:

1. **Area Error**: Relative error between reconstructed and analytical/ground-truth interface area
   - Normalized by true area: |A_reconstructed - A_true| / A_true

2. **Curvature Error**: Mean absolute error in reconstructed interface curvature
   - For circles: MAE with respect to κ = 1/R
   - For ellipses: average over varying curvatures along boundary

3. **Facet Gap**: Average minimum distance between endpoints of adjacent facets
   - Measures continuity and connectivity of reconstructed interface

4. **Edge Alignment Error**: Perpendicular distance from reconstructed facet midpoints to true geometric edges
   - Normalized by feature scale for dimensional consistency

5. **Hausdorff Distance**: Maximum geometric deviation between true and reconstructed interfaces
   - Provides worst-case error bounds for shape fidelity

### Convergence Analysis

Performance is evaluated across multiple resolutions to assess convergence behavior:
- Logarithmic spacing of resolution values
- Expected convergence: O(h²) for area preservation, O(h) or better for geometric fidelity
- Algorithm comparison via convergence rate analysis in log-log plots

---

## Test Case 1: Line Reconstruction

### Purpose

Lines serve as the fundamental baseline test, evaluating algorithms on zero-curvature interfaces where exact reconstruction should be achievable.

### Geometry Specification

- **Shape**: Infinite straight line segment
- **Sampling**: 
  - Line endpoints: p₁ = (x₁, y₁) with x₁, y₁ ∈ [50, 51]
  - Endpoint displacement: p₂ = (x₁ + 0.2, y₁ + tan(θ) × 0.2)
  - Orientation angle: θ ∈ [0, 2π) with 25 uniform samples

### Ground Truth

For each mesh cell, the true line interface is computed analytically as the linear intersection with the cell polygon, yielding exact area fractions.

### Evaluation Metrics

- **Primary**: Hausdorff distance between true line and reconstructed facets
- **Secondary**: Facet gap measurement

### Expected Results

Ideally, all methods should achieve machine-precision accuracy. Deviations indicate numerical stability issues or suboptimal facet fitting in the reconstruction process.

---

## Test Case 2: Square Reconstruction

### Purpose

Squares test reconstruction of piecewise linear geometries with sharp corners and discontinuities in interface orientation.

### Geometry Specification

- **Shape**: Axis-aligned square rotated by random angle θ ∈ [0, π/2)
- **Size variation**: Side length s ∈ [10, 30] with 25 uniform samples
- **Position**: Random center c ∈ [50, 51] × [50, 51]
- **Rotation**: Random angle θ ∈ [0, π/2)

### Ground Truth

The true interface consists of four linear segments forming the square boundary. Reconstruction quality is assessed both globally (total area) and locally (edge alignment).

### Evaluation Metrics

1. **Area Error**: |A_reconstructed - s²| / s²
2. **Edge Alignment Error**: Average perpendicular distance from reconstructed facet midpoints to nearest square edge, normalized by side length

### Expected Results

Methods with cell merging and corner detection should demonstrate superior performance, particularly at coarser resolutions where single-cell facet fitting fails to capture corner geometry.

---

## Test Case 3: Circle Reconstruction

### Purpose

Circles evaluate algorithms on constant-curvature interfaces, serving as the ideal test case for circular arc reconstruction methods.

### Geometry Specification

- **Shape**: Circle with constant curvature
- **Radius**: R = 10.0 (fixed across all test cases)
- **Position**: Random center c ∈ [50, 51] × [50, 51] (25 samples)

### Ground Truth

Analytical circle with κ = 1/R = 0.1. Exact intersections between circular arc and cell polygons computed geometrically.

### Evaluation Metrics

1. **Curvature Error**: MAE of |κ_reconstructed - 1/R|
2. **Facet Gap**: Average minimum distance between adjacent circular facet endpoints
3. **Hausdorff Distance**: Geometric deviation from true circular arc

### Expected Results

Circular facet methods should achieve optimal performance with exact curvature capture. Linear methods will exhibit constant curvature error regardless of resolution due to geometric limitation.

---

## Test Case 4: Ellipse Reconstruction

### Purpose

Ellipses introduce spatially varying curvature, testing algorithm robustness under non-uniform geometric characteristics.

### Geometry Specification

- **Shape**: Ellipse with varying aspect ratio
- **Major axis**: a = 30.0 (fixed)
- **Aspect ratio**: b/a ∈ [1/3, 2/3] (minor axis b ∈ [10, 20])
  - 25 uniform samples of aspect ratio
- **Position**: Random center c ∈ [50, 51] × [50, 51]
- **Orientation**: Random angle θ ∈ [0, π/2)

### Ground Truth

Ellipse boundary defined by parametric form with spatially varying curvature:
- κ(φ) = (ab) / (a²sin²(φ - θ) + b²cos²(φ - θ))³/²
- Arc length parameterization employed for Hausdorff computation

### Evaluation Metrics

1. **Curvature Error**: Average absolute error over all reconstructed facets
2. **Facet Gap**: Connectivity measure across varying interface geometry
3. **Hausdorff Distance**: Computed via ellipse-to-circle coordinate transformation

### Expected Results

Most challenging test case due to combined effects of:
- Curvature variation along interface
- Orientation-dependent sampling
- Potential numerical instability at high-curvature regions

---

## Test Case 5: Zalesak Disk

### Purpose

Zalesak's slotless disk test combines curved and straight interface segments with complex topology, evaluating reconstruction quality at geometric singularities.

### Geometry Specification

- **Base shape**: Circle with R = 15.0
- **Slot geometry**:
  - Width: W = 5.0
  - Orientation: Axis-aligned vertical slot, then rotated by θ ∈ [0, π/2)
  - Top extent: y_top_rel = 10.0 below circle radius
  - Bottom extent: Extends below circle perimeter
- **Position**: Random center c ∈ [50, 51] × [50, 51]
- **Rotation**: Random angle θ ∈ [0, π/2)

### Analytical Ground Truth

For the corrected Zalesak configuration (slot top strictly interior to circle):

A_total = πR² - A_removed

A_removed = W·y_top_rel + a·√(R² - a²) + R²·arcsin(a/R)

where a = min(W/2, R)

### Evaluation Metrics

1. **Area Error**: |A_reconstructed - A_analytical| / A_analytical
2. **Facet Gap**: Critical for continuity across circle-slot junctions

### Expected Results

Tests algorithms at the intersection of curvature (circular arc) and corners (slot edges). Methods with corner modeling and robust merging strategies should outperform baseline linear reconstructions.

---

## Visualization and Output

### VTK Output Structure

Each experiment generates comprehensive visualization data:

```
plots/{save_name}/
  vtk/
    mesh.vtk                                    # Computational mesh
    true/                                       # Ground truth interfaces
      true_geometry{i}.vtp                      
    reconstructed/                              # Algorithm outputs
      facets/
        geometry{i}_algorithm_resolution.vtp    
  plt/                                          # Matplotlib plots
    areas/                                      # Volume fraction fields
      initial_geometry{i}.png
    partial_areas/                              # Mixed cell detail
      initial_geometry{i}.png
  metrics/                                      # Quantitative results
    {metric}.txt                                # Raw numerical data
```

### Comparative Visualization

For publication-ready figures:
- **True interfaces**: Black solid lines (ground truth)
- **Reconstructed interfaces**: Color-coded by algorithm
- **Grid overlay**: Transparent for mesh context
- **Legend**: Algorithm names with line style/color conventions

### Performance Plots

Standard presentation format:
- **X-axis**: Resolution (log scale, base 2)
- **Y-axis**: Error metric (log scale)
- **Multiple curves**: One per algorithm
- **Convergence lines**: Reference slopes for visual comparison
- **Statistics**: Error bars from 25-sample ensemble

---

## Statistical Considerations

### Ensemble Size Justification

Twenty-five random configurations per test:
- Ensures statistical significance while maintaining computational efficiency
- Provides sufficient sampling across orientation and position space
- Balanced trade-off between runtime and robustness

### Metric Robustness

Multi-metric evaluation addresses:
- **Volume conservation**: Area error
- **Geometric fidelity**: Hausdorff distance, edge alignment
- **Differential properties**: Curvature error
- **Topological quality**: Facet gap

### Convergence Criteria

Successful reconstruction requires:
- Monotonic error decrease with resolution refinement
- Convergence rate consistent with theoretical expectations
- Consistent ranking across multiple metrics

---

## Implementation Notes

### Computational Framework

- **Language**: Python 3 with NumPy, SciPy, Matplotlib, VTK
- **Random number generation**: NumPy `default_rng` with fixed seeds per experiment
- **Geometric primitives**: Custom implementations for exact polygon-circle, polygon-line intersections
- **VTK I/O**: Standard VTK XML format for interoperability

### Performance Considerations

- Cell merging: Increases accuracy at ~2-3× computational cost
- Resolution scaling: Memory O(N²), compute O(N² log N) for N×N grid
- Parallelization: Embarrassingly parallel across test configurations

### Reproducibility

Complete reproducibility via:
- Fixed random seeds in all stochastic operations
- Version-controlled configuration files
- Deterministic floating-point arithmetic
- Comprehensive logging of all geometric parameters

---

## Paper Integration

This experimental framework provides direct input for:

1. **Algorithm comparison**: Quantitative ranking via convergence analysis
2. **Method validation**: Robustness across geometric complexity spectrum
3. **Failure mode analysis**: Systematic identification of algorithmic limitations
4. **Computational efficiency**: Performance trade-offs of merging and corner modeling

Suggested paper sections:
- **Experimental Methods**: Copy Test Case descriptions with specifics for your geometries
- **Results**: Use performance plots with convergence analysis
- **Discussion**: Algorithm-appropriate selection based on test case performance
- **Supplementary Material**: Full parameter sweeps and statistical summaries

