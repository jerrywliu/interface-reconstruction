"""
Ellipse Reconstruction Experiment

This module performs ellipse reconstruction experiments to evaluate the performance of different
interface reconstruction algorithms on elliptical interfaces.

EXPERIMENT OVERVIEW:
- Tests reconstruction of ellipses with varying aspect ratios (1.5 to 3.0)
- Compares different facet reconstruction algorithms: Youngs, LVIRA, safe_linear, linear, safe_circle, and circular
- Evaluates performance using curvature error and facet gap measurements
- Supports both single experiments and comprehensive parameter sweeps

PARAMETER SWEEP FUNCTIONALITY:
When run with --sweep flag, performs a comprehensive parameter sweep across:

1. Mesh Resolution (6 values):
   - Fine resolutions: [0.32, 0.50, 0.64]
   - Coarse resolutions: [1.00, 1.28, 2.00]
   - Tests convergence behavior as resolution increases

2. Facet Reconstruction Algorithms (6 algorithms):
   - Youngs: Classic Youngs' method for interface reconstruction
   - LVIRA: Least Squares Volume-of-Fluid Interface Reconstruction Algorithm  
   - safe_linear: Linear reconstruction method without cell merging (faster but potentially less accurate)
   - linear: Our linear reconstruction method with cell merging
   - safe_circle: Circular reconstruction method without cell merging (faster but potentially less accurate)
   - circular: Circular reconstruction method with cell merging for improved accuracy

ALGORITHM DIFFERENCES:
- safe_linear: Skips cell merging for faster execution, but may be less accurate
- linear: Performs cell merging to improve accuracy at the cost of computational speed
- safe_circle: Skips cell merging for faster execution, but may be less accurate
- circular: Performs cell merging for better accuracy in circular reconstructions

SWEEP EXECUTION:
- For each (resolution, algorithm) combination:
  - Tests 25 different ellipse aspect ratios (1.5 to 3.0)
  - Calculates curvature errors and facet gaps
  - Averages results across all ellipse configurations
- Generates performance plots and results files
- Creates 6×6 grid of experiments (6 resolutions × 6 algorithms)

USAGE:
Single experiment:
    python ellipses.py --config <config> [--resolution <res>] [--facet_algo <algo>]

Parameter sweep:
    python ellipses.py --config <config> --sweep [--num_ellipses <n>]

OUTPUTS:
- ellipse_reconstruction_curvature.png: Curvature error comparison plot
- ellipse_reconstruction_gaps.png: Facet gap comparison plot
- ellipse_reconstruction_results.txt: Raw numerical results
- Individual experiment outputs in specified save directories
"""

import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt

from main.structs.meshes.merge_mesh import MergeMesh
from util.metrics.metrics import (
    calculate_facet_gaps,
    hausdorffFacets,
    interface_turning_curvature,
    polyline_average_curvature,
    tangent_error_to_curve,
)
from main.structs.interface import Interface
from main.geoms.circular_facet import getCircleIntersectArea
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.linear_facet import LinearFacet

from util.config import read_yaml
from util.io.setup import setupOutputDirs
from util.reconstruction import runReconstruction
from util.initialize.mesh_factory import make_points_from_config, apply_mesh_overrides
from util.initialize.areas import initializeEllipse
from util.plotting.plt_utils import plotAreas, plotPartialAreas
from util.plotting.vtk_utils import writeMesh

# === Ellipse Hausdorff helpers ===
def get_ellipse_to_circle_matrix(major_axis, minor_axis, theta):
    """Return the 2x2 matrix that maps ellipse space to circle space."""
    return np.linalg.inv(np.array([
        [major_axis * math.cos(theta) ** 2 + minor_axis * math.sin(theta) ** 2,
         (major_axis - minor_axis) * math.cos(theta) * math.sin(theta)],
        [(major_axis - minor_axis) * math.cos(theta) * math.sin(theta),
         major_axis * math.sin(theta) ** 2 + minor_axis * math.cos(theta) ** 2],
    ]))

def get_circle_to_ellipse_matrix(major_axis, minor_axis, theta):
    """Return the 2x2 matrix that maps circle space to ellipse space."""
    return np.array([
        [major_axis * math.cos(theta) ** 2 + minor_axis * math.sin(theta) ** 2,
         (major_axis - minor_axis) * math.cos(theta) * math.sin(theta)],
        [(major_axis - minor_axis) * math.cos(theta) * math.sin(theta),
         major_axis * math.sin(theta) ** 2 + minor_axis * math.cos(theta) ** 2],
    ])

def transform_points(points, matrix, center):
    """Apply a 2x2 matrix and translation to a list of points."""
    arr = np.array(points) - np.array(center)
    return (arr @ matrix.T)  # shape (N,2)

def inverse_transform_points(points, matrix, center):
    """Apply the inverse transform (from circle space back to ellipse space)."""
    arr = np.array(points) @ matrix + np.array(center)
    return arr

def sample_arc_points(center, radius, p1, p2, n=100):
    """Sample n points along the arc from p1 to p2 (in circle space)."""
    # Compute angles
    a1 = math.atan2(p1[1] - center[1], p1[0] - center[0])
    a2 = math.atan2(p2[1] - center[1], p2[0] - center[0])
    # Ensure shortest direction
    if a2 < a1:
        a2 += 2 * math.pi
    angles = np.linspace(a1, a2, n)
    return np.stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ], axis=-1)

def sample_linear_facet_points(facet, n=100):
    return np.linspace(facet.pLeft, facet.pRight, n)

def plot_hausdorff_comparison(true_points, reconstructed_points, title, save_path=None):
    """Plot the true and reconstructed curves for Hausdorff comparison."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    # Plot true curve
    plt.plot(true_points[:, 0], true_points[:, 1], 'b-', linewidth=3, label='True curve', alpha=0.8)
    plt.plot(true_points[0, 0], true_points[0, 1], 'bo', markersize=8, label='True start')
    plt.plot(true_points[-1, 0], true_points[-1, 1], 'bs', markersize=8, label='True end')
    
    # Plot reconstructed curve
    plt.plot(reconstructed_points[:, 0], reconstructed_points[:, 1], 'r--', linewidth=3, label='Reconstructed curve', alpha=0.8)
    plt.plot(reconstructed_points[0, 0], reconstructed_points[0, 1], 'ro', markersize=8, label='Reconstructed start')
    plt.plot(reconstructed_points[-1, 0], reconstructed_points[-1, 1], 'rs', markersize=8, label='Reconstructed end')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def ellipse_hausdorff(mesh, reconstructed_facets, major_axis, minor_axis, theta, center, n=100):
    """
    Compute the average Hausdorff distance between reconstructed facets and true ellipse arcs using the circle trick.
    The Hausdorff distance is computed in the original (ellipse) space.
    """
    ellipse_to_circle = get_ellipse_to_circle_matrix(major_axis, minor_axis, theta)
    circle_to_ellipse = get_circle_to_ellipse_matrix(major_axis, minor_axis, theta)
    total_hausdorff = 0
    cnt_hausdorff = 0
    
    for poly, reconstructed_facet in zip(mesh.merged_polys.values(), reconstructed_facets):
        # Transform cell polygon to circle space
        poly_points_circ = transform_points(poly.points, ellipse_to_circle, center)
        # Get intersection points in circle space
        area, arcpoints = getCircleIntersectArea([0, 0], 1, poly_points_circ)
        if len(arcpoints) >= 2:
            # Sample points along the true arc in circle space
            arc_samples_circ = sample_arc_points([0, 0], 1, arcpoints[0], arcpoints[-1], n)
            # Transform true arc samples back to ellipse space
            arc_samples_ellipse = inverse_transform_points(arc_samples_circ, circle_to_ellipse, center)
            # Sample points along the reconstructed facet in ellipse space
            if isinstance(reconstructed_facet, LinearFacet):
                rec_points_ellipse = np.linspace(reconstructed_facet.pLeft, reconstructed_facet.pRight, n)
            elif isinstance(reconstructed_facet, ArcFacet):
                rec_points_ellipse = sample_arc_points(reconstructed_facet.center, abs(reconstructed_facet.radius), reconstructed_facet.pLeft, reconstructed_facet.pRight, n)
            else:
                continue  # skip unsupported facet types
            # Compute Hausdorff distance in ellipse space
            d1 = np.max([np.min(np.linalg.norm(rec_points_ellipse - p, axis=1)) for p in arc_samples_ellipse])
            d2 = np.max([np.min(np.linalg.norm(arc_samples_ellipse - p, axis=1)) for p in rec_points_ellipse])
            hausdorff = max(d1, d2)
            total_hausdorff += hausdorff
            cnt_hausdorff += 1
    return total_hausdorff / cnt_hausdorff if cnt_hausdorff > 0 else 0

# Global seed for reproducibility
RANDOM_SEED = 42


def main(
    config_setting,
    resolution=None,
    facet_algo=None,
    save_name=None,
    num_ellipses=25,
    mesh_type=None,
    perturb_wiggle=None,
    perturb_seed=None,
    perturb_fix_boundary=None,
    perturb_max_tries=None,
    perturb_type=None,
    **kwargs,
):
    # Read config
    config = read_yaml(f"config/{config_setting}.yaml")

    # Test settings
    save_name = save_name if save_name is not None else config["TEST"]["SAVE_NAME"]
    # Mesh settings
    grid_size = config["MESH"]["GRID_SIZE"]
    resolution = resolution if resolution is not None else config["MESH"]["RESOLUTION"]

    # Area and facet settings
    facet_algo = facet_algo if facet_algo is not None else config["GEOMS"]["FACET_ALGO"]
    threshold = config["GEOMS"]["THRESHOLD"]
    do_c0 = config["GEOMS"]["DO_C0"]

    # Setup output directories
    output_dirs = setupOutputDirs(save_name)

    # Generate ellipses with different aspect ratios
    aspect_ratios = np.linspace(1.5, 3.0, num_ellipses)  # Major axis / minor axis
    major_axis = 30  # Fixed major axis length

    # Initialize mesh once
    print("Generating mesh...")
    if isinstance(perturb_fix_boundary, int):
        perturb_fix_boundary = bool(perturb_fix_boundary)
    mesh_cfg = apply_mesh_overrides(
        config["MESH"],
        resolution=resolution,
        mesh_type=mesh_type,
        perturb_wiggle=perturb_wiggle,
        perturb_seed=perturb_seed,
        perturb_fix_boundary=perturb_fix_boundary,
        perturb_max_tries=perturb_max_tries,
        perturb_type=perturb_type,
    )
    opoints = make_points_from_config(mesh_cfg)
    m = MergeMesh(opoints, threshold)
    writeMesh(m, os.path.join(output_dirs["vtk"], f"mesh.vtk"))

    # Random number generator for reproducibility
    rng = np.random.default_rng(RANDOM_SEED)

    # Store metrics for all ellipses
    curvature_errors = []
    facet_gaps = []
    hausdorff_distances = []
    tangent_errors = []
    curvature_proxy_errors = []

    for i, aspect_ratio in enumerate(aspect_ratios):
        print(f"Processing ellipse {i+1}/{num_ellipses}")

        # Re-initialize mesh
        m = MergeMesh(opoints, threshold)

        # Random center in [50,50] to [51,51] square
        center = [rng.uniform(50, 51), rng.uniform(50, 51)]
        # Random angle between 0 and pi/2
        theta = rng.uniform(0, math.pi / 2)
        # Calculate minor axis based on aspect ratio
        minor_axis = major_axis / aspect_ratio

        # Initialize ellipse fractions
        fractions = initializeEllipse(m, major_axis, minor_axis, theta, center)
        m.initializeFractions(fractions)

        # Plot initial areas
        plotAreas(m, os.path.join(output_dirs["plt_areas"], f"initial_ellipse{i}.png"))
        plotPartialAreas(
            m, os.path.join(output_dirs["plt_partial"], f"initial_ellipse{i}.png")
        )

        # Run reconstruction
        print(f"Reconstructing ellipse {i+1}")
        reconstructed_facets = runReconstruction(
            m,
            facet_algo,
            do_c0,
            i,
            output_dirs,
            algo_kwargs={},
        )

        # Calculate curvature error
        # For an ellipse, curvature varies along the boundary
        # We'll calculate the average curvature error across all facets
        avg_curvature_error = 0
        cnt_curvature = 0

        # Hausdorff distance calculation
        total_hausdorff = 0
        cnt_hausdorff = 0

        for poly, reconstructed_facet in zip(
            m.merged_polys.values(), reconstructed_facets
        ):
            # Curvature error (existing)
            facet_center = [
                (reconstructed_facet.pLeft[0] + reconstructed_facet.pRight[0]) / 2,
                (reconstructed_facet.pLeft[1] + reconstructed_facet.pRight[1]) / 2,
            ]
            dx = facet_center[0] - center[0]
            dy = facet_center[1] - center[1]
            phi = math.atan2(dy, dx) - theta
            true_curvature = (major_axis * minor_axis) / (
                major_axis**2 * math.sin(phi) ** 2 + minor_axis**2 * math.cos(phi) ** 2
            ) ** (3 / 2)
            curvature_error = abs(reconstructed_facet.curvature - true_curvature)
            avg_curvature_error += curvature_error
            cnt_curvature += 1

            # Hausdorff distance calculation (ellipse arc vs reconstructed facet)
            # Use the circle trick: transform cell polygon to circle space, get arcpoints, then back to ellipse space
            ellipse_to_circle = get_ellipse_to_circle_matrix(major_axis, minor_axis, theta)
            circle_to_ellipse = get_circle_to_ellipse_matrix(major_axis, minor_axis, theta)
            poly_points_circ = transform_points(poly.points, ellipse_to_circle, center)
            area, arcpoints = getCircleIntersectArea([0, 0], 1, poly_points_circ)
            if len(arcpoints) >= 2:
                # True arc in ellipse space
                arc_samples_circ = sample_arc_points([0, 0], 1, arcpoints[0], arcpoints[-1], 100)
                arc_samples_ellipse = inverse_transform_points(arc_samples_circ, circle_to_ellipse, center)
                # Reconstructed facet points in ellipse space
                if isinstance(reconstructed_facet, LinearFacet):
                    rec_points_ellipse = np.linspace(reconstructed_facet.pLeft, reconstructed_facet.pRight, 100)
                elif isinstance(reconstructed_facet, ArcFacet):
                    rec_points_ellipse = sample_arc_points(reconstructed_facet.center, abs(reconstructed_facet.radius), reconstructed_facet.pLeft, reconstructed_facet.pRight, 100)
                else:
                    continue
                # Hausdorff distance
                d1 = np.max([np.min(np.linalg.norm(rec_points_ellipse - p, axis=1)) for p in arc_samples_ellipse])
                d2 = np.max([np.min(np.linalg.norm(arc_samples_ellipse - p, axis=1)) for p in rec_points_ellipse])
                hausdorff = max(d1, d2)
                total_hausdorff += hausdorff
                cnt_hausdorff += 1

        avg_error = avg_curvature_error / cnt_curvature
        print(f"Average curvature error for ellipse {i+1}: {avg_error:.3e}")

        avg_hausdorff = total_hausdorff / cnt_hausdorff if cnt_hausdorff > 0 else 0
        print(f"Average Hausdorff distance for ellipse {i+1}: {avg_hausdorff:.3e}")

        # Calculate facet gaps
        avg_gap = calculate_facet_gaps(m, reconstructed_facets)
        print(f"Average facet gap for ellipse {i+1}: {avg_gap:.3e}")

        # Tangent error + curvature proxy error
        sample_count = 256
        true_points = []
        true_tangents = []
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        for k in range(sample_count):
            t = 2 * math.pi * k / sample_count
            x_local = major_axis * math.cos(t)
            y_local = minor_axis * math.sin(t)
            x = center[0] + cos_t * x_local - sin_t * y_local
            y = center[1] + sin_t * x_local + cos_t * y_local
            true_points.append([x, y])

            dx_local = -major_axis * math.sin(t)
            dy_local = minor_axis * math.cos(t)
            dx = cos_t * dx_local - sin_t * dy_local
            dy = sin_t * dx_local + cos_t * dy_local
            true_tangents.append([dx, dy])

        tangent_stats = tangent_error_to_curve(
            reconstructed_facets, true_points, true_tangents, n_per_facet=25
        )
        tangent_error = tangent_stats["mean"]

        interface = Interface.from_merge_mesh(
            m, reconstructed_facets=reconstructed_facets, infer_missing_neighbors=True
        )
        curvature_stats = interface_turning_curvature(interface)
        recon_curvature = curvature_stats["mean"]
        true_curvature = polyline_average_curvature(true_points, closed=True)
        curvature_proxy_error = abs(recon_curvature - true_curvature)

        print(f"Tangent error for ellipse {i+1}: {tangent_error:.3e}")
        print(
            f"Curvature proxy error for ellipse {i+1}: {curvature_proxy_error:.3e}"
        )

        # Save metrics to file
        with open(
            os.path.join(output_dirs["metrics"], "curvature_error.txt"), "a"
        ) as f:
            f.write(f"{avg_error}\n")
        with open(os.path.join(output_dirs["metrics"], "facet_gap.txt"), "a") as f:
            f.write(f"{avg_gap}\n")
        with open(os.path.join(output_dirs["metrics"], "hausdorff.txt"), "a") as f:
            f.write(f"{avg_hausdorff}\n")
        with open(os.path.join(output_dirs["metrics"], "tangent_error.txt"), "a") as f:
            f.write(f"{tangent_error}\n")
        with open(
            os.path.join(output_dirs["metrics"], "curvature_proxy_error.txt"), "a"
        ) as f:
            f.write(f"{curvature_proxy_error}\n")

        curvature_errors.append(avg_error)
        facet_gaps.append(avg_gap)
        hausdorff_distances.append(avg_hausdorff)
        tangent_errors.append(tangent_error)
        curvature_proxy_errors.append(curvature_proxy_error)

    return (
        curvature_errors,
        facet_gaps,
        hausdorff_distances,
        tangent_errors,
        curvature_proxy_errors,
    )


def create_combined_plot(resolutions, curvature_results, gap_results, 
                        save_path="results/static/ellipse_reconstruction_combined.png", drop_resolution_200=False):
    """
    Create a combined plot with facet gaps and curvature error subplots.
    
    Args:
        resolutions: List of resolution values
        curvature_results: Dictionary of curvature error results
        gap_results: Dictionary of facet gap results
        save_path: Path to save the plot
        drop_resolution_200: Whether to drop the highest resolution (200) data point
    """
    # Set up matplotlib for better looking plots
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
    })
    
    # Filter out resolution 200 if requested
    if drop_resolution_200 and len(resolutions) > 0 and resolutions[-1] == 2.00:
        resolutions = resolutions[:-1]
        for algo in curvature_results:
            curvature_results[algo] = curvature_results[algo][:-1]
        for algo in gap_results:
            gap_results[algo] = gap_results[algo][:-1]
    
    # Convert resolutions to integers (100*r)
    x_values = [int(100 * r) for r in resolutions]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot facet gaps (left subplot)
    for algo, values in gap_results.items():
        if algo == "Youngs":
            plt.sca(ax1)
            plt.plot(x_values, values, marker='o', label="Youngs", 
                    linewidth=2.5, markersize=8, linestyle='-')
        elif algo == "LVIRA":
            plt.sca(ax1)
            plt.plot(x_values, values, marker='s', label="LVIRA", 
                    linewidth=2.5, markersize=8, linestyle='--')
        elif algo == "safe_linear":
            plt.sca(ax1)
            plt.plot(x_values, values, marker='o', label="Ours (linear, no merging)", 
                    linewidth=2.5, markersize=8, linestyle='-')
        elif algo == "linear":
            plt.sca(ax1)
            plt.plot(x_values, values, marker='s', label="Ours (linear, with merging)", 
                    linewidth=2.5, markersize=8, linestyle='--')
        elif algo == "safe_circle":
            plt.sca(ax1)
            plt.plot(x_values, values, marker='^', label="Ours (circular, no merging)", 
                    linewidth=2.5, markersize=8, linestyle='-')
        elif algo == "circular":
            plt.sca(ax1)
            plt.plot(x_values, values, marker='v', label="Ours (circular, with merging)", 
                    linewidth=2.5, markersize=8, linestyle='--')
    
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(r"Resolution", fontsize=14)
    ax1.set_yscale("log")
    ax1.set_ylabel("Average Facet Gap", fontsize=14)
    ax1.set_title("Facet Gaps", fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=False, loc='center left', bbox_to_anchor=(0.02, 0.4))
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    ax1.grid(True, which="minor", ls=":", alpha=0.2)
    ax1.set_xticks(x_values)
    ax1.set_xticklabels([str(x) for x in x_values])
    
    # Plot curvature errors (right subplot)
    for algo, values in curvature_results.items():
        if algo == "Youngs":
            plt.sca(ax2)
            plt.plot(x_values, values, marker='o', label="Youngs", 
                    linewidth=2.5, markersize=8, linestyle='-')
        elif algo == "LVIRA":
            plt.sca(ax2)
            plt.plot(x_values, values, marker='s', label="LVIRA", 
                    linewidth=2.5, markersize=8, linestyle='--')
        elif algo == "safe_linear":
            plt.sca(ax2)
            plt.plot(x_values, values, marker='o', label="Ours (linear, no merging)", 
                    linewidth=2.5, markersize=8, linestyle='-')
        elif algo == "linear":
            plt.sca(ax2)
            plt.plot(x_values, values, marker='s', label="Ours (linear, with merging)", 
                    linewidth=2.5, markersize=8, linestyle='--')
        elif algo == "safe_circle":
            plt.sca(ax2)
            plt.plot(x_values, values, marker='^', label="Ours (circular, no merging)", 
                    linewidth=2.5, markersize=8, linestyle='-')
        elif algo == "circular":
            plt.sca(ax2)
            plt.plot(x_values, values, marker='v', label="Ours (circular, with merging)", 
                    linewidth=2.5, markersize=8, linestyle='--')
    
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel(r"Resolution", fontsize=14)
    ax2.set_yscale("log")
    ax2.set_ylabel("Average Curvature Error", fontsize=14)
    ax2.set_title("Curvature", fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=False, loc='center left', bbox_to_anchor=(0.02, 0.4))
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.grid(True, which="minor", ls=":", alpha=0.2)
    ax2.set_xticks(x_values)
    ax2.set_xticklabels([str(x) for x in x_values])
    
    plt.suptitle("Ellipse Static Reconstruction", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_hausdorff_plot(resolutions, hausdorff_results, save_path="results/static/ellipse_reconstruction_hausdorff.png"):
    """
    Create a plot for Hausdorff distance vs. resolution for all algorithms.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
    })
    plt.figure(figsize=(8, 6))
    x_values = [int(100 * r) for r in resolutions]
    for algo, values in hausdorff_results.items():
        plt.plot(x_values, values, marker='o', label=algo, linewidth=2.5, markersize=8)
    plt.xscale("log", base=2)
    plt.xlabel(r"Resolution", fontsize=14)
    plt.yscale("log")
    plt.ylabel("Average Hausdorff Distance", fontsize=14)
    plt.title("Hausdorff Distance (Ellipse)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=False, loc='center left', bbox_to_anchor=(0.02, 0.4))
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xticks(x_values, [str(x) for x in x_values])
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_tangent_plot(
    resolutions,
    tangent_results,
    save_path="results/static/ellipse_reconstruction_tangent.png",
):
    """
    Create a plot for tangent error vs. resolution for all algorithms.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
    })
    plt.figure(figsize=(8, 6))
    x_values = [int(100 * r) for r in resolutions]
    for algo, values in tangent_results.items():
        plt.plot(x_values, values, marker='o', label=algo, linewidth=2.5, markersize=8)
    plt.xscale("log", base=2)
    plt.xlabel(r"Resolution", fontsize=14)
    plt.yscale("log")
    plt.ylabel("Mean Tangent Error (radians)", fontsize=14)
    plt.title("Tangent Error (Ellipse)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=False, loc='center left', bbox_to_anchor=(0.02, 0.4))
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xticks(x_values, [str(x) for x in x_values])
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_curvature_proxy_plot(
    resolutions,
    curvature_proxy_results,
    save_path="results/static/ellipse_reconstruction_curvature_proxy.png",
):
    """
    Create a plot for curvature proxy error vs. resolution for all algorithms.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
    })
    plt.figure(figsize=(8, 6))
    x_values = [int(100 * r) for r in resolutions]
    for algo, values in curvature_proxy_results.items():
        plt.plot(x_values, values, marker='o', label=algo, linewidth=2.5, markersize=8)
    plt.xscale("log", base=2)
    plt.xlabel(r"Resolution", fontsize=14)
    plt.yscale("log")
    plt.ylabel("Curvature Proxy Error", fontsize=14)
    plt.title("Curvature Proxy Error (Ellipse)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=False, loc='center left', bbox_to_anchor=(0.02, 0.4))
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xticks(x_values, [str(x) for x in x_values])
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_results_from_file(file_path):
    """
    Load results from a summary results file.
    
    Args:
        file_path: Path to the results file
        
    Returns:
        tuple: (resolutions, curvature_results, gap_results)
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Parse the file content
    lines = content.strip().split('\n')
    
    # Extract resolutions
    resolutions_line = lines[0]
    resolutions_str = resolutions_line.split('Resolutions: ')[1]
    resolutions = eval(resolutions_str)  # Safe for this specific format
    
    # Extract curvature results
    curvature_line = lines[1]
    curvature_str = curvature_line.split('Curvature Results: ')[1]
    curvature_results = eval(curvature_str)  # Safe for this specific format
    
    # Extract gap results
    gap_line = lines[2]
    gap_str = gap_line.split('Gap Results: ')[1]
    gap_results = eval(gap_str)  # Safe for this specific format
    
    return resolutions, curvature_results, gap_results


def plot_from_results_file(file_path="results/static/ellipse_reconstruction_results.txt", drop_resolution_200=False):
    """
    Load results from file and create combined performance plot and Hausdorff plot.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.read().strip().split('\n')
        resolutions_line = lines[0]
        resolutions_str = resolutions_line.split('Resolutions: ')[1]
        resolutions = eval(resolutions_str)
        curvature_line = lines[1]
        curvature_str = curvature_line.split('Curvature Results: ')[1]
        curvature_results = eval(curvature_str)
        gap_line = lines[2]
        gap_str = gap_line.split('Gap Results: ')[1]
        gap_results = eval(gap_str)
        hausdorff_line = lines[3]
        hausdorff_str = hausdorff_line.split('Hausdorff Results: ')[1]
        hausdorff_results = eval(hausdorff_str)
        create_combined_plot(resolutions, curvature_results, gap_results, drop_resolution_200=drop_resolution_200)
        create_hausdorff_plot(resolutions, hausdorff_results)
        print(f"Combined and Hausdorff plots created from {file_path}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"Error loading results: {e}")


def run_parameter_sweep(config_setting, num_ellipses=25, drop_resolution_200=False):
    MIN_ERROR = 1e-14

    # Define parameter ranges
    resolutions = [0.32, 0.50, 0.64, 1.00, 1.28, 1.50]
    facet_algos = ["Youngs", "LVIRA", "safe_linear", "linear", "safe_circle", "circular"]
    save_names = [
        "ellipse_youngs",
        "ellipse_lvira",
        "ellipse_safelinear",
        "ellipse_linear",
        "ellipse_safecircle",
        "ellipse_mergecircle",
    ]

    # Store results
    curvature_results = {algo: [] for algo in facet_algos}
    gap_results = {algo: [] for algo in facet_algos}
    hausdorff_results = {algo: [] for algo in facet_algos}
    tangent_results = {algo: [] for algo in facet_algos}
    curvature_proxy_results = {algo: [] for algo in facet_algos}

    # Run experiments
    for resolution in resolutions:
        print(f"\nRunning experiments for resolution {resolution}")
        for algo, save_name in zip(facet_algos, save_names):
            print(f"Testing {algo} algorithm...")
            errors, gaps, hausdorffs, tangent_errors, curvature_proxy_errors = main(
                config_setting=config_setting,
                resolution=resolution,
                facet_algo=algo,
                save_name=save_name,
                num_ellipses=num_ellipses,
            )
            curvature_results[algo].append(max(np.mean(np.array(errors)), MIN_ERROR))
            gap_results[algo].append(max(np.mean(np.array(gaps)), MIN_ERROR))
            hausdorff_results[algo].append(max(np.mean(np.array(hausdorffs)), MIN_ERROR))
            tangent_results[algo].append(
                max(np.mean(np.array(tangent_errors)), MIN_ERROR)
            )
            curvature_proxy_results[algo].append(
                max(np.mean(np.array(curvature_proxy_errors)), MIN_ERROR)
            )

    # Create combined plot
    create_combined_plot(resolutions, curvature_results, gap_results, drop_resolution_200=drop_resolution_200)
    create_hausdorff_plot(resolutions, hausdorff_results)
    create_tangent_plot(resolutions, tangent_results)
    create_curvature_proxy_plot(resolutions, curvature_proxy_results)

    # Dump results to file
    with open("results/static/ellipse_reconstruction_results.txt", "w") as f:
        f.write(f"Resolutions: {resolutions}\n")
        f.write(f"Curvature Results: {curvature_results}\n")
        f.write(f"Gap Results: {gap_results}\n")
        f.write(f"Hausdorff Results: {hausdorff_results}\n")
        f.write(f"Tangent Results: {tangent_results}\n")
        f.write(f"Curvature Proxy Results: {curvature_proxy_results}\n")

    return curvature_results, gap_results, hausdorff_results


def test_ellipse_hausdorff():
    print("Running ellipse_hausdorff test (single case)...")
    import numpy as np
    class DummyPolys(dict):
        def values(self):
            return [self[0]]
    class DummyMesh:
        def __init__(self, cell):
            self.merged_polys = DummyPolys({0: cell})

    # Test case parameters (match test_plot_hausdorff_case)
    major_axis = 5.3
    minor_axis = 2.0
    theta = 0.0
    center = [0.0, 0.0]
    poly_points = [[5.0, 0.0], [7.0, 0.0], [7.0, 2.0], [5.0, 2.0]]

    # Intersect the polygon with the reference circle (r=5.5)
    from main.geoms.circular_facet import getCircleIntersectArea
    area_circ, arcpoints_circ = getCircleIntersectArea(center, 5.5, poly_points)
    if len(arcpoints_circ) < 2:
        print("Reference circle does not intersect polygon in two points!")
        return
    # Create the reference circle arc facet
    ref_facet = ArcFacet(center, 5.5, arcpoints_circ[0], arcpoints_circ[-1])

    # Create the mesh and cell for the ellipse
    cell = type('Poly', (), {})()
    cell.points = poly_points
    mesh = DummyMesh(cell)

    # Compute Hausdorff distance between ellipse arc and reference circle arc within the polygon
    reconstructed_facets = [ref_facet]
    hausdorff = ellipse_hausdorff(mesh, reconstructed_facets, major_axis, minor_axis, theta, center)
    print(f"Hausdorff distance (ellipse arc vs reference circle arc within polygon): {hausdorff:.6f}")
    print("Test completed!")


def test_plot_ellipse_arc():
    import numpy as np
    import matplotlib.pyplot as plt
    print("Plotting circle arc and corresponding ellipse arc...")
    # Ellipse parameters
    major_axis = 2.0
    minor_axis = 1.0
    theta = np.pi / 6  # 30 degrees
    center = [1.0, -1.0]
    circle_to_ellipse = get_circle_to_ellipse_matrix(major_axis, minor_axis, theta)

    # Arc endpoints in circle space (unit circle)
    angle1 = np.pi / 4
    angle2 = 3 * np.pi / 4
    n = 100
    arc_points_circ = np.stack([
        np.cos(np.linspace(angle1, angle2, n)),
        np.sin(np.linspace(angle1, angle2, n))
    ], axis=-1)

    # Transform arc points to ellipse space
    arc_points_ellipse = (arc_points_circ @ circle_to_ellipse) + np.array(center)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(arc_points_circ[:, 0], arc_points_circ[:, 1], 'b.-', label='Circle arc')
    plt.gca().set_aspect('equal')
    plt.title('Arc in Circle Space')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(arc_points_ellipse[:, 0], arc_points_ellipse[:, 1], 'r.-', label='Ellipse arc')
    plt.gca().set_aspect('equal')
    plt.title('Arc in Ellipse Space')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/tests/ellipse_arc_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plot saved as results/tests/ellipse_arc_comparison.png")


def test_plot_hausdorff_case():
    """Plot the specific test case to visualize ellipse vs circle points, using only the arc segments within the polygon for both."""
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    print("Plotting Hausdorff test case...")
    
    # Test case parameters
    major_axis = 5.3
    minor_axis = 2.0
    theta = 0.0
    center = [0.0, 0.0]
    
    # Polygon
    poly_points = [[5.0, 0.0], [7.0, 0.0], [7.0, 2.0], [5.0, 2.0]]
    
    # Transform to circle space
    ellipse_to_circle = get_ellipse_to_circle_matrix(major_axis, minor_axis, theta)
    circle_to_ellipse = get_circle_to_ellipse_matrix(major_axis, minor_axis, theta)
    
    poly_circ = transform_points(poly_points, ellipse_to_circle, center)
    
    # Get intersection points for ellipse (unit circle in circle space)
    area, arcpoints = getCircleIntersectArea([0, 0], 1, poly_circ)
    print(f"Ellipse/circle space intersection points: {arcpoints}")
    
    # Sample points along the true arc in circle space
    if len(arcpoints) >= 2:
        arc_samples_circ = sample_arc_points([0, 0], 1, arcpoints[0], arcpoints[-1], 100)
        # Transform back to ellipse space
        arc_samples_ellipse = inverse_transform_points(arc_samples_circ, circle_to_ellipse, center)
        
        # Now do the same for the reference circle (r=5.5)
        # Intersect the polygon with the reference circle
        area_circ, arcpoints_circ = getCircleIntersectArea(center, 5.5, poly_points)
        print(f"Reference circle intersection points: {arcpoints_circ}")
        if len(arcpoints_circ) >= 2:
            arc_samples_circle = sample_arc_points(center, 5.5, arcpoints_circ[0], arcpoints_circ[-1], 100)
        else:
            arc_samples_circle = None
        
        # Create the plot
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Circle space
        plt.subplot(1, 2, 1)
        # Draw unit circle
        circle_theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(circle_theta)
        circle_y = np.sin(circle_theta)
        plt.plot(circle_x, circle_y, 'k-', alpha=0.5, label='Unit circle')
        
        # Draw polygon in circle space
        poly_circ_array = np.array(poly_circ)
        plt.plot(poly_circ_array[:, 0], poly_circ_array[:, 1], 'b-', linewidth=2, label='Transformed polygon')
        plt.plot([poly_circ_array[-1, 0], poly_circ_array[0, 0]], [poly_circ_array[-1, 1], poly_circ_array[0, 1]], 'b-', linewidth=2)
        
        # Draw intersection points
        if arcpoints:
            arcpoints_array = np.array(arcpoints)
            plt.plot(arcpoints_array[:, 0], arcpoints_array[:, 1], 'ro', markersize=8, label='Intersection points')
        
        # Draw sampled arc points
        plt.plot(arc_samples_circ[:, 0], arc_samples_circ[:, 1], 'g.-', linewidth=2, markersize=4, label='Sampled ellipse arc points')
        
        plt.title('Circle Space')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Ellipse space
        plt.subplot(1, 2, 2)
        # Draw ellipse
        ellipse_theta = np.linspace(0, 2*np.pi, 100)
        ellipse_x = center[0] + major_axis * np.cos(ellipse_theta) * np.cos(theta) - minor_axis * np.sin(ellipse_theta) * np.sin(theta)
        ellipse_y = center[1] + major_axis * np.cos(ellipse_theta) * np.sin(theta) + minor_axis * np.sin(ellipse_theta) * np.cos(theta)
        plt.plot(ellipse_x, ellipse_y, 'k-', alpha=0.5, label='Ellipse')
        
        # Draw original polygon
        poly_array = np.array(poly_points)
        plt.plot(poly_array[:, 0], poly_array[:, 1], 'b-', linewidth=2, label='Original polygon')
        plt.plot([poly_array[-1, 0], poly_array[0, 0]], [poly_array[-1, 1], poly_array[0, 1]], 'b-', linewidth=2)
        
        # Draw sampled ellipse arc points
        plt.plot(arc_samples_ellipse[:, 0], arc_samples_ellipse[:, 1], 'g.-', linewidth=2, markersize=4, label='Ellipse arc points')
        
        # Draw the reference circle arc segment and its points
        if arc_samples_circle is not None:
            plt.plot(arc_samples_circle[:, 0], arc_samples_circle[:, 1], 'r.-', linewidth=2, markersize=4, label='Reference circle arc points')
        else:
            # fallback: plot the full circle arc for reference
            circle_arc_theta = np.linspace(-np.pi/2, np.pi/2, 100)
            circle_arc_x = center[0] + 5.5 * np.cos(circle_arc_theta)
            circle_arc_y = center[1] + 5.5 * np.sin(circle_arc_theta)
            plt.plot(circle_arc_x, circle_arc_y, 'r--', linewidth=2, label='Reference circle arc (r=5.5)')
        
        plt.title('Ellipse Space')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs('results/tests', exist_ok=True)
        plt.savefig('results/tests/hausdorff_test_case_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Plot saved as results/tests/hausdorff_test_case_visualization.png")
    else:
        print("No intersection points found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ellipse reconstruction tests.")
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument("--resolution", type=float, help="resolution", required=False)
    parser.add_argument("--facet_algo", type=str, help="facet_algo", required=False)
    parser.add_argument("--save_name", type=str, help="save_name", required=False)
    parser.add_argument(
        "--num_ellipses", type=int, help="number of ellipses to test", default=25
    )
    parser.add_argument("--mesh_type", type=str, help="mesh type override", default=None)
    parser.add_argument(
        "--perturb_wiggle",
        type=float,
        help="perturbation amplitude (fraction of cell size)",
        default=None,
    )
    parser.add_argument(
        "--perturb_seed", type=int, help="perturbation RNG seed", default=None
    )
    parser.add_argument(
        "--perturb_fix_boundary",
        type=int,
        choices=[0, 1],
        help="fix boundary nodes (1=yes, 0=no)",
        default=None,
    )
    parser.add_argument(
        "--perturb_max_tries",
        type=int,
        help="max attempts to generate non-inverted mesh",
        default=None,
    )
    parser.add_argument(
        "--perturb_type",
        type=str,
        help="perturbation type (e.g., random)",
        default=None,
    )
    parser.add_argument(
        "--sweep", action="store_true", help="run parameter sweep", default=False
    )
    parser.add_argument(
        "--plot_only", action="store_true", help="load results and create plot only", default=False
    )
    parser.add_argument(
        "--results_file", type=str, help="path to results file for plotting", 
        default="results/static/ellipse_reconstruction_results.txt"
    )
    parser.add_argument(
        "--drop_resolution_200", action="store_true", help="drop the highest resolution (200) data point from plots", 
        default=False
    )
    parser.add_argument(
        "--test_ellipse_hausdorff", action="store_true", help="run ellipse_hausdorff tests and exit", default=False
    )
    parser.add_argument(
        "--test_plot_ellipse_arc", action="store_true", help="plot circle and ellipse arc for visual check", default=False
    )
    parser.add_argument(
        "--test_plot_hausdorff_case", action="store_true", help="plot the specific test case for Hausdorff visualization", default=False
    )

    args = parser.parse_args()

    if args.test_ellipse_hausdorff:
        test_ellipse_hausdorff()
        exit(0)
    if args.test_plot_ellipse_arc:
        test_plot_ellipse_arc()
        exit(0)
    if args.test_plot_hausdorff_case:
        test_plot_hausdorff_case()
        exit(0)

    if args.plot_only:
        plot_from_results_file(args.results_file, args.drop_resolution_200)
    elif args.sweep:
        curvature_results, gap_results, hausdorff_results = run_parameter_sweep(
            args.config, args.num_ellipses, args.drop_resolution_200
        )
        print("\nParameter sweep results:")
        print("\nCurvature Error:")
        for algo, values in curvature_results.items():
            print(f"{algo}: {values}")
        print("\nFacet Gaps:")
        for algo, values in gap_results.items():
            print(f"{algo}: {values}")
        print("\nHausdorff Distances:")
        for algo, values in hausdorff_results.items():
            print(f"{algo}: {values}")
    else:
        main(
            config_setting=args.config,
            resolution=args.resolution,
            facet_algo=args.facet_algo,
            save_name=args.save_name,
            num_ellipses=args.num_ellipses,
            mesh_type=args.mesh_type,
            perturb_wiggle=args.perturb_wiggle,
            perturb_seed=args.perturb_seed,
            perturb_fix_boundary=args.perturb_fix_boundary,
            perturb_max_tries=args.perturb_max_tries,
            perturb_type=args.perturb_type,
        )
