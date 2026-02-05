"""
Circle Reconstruction Experiment

This module performs circle reconstruction experiments to evaluate the performance of different
interface reconstruction algorithms on circular interfaces.

EXPERIMENT OVERVIEW:
- Tests reconstruction of circles with varying centers (random positions)
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
  - Tests 25 different circle positions (random centers)
  - Calculates curvature errors and facet gaps
  - Averages results across all circle positions
- Generates performance plots and results files
- Creates 6×6 grid of experiments (6 resolutions × 6 algorithms)

USAGE:
Single experiment:
    python circles.py --config <config> [--resolution <res>] [--facet_algo <algo>]

Parameter sweep:
    python circles.py --config <config> --sweep [--num_circles <n>] [--radius <r>]

OUTPUTS:
- circle_reconstruction_curvature.png: Curvature error comparison plot
- circle_reconstruction_gaps.png: Facet gap comparison plot
- circle_reconstruction_results.txt: Raw numerical results
- Individual experiment outputs in specified save directories
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from main.structs.meshes.merge_mesh import MergeMesh
from main.geoms.circular_facet import getCircleIntersectArea
from main.structs.facets.circular_facet import ArcFacet
from util.metrics.metrics import (
    calculate_facet_gaps,
    hausdorffFacets,
    interface_turning_curvature,
    polyline_average_curvature,
    tangent_error_to_curve,
)
from main.structs.interface import Interface

from util.config import read_yaml
from util.io.setup import setupOutputDirs
from util.reconstruction import runReconstruction
from util.initialize.mesh_factory import make_points_from_config, apply_mesh_overrides
from util.initialize.areas import initializeCircle
from util.plotting.plt_utils import plotAreas, plotPartialAreas
from util.plotting.vtk_utils import writeMesh
from util.write_facets import writeFacets

# Global seed for reproducibility
RANDOM_SEED = 41


def create_true_facets_circle(m, center, radius):
    """
    Create true facets for a circle geometry.

    Args:
        m: MergeMesh object
        center: Circle center [x, y]
        radius: Circle radius

    Returns:
        List of ArcFacet objects representing the circle intersection with mesh cells
    """
    true_facets = []
    for poly in m.merged_polys.values():
        # Get circle intersection with polygon
        _, arcpoints = getCircleIntersectArea(center, radius, poly.points)
        if len(arcpoints) >= 2:
            # Create arc facet for this cell
            true_facets.append(ArcFacet(center, radius, arcpoints[0], arcpoints[-1]))
    return true_facets


def main(
    config_setting,
    resolution=None,
    facet_algo=None,
    save_name=None,
    num_circles=25,
    radius=10.0,  # Fixed radius for all circles
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

    # Store metrics for all circles
    curvature_errors = []
    facet_gaps = []
    hausdorff_distances = []
    tangent_errors = []
    curvature_proxy_errors = []

    for i in range(num_circles):
        print(f"Processing circle {i+1}/{num_circles}")

        # Re-initialize mesh
        m = MergeMesh(opoints, threshold)

        # Random center in [50,50] to [51,51] square
        center = [rng.uniform(50, 51), rng.uniform(50, 51)]

        # Initialize circle fractions
        fractions = initializeCircle(m, center, radius)
        m.initializeFractions(fractions)

        # Plot initial areas
        plotAreas(m, os.path.join(output_dirs["plt_areas"], f"initial_circle{i}.png"))
        plotPartialAreas(
            m, os.path.join(output_dirs["plt_partial"], f"initial_circle{i}.png")
        )

        # Run reconstruction
        print(f"Reconstructing circle {i+1}")
        reconstructed_facets = runReconstruction(
            m,
            facet_algo,
            do_c0,
            i,
            output_dirs,
            algo_kwargs={},
        )

        # ---------- Save true facets to VTK ----------
        true_facets = create_true_facets_circle(m, center, radius)
        writeFacets(
            true_facets,
            os.path.join(output_dirs["vtk_true"], f"true_circle{i}.vtp"),
        )

        # Calculate curvature error
        true_curvature = 1.0 / radius  # Curvature = 1/R
        avg_curvature_error = 0
        cnt_curvature = 0

        # Calculate Hausdorff distance
        total_hausdorff = 0
        cnt_hausdorff = 0

        for poly, reconstructed_facet in zip(
            m.merged_polys.values(), reconstructed_facets
        ):
            # Take absolute error in curvature
            curvature_error = abs(reconstructed_facet.curvature - true_curvature)
            avg_curvature_error += curvature_error
            cnt_curvature += 1

            # Hausdorff distance calculation
            area, arcpoints = getCircleIntersectArea(center, radius, poly.points)
            if len(arcpoints) >= 2:
                true_facet = ArcFacet(center, radius, arcpoints[0], arcpoints[-1])
                hausdorff_dist = hausdorffFacets(true_facet, reconstructed_facet)
                total_hausdorff += hausdorff_dist
                cnt_hausdorff += 1

        avg_error = avg_curvature_error / cnt_curvature
        print(f"Average curvature error for circle {i+1}: {avg_error:.3e}")

        avg_hausdorff = total_hausdorff / cnt_hausdorff if cnt_hausdorff > 0 else 0
        print(f"Average Hausdorff distance for circle {i+1}: {avg_hausdorff:.3e}")

        # Calculate facet gaps
        avg_gap = calculate_facet_gaps(m, reconstructed_facets)
        print(f"Average facet gap for circle {i+1}: {avg_gap:.3e}")

        # Tangent error + curvature proxy error
        sample_count = 256
        true_points = []
        true_tangents = []
        for k in range(sample_count):
            t = 2 * math.pi * k / sample_count
            x = center[0] + radius * math.cos(t)
            y = center[1] + radius * math.sin(t)
            true_points.append([x, y])
            true_tangents.append([-math.sin(t), math.cos(t)])

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

        print(f"Tangent error for circle {i+1}: {tangent_error:.3e}")
        print(
            f"Curvature proxy error for circle {i+1}: {curvature_proxy_error:.3e}"
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


def create_combined_plot(
    resolutions,
    curvature_results,
    gap_results,
    radius=10.0,
    save_path="results/static/circle_reconstruction_combined.png",
):
    """
    Create a combined plot with facet gaps and curvature error subplots.

    Args:
        resolutions: List of resolution values
        curvature_results: Dictionary of curvature error results
        gap_results: Dictionary of facet gap results
        radius: Circle radius for title
        save_path: Path to save the plot
    """
    # Set up matplotlib for better looking plots
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        }
    )

    # Convert resolutions to integers (100*r)
    x_values = [int(100 * r) for r in resolutions]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot facet gaps (left subplot)
    for algo, values in gap_results.items():
        plt.sca(ax1)
        plt.plot(x_values, values, marker="o", label=algo, linewidth=2.5, markersize=8)

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(r"Resolution", fontsize=14)
    ax1.set_yscale("log")
    ax1.set_ylabel("Average Facet Gap", fontsize=14)
    ax1.set_title("Facet Gaps", fontsize=16, fontweight="bold")
    ax1.legend(
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=False,
        loc="center left",
        bbox_to_anchor=(0.02, 0.4),
    )
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    ax1.grid(True, which="minor", ls=":", alpha=0.2)
    ax1.set_xticks(x_values)
    ax1.set_xticklabels([str(x) for x in x_values])

    # Plot curvature errors (right subplot)
    for algo, values in curvature_results.items():
        plt.sca(ax2)
        plt.plot(x_values, values, marker="o", label=algo, linewidth=2.5, markersize=8)

    # Add PLIC reference line (1/r) for curvature
    plic_values = [1 / radius for r in resolutions]
    plt.sca(ax2)
    plt.plot(
        x_values,
        plic_values,
        marker="",
        label="PLIC",
        linewidth=2.5,
        linestyle=":",
        color="gray",
    )

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel(r"Resolution", fontsize=14)
    ax2.set_yscale("log")
    ax2.set_ylabel("Average Curvature Error", fontsize=14)
    ax2.set_title("Curvature", fontsize=16, fontweight="bold")
    ax2.legend(
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=False,
        loc="center left",
        bbox_to_anchor=(0.02, 0.4),
    )
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.grid(True, which="minor", ls=":", alpha=0.2)
    ax2.set_xticks(x_values)
    ax2.set_xticklabels([str(x) for x in x_values])

    plt.suptitle(
        f"Circle Static Reconstruction (Radius = {radius})",
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_hausdorff_plot(
    resolutions,
    hausdorff_results,
    radius=10.0,
    save_path="results/static/circle_reconstruction_hausdorff.png",
):
    """
    Create a plot for Hausdorff distance vs. resolution.
    """
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        }
    )
    plt.figure(figsize=(8, 6))
    x_values = [int(100 * r) for r in resolutions]
    for algo, values in hausdorff_results.items():
        plt.plot(x_values, values, marker="o", label=algo, linewidth=2.5, markersize=8)
    plt.xscale("log", base=2)
    plt.xlabel(r"Resolution", fontsize=14)
    plt.yscale("log")
    plt.ylabel("Average Hausdorff Distance", fontsize=14)
    plt.title(f"Hausdorff Distance (Radius = {radius})", fontsize=16, fontweight="bold")
    plt.legend(
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=False,
        loc="center left",
        bbox_to_anchor=(0.02, 0.4),
    )
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xticks(x_values, [str(x) for x in x_values])
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_tangent_plot(
    resolutions,
    tangent_results,
    radius=10.0,
    save_path="results/static/circle_reconstruction_tangent.png",
):
    plt.figure(figsize=(8, 6))
    x_values = [int(100 * r) for r in resolutions]
    for algo, values in tangent_results.items():
        plt.plot(x_values, values, marker="o", label=algo, linewidth=2.5, markersize=8)
    plt.xscale("log", base=2)
    plt.xlabel(r"Resolution", fontsize=14)
    plt.yscale("log")
    plt.ylabel("Mean Tangent Error (radians)", fontsize=14)
    plt.title("Tangent Error vs. Resolution", fontsize=16, fontweight="bold")
    plt.legend(
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=False,
        loc="center left",
        bbox_to_anchor=(0.02, 0.4),
    )
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xticks(x_values, [str(x) for x in x_values])
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_curvature_proxy_plot(
    resolutions,
    curvature_proxy_results,
    radius=10.0,
    save_path="results/static/circle_reconstruction_curvature_proxy.png",
):
    plt.figure(figsize=(8, 6))
    x_values = [int(100 * r) for r in resolutions]
    for algo, values in curvature_proxy_results.items():
        plt.plot(x_values, values, marker="o", label=algo, linewidth=2.5, markersize=8)
    plt.xscale("log", base=2)
    plt.xlabel(r"Resolution", fontsize=14)
    plt.yscale("log")
    plt.ylabel("Curvature Proxy Error", fontsize=14)
    plt.title("Curvature Proxy Error vs. Resolution", fontsize=16, fontweight="bold")
    plt.legend(
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=False,
        loc="center left",
        bbox_to_anchor=(0.02, 0.4),
    )
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xticks(x_values, [str(x) for x in x_values])
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_results_from_file(file_path):
    """
    Load results from a summary results file.
    """
    with open(file_path, "r") as f:
        content = f.read()
    lines = content.strip().split("\n")
    resolutions_line = lines[0]
    resolutions_str = resolutions_line.split("Resolutions: ")[1]
    resolutions = eval(resolutions_str)
    curvature_line = lines[1]
    curvature_str = curvature_line.split("Curvature Results: ")[1]
    curvature_results = eval(curvature_str)
    gap_line = lines[2]
    gap_str = gap_line.split("Gap Results: ")[1]
    gap_results = eval(gap_str)
    hausdorff_line = lines[3]
    hausdorff_str = hausdorff_line.split("Hausdorff Results: ")[1]
    hausdorff_results = eval(hausdorff_str)
    return resolutions, curvature_results, gap_results, hausdorff_results


def plot_from_results_file(
    file_path="results/static/circle_reconstruction_results.txt", radius=10.0
):
    """
    Load results from file and create combined performance plot.
    """
    try:
        resolutions, curvature_results, gap_results, hausdorff_results = (
            load_results_from_file(file_path)
        )
        create_combined_plot(resolutions, curvature_results, gap_results, radius)
        create_hausdorff_plot(resolutions, hausdorff_results, radius)
        print(f"Combined and Hausdorff plots created from {file_path}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"Error loading results: {e}")


def run_parameter_sweep(config_setting, num_circles=25, radius=10.0):
    MIN_ERROR = 1e-14
    resolutions = [0.32, 0.50, 0.64, 1.00, 1.28, 1.50]
    facet_algos = [
        "Youngs",
        "LVIRA",
        "safe_linear",
        "linear",
        "safe_circle",
        "circular",
    ]
    save_names = [
        "circle_youngs",
        "circle_lvira",
        "circle_safelinear",
        "circle_linear",
        "circle_safecircle",
        "circle_mergecircle",
    ]
    curvature_results = {algo: [] for algo in facet_algos}
    gap_results = {algo: [] for algo in facet_algos}
    hausdorff_results = {algo: [] for algo in facet_algos}
    tangent_results = {algo: [] for algo in facet_algos}
    curvature_proxy_results = {algo: [] for algo in facet_algos}
    for resolution in resolutions:
        print(f"\nRunning experiments for resolution {resolution}")
        for algo, save_name in zip(facet_algos, save_names):
            print(f"Testing {algo} algorithm...")
            errors, gaps, hausdorffs, tangent_errors, curvature_proxy_errors = main(
                config_setting=config_setting,
                resolution=resolution,
                facet_algo=algo,
                save_name=save_name,
                num_circles=num_circles,
                radius=radius,
            )
            curvature_results[algo].append(max(np.mean(np.array(errors)), MIN_ERROR))
            gap_results[algo].append(max(np.mean(np.array(gaps)), MIN_ERROR))
            hausdorff_results[algo].append(
                max(np.mean(np.array(hausdorffs)), MIN_ERROR)
            )
            tangent_results[algo].append(
                max(np.mean(np.array(tangent_errors)), MIN_ERROR)
            )
            curvature_proxy_results[algo].append(
                max(np.mean(np.array(curvature_proxy_errors)), MIN_ERROR)
            )
    create_combined_plot(resolutions, curvature_results, gap_results, radius)
    create_hausdorff_plot(resolutions, hausdorff_results, radius)
    create_tangent_plot(resolutions, tangent_results, radius)
    create_curvature_proxy_plot(resolutions, curvature_proxy_results, radius)
    with open("results/static/circle_reconstruction_results.txt", "w") as f:
        f.write(f"Resolutions: {resolutions}\n")
        f.write(f"Curvature Results: {curvature_results}\n")
        f.write(f"Gap Results: {gap_results}\n")
        f.write(f"Hausdorff Results: {hausdorff_results}\n")
        f.write(f"Tangent Results: {tangent_results}\n")
        f.write(f"Curvature Proxy Results: {curvature_proxy_results}\n")
    return curvature_results, gap_results, hausdorff_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Circle reconstruction tests.")
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument("--resolution", type=float, help="resolution", required=False)
    parser.add_argument("--facet_algo", type=str, help="facet_algo", required=False)
    parser.add_argument("--save_name", type=str, help="save_name", required=False)
    parser.add_argument(
        "--num_circles", type=int, help="number of circles to test", default=25
    )
    parser.add_argument("--radius", type=float, help="circle radius", default=10.0)
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
        "--plot_only",
        action="store_true",
        help="load results and create plot only",
        default=False,
    )
    parser.add_argument(
        "--results_file",
        type=str,
        help="path to results file for plotting",
        default="results/static/circle_reconstruction_results.txt",
    )

    args = parser.parse_args()

    if args.plot_only:
        plot_from_results_file(args.results_file, args.radius)
    elif args.sweep:
        curvature_results, gap_results, hausdorff_results = run_parameter_sweep(
            args.config, args.num_circles, args.radius
        )
        print("\nParameter sweep results:")
        print("\nCurvature Error:")
        for algo, values in curvature_results.items():
            print(f"{algo}: {values}")
        print("\nFacet Gaps:")
        for algo, values in gap_results.items():
            print(f"{algo}: {values}")
        print("\nHausdorff Distance:")
        for algo, values in hausdorff_results.items():
            print(f"{algo}: {values}")
    else:
        main(
            config_setting=args.config,
            resolution=args.resolution,
            facet_algo=args.facet_algo,
            save_name=args.save_name,
            num_circles=args.num_circles,
            radius=args.radius,
            mesh_type=args.mesh_type,
            perturb_wiggle=args.perturb_wiggle,
            perturb_seed=args.perturb_seed,
            perturb_fix_boundary=args.perturb_fix_boundary,
            perturb_max_tries=args.perturb_max_tries,
            perturb_type=args.perturb_type,
        )
