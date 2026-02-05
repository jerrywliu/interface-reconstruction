"""
Line Reconstruction Experiment

This module performs line reconstruction experiments to evaluate the performance of different
interface reconstruction algorithms on straight line interfaces.

EXPERIMENT OVERVIEW:
- Tests reconstruction of straight lines with varying orientations (0 to 2π)
- Compares different facet reconstruction algorithms: Youngs, LVIRA, linear, and safe_linear
- Evaluates performance using Hausdorff distance between true and reconstructed facets
- Supports both single experiments and comprehensive parameter sweeps

PARAMETER SWEEP FUNCTIONALITY:
When run with --sweep flag, performs a comprehensive parameter sweep across:

1. Mesh Resolution (6 values):
   - Fine resolutions: [0.32, 0.50, 0.64]
   - Coarse resolutions: [1.00, 1.28, 2.00]
   - Tests convergence behavior as resolution increases

2. Facet Reconstruction Algorithms (4 algorithms):
   - Youngs: Classic Youngs' method for interface reconstruction
   - LVIRA: Least Squares Volume-of-Fluid Interface Reconstruction Algorithm
   - linear: Our linear reconstruction method with cell merging
   - safe_linear: Linear reconstruction method without cell merging (faster but potentially less accurate)

ALGORITHM DIFFERENCES:
- safe_linear: Skips cell merging for faster execution, but may be less accurate
- linear: Performs cell merging to improve accuracy at the cost of computational speed

SWEEP EXECUTION:
- For each (resolution, algorithm) combination:
  - Tests 25 different line orientations (angles 0 to 2π)
  - Calculates Hausdorff distances between true and reconstructed facets
  - Averages results across all orientations
- Generates performance plots and results files
- Creates 6×4 grid of experiments (6 resolutions × 4 algorithms)

USAGE:
Single experiment:
    python lines.py --config <config> [--resolution <res>] [--facet_algo <algo>]

Parameter sweep:
    python lines.py --config <config> --sweep [--num_lines <n>]

OUTPUTS:
- line_reconstruction_hausdorff.png: Performance comparison plot
- line_reconstruction_results.txt: Raw numerical results
- Individual experiment outputs in specified save directories
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from main.structs.meshes.merge_mesh import MergeMesh
from main.structs.facets.linear_facet import LinearFacet
from main.geoms.geoms import getPolyLineIntersects
from util.metrics.metrics import hausdorffFacets, calculate_facet_gaps

from util.config import read_yaml
from util.io.setup import setupOutputDirs
from util.reconstruction import runReconstruction
from util.initialize.mesh_factory import make_points_from_config, apply_mesh_overrides
from util.initialize.areas import initializeLine
from util.plotting.plt_utils import plotAreas, plotPartialAreas
from util.plotting.vtk_utils import writeMesh
from util.write_facets import writeFacets

# Global seed for reproducibility
RANDOM_SEED = 42


def create_true_facets_line(x1, y1, x2, y2):
    """
    Create true facets for a line geometry.

    Args:
        x1, y1, x2, y2: Endpoints of the line

    Returns:
        List with single LinearFacet object representing the line
    """
    return [LinearFacet([x1, y1], [x2, y2])]


def create_performance_plot(
    resolutions,
    results,
    title="Line Static Reconstruction",
    ylabel="Average Hausdorff Distance",
    save_path="results/static/line_reconstruction_hausdorff.png",
):
    """
    Create a performance comparison plot from results data.

    Args:
        resolutions: List of resolution values
        results: Dictionary of algorithm results
        title: Plot title
        ylabel: Y-axis label
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

    plt.figure(figsize=(8, 6))

    # Convert resolutions to integers (100*r) and flip x-axis
    x_values = [int(100 * r) for r in resolutions]

    # Plot algorithms with better labels and styling
    for algo, values in results.items():
        if algo == "safe_linear":
            plt.plot(
                x_values,
                values,
                marker="o",
                label="Ours (no merging)",
                linewidth=2.5,
                markersize=8,
                linestyle="-",
            )
        elif algo == "linear":
            plt.plot(
                x_values,
                values,
                marker="s",
                label="Ours (with merging)",
                linewidth=2.5,
                markersize=8,
                linestyle="--",
            )
        else:
            plt.plot(
                x_values, values, marker="o", label=algo, linewidth=2.5, markersize=8
            )

    plt.xscale("log", base=2)
    plt.xlabel(r"Resolution", fontsize=14)
    plt.yscale("log")
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=False,
        loc="center left",
        bbox_to_anchor=(0.02, 0.4),
    )
    plt.grid(True, which="both", ls="-", alpha=0.3)

    # Set x-axis ticks to show resolution values as integers
    plt.xticks(x_values, [str(x) for x in x_values])

    # Add minor grid
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
        tuple: (resolutions, results_dict)
    """
    with open(file_path, "r") as f:
        content = f.read()

    # Parse the file content
    lines = content.strip().split("\n")

    # Extract resolutions
    resolutions_line = lines[0]
    resolutions_str = resolutions_line.split("Resolutions: ")[1]
    resolutions = eval(resolutions_str)  # Safe for this specific format

    # Extract results
    results_line = lines[1]
    results_str = results_line.split("Results: ")[1]
    results = eval(results_str)  # Safe for this specific format

    return resolutions, results


def plot_from_results_file(file_path="results/static/line_reconstruction_results.txt"):
    """
    Load results from file and create performance plot.

    Args:
        file_path: Path to the results file (default: line_reconstruction_results.txt)
    """
    try:
        resolutions, results = load_results_from_file(file_path)
        create_performance_plot(resolutions, results)
        print(f"Plot created from {file_path}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"Error loading results: {e}")


def main(
    config_setting,
    resolution=None,
    facet_algo=None,
    save_name=None,
    num_lines=25,
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

    # Initialize metrics files with headers
    with open(os.path.join(output_dirs["metrics"], "hausdorff.txt"), "w") as f:
        f.write("# Line reconstruction Hausdorff distances\n")
        f.write("# Format: line_{line_number}_angle_{angle}_hausdorff_{distance}\n")
    with open(os.path.join(output_dirs["metrics"], "facet_gap.txt"), "w") as f:
        f.write("# Line reconstruction facet gaps\n")
        f.write("# Format: line_{line_number}_angle_{angle}_gap_{distance}\n")
    with open(os.path.join(output_dirs["metrics"], "facet_details.txt"), "w") as f:
        f.write("# Detailed facet information for each line\n")
        f.write(
            "# Includes true vs reconstructed facet coordinates and individual cell Hausdorff distances\n"
        )

    # Generate lines with different slopes
    angles = np.linspace(0, 2 * np.pi, num_lines + 1)[:-1]

    # Initialize mesh
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
    # Write it once
    writeMesh(m, os.path.join(output_dirs["vtk"], f"mesh.vtk"))

    # Random number generator for reproducibility
    rng = np.random.default_rng(RANDOM_SEED)

    hausdorff_distances = []
    facet_gaps = []

    for i, angle in enumerate(angles):
        print(f"Processing line {i+1}/{num_lines}")

        # Re-initialize mesh
        m = MergeMesh(opoints, threshold)

        # Calculate line endpoints
        x1, y1 = rng.uniform(50, 51), rng.uniform(50, 51)
        x2 = x1 + 0.2  # Small x displacement
        y2 = y1 + np.tan(angle) * (x2 - x1)

        # Initialize line fractions
        fractions = initializeLine(m, [x1, y1], [x2, y2])
        m.initializeFractions(fractions)

        # Plot initial areas
        plotAreas(m, os.path.join(output_dirs["plt_areas"], f"initial_line{i}.png"))
        plotPartialAreas(
            m, os.path.join(output_dirs["plt_partial"], f"initial_line{i}.png")
        )

        # Run reconstruction
        print(f"Reconstructing line {i+1}")
        reconstructed_facets = runReconstruction(
            m,
            facet_algo,
            do_c0,
            i,
            output_dirs,
            algo_kwargs={
                "fit_1neighbor": True
            },  # Fit 1-neighbor to handle boundary cells
        )

        # ---------- Save true facets to VTK ----------
        true_facets = create_true_facets_line(x1, y1, x2, y2)
        writeFacets(
            true_facets,
            os.path.join(output_dirs["vtk_true"], f"true_line{i}.vtp"),
        )

        # Calculate Hausdorff distance
        # For each cell, calculate the true LinearFacet and compare with reconstructed
        avg_hausdorff = 0
        cnt_hausdorff = 0
        for poly, reconstructed_facet in zip(
            m.merged_polys.values(), reconstructed_facets
        ):
            intersects = getPolyLineIntersects(poly.points, [x1, y1], [x2, y2])
            if intersects:
                true_facet = LinearFacet(intersects[0], intersects[-1])
                avg_hausdorff += hausdorffFacets(true_facet, reconstructed_facet)
                cnt_hausdorff += 1
        print(
            f"Average Hausdorff distance for line {i+1}: {avg_hausdorff/cnt_hausdorff:.3e}"
        )

        # Save metric to file with detailed information
        with open(os.path.join(output_dirs["metrics"], "hausdorff.txt"), "a") as f:
            f.write(
                f"line_{i+1}_angle_{angle:.4f}_hausdorff_{avg_hausdorff/cnt_hausdorff:.6e}\n"
            )

        # Save detailed facet information
        with open(os.path.join(output_dirs["metrics"], "facet_details.txt"), "a") as f:
            f.write(f"=== Line {i+1} (angle: {angle:.4f}) ===\n")
            f.write(f"True line: ({x1:.4f}, {y1:.4f}) to ({x2:.4f}, {y2:.4f})\n")
            f.write(f"Average Hausdorff: {avg_hausdorff/cnt_hausdorff:.6e}\n")
            f.write(f"Number of cells with facets: {cnt_hausdorff}\n")

            # Write individual cell facet information
            for j, (poly, reconstructed_facet) in enumerate(
                zip(m.merged_polys.values(), reconstructed_facets)
            ):
                intersects = getPolyLineIntersects(poly.points, [x1, y1], [x2, y2])
                if intersects:
                    true_facet = LinearFacet(intersects[0], intersects[-1])
                    cell_hausdorff = hausdorffFacets(true_facet, reconstructed_facet)
                    f.write(
                        f"  Cell {j}: Hausdorff={cell_hausdorff:.6e}, "
                        f"True=({true_facet.pLeft[0]:.4f},{true_facet.pLeft[1]:.4f})-({true_facet.pRight[0]:.4f},{true_facet.pRight[1]:.4f}), "
                        f"Reconstructed=({reconstructed_facet.pLeft[0]:.4f},{reconstructed_facet.pLeft[1]:.4f})-({reconstructed_facet.pRight[0]:.4f},{reconstructed_facet.pRight[1]:.4f})\n"
                    )
            f.write("\n")

        hausdorff_distances.append(avg_hausdorff / cnt_hausdorff)

        # Calculate facet gap
        avg_gap = calculate_facet_gaps(m, reconstructed_facets)
        print(f"Average facet gap for line {i+1}: {avg_gap:.3e}")
        with open(os.path.join(output_dirs["metrics"], "facet_gap.txt"), "a") as f:
            f.write(f"line_{i+1}_angle_{angle:.4f}_gap_{avg_gap:.6e}\n")
        facet_gaps.append(avg_gap)

    return hausdorff_distances, facet_gaps


def run_parameter_sweep(config_setting, num_lines=25):
    MIN_ERROR = 1e-14

    # Define parameter ranges
    resolutions = [0.32, 0.50, 0.64, 1.00, 1.28, 1.50]
    facet_algos = ["Youngs", "LVIRA", "safe_linear", "linear"]
    save_names = [
        "line_youngs",
        "line_lvira",
        "line_safelinear",
        "line_mergelinear",
    ]

    # Store results
    results = {algo: [] for algo in facet_algos}
    gap_results = {algo: [] for algo in facet_algos}

    # Run experiments
    for resolution in resolutions:
        print(f"\nRunning experiments for resolution {resolution}")
        for algo, save_name in zip(facet_algos, save_names):
            print(f"Testing {algo} algorithm...")
            hausdorff, gaps = main(
                config_setting=config_setting,
                resolution=resolution,
                facet_algo=algo,
                save_name=save_name,
                num_lines=num_lines,
            )
            results[algo].append(max(np.mean(np.array(hausdorff)), MIN_ERROR))
            gap_results[algo].append(max(np.mean(np.array(gaps)), MIN_ERROR))

    # Create summary plot
    create_performance_plot(resolutions, results)

    # Add a facet gap plot if desired
    plt.figure(figsize=(8, 6))
    x_values = [int(100 * r) for r in resolutions]
    for algo, values in gap_results.items():
        plt.plot(x_values, values, marker="o", label=algo, linewidth=2.5, markersize=8)
    plt.xscale("log", base=2)
    plt.xlabel(r"Resolution", fontsize=14)
    plt.yscale("log")
    plt.ylabel("Average Facet Gap", fontsize=14)
    plt.title("Facet Gap vs. Resolution", fontsize=16, fontweight="bold")
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
    plt.savefig(
        "results/static/line_reconstruction_facet_gap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Dump results to file
    with open("results/static/line_reconstruction_results.txt", "w") as f:
        f.write(f"Resolutions: {resolutions}\n")
        f.write(f"Results: {results}\n")
        f.write(f"Facet Gaps: {gap_results}\n")

    return results, gap_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Line reconstruction tests.")
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument("--resolution", type=float, help="resolution", required=False)
    parser.add_argument("--facet_algo", type=str, help="facet_algo", required=False)
    parser.add_argument("--save_name", type=str, help="save_name", required=False)
    parser.add_argument(
        "--num_lines", type=int, help="number of lines to test", default=25
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
        "--plot_only",
        action="store_true",
        help="load results and create plot only",
        default=False,
    )
    parser.add_argument(
        "--results_file",
        type=str,
        help="path to results file for plotting",
        default="results/static/line_reconstruction_results.txt",
    )

    args = parser.parse_args()

    if args.plot_only:
        plot_from_results_file(args.results_file)
    elif args.sweep:
        results, gap_results = run_parameter_sweep(args.config, args.num_lines)
        print("\nParameter sweep results:")
        for algo, values in results.items():
            print(f"{algo}: {values}")
        print("\nFacet Gaps:")
        for algo, values in gap_results.items():
            print(f"{algo}: {values}")
    else:
        main(
            config_setting=args.config,
            resolution=args.resolution,
            facet_algo=args.facet_algo,
            save_name=args.save_name,
            num_lines=args.num_lines,
            mesh_type=args.mesh_type,
            perturb_wiggle=args.perturb_wiggle,
            perturb_seed=args.perturb_seed,
            perturb_fix_boundary=args.perturb_fix_boundary,
            perturb_max_tries=args.perturb_max_tries,
            perturb_type=args.perturb_type,
        )
