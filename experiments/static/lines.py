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
from main.structs.facets.base_facet import LinearFacet, hausdorffLinearFacets
from main.geoms.geoms import getPolyLineIntersects

from util.config import read_yaml
from util.io.setup import setupOutputDirs
from util.reconstruction import runReconstruction
from util.initialize.points import makeFineCartesianGrid
from util.initialize.areas import initializeLine
from util.plotting.plt_utils import plotAreas, plotPartialAreas
from util.plotting.vtk_utils import writeMesh

# Global seed for reproducibility
RANDOM_SEED = 42


def main(
    config_setting,
    resolution=None,
    facet_algo=None,
    save_name=None,
    num_lines=25,
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

    # Generate lines with different slopes
    angles = np.linspace(0, 2 * np.pi, num_lines + 1)[:-1]

    # Initialize mesh
    print("Generating mesh...")
    opoints = makeFineCartesianGrid(grid_size, resolution)
    m = MergeMesh(opoints, threshold)
    # Write it once
    writeMesh(m, os.path.join(output_dirs["vtk"], f"mesh.vtk"))

    # Random number generator for reproducibility
    rng = np.random.default_rng(RANDOM_SEED)

    hausdorff_distances = []

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
                avg_hausdorff += hausdorffLinearFacets(true_facet, reconstructed_facet)
                cnt_hausdorff += 1
        print(
            f"Average Hausdorff distance for line {i+1}: {avg_hausdorff/cnt_hausdorff:.3e}"
        )

        # Save metric to file
        with open(os.path.join(output_dirs["metrics"], "hausdorff.txt"), "a") as f:
            f.write(f"{avg_hausdorff/cnt_hausdorff}\n")

        hausdorff_distances.append(avg_hausdorff / cnt_hausdorff)

    return hausdorff_distances


def run_parameter_sweep(config_setting, num_lines=25):
    MIN_ERROR = 1e-14

    # Define parameter ranges
    resolutions = [0.32, 0.50, 0.64, 1.00, 1.28, 2.00]
    facet_algos = ["Youngs", "LVIRA", "safe_linear", "linear"]
    save_names = [
        "line_youngs",
        "line_lvira",
        "line_safelinear",
        "line_mergelinear",
    ]

    # Store results
    results = {algo: [] for algo in facet_algos}

    # Run experiments
    for resolution in resolutions:
        print(f"\nRunning experiments for resolution {resolution}")
        for algo, save_name in zip(facet_algos, save_names):
            print(f"Testing {algo} algorithm...")
            hausdorff = main(
                config_setting=config_setting,
                resolution=resolution,
                facet_algo=algo,
                save_name=save_name,
                num_lines=num_lines,
            )
            results[algo].append(max(np.mean(np.array(hausdorff)), MIN_ERROR))

    # Create summary plot
    plt.figure(figsize=(10, 6))
    for algo in facet_algos:
        plt.plot([1 / r for r in resolutions], results[algo], marker="o", label=algo)

    plt.xscale("log", base=2)
    plt.xlabel("1/Resolution")
    plt.yscale("log")
    plt.ylabel("Average Hausdorff Distance")
    plt.title("Line Reconstruction Performance")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    # Set x-axis ticks to powers of 2
    plt.xticks([1 / r for r in resolutions], [f"1/{r:.2f}" for r in resolutions])
    plt.savefig("line_reconstruction_hausdorff.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Dump results to file
    with open("line_reconstruction_results.txt", "w") as f:
        f.write(f"Resolutions: {resolutions}\n")
        f.write(f"Results: {results}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Line reconstruction tests.")
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument("--resolution", type=float, help="resolution", required=False)
    parser.add_argument("--facet_algo", type=str, help="facet_algo", required=False)
    parser.add_argument("--save_name", type=str, help="save_name", required=False)
    parser.add_argument(
        "--num_lines", type=int, help="number of lines to test", default=25
    )
    parser.add_argument(
        "--sweep", action="store_true", help="run parameter sweep", default=False
    )

    args = parser.parse_args()

    if args.sweep:
        results = run_parameter_sweep(args.config, args.num_lines)
        print("\nParameter sweep results:")
        for algo, values in results.items():
            print(f"{algo}: {values}")
    else:
        main(
            config_setting=args.config,
            resolution=args.resolution,
            facet_algo=args.facet_algo,
            save_name=args.save_name,
            num_lines=args.num_lines,
        )
