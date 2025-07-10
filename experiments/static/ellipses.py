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
from util.metrics import calculate_facet_gaps

from util.config import read_yaml
from util.io.setup import setupOutputDirs
from util.reconstruction import runReconstruction
from util.initialize.points import makeFineCartesianGrid
from util.initialize.areas import initializeEllipse
from util.plotting.plt_utils import plotAreas, plotPartialAreas
from util.plotting.vtk_utils import writeMesh

# Global seed for reproducibility
RANDOM_SEED = 42


def main(
    config_setting,
    resolution=None,
    facet_algo=None,
    save_name=None,
    num_ellipses=25,
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
    opoints = makeFineCartesianGrid(grid_size, resolution)
    m = MergeMesh(opoints, threshold)
    writeMesh(m, os.path.join(output_dirs["vtk"], f"mesh.vtk"))

    # Random number generator for reproducibility
    rng = np.random.default_rng(RANDOM_SEED)

    # Store metrics for all ellipses
    curvature_errors = []
    facet_gaps = []

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
            algo_kwargs=None,
        )

        # Calculate curvature error
        # For an ellipse, curvature varies along the boundary
        # We'll calculate the average curvature error across all facets
        avg_curvature_error = 0
        cnt_curvature = 0

        for poly, reconstructed_facet in zip(
            m.merged_polys.values(), reconstructed_facets
        ):
            # For each facet, calculate the true curvature at its center
            # The curvature of an ellipse at angle phi is:
            # k(phi) = (a*b) / (a^2 sin^2(phi) + b^2 cos^2(phi))^(3/2)
            # where a is major axis, b is minor axis
            facet_center = [
                (reconstructed_facet.pLeft[0] + reconstructed_facet.pRight[0]) / 2,
                (reconstructed_facet.pLeft[1] + reconstructed_facet.pRight[1]) / 2,
            ]

            # Calculate angle from center to facet center
            dx = facet_center[0] - center[0]
            dy = facet_center[1] - center[1]
            phi = math.atan2(dy, dx) - theta  # Adjust for ellipse rotation

            # Calculate true curvature
            true_curvature = (major_axis * minor_axis) / (
                major_axis**2 * math.sin(phi) ** 2 + minor_axis**2 * math.cos(phi) ** 2
            ) ** (3 / 2)

            # Take absolute error in curvature
            curvature_error = abs(reconstructed_facet.curvature - true_curvature)
            avg_curvature_error += curvature_error
            cnt_curvature += 1

        avg_error = avg_curvature_error / cnt_curvature
        print(f"Average curvature error for ellipse {i+1}: {avg_error:.3e}")

        # Calculate facet gaps
        avg_gap = calculate_facet_gaps(m, reconstructed_facets)
        print(f"Average facet gap for ellipse {i+1}: {avg_gap:.3e}")

        # Save metrics to file
        with open(
            os.path.join(output_dirs["metrics"], "curvature_error.txt"), "a"
        ) as f:
            f.write(f"{avg_error}\n")
        with open(os.path.join(output_dirs["metrics"], "facet_gap.txt"), "a") as f:
            f.write(f"{avg_gap}\n")

        curvature_errors.append(avg_error)
        facet_gaps.append(avg_gap)

    return curvature_errors, facet_gaps


def run_parameter_sweep(config_setting, num_ellipses=25):
    MIN_ERROR = 1e-14

    # Define parameter ranges
    resolutions = [0.32, 0.50, 0.64, 1.00, 1.28, 2.00]
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

    # Run experiments
    for resolution in resolutions:
        print(f"\nRunning experiments for resolution {resolution}")
        for algo, save_name in zip(facet_algos, save_names):
            print(f"Testing {algo} algorithm...")
            errors, gaps = main(
                config_setting=config_setting,
                resolution=resolution,
                facet_algo=algo,
                save_name=save_name,
                num_ellipses=num_ellipses,
            )
            curvature_results[algo].append(max(np.mean(np.array(errors)), MIN_ERROR))
            gap_results[algo].append(max(np.mean(np.array(gaps)), MIN_ERROR))

    # Create summary plots
    # Curvature error plot
    plt.figure(figsize=(10, 6))
    for algo in facet_algos:
        plt.plot(
            [1 / r for r in resolutions],
            curvature_results[algo],
            marker="o",
            label=algo,
        )

    plt.xscale("log", base=2)
    plt.xlabel("1/Resolution")
    plt.yscale("log")
    plt.ylabel("Average Curvature Error")
    plt.title("Ellipse Reconstruction Performance")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    # Set x-axis ticks to powers of 2
    plt.xticks([1 / r for r in resolutions], [f"1/{r:.2f}" for r in resolutions])
    plt.savefig("ellipse_reconstruction_curvature.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Facet gap plot
    plt.figure(figsize=(10, 6))
    for algo in facet_algos:
        plt.plot(
            [1 / r for r in resolutions], gap_results[algo], marker="o", label=algo
        )

    plt.xscale("log", base=2)
    plt.xlabel("1/Resolution")
    plt.yscale("log")
    plt.ylabel("Average Facet Gap")
    plt.title("Ellipse Reconstruction Facet Gaps")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    # Set x-axis ticks to powers of 2
    plt.xticks([1 / r for r in resolutions], [f"1/{r:.2f}" for r in resolutions])
    plt.savefig("ellipse_reconstruction_gaps.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Dump results to file
    with open("ellipse_reconstruction_results.txt", "w") as f:
        f.write(f"Resolutions: {resolutions}\n")
        f.write(f"Curvature Results: {curvature_results}\n")
        f.write(f"Gap Results: {gap_results}\n")

    return curvature_results, gap_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ellipse reconstruction tests.")
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument("--resolution", type=float, help="resolution", required=False)
    parser.add_argument("--facet_algo", type=str, help="facet_algo", required=False)
    parser.add_argument("--save_name", type=str, help="save_name", required=False)
    parser.add_argument(
        "--num_ellipses", type=int, help="number of ellipses to test", default=25
    )
    parser.add_argument(
        "--sweep", action="store_true", help="run parameter sweep", default=False
    )

    args = parser.parse_args()

    if args.sweep:
        curvature_results, gap_results = run_parameter_sweep(
            args.config, args.num_ellipses
        )
        print("\nParameter sweep results:")
        print("\nCurvature Error:")
        for algo, values in curvature_results.items():
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
            num_ellipses=args.num_ellipses,
        )
