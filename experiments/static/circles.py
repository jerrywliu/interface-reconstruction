import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from main.structs.meshes.merge_mesh import MergeMesh
from main.geoms.geoms import getDistance

from util.config import read_yaml
from util.io.setup import setupOutputDirs
from util.reconstruction import runReconstruction
from util.initialize.points import makeFineCartesianGrid
from util.initialize.areas import initializeCircle
from util.plotting.plt_utils import plotAreas, plotPartialAreas
from util.plotting.vtk_utils import writeMesh

# Global seed for reproducibility
RANDOM_SEED = 42


def calculate_facet_gaps(m, reconstructed_facets):
    """Calculate the minimum distance between left facet endpoint and neighboring facets' endpoints.

    Args:
        m: MergeMesh object
        reconstructed_facets: List of reconstructed facets for each cell

    Returns:
        avg_gap: Average minimum gap distance across all mixed cells
    """
    total_gap = 0
    cnt_gap = 0

    # Create a map from cell index to its reconstructed facet
    merged_polys = list(m.merged_polys.values())

    for i, (poly, facet) in enumerate(
        zip(m.merged_polys.values(), reconstructed_facets)
    ):
        if not facet:  # Skip if no facet
            continue

        # Get left endpoint of current facet
        left_endpoint = facet.pLeft

        # Find all neighboring mixed cells that have facets
        min_gap = float("inf")
        for neighbor in poly.adjacent_polys:
            if neighbor in merged_polys:
                # Get neighbor facet
                i = merged_polys.index(neighbor)
                neighbor_facet = reconstructed_facets[i]
                assert neighbor_facet == neighbor.getFacet()
                # Calculate distance to both endpoints of neighbor's facet
                dist_left = getDistance(left_endpoint, neighbor_facet.pLeft)
                dist_right = getDistance(left_endpoint, neighbor_facet.pRight)
                # Update minimum gap
                min_gap = min(min_gap, dist_left, dist_right)

        if min_gap != float("inf"):
            total_gap += min_gap
            cnt_gap += 1

    return total_gap / cnt_gap if cnt_gap > 0 else 0


def main(
    config_setting,
    resolution=None,
    facet_algo=None,
    save_name=None,
    num_circles=25,
    radius=10.0,  # Fixed radius for all circles
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
    opoints = makeFineCartesianGrid(grid_size, resolution)
    m = MergeMesh(opoints, threshold)
    writeMesh(m, os.path.join(output_dirs["vtk"], f"mesh.vtk"))

    # Random number generator for reproducibility
    rng = np.random.default_rng(RANDOM_SEED)

    # Store metrics for all circles
    curvature_errors = []
    facet_gaps = []

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
            algo_kwargs=None,
        )

        # Calculate curvature error
        true_curvature = 1.0 / radius  # Curvature = 1/R
        avg_curvature_error = 0
        cnt_curvature = 0

        for poly, reconstructed_facet in zip(
            m.merged_polys.values(), reconstructed_facets
        ):
            # Take absolute error in curvature
            curvature_error = abs(reconstructed_facet.curvature - true_curvature)
            avg_curvature_error += curvature_error
            cnt_curvature += 1

        avg_error = avg_curvature_error / cnt_curvature
        print(f"Average curvature error for circle {i+1}: {avg_error:.3e}")

        # Calculate facet gaps
        avg_gap = calculate_facet_gaps(m, reconstructed_facets)
        print(f"Average facet gap for circle {i+1}: {avg_gap:.3e}")

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


def run_parameter_sweep(config_setting, num_circles=25, radius=10.0):
    MIN_ERROR = 1e-14

    # Define parameter ranges
    resolutions = [0.32, 0.50, 0.64, 1.00, 1.28, 2.00]
    facet_algos = ["Youngs", "LVIRA", "linear", "safe_circle", "circular"]
    save_names = [
        "circle_youngs",
        "circle_lvira",
        "circle_linear",
        "circle_safecircle",
        "circle_mergecircle",
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
                num_circles=num_circles,
                radius=radius,
            )
            curvature_results[algo].append(max(np.mean(np.array(errors)), MIN_ERROR))
            gap_results[algo].append(np.mean(np.array(gaps)))

    # Create summary plots
    # Curvature error plot
    plt.figure(figsize=(10, 6))
    for algo in facet_algos:
        plt.plot(resolutions, curvature_results[algo], marker="o", label=algo)

    plt.xscale("log")
    plt.xlabel("Resolution")
    plt.ylabel("Average Curvature Error")
    plt.title(f"Circle Reconstruction Performance (Radius = {radius})")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig("circle_reconstruction_curvature.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Facet gap plot
    plt.figure(figsize=(10, 6))
    for algo in facet_algos:
        plt.plot(resolutions, gap_results[algo], marker="o", label=algo)

    plt.xscale("log")
    plt.xlabel("Resolution")
    plt.ylabel("Average Facet Gap")
    plt.title(f"Circle Reconstruction Facet Gaps (Radius = {radius})")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig("circle_reconstruction_gaps.png", dpi=300, bbox_inches="tight")
    plt.close()

    return curvature_results, gap_results


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
    parser.add_argument(
        "--sweep", action="store_true", help="run parameter sweep", default=False
    )

    args = parser.parse_args()

    if args.sweep:
        curvature_results, gap_results = run_parameter_sweep(
            args.config, args.num_circles, args.radius
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
            num_circles=args.num_circles,
            radius=args.radius,
        )
