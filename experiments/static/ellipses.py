import argparse
import os
import numpy as np
import math

from main.structs.meshes.merge_mesh import MergeMesh

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
    num_ellipses=5,
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

        print(
            f"Average curvature error for ellipse {i+1}: {avg_curvature_error/cnt_curvature:.3f}"
        )

        # Save metric to file
        with open(
            os.path.join(output_dirs["metrics"], "curvature_error.txt"), "a"
        ) as f:
            f.write(f"{avg_curvature_error/cnt_curvature}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ellipse reconstruction tests.")
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument("--resolution", type=float, help="resolution", required=False)
    parser.add_argument("--facet_algo", type=str, help="facet_algo", required=False)
    parser.add_argument("--save_name", type=str, help="save_name", required=False)
    parser.add_argument(
        "--num_ellipses", type=int, help="number of ellipses to test", default=5
    )

    args = parser.parse_args()

    main(
        config_setting=args.config,
        resolution=args.resolution,
        facet_algo=args.facet_algo,
        save_name=args.save_name,
        num_ellipses=args.num_ellipses,
    )
