import argparse
import os
import numpy as np

from main.structs.meshes.merge_mesh import MergeMesh

from util.config import read_yaml
from util.io.setup import setupOutputDirs
from util.reconstruction import runReconstruction
from util.initialize.points import makeFineCartesianGrid
from util.initialize.areas import initializeCircle
from util.plotting.plt_utils import plotAreas, plotPartialAreas
from util.plotting.vtk_utils import writeMesh

# Global seed for reproducibility
RANDOM_SEED = 42


def main(
    config_setting,
    resolution=None,
    facet_algo=None,
    save_name=None,
    num_circles=5,
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

    # Generate circles with different radii
    radii = np.linspace(3, 45, num_circles)

    # Initialize mesh once
    print("Generating mesh...")
    opoints = makeFineCartesianGrid(grid_size, resolution)
    m = MergeMesh(opoints, threshold)
    writeMesh(m, os.path.join(output_dirs["vtk"], f"mesh.vtk"))

    # Random number generator for reproducibility
    rng = np.random.default_rng(RANDOM_SEED)

    for i, radius in enumerate(radii):
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

        print(
            f"Average curvature error for circle {i+1}: {avg_curvature_error/cnt_curvature:.3f}"
        )

        # Save metric to file
        with open(
            os.path.join(output_dirs["metrics"], "curvature_error.txt"), "a"
        ) as f:
            f.write(f"{avg_curvature_error/cnt_curvature}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Circle reconstruction tests.")
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument("--resolution", type=float, help="resolution", required=False)
    parser.add_argument("--facet_algo", type=str, help="facet_algo", required=False)
    parser.add_argument("--save_name", type=str, help="save_name", required=False)
    parser.add_argument(
        "--num_circles", type=int, help="number of circles to test", default=5
    )

    args = parser.parse_args()

    main(
        config_setting=args.config,
        resolution=args.resolution,
        facet_algo=args.facet_algo,
        save_name=args.save_name,
        num_circles=args.num_circles,
    )
