import argparse
import os
import numpy as np
import math

from main.structs.meshes.merge_mesh import MergeMesh
from main.geoms.geoms import getArea, getPolyLineArea

from util.config import read_yaml
from util.io.setup import setupOutputDirs
from util.reconstruction import runReconstruction
from util.initialize.points import makeFineCartesianGrid
from util.initialize.areas import initializePoly
from util.plotting.plt_utils import plotAreas, plotPartialAreas
from util.plotting.vtk_utils import writeMesh

# Global seed for reproducibility
RANDOM_SEED = 42


def main(
    config_setting,
    resolution=None,
    facet_algo=None,
    save_name=None,
    num_squares=5,
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

    # Generate squares with different sizes
    side_lengths = np.linspace(10, 30, num_squares)  # Vary side length from 10 to 30

    # Initialize mesh once
    print("Generating mesh...")
    opoints = makeFineCartesianGrid(grid_size, resolution)
    m = MergeMesh(opoints, threshold)
    writeMesh(m, os.path.join(output_dirs["vtk"], f"mesh.vtk"))

    # Random number generator for reproducibility
    rng = np.random.default_rng(RANDOM_SEED)

    for i, side_length in enumerate(side_lengths):
        print(f"Processing square {i+1}/{num_squares}")

        # Re-initialize mesh
        m = MergeMesh(opoints, threshold)

        # Random center in [50,50] to [51,51] square
        center = [rng.uniform(50, 51), rng.uniform(50, 51)]
        # Random angle between 0 and pi/2
        theta = rng.uniform(0, math.pi / 2)

        # Create square vertices
        half_side = side_length / 2
        # Start with unrotated square centered at origin
        square = [
            [-half_side, -half_side],
            [half_side, -half_side],
            [half_side, half_side],
            [-half_side, half_side],
        ]

        # Rotate and translate square
        rotated_square = []
        for point in square:
            # Rotate
            x = point[0] * math.cos(theta) - point[1] * math.sin(theta)
            y = point[0] * math.sin(theta) + point[1] * math.cos(theta)
            # Translate
            rotated_square.append([x + center[0], y + center[1]])

        # Initialize square fractions
        fractions = initializePoly(m, rotated_square)
        m.initializeFractions(fractions)

        # Plot initial areas
        plotAreas(m, os.path.join(output_dirs["plt_areas"], f"initial_square{i}.png"))
        plotPartialAreas(
            m, os.path.join(output_dirs["plt_partial"], f"initial_square{i}.png")
        )

        # Run reconstruction
        print(f"Reconstructing square {i+1}")
        reconstructed_facets = runReconstruction(
            m,
            facet_algo,
            do_c0,
            i,
            output_dirs,
            algo_kwargs=None,
        )

        # Calculate error metrics
        # For squares, we can measure:
        # 1. Area error (total area should be side_length^2)
        # 2. Edge alignment error (how well the reconstructed facets align with the true edges)
        total_area = 0
        edge_alignment_error = 0
        cnt_edges = 0

        for poly, reconstructed_facet in zip(
            m.merged_polys.values(), reconstructed_facets
        ):
            # Calculate area error
            # For linear facets, calculate area using the facet line and polygon
            facet_area = getPolyLineArea(
                poly.points, reconstructed_facet.pLeft, reconstructed_facet.pRight
            )
            total_area += facet_area

            # Calculate edge alignment error
            # For each reconstructed facet, find the closest true edge
            facet_center = [
                (reconstructed_facet.pLeft[0] + reconstructed_facet.pRight[0]) / 2,
                (reconstructed_facet.pLeft[1] + reconstructed_facet.pRight[1]) / 2,
            ]

            # Find closest true edge
            min_dist = float("inf")
            for j in range(4):
                edge_start = rotated_square[j]
                edge_end = rotated_square[(j + 1) % 4]

                # Calculate distance from facet center to edge
                edge_vec = [edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]]
                edge_len = math.sqrt(edge_vec[0] ** 2 + edge_vec[1] ** 2)
                edge_normal = [-edge_vec[1] / edge_len, edge_vec[0] / edge_len]

                # Project facet center onto edge
                proj = (
                    (facet_center[0] - edge_start[0]) * edge_vec[0]
                    + (facet_center[1] - edge_start[1]) * edge_vec[1]
                ) / edge_len

                if 0 <= proj <= edge_len:  # Projection is on the edge
                    dist = abs(
                        (facet_center[0] - edge_start[0]) * edge_normal[0]
                        + (facet_center[1] - edge_start[1]) * edge_normal[1]
                    )
                    min_dist = min(min_dist, dist)

            edge_alignment_error += min_dist
            cnt_edges += 1

        area_error = abs(total_area - side_length**2) / side_length**2
        avg_edge_error = edge_alignment_error / cnt_edges

        print(f"Area error for square {i+1}: {area_error:.3f}")
        print(f"Average edge alignment error for square {i+1}: {avg_edge_error:.3f}")

        # Save metrics to file
        with open(os.path.join(output_dirs["metrics"], "area_error.txt"), "a") as f:
            f.write(f"{area_error}\n")

        with open(
            os.path.join(output_dirs["metrics"], "edge_alignment_error.txt"), "a"
        ) as f:
            f.write(f"{avg_edge_error}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Square reconstruction tests.")
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument("--resolution", type=float, help="resolution", required=False)
    parser.add_argument("--facet_algo", type=str, help="facet_algo", required=False)
    parser.add_argument("--save_name", type=str, help="save_name", required=False)
    parser.add_argument(
        "--num_squares", type=int, help="number of squares to test", default=5
    )

    args = parser.parse_args()

    main(
        config_setting=args.config,
        resolution=args.resolution,
        facet_algo=args.facet_algo,
        save_name=args.save_name,
        num_squares=args.num_squares,
    )
