import argparse
import os
import numpy as np

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


def main(
    config_setting,
    resolution=None,
    facet_algo=None,
    save_name=None,
    num_lines=5,
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

    for i, angle in enumerate(angles):
        print(f"Processing line {i+1}/{num_lines}")

        # Re-initialize mesh
        m = MergeMesh(opoints, threshold)

        # Calculate line endpoints
        x1, y1 = 50.2, 50.3
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
            f"Average Hausdorff distance for line {i+1}: {avg_hausdorff/cnt_hausdorff:.3f}"
        )

        # Save metric to file
        with open(os.path.join(output_dirs["metrics"], "hausdorff.txt"), "a") as f:
            f.write(f"{avg_hausdorff/cnt_hausdorff}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Line reconstruction tests.")
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument("--resolution", type=float, help="resolution", required=False)
    parser.add_argument("--facet_algo", type=str, help="facet_algo", required=False)
    parser.add_argument("--save_name", type=str, help="save_name", required=False)
    parser.add_argument(
        "--num_lines", type=int, help="number of lines to test", default=5
    )

    args = parser.parse_args()

    main(
        config_setting=args.config,
        resolution=args.resolution,
        facet_algo=args.facet_algo,
        save_name=args.save_name,
        num_lines=args.num_lines,
    )
