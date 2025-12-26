import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt

from main.structs.meshes.merge_mesh import MergeMesh
from main.geoms.geoms import getArea, getPolyLineArea, getPolyIntersectArea
from main.structs.facets.base_facet import LinearFacet

from util.config import read_yaml
from util.io.setup import setupOutputDirs
from util.reconstruction import runReconstruction
from util.initialize.points import makeFineCartesianGrid
from util.initialize.areas import initializePoly
from util.plotting.plt_utils import plotAreas, plotPartialAreas
from util.plotting.vtk_utils import writeMesh
from util.write_facets import writeFacets

# Global seed for reproducibility
RANDOM_SEED = 42


def true_area_from_polygon_over_mesh(m, polygon):
    """Sum exact polygon∩cell areas (merged cells)."""
    total = 0.0
    for poly in m.merged_polys.values():
        intersects = getPolyIntersectArea(polygon, poly.points)
        if not intersects:
            continue
        for inter in intersects:
            total += abs(getArea(inter))
    return total


def _point_segment_distance(px, py, ax, ay, bx, by):
    """Euclidean distance from point P to segment AB."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    if vv == 0.0:  # degenerate edge
        return math.hypot(px - ax, py - ay)
    t = (wx * vx + wy * vy) / vv
    if t <= 0.0:
        dx, dy = px - ax, py - ay
    elif t >= 1.0:
        dx, dy = px - bx, py - by
    else:
        projx, projy = ax + t * vx, ay + t * vy
        dx, dy = px - projx, py - projy
    return math.hypot(dx, dy)


def _facet_midpoint(pL, pR):
    return 0.5 * (pL[0] + pR[0]), 0.5 * (pL[1] + pR[1])


def compute_edge_alignment_error_for_facet(rotated_square, p_left, p_right):
    """
    Compute the edge alignment error for a single facet defined by endpoints p_left and p_right
    against the nearest edge of the true (possibly rotated) square polygon provided in order.

    Returns:
        float: minimum perpendicular distance from the facet midpoint to any valid projected edge.
               If no valid projection exists on any edge, returns np.inf.
    """
    facet_center = [
        (p_left[0] + p_right[0]) / 2,
        (p_left[1] + p_right[1]) / 2,
    ]
    min_dist = float("inf")
    for j in range(4):
        edge_start = rotated_square[j]
        edge_end = rotated_square[(j + 1) % 4]
        edge_vec = [edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]]
        edge_len = math.sqrt(edge_vec[0] ** 2 + edge_vec[1] ** 2)
        if edge_len == 0:
            continue
        edge_normal = [-edge_vec[1] / edge_len, edge_vec[0] / edge_len]
        proj = (
            (facet_center[0] - edge_start[0]) * edge_vec[0]
            + (facet_center[1] - edge_start[1]) * edge_vec[1]
        ) / edge_len
        if 0 <= proj <= edge_len:
            dist = abs(
                (facet_center[0] - edge_start[0]) * edge_normal[0]
                + (facet_center[1] - edge_start[1]) * edge_normal[1]
            )
            min_dist = min(min_dist, dist)
    return min_dist


def create_true_facets_square(rotated_square):
    """
    Create true facets for a square geometry.

    Args:
        rotated_square: List of 4 vertices [x, y] defining the square

    Returns:
        List of LinearFacet objects representing the 4 edges of the square
    """
    true_facets = []
    for j in range(4):
        edge_start = rotated_square[j]
        edge_end = rotated_square[(j + 1) % 4]
        true_facets.append(LinearFacet(edge_start, edge_end))
    return true_facets


def main(
    config_setting,
    resolution=None,
    facet_algo=None,
    save_name=None,
    num_squares=25,
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

    # Store metrics for all squares
    area_errors = []
    edge_alignment_errors = []

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
            algo_kwargs={},
        )

        # ---------- Save true facets to VTK ----------
        true_facets = create_true_facets_square(rotated_square)
        writeFacets(
            true_facets,
            os.path.join(output_dirs["vtk_true"], f"true_square{i}.vtp"),
        )

        # ---------- Ground-truth area over the mesh (polygon ∩ cells) ----------
        true_total_area = true_area_from_polygon_over_mesh(m, rotated_square)

        # ---------- Reconstructed total area: mixed (via facet) + full (via exact ∩) ----------
        reconstructed_total_area = 0.0
        mixed_ids = set()

        for poly, reconstructed_facet in zip(
            m.merged_polys.values(), reconstructed_facets
        ):
            try:
                reconstructed_total_area += getPolyLineArea(
                    poly.points, reconstructed_facet.pLeft, reconstructed_facet.pRight
                )
                mixed_ids.add(id(poly))
            except Exception:
                # facet might be None for some algos/corner cases
                continue

        # Add area for cells without a facet (fully inside, or otherwise no mixed boundary)
        for poly in m.merged_polys.values():
            if id(poly) in mixed_ids:
                continue
            intersects = getPolyIntersectArea(rotated_square, poly.points)
            if not intersects:
                continue
            inside_area = sum(abs(getArea(p)) for p in intersects)
            reconstructed_total_area += inside_area

        # Final area error (vs exact per-cell truth)
        area_error = abs(reconstructed_total_area - true_total_area) / max(
            true_total_area, 1e-12
        )

        edge_alignment_error_sum = 0.0
        cnt_edges = 0

        for poly, reconstructed_facet in zip(
            m.merged_polys.values(), reconstructed_facets
        ):
            try:
                mx, my = _facet_midpoint(
                    reconstructed_facet.pLeft, reconstructed_facet.pRight
                )
            except Exception:
                continue

            # min distance from facet midpoint to any of the 4 square edges (as segments)
            mind = float("inf")
            for j in range(4):
                ax, ay = rotated_square[j]
                bx, by = rotated_square[(j + 1) % 4]
                d = _point_segment_distance(mx, my, ax, ay, bx, by)
                if d < mind:
                    mind = d

            # Normalize by side length to make scale-free
            edge_alignment_error_sum += mind / max(side_length, 1e-12)
            cnt_edges += 1

        avg_edge_error = edge_alignment_error_sum / max(cnt_edges, 1)

        print(f"Area error for square {i+1}: {area_error:.3e}")
        print(f"Average edge alignment error for square {i+1}: {avg_edge_error:.3e}")

        # Save metrics to file
        with open(os.path.join(output_dirs["metrics"], "area_error.txt"), "a") as f:
            f.write(f"{area_error}\n")

        with open(
            os.path.join(output_dirs["metrics"], "edge_alignment_error.txt"), "a"
        ) as f:
            f.write(f"{avg_edge_error}\n")

        area_errors.append(area_error)
        edge_alignment_errors.append(avg_edge_error)

    return area_errors, edge_alignment_errors


def run_parameter_sweep(config_setting, num_squares=25):
    MIN_ERROR = 1e-14

    # Define parameter ranges
    resolutions = [0.50, 0.64, 1.00, 1.28, 1.50]
    facet_algos = [
        "Youngs",
        "LVIRA",
        "safe_linear",
        "linear",
        # "safe_linear_corner",
        "linear+corner",
        "safe_circle",
        "circular",
    ]
    save_names = [
        "square_youngs",
        "square_lvira",
        "square_safelinear",
        "square_linear",
        # "square_safelinearcorner",
        "square_linear+corner",
        "square_safecircle",
        "square_mergecircle",
    ]

    # Store results
    area_results = {algo: [] for algo in facet_algos}
    edge_results = {algo: [] for algo in facet_algos}

    # Run experiments
    for resolution in resolutions:
        print(f"\nRunning experiments for resolution {resolution}")
        for algo, save_name in zip(facet_algos, save_names):
            print(f"Testing {algo} algorithm...")
            area_errors, edge_errors = main(
                config_setting=config_setting,
                resolution=resolution,
                facet_algo=algo,
                save_name=save_name,
                num_squares=num_squares,
            )
            area_results[algo].append(max(np.mean(np.array(area_errors)), MIN_ERROR))
            edge_results[algo].append(max(np.mean(np.array(edge_errors)), MIN_ERROR))

    # Create summary plots
    # Area error plot
    plt.figure(figsize=(10, 6))
    for algo in facet_algos:
        plt.plot(resolutions, area_results[algo], marker="o", label=algo)

    plt.xscale("log")
    plt.xlabel("Resolution")
    plt.yscale("log")
    plt.ylabel("Average Area Error")
    plt.title("Square Reconstruction Performance")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(
        "results/static/square_reconstruction_area.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Edge alignment error plot
    plt.figure(figsize=(10, 6))
    for algo in facet_algos:
        plt.plot(resolutions, edge_results[algo], marker="o", label=algo)

    plt.xscale("log")
    plt.xlabel("Resolution")
    plt.yscale("log")
    plt.ylabel("Average Edge Alignment Error")
    plt.title("Square Reconstruction Edge Alignment")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(
        "results/static/square_reconstruction_edge.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Dump results to file
    with open("results/static/square_reconstruction_results.txt", "w") as f:
        f.write(f"Resolutions: {resolutions}\n")
        f.write(f"Area Results: {area_results}\n")
        f.write(f"Edge Results: {edge_results}\n")

    return area_results, edge_results


def create_plots(
    resolutions,
    area_results,
    edge_results,
    area_save_path="results/static/square_reconstruction_area.png",
    edge_save_path="results/static/square_reconstruction_edge.png",
):
    # Area error plot
    plt.figure(figsize=(10, 6))
    for algo, values in area_results.items():
        plt.plot(resolutions, values, marker="o", label=algo)
    plt.xscale("log")
    plt.xlabel("Resolution")
    plt.yscale("log")
    plt.ylabel("Average Area Error")
    plt.title("Square Reconstruction Performance")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(area_save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Edge alignment error plot
    plt.figure(figsize=(10, 6))
    for algo, values in edge_results.items():
        plt.plot(resolutions, values, marker="o", label=algo)
    plt.xscale("log")
    plt.xlabel("Resolution")
    plt.yscale("log")
    plt.ylabel("Average Edge Alignment Error")
    plt.title("Square Reconstruction Edge Alignment")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(edge_save_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_results_from_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    lines = content.strip().split("\n")
    resolutions_line = lines[0]
    resolutions_str = resolutions_line.split("Resolutions: ")[1]
    resolutions = eval(resolutions_str)
    area_line = lines[1]
    area_str = area_line.split("Area Results: ")[1]
    area_results = eval(area_str)
    edge_line = lines[2]
    edge_str = edge_line.split("Edge Results: ")[1]
    edge_results = eval(edge_str)
    return resolutions, area_results, edge_results


def plot_from_results_file(
    file_path="results/static/square_reconstruction_results.txt",
):
    try:
        resolutions, area_results, edge_results = load_results_from_file(file_path)
        create_plots(resolutions, area_results, edge_results)
        print(f"Plots created from {file_path}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"Error loading results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Square reconstruction tests.")
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument("--resolution", type=float, help="resolution", required=False)
    parser.add_argument("--facet_algo", type=str, help="facet_algo", required=False)
    parser.add_argument("--save_name", type=str, help="save_name", required=False)
    parser.add_argument(
        "--num_squares", type=int, help="number of squares to test", default=25
    )
    parser.add_argument(
        "--sweep", action="store_true", help="run parameter sweep", default=False
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="load results and create plots only",
        default=False,
    )
    parser.add_argument(
        "--results_file",
        type=str,
        help="path to results file for plotting",
        default="results/static/square_reconstruction_results.txt",
    )
    parser.add_argument(
        "--test_edge_metric",
        action="store_true",
        help="run edge alignment metric tests and exit",
        default=False,
    )

    args = parser.parse_args()

    if args.test_edge_metric:
        # Define a unit square centered at origin (side length 2)
        rot_sq = [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
        # Facet exactly on top edge: expect ~0
        pL1, pR1 = [-0.5, 1.0], [0.5, 1.0]
        d1 = compute_edge_alignment_error_for_facet(rot_sq, pL1, pR1)
        print(f"Top-edge-aligned facet distance: {d1}")
        assert np.isclose(d1, 0.0, atol=1e-12)

        # Facet shifted inward by 0.25 from left edge: expect ~0.25
        pL2, pR2 = [-0.75, -0.5], [-0.75, 0.5]
        d2 = compute_edge_alignment_error_for_facet(rot_sq, pL2, pR2)
        print(f"Left-edge-offset facet distance: {d2}")
        assert np.isclose(d2, 0.25, atol=1e-12)

        print("Edge alignment metric tests passed.")
    elif args.plot_only:
        plot_from_results_file(args.results_file)
    elif args.sweep:
        area_results, edge_results = run_parameter_sweep(args.config, args.num_squares)
        print("\nParameter sweep results:")
        print("\nArea Error:")
        for algo, values in area_results.items():
            print(f"{algo}: {values}")
        print("\nEdge Alignment Error:")
        for algo, values in edge_results.items():
            print(f"{algo}: {values}")
    else:
        main(
            config_setting=args.config,
            resolution=args.resolution,
            facet_algo=args.facet_algo,
            save_name=args.save_name,
            num_squares=args.num_squares,
        )
