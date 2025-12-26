import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from main.structs.meshes.merge_mesh import MergeMesh
from util.metrics.metrics import calculate_facet_gaps
from main.geoms.geoms import getArea, getPolyIntersectArea
from main.geoms.circular_facet import getCircleIntersectArea

from util.config import read_yaml
from util.io.setup import setupOutputDirs
from util.reconstruction import runReconstruction
from util.initialize.points import makeFineCartesianGrid
from util.plotting.plt_utils import plotAreas, plotPartialAreas
from util.plotting.vtk_utils import writeMesh
from util.write_facets import writeFacets
from main.structs.facets.base_facet import LinearFacet, ArcFacet

# Global seed for reproducibility
RANDOM_SEED = 43


def rotate_point_around_center(point, center, theta):
    x, y = point[0] - center[0], point[1] - center[1]
    xr = x * math.cos(theta) - y * math.sin(theta)
    yr = x * math.sin(theta) + y * math.cos(theta)
    return [xr + center[0], yr + center[1]]


def create_true_facets_zalesak(center, radius, slot_rect, theta=0.0):
    """
    Create true facets for Zalesak geometry (circle with rectangle slot).

    The true geometry consists of:
    - A circular arc (the main part of the circle)
    - Three edges of the rectangle slot

    Args:
        center: Circle center [x, y]
        radius: Circle radius
        slot_rect: List of 4 vertices [x, y] defining the slot rectangle
        theta: Rotation angle (already applied to slot_rect)

    Returns:
        List of ArcFacet and LinearFacet objects representing the true interface
    """
    from main.geoms.circular_facet import getCircleLineIntersects

    true_facets = []

    # Get intersection points between rectangle and circle
    # We need to find where each edge of the rectangle intersects the circle
    intersections = []
    for i in range(4):
        p1 = slot_rect[i]
        p2 = slot_rect[(i + 1) % 4]
        edge_intersects = getCircleLineIntersects(
            p1, p2, center, radius, checkWithinLine=True
        )
        if len(edge_intersects) == 2:
            intersections.extend(edge_intersects)

    # If we have exactly 2 intersections, the slot divides the circle
    # Create the large arc from intersection[0] to intersection[1]
    if len(intersections) == 2:
        # Create the large arc (the part of circle not cut by slot)
        true_facets.append(ArcFacet(center, radius, intersections[0], intersections[1]))
    elif len(intersections) > 2:
        # Multiple intersections - handle edge cases
        # For now, just take first two
        true_facets.append(ArcFacet(center, radius, intersections[0], intersections[1]))

    # Add rectangle edges. The slot typically has 3 visible edges in the interface:
    # left (from bottom-left to top-left), top (from top-left to top-right),
    # right (from top-right to bottom-right)
    # The bottom edge is typically below the circle and not part of the interface

    # For a properly aligned Zalesak slot, intersections occur on the left and right edges
    # We want to show the 3 edges: left, top, right
    # Slot rect vertices: [bottom-left, bottom-right, top-right, top-left]

    # Check if we have intersections on left/right edges to determine what to show
    has_intersections = len(intersections) >= 2

    if has_intersections:
        # Add the three visible edges: left, top, right
        true_facets.append(LinearFacet(slot_rect[0], slot_rect[3]))  # left edge
        true_facets.append(LinearFacet(slot_rect[3], slot_rect[2]))  # top edge
        true_facets.append(LinearFacet(slot_rect[2], slot_rect[1]))  # right edge
    else:
        # Fallback: add all edges if no clear intersections
        for i in range(4):
            p1 = slot_rect[i]
            p2 = slot_rect[(i + 1) % 4]
            true_facets.append(LinearFacet(p1, p2))

    return true_facets


def zalesak_removed_area(radius: float, slot_width: float, y_top_rel: float) -> float:
    """
    Removed area for the corrected Zalesak:
      - vertical, axis-aligned slot centered at x = Cx
      - bottom is at or below the bottom rim (y0_rel <= -R)
      - top is strictly inside: y_top_rel in (-R, R)
    Formula: A_rem = W * y_top_rel + a*sqrt(R^2 - a^2) + R^2 * asin(a/R),
      where a = min(max(W/2, 0), R)
    """
    a = min(max(slot_width * 0.5, 0.0), radius)
    return (
        slot_width * y_top_rel
        + a * math.sqrt(radius * radius - a * a)
        + radius * radius * math.asin(a / radius)
    )


def zalesak_total_area(radius: float, slot_width: float, y_top_rel: float) -> float:
    return math.pi * radius * radius - zalesak_removed_area(
        radius, slot_width, y_top_rel
    )


def initialize_zalesak(m, center, radius, slot_width, y_top_rel=10.0, theta=0.0):
    """
    Corrected Zalesak disk:
      - Circle of radius `radius` at `center`.
      - Slot is an axis-aligned vertical rectangle centered horizontally at Cx:
            x in [Cx - W/2, Cx + W/2],
            y in [y_bottom, Cy + y_top_rel],
        with y_bottom chosen at/below the circle bottom so it fully intersects.
      - The rectangle is then rotated by angle `theta` about `center`.
    Assumption (per request): y_top_rel < R (top does not reach the rim).
    """
    Cx, Cy = center
    half_w = slot_width * 0.5

    # Put the bottom at/below the circle’s bottom; exact amount doesn’t affect the intersection
    y_bottom = Cy - radius - 1.0e-6  # strictly below
    y_top = Cy + y_top_rel  # strictly inside the circle by assumption

    # Axis-aligned rectangle before rotation
    rect = [
        [Cx - half_w, y_bottom],
        [Cx + half_w, y_bottom],
        [Cx + half_w, y_top],
        [Cx - half_w, y_top],
    ]
    # Rotate rectangle by theta around center
    rect = [rotate_point_around_center(p, center, theta) for p in rect]

    areas = [[0.0] * len(m.polys[0]) for _ in range(len(m.polys))]
    for ix in range(len(areas)):
        for iy in range(len(areas[0])):
            poly = m.polys[ix][iy]

            # Circle contribution
            c_area, _ = getCircleIntersectArea(center, radius, poly.points)
            cell_area = c_area

            # Subtract slot rectangle overlap
            rect_intersects = getPolyIntersectArea(rect, poly.points)
            for inter in rect_intersects:
                cell_area -= abs(getArea(inter))

            # Clamp and normalize
            areas[ix][iy] = max(0.0, cell_area) / poly.getMaxArea()

    return areas


def main(
    config_setting,
    resolution=None,
    facet_algo=None,
    save_name=None,
    num_cases=25,
    radius=15.0,
    slot_width=5.0,
    slot_top_rel=10.0,
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

    rng = np.random.default_rng(RANDOM_SEED)

    # Store metrics across cases
    area_errors = []
    facet_gaps = []

    # True reference area (top strictly inside, bottom at/below rim)
    true_area = zalesak_total_area(radius, slot_width, slot_top_rel)

    for i in range(num_cases):
        print(f"Processing Zalesak {i+1}/{num_cases}")

        # Re-initialize mesh
        m = MergeMesh(opoints, threshold)

        # Random center and rotation
        center = [rng.uniform(50, 51), rng.uniform(50, 51)]
        theta = rng.uniform(0, math.pi / 2)

        # Initialize Zalesak fractions
        fractions = initialize_zalesak(
            m, center, radius, slot_width, y_top_rel=slot_top_rel, theta=theta
        )
        m.initializeFractions(fractions)

        # Plot initial areas
        plotAreas(m, os.path.join(output_dirs["plt_areas"], f"initial_zalesak{i}.png"))
        plotPartialAreas(
            m, os.path.join(output_dirs["plt_partial"], f"initial_zalesak{i}.png")
        )

        # Run reconstruction
        print(f"Reconstructing Zalesak {i+1}")
        reconstructed_facets = runReconstruction(
            m,
            facet_algo,
            do_c0,
            i,
            output_dirs,
            algo_kwargs={},
        )

        # ---------- Save true facets to VTK ----------
        # Reconstruct the slot rectangle
        Cx, Cy = center
        half_w = slot_width * 0.5
        y_bottom = Cy - radius - 1.0e-6
        y_top = Cy + slot_top_rel
        rect = [
            [Cx - half_w, y_bottom],
            [Cx + half_w, y_bottom],
            [Cx + half_w, y_top],
            [Cx - half_w, y_top],
        ]
        # Rotate rectangle by theta around center
        rect = [rotate_point_around_center(p, center, theta) for p in rect]

        true_facets = create_true_facets_zalesak(center, radius, rect, theta)
        writeFacets(
            true_facets,
            os.path.join(output_dirs["vtk_true"], f"true_zalesak{i}.vtp"),
        )

        # Area error vs analytical area
        reconstructed_total_area = 0.0
        for poly, facet in zip(m.merged_polys.values(), reconstructed_facets):
            # Approximate area from facet by splitting polygon with linear facet endpoints
            try:
                from main.geoms.geoms import getPolyLineArea

                reconstructed_total_area += getPolyLineArea(
                    poly.points, facet.pLeft, facet.pRight
                )
            except Exception:
                continue

        area_error = abs(reconstructed_total_area - true_area) / max(true_area, 1e-12)
        print(f"Area error for case {i+1}: {area_error:.3e}")

        # Facet gaps
        avg_gap = calculate_facet_gaps(m, reconstructed_facets)
        print(f"Average facet gap for case {i+1}: {avg_gap:.3e}")

        # Save metrics
        with open(os.path.join(output_dirs["metrics"], "area_error.txt"), "a") as f:
            f.write(f"{area_error}\n")
        with open(os.path.join(output_dirs["metrics"], "facet_gap.txt"), "a") as f:
            f.write(f"{avg_gap}\n")

        area_errors.append(area_error)
        facet_gaps.append(avg_gap)

    return area_errors, facet_gaps


def create_combined_plot(
    resolutions,
    area_results,
    gap_results,
    save_path="results/static/zalesak_reconstruction_combined.png",
):
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        }
    )

    x_values = [int(100 * r) for r in resolutions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Facet gaps
    for algo, values in gap_results.items():
        plt.sca(ax1)
        plt.plot(x_values, values, marker="o", label=algo)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(r"Resolution", fontsize=14)
    ax1.set_yscale("log")
    ax1.set_ylabel("Average Facet Gap", fontsize=14)
    ax1.set_title("Facet Gaps", fontsize=16, fontweight="bold")
    ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=False, loc="best")
    ax1.grid(True, which="both", ls="-", alpha=0.3)

    # Area errors
    for algo, values in area_results.items():
        plt.sca(ax2)
        plt.plot(x_values, values, marker="o", label=algo)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel(r"Resolution", fontsize=14)
    ax2.set_yscale("log")
    ax2.set_ylabel("Average Area Error", fontsize=14)
    ax2.set_title("Area Error", fontsize=16, fontweight="bold")
    ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=False, loc="best")
    ax2.grid(True, which="both", ls="-", alpha=0.3)

    plt.suptitle("Zalesak Static Reconstruction", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_results_from_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    lines = content.strip().split("\n")
    resolutions_line = lines[0]
    resolutions = eval(resolutions_line.split("Resolutions: ")[1])
    area_results = eval(lines[1].split("Area Results: ")[1])
    gap_results = eval(lines[2].split("Gap Results: ")[1])
    return resolutions, area_results, gap_results


def plot_from_results_file(
    file_path="results/static/zalesak_reconstruction_results.txt",
):
    try:
        resolutions, area_results, gap_results = load_results_from_file(file_path)
        create_combined_plot(resolutions, area_results, gap_results)
        print(f"Combined plot created from {file_path}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"Error loading results: {e}")


def run_parameter_sweep(
    config_setting, num_cases=25, radius=15.0, slot_width=5.0, slot_top_rel=10.0
):
    MIN_ERROR = 1e-14
    resolutions = [0.50, 0.64, 1.00, 1.28, 1.50]
    facet_algos = [
        "Youngs",
        "LVIRA",
        "safe_linear",
        "linear",
        "safe_circle",
        "circular",
        "circular+corner",
    ]
    save_names = [
        "zalesak_youngs",
        "zalesak_lvira",
        "zalesak_safelinear",
        "zalesak_linear",
        "zalesak_safecircle",
        "zalesak_mergecircle",
        "zalesak_ccorner",
    ]
    area_results = {algo: [] for algo in facet_algos}
    gap_results = {algo: [] for algo in facet_algos}
    for resolution in resolutions:
        print(f"\nRunning experiments for resolution {resolution}")
        for algo, save_name in zip(facet_algos, save_names):
            print(f"Testing {algo} algorithm...")
            areas, gaps = main(
                config_setting=config_setting,
                resolution=resolution,
                facet_algo=algo,
                save_name=save_name,
                num_cases=num_cases,
                radius=radius,
                slot_width=slot_width,
                slot_top_rel=slot_top_rel,
            )
            area_results[algo].append(max(np.mean(np.array(areas)), MIN_ERROR))
            gap_results[algo].append(max(np.mean(np.array(gaps)), MIN_ERROR))

    create_combined_plot(resolutions, area_results, gap_results)
    with open("results/static/zalesak_reconstruction_results.txt", "w") as f:
        f.write(f"Resolutions: {resolutions}\n")
        f.write(f"Area Results: {area_results}\n")
        f.write(f"Gap Results: {gap_results}\n")
    return area_results, gap_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zalesak static reconstruction tests.")
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument("--resolution", type=float, help="resolution", required=False)
    parser.add_argument("--facet_algo", type=str, help="facet_algo", required=False)
    parser.add_argument("--save_name", type=str, help="save_name", required=False)
    parser.add_argument(
        "--num_cases", type=int, help="number of randomized cases", default=25
    )
    parser.add_argument("--radius", type=float, help="disk radius", default=15.0)
    parser.add_argument("--slot_width", type=float, help="slot width", default=5.0)
    parser.add_argument(
        "--slot_top_rel", type=float, help="slot top relative to center", default=10.0
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
        default="results/static/zalesak_reconstruction_results.txt",
    )

    args = parser.parse_args()

    if args.plot_only:
        plot_from_results_file(args.results_file)
    elif args.sweep:
        area_results, gap_results = run_parameter_sweep(
            args.config, args.num_cases, args.radius, args.slot_width
        )
        print("\nParameter sweep results:")
        print("\nArea Error:")
        for algo, values in area_results.items():
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
            num_cases=args.num_cases,
            radius=args.radius,
            slot_width=args.slot_width,
            slot_top_rel=args.slot_top_rel,
        )
