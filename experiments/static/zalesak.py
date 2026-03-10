import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from main.structs.meshes.merge_mesh import MergeMesh
from main.geoms.geoms import (
    getArea,
    getDistance,
    getPolyIntersectArea,
    getPolyLineArea,
    pointInPoly,
)
from main.geoms.circular_facet import getCircleIntersectArea, getCircleLineIntersects
from main.geoms.corner_facet import getPolyCurvedCornerArea

from util.config import read_yaml
from util.io.setup import setupOutputDirs
from util.reconstruction import runReconstruction
from util.initialize.mesh_factory import make_points_from_config, apply_mesh_overrides
from util.metrics.metrics import calculate_facet_gaps, hausdorff_interface
from util.plotting.plt_utils import plotAreas, plotPartialAreas
from util.plotting.vtk_utils import writeMesh
from util.write_facets import writeFacets
from util.logging.get_arc_facet_logger import arc_facet_log_context
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.corner_facet import CornerFacet
from main.structs.facets.linear_facet import LinearFacet

# Global seed for reproducibility
RANDOM_SEED = 43
ZALESAK_POINT_TOL = 1e-8


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
    def _dedupe_points(points):
        unique = []
        for point in points:
            if not any(getDistance(point, existing) < ZALESAK_POINT_TOL for existing in unique):
                unique.append(list(point))
        return unique

    def _pick_wall_intersection(edge_start, intersections):
        candidates = _dedupe_points(intersections)
        if not candidates:
            return None
        return min(candidates, key=lambda point: getDistance(point, edge_start))

    left_hits = getCircleLineIntersects(
        slot_rect[0], slot_rect[3], center, radius, checkWithinLine=True
    )
    right_hits = getCircleLineIntersects(
        slot_rect[1], slot_rect[2], center, radius, checkWithinLine=True
    )
    left_intersection = _pick_wall_intersection(slot_rect[0], left_hits)
    right_intersection = _pick_wall_intersection(slot_rect[1], right_hits)

    if left_intersection is None or right_intersection is None:
        raise RuntimeError(
            "Failed to build Zalesak true facets: expected one circle-slot intersection on each wall"
        )

    # The correct outer interface is the long circular arc that avoids the slot interior.
    arc_candidates = [
        ArcFacet(center, radius, right_intersection, left_intersection),
        ArcFacet(center, radius, left_intersection, right_intersection),
    ]
    outer_arc = next(
        (candidate for candidate in arc_candidates if not pointInPoly(candidate.midpoint, slot_rect)),
        None,
    )
    if outer_arc is None:
        slot_center = [
            sum(point[0] for point in slot_rect) / len(slot_rect),
            sum(point[1] for point in slot_rect) / len(slot_rect),
        ]
        outer_arc = max(
            arc_candidates,
            key=lambda candidate: getDistance(candidate.midpoint, slot_center),
        )

    return [
        outer_arc,
        LinearFacet(left_intersection, slot_rect[3]),
        LinearFacet(slot_rect[3], slot_rect[2]),
        LinearFacet(slot_rect[2], right_intersection),
    ]


def _reconstructed_facet_area(poly, facet, target_area=None):
    if isinstance(facet, CornerFacet):
        area = getPolyCurvedCornerArea(
            poly,
            facet.pLeft,
            facet.corner,
            facet.pRight,
            facet.radiusLeft,
            facet.radiusRight,
        )
    elif isinstance(facet, ArcFacet):
        area = facet.getPolyIntersectArea(poly)
    else:
        area = getPolyLineArea(poly, facet.pLeft, facet.pRight)

    poly_area = abs(getArea(poly))
    area = min(max(area, 0.0), poly_area)
    if target_area is None:
        return area

    complement = poly_area - area
    if abs(complement - target_area) < abs(area - target_area):
        return complement
    return area


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
    output_dirs = setupOutputDirs(save_name, clean_existing=True)

    # Initialize mesh once
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
    writeMesh(m, os.path.join(output_dirs["vtk"], f"mesh.vtk"))

    rng = np.random.default_rng(RANDOM_SEED)

    # Store metrics across cases
    area_errors = []
    facet_gaps = []
    hausdorff_distances = []

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

        # Run reconstruction and any optional geometry logging under one case context.
        print(f"Reconstructing Zalesak {i+1}")
        with arc_facet_log_context(
            experiment="zalesak",
            algo=facet_algo,
            resolution=resolution,
            wiggle=perturb_wiggle,
            seed=perturb_seed,
            save_name=save_name,
            case_index=i,
        ):
            reconstructed_facets, reconstructed_polys = runReconstruction(
                m,
                facet_algo,
                do_c0,
                i,
                output_dirs,
                algo_kwargs={},
                return_polys=True,
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
            for row in m.polys:
                for poly in row:
                    if poly.getFraction() >= 1 - threshold:
                        reconstructed_total_area += poly.getMaxArea()

            for facet_index, (poly, facet) in enumerate(
                zip(reconstructed_polys, reconstructed_facets)
            ):
                try:
                    with arc_facet_log_context(
                        metric_stage="area_error",
                        facet_index=facet_index,
                    ):
                        reconstructed_total_area += _reconstructed_facet_area(
                            poly.points,
                            facet,
                            target_area=poly.getFraction() * poly.getMaxArea(),
                        )
                except Exception:
                    continue

            area_error = abs(reconstructed_total_area - true_area) / max(true_area, 1e-12)
            print(f"Area error for case {i+1}: {area_error:.3e}")

            # Facet gaps / Hausdorff
            avg_gap = calculate_facet_gaps(m, reconstructed_facets)
            hausdorff_distance = hausdorff_interface(true_facets, reconstructed_facets)
            print(f"Average facet gap for case {i+1}: {avg_gap:.3e}")
            print(f"Hausdorff distance for case {i+1}: {hausdorff_distance:.3e}")

            # Save metrics
            with open(os.path.join(output_dirs["metrics"], "area_error.txt"), "a") as f:
                f.write(f"{area_error}\n")
            with open(os.path.join(output_dirs["metrics"], "facet_gap.txt"), "a") as f:
                f.write(f"{avg_gap}\n")
            with open(os.path.join(output_dirs["metrics"], "hausdorff.txt"), "a") as f:
                f.write(f"{hausdorff_distance}\n")

            area_errors.append(area_error)
            facet_gaps.append(avg_gap)
            hausdorff_distances.append(hausdorff_distance)

    return area_errors, facet_gaps, hausdorff_distances


def create_combined_plot(
    resolutions,
    hausdorff_results,
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

    # Hausdorff distances
    for algo, values in hausdorff_results.items():
        plt.sca(ax2)
        plt.plot(x_values, values, marker="o", label=algo)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel(r"Resolution", fontsize=14)
    ax2.set_yscale("log")
    ax2.set_ylabel("Average Hausdorff Distance", fontsize=14)
    ax2.set_title("Hausdorff Distance", fontsize=16, fontweight="bold")
    ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=False, loc="best")
    ax2.grid(True, which="both", ls="-", alpha=0.3)

    plt.suptitle("Zalesak Static Reconstruction", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_area_plot(
    resolutions,
    area_results,
    save_path="results/static/zalesak_reconstruction_area.png",
):
    plt.figure(figsize=(8, 6))
    x_values = [int(100 * r) for r in resolutions]
    for algo, values in area_results.items():
        plt.plot(x_values, values, marker="o", label=algo)
    plt.xscale("log", base=2)
    plt.xlabel(r"Resolution", fontsize=14)
    plt.yscale("log")
    plt.ylabel("Average Area Error", fontsize=14)
    plt.title("Zalesak Area Error", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=False, loc="best")
    plt.grid(True, which="both", ls="-", alpha=0.3)
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
    hausdorff_results = eval(lines[3].split("Hausdorff Results: ")[1])
    return resolutions, area_results, gap_results, hausdorff_results


def plot_from_results_file(
    file_path="results/static/zalesak_reconstruction_results.txt",
):
    try:
        resolutions, area_results, gap_results, hausdorff_results = load_results_from_file(
            file_path
        )
        create_combined_plot(resolutions, hausdorff_results, gap_results)
        create_area_plot(resolutions, area_results)
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
    hausdorff_results = {algo: [] for algo in facet_algos}
    for resolution in resolutions:
        print(f"\nRunning experiments for resolution {resolution}")
        for algo, save_name in zip(facet_algos, save_names):
            print(f"Testing {algo} algorithm...")
            areas, gaps, hausdorff_values = main(
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
            hausdorff_results[algo].append(
                max(np.mean(np.array(hausdorff_values)), MIN_ERROR)
            )

    create_combined_plot(resolutions, hausdorff_results, gap_results)
    create_area_plot(resolutions, area_results)
    with open("results/static/zalesak_reconstruction_results.txt", "w") as f:
        f.write(f"Resolutions: {resolutions}\n")
        f.write(f"Area Results: {area_results}\n")
        f.write(f"Gap Results: {gap_results}\n")
        f.write(f"Hausdorff Results: {hausdorff_results}\n")
    return area_results, gap_results, hausdorff_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zalesak static reconstruction tests.")
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument("--resolution", type=float, help="resolution", required=False)
    parser.add_argument("--facet_algo", type=str, help="facet_algo", required=False)
    parser.add_argument("--save_name", type=str, help="save_name", required=False)
    parser.add_argument(
        "--num_cases", type=int, help="number of randomized cases", default=25
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
        area_results, gap_results, hausdorff_results = run_parameter_sweep(
            args.config, args.num_cases, args.radius, args.slot_width
        )
        print("\nParameter sweep results:")
        print("\nArea Error:")
        for algo, values in area_results.items():
            print(f"{algo}: {values}")
        print("\nFacet Gaps:")
        for algo, values in gap_results.items():
            print(f"{algo}: {values}")
        print("\nHausdorff Distance:")
        for algo, values in hausdorff_results.items():
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
            mesh_type=args.mesh_type,
            perturb_wiggle=args.perturb_wiggle,
            perturb_seed=args.perturb_seed,
            perturb_fix_boundary=args.perturb_fix_boundary,
            perturb_max_tries=args.perturb_max_tries,
            perturb_type=args.perturb_type,
        )
