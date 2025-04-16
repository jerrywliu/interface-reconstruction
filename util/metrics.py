import os

from main.geoms.geoms import getDistance
from util.plotting.plt_utils import plotInitialAreaCompare


def L2ErrorFractions(final, true):
    assert len(final) == len(true) and len(final[0]) == len(true[0])
    l2_error = 0
    count = 0
    for x in range(len(final)):
        for y in range(len(final[0])):
            if (final[x][y] < 1 and final[x][y] > 0) or (
                true[x][y] < 1 and true[x][y] > 0
            ):
                l2_error += (final[x][y] - true[x][y]) ** 2
                count += 1
    return l2_error / count, count


# Metric from "An efficient bilinear interface reconstruction algorithm and
# consistent multidimensional unsplit advection scheme for accurate
# capturing of highly-curved interfacial shapes on structured grids"
# van der Eijk et al. 2024, used for vortex problem.
# Average over all grid cells and multiply by dx^2
def L2ErrorFractionsbyGridSpacing(final, true, dx):
    assert len(final) == len(true) and len(final[0]) == len(true[0])
    l2_error = 0
    for x in range(len(final)):
        for y in range(len(final[0])):
            l2_error += abs(final[x][y] - true[x][y])
    return l2_error / (len(final) * len(final[0]))


def LinfErrorFractions(final, true):
    assert len(final) == len(true) and len(final[0]) == len(true[0])
    linf_error = 0
    count = 0
    for x in range(len(final)):
        for y in range(len(final[0])):
            if (final[x][y] < 1 and final[x][y] > 0) or (
                true[x][y] < 1 and true[x][y] > 0
            ):
                linf_error = max(linf_error, abs(final[x][y] - true[x][y]))
                count += 1
    return linf_error, count


def calculate_facet_gaps(mesh, reconstructed_facets):
    """Calculate the minimum distance between left facet endpoint and neighboring facets' endpoints.

    Args:
        mesh: MergeMesh object
        reconstructed_facets: List of reconstructed facets for each cell

    Returns:
        avg_gap: Average minimum gap distance across all mixed cells
    """
    total_gap = 0
    cnt_gap = 0

    # Get list of merged polygons for easier indexing
    merged_polys = list(mesh.merged_polys.values())

    for i, (poly, facet) in enumerate(zip(merged_polys, reconstructed_facets)):
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


def computeFinalMetrics(m, true_final_areas, output_dirs):
    """
    Compute and save final error metrics.

    Args:
        m: MergeMesh object
        true_final_areas: Array of true final areas
        output_dirs: Dictionary of output directories
    """
    # Volume errors
    volume_l2_error, l2_mixed_count = L2ErrorFractions(
        m.getFractions(), true_final_areas
    )

    with open(os.path.join(output_dirs["base"], "volume_l2_error.txt"), "w") as f:
        f.write(f"{volume_l2_error}\n{l2_mixed_count}\n")

    # Volume errors by grid spacing
    dx = 1 / len(m.getFractions())
    volume_l2_error_by_grid_spacing = L2ErrorFractionsbyGridSpacing(
        m.getFractions(), true_final_areas, dx
    )

    with open(
        os.path.join(output_dirs["base"], "volume_l2_error_by_grid_spacing.txt"), "w"
    ) as f:
        f.write(
            f"{volume_l2_error_by_grid_spacing}\n"
        )  # TODO JL 3/13/25 hack to set dx. To fix, need to adjust definition of vortex problem to be on 1x1 domain.

    volume_linf_error, _ = LinfErrorFractions(m.getFractions(), true_final_areas)

    with open(os.path.join(output_dirs["base"], "volume_linf_error.txt"), "w") as f:
        f.write(f"{volume_linf_error}\n{l2_mixed_count}\n")

    plotInitialAreaCompare(m, os.path.join(output_dirs["plt"], f"initial_compare.png"))
