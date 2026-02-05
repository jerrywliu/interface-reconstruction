import os

import numpy as np

from main.geoms.geoms import getDistance
from main.structs.interface import Interface
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
    """Calculate average gap distance between adjacent facets."""
    interface = Interface.from_merge_mesh(
        mesh, reconstructed_facets=reconstructed_facets, infer_missing_neighbors=False
    )
    has_oriented = any(
        record.left_cell_id is not None or record.right_cell_id is not None
        for component in interface.components
        for record in component.records
    )
    if not has_oriented:
        interface = Interface.from_merge_mesh(
            mesh,
            reconstructed_facets=reconstructed_facets,
            infer_missing_neighbors=True,
        )

    gaps = []
    for component in interface.components:
        records = component.records
        if len(records) < 2:
            continue
        for i in range(len(records)):
            j = (i + 1) % len(records) if component.is_closed else i + 1
            if j >= len(records):
                continue
            p_right = records[i].right_point()
            p_left = records[j].left_point()
            gaps.append(getDistance(p_right, p_left))

    if not gaps:
        return 0
    return float(np.mean(np.asarray(gaps)))


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
