import os

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

    volume_linf_error, _ = LinfErrorFractions(m.getFractions(), true_final_areas)

    with open(os.path.join(output_dirs["base"], "volume_linf_error.txt"), "w") as f:
        f.write(f"{volume_linf_error}\n{l2_mixed_count}\n")

    plotInitialAreaCompare(m, os.path.join(output_dirs["plt"], f"initial_compare.png"))
