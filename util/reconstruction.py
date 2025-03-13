import os

from main.structs.meshes.merge_mesh import MergeMesh

from util.plotting.plt_utils import plotAreas, plotPartialAreas
from util.plotting.vtk_utils import writePartialCells, writeFacets


def runReconstruction(
    m: MergeMesh, facet_algo, do_c0, iter, output_dirs, algo_kwargs={}
):
    """
    Run interface reconstruction based on specified algorithm.

    Args:
        m: MergeMesh object
        facet_algo: String specifying reconstruction algorithm
        do_c0: Boolean for C0 continuity enforcement
        iter: Current iteration number
        output_dirs: Dictionary of output directories
        algo_kwargs: Dictionary of algorithm-specific keyword arguments

    Returns:
        reconstructed_facets: List of reconstructed facets
    """
    # Algorithms that don't merge cells
    no_merge_algos = [
        "Youngs",
        "LVIRA",
        "safe_linear",
        "safe_circle",
        "safe_linear_corner",
    ]

    # Plot areas regardless of algorithm
    plotAreas(m, os.path.join(output_dirs["plt_areas"], f"{iter}.png"))
    plotPartialAreas(m, os.path.join(output_dirs["plt_partial"], f"{iter}.png"))

    if facet_algo in no_merge_algos:
        reconstructed_facets = _run_no_merge(
            m, facet_algo, iter, output_dirs, algo_kwargs
        )
    else:
        reconstructed_facets = _run_with_merge(
            m, facet_algo, do_c0, iter, output_dirs, algo_kwargs
        )

    # Write final reconstructed facets
    writeFacets(
        reconstructed_facets,
        os.path.join(output_dirs["vtk_reconstructed_facets"], f"{iter}.vtp"),
    )

    return reconstructed_facets


def _run_no_merge(m: MergeMesh, facet_algo, iter, output_dirs, algo_kwargs={}):
    """
    Run reconstruction for algorithms that operate on individual cells.
    These algorithms reconstruct interfaces without merging cells.
    """
    m.createMergedPolys()
    writePartialCells(
        m, os.path.join(output_dirs["vtk_reconstructed_mixed"], f"{iter}.vtp")
    )

    if facet_algo == "Youngs":
        m.runYoungs()
    elif facet_algo == "LVIRA":
        m.runLVIRA()
    elif facet_algo == "safe_linear":
        m.runSafeLinear(**algo_kwargs)
    elif facet_algo == "safe_circle":
        m.runSafeCircle()
    elif facet_algo == "safe_linear_corner":
        _ = m.findSafeOrientations()  # basic orientation finding
        m.runSafeLinearCorner()

    return [p.getFacet() for p in m.merged_polys.values()]


def _run_with_merge(
    m: MergeMesh, facet_algo, do_c0, iter, output_dirs, algo_kwargs=None
):
    """
    Run reconstruction for algorithms that merge cells.
    These algorithms first merge neighboring cells, then fit interfaces.
    """
    m.merge1Neighbors()
    merge_ids = m.findOrientations()

    m.updatePlots()
    writePartialCells(
        m, os.path.join(output_dirs["vtk_reconstructed_mixed"], f"{iter}.vtp")
    )

    merged_polys = m.fitFacets(merge_ids, setting=facet_algo)
    reconstructed_facets = [p.getFacet() for p in merged_polys]

    if do_c0:
        merged_polys = m.makeC0(merged_polys)
        C0_facets = [p.getFacet() for p in merged_polys]
        writeFacets(
            C0_facets, os.path.join(output_dirs["vtk_reconstructed_c0"], f"{iter}.vtp")
        )

    return reconstructed_facets
