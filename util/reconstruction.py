import os

from main.structs.meshes.merge_mesh import MergeMesh

from util.plotting.plt_utils import plotAreas, plotPartialAreas
from util.plotting.vtk_utils import writePartialCells, writeFacets
from util.reconstruction_diagnostics import write_reconstruction_diagnostics


class IncompleteReconstructionError(RuntimeError):
    """Raised when the final facet list does not cover every active mixed component."""


def _active_merged_polygons(m: MergeMesh):
    active_merge_ids = []
    seen = set()
    for merge_id_row in m.coords_to_merge_id:
        for merge_id in merge_id_row:
            if merge_id is None or merge_id in seen:
                continue
            seen.add(merge_id)
            active_merge_ids.append(merge_id)

    missing_ids = [
        merge_id for merge_id in active_merge_ids if merge_id not in m.merged_polys
    ]
    if missing_ids:
        raise IncompleteReconstructionError(
            f"Active merge ids have no polygon objects: {missing_ids}"
        )
    return active_merge_ids, [m.merged_polys[merge_id] for merge_id in active_merge_ids]


def _validate_active_reconstruction(m, reconstructed_polys, reconstructed_facets):
    active_merge_ids, active_polys = _active_merged_polygons(m)
    returned_polys = list(reconstructed_polys)
    returned_facets = list(reconstructed_facets)

    if len(returned_polys) != len(returned_facets):
        raise IncompleteReconstructionError(
            "Final reconstruction returned "
            f"{len(returned_facets)} facets for {len(returned_polys)} polygons"
        )

    active_by_object = {
        id(poly): merge_id for merge_id, poly in zip(active_merge_ids, active_polys)
    }
    returned_objects = [id(poly) for poly in returned_polys]
    missing_active_ids = [
        active_by_object[object_id]
        for object_id in active_by_object
        if object_id not in returned_objects
    ]
    extra_count = sum(
        object_id not in active_by_object for object_id in returned_objects
    )
    duplicate_count = len(returned_objects) - len(set(returned_objects))
    if missing_active_ids or extra_count or duplicate_count:
        raise IncompleteReconstructionError(
            "Final reconstruction does not match the active mixed partition: "
            f"missing active merge ids={missing_active_ids}, "
            f"extra polygons={extra_count}, duplicate polygons={duplicate_count}"
        )

    missing_facet_indices = [
        index for index, facet in enumerate(returned_facets) if facet is None
    ]
    if missing_facet_indices:
        raise IncompleteReconstructionError(
            f"Final reconstruction has missing facets at indices {missing_facet_indices}"
        )


def runReconstruction(
    m: MergeMesh, facet_algo, do_c0, iter, output_dirs, algo_kwargs={}, return_polys=False
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
        "ELVIRA",
        "LVIRA",
        "safe_linear",
        "safe_circle",
        "safe_linear_corner",
    ]

    # Plot areas regardless of algorithm
    plotAreas(m, os.path.join(output_dirs["plt_areas"], f"{iter}.png"))
    plotPartialAreas(m, os.path.join(output_dirs["plt_partial"], f"{iter}.png"))

    if facet_algo in no_merge_algos:
        m._provenance_stage = facet_algo
        reconstructed_facets, reconstructed_polys = _run_no_merge(
            m, facet_algo, iter, output_dirs, algo_kwargs
        )
    else:
        reconstructed_facets, reconstructed_polys = _run_with_merge(
            m, facet_algo, do_c0, iter, output_dirs, algo_kwargs
        )

    # Write final reconstructed facets. If C0 is enabled, the returned facets should
    # reflect the adjusted reconstruction because downstream static metrics and plots
    # interpret the return value as the final interface.
    writeFacets(
        reconstructed_facets,
        os.path.join(output_dirs["vtk_reconstructed_facets"], f"{iter}.vtp"),
    )
    write_reconstruction_diagnostics(m, iter, output_dirs)
    _validate_active_reconstruction(m, reconstructed_polys, reconstructed_facets)

    if return_polys:
        return reconstructed_facets, reconstructed_polys
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
    elif facet_algo == "ELVIRA":
        m.runELVIRA()
    elif facet_algo == "LVIRA":
        m.runLVIRA()
    elif facet_algo == "safe_linear":
        safe_linear_kwargs = {
            key: value
            for key, value in algo_kwargs.items()
            if key in {"check_threshold", "default_to_youngs", "fit_1neighbor"}
        }
        m.runSafeLinear(**safe_linear_kwargs)
    elif facet_algo == "safe_circle":
        m.runSafeCircle(
            plic_fallback=algo_kwargs.get("plic_fallback", "LVIRA"),
            arc_failure_fallback=algo_kwargs.get(
                "arc_failure_fallback", "local_linear"
            ),
        )
    elif facet_algo == "safe_linear_corner":
        _ = m.findSafeOrientations()  # basic orientation finding
        m.runSafeLinearCorner()

    merged_polys = list(m.merged_polys.values())
    return [p.getFacet() for p in merged_polys], merged_polys


def _run_with_merge(
    m: MergeMesh, facet_algo, do_c0, iter, output_dirs, algo_kwargs=None
):
    """
    Run reconstruction for algorithms that merge cells.
    These algorithms first merge neighboring cells, then fit interfaces.
    """
    algo_kwargs = algo_kwargs or {}

    m.configure_corner_behavior(
        algo_kwargs.get(
            "corner_behavior_profile", MergeMesh.default_corner_behavior_profile
        )
    )
    m.merge1Neighbors()
    merge_ids = m.findOrientations()

    m.updatePlots()
    writePartialCells(
        m, os.path.join(output_dirs["vtk_reconstructed_mixed"], f"{iter}.vtp")
    )

    merged_polys = m.fitFacets(
        merge_ids,
        setting=facet_algo,
        plic_fallback=algo_kwargs.get("plic_fallback", "LVIRA"),
        rescue_profile=algo_kwargs.get(
            "rescue_profile", MergeMesh.default_rescue_profile
        ),
    )
    reconstructed_facets = [p.getFacet() for p in merged_polys]

    if do_c0:
        merged_polys = m.makeC0(merged_polys)
        C0_facets = [p.getFacet() for p in merged_polys]
        writeFacets(
            C0_facets, os.path.join(output_dirs["vtk_reconstructed_c0"], f"{iter}.vtp")
        )
        reconstructed_facets = C0_facets

    return reconstructed_facets, merged_polys
