import math
from typing import Iterable, Sequence, Union

import numpy as np
from scipy.spatial import cKDTree

from main.geoms.geoms import getDistance
from main.structs.facets.base_facet import getNormal
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.corner_facet import CornerFacet
from main.structs.facets.linear_facet import LinearFacet
from main.structs.interface import Interface


def hausdorffFacets(
    facet1: Union[LinearFacet, ArcFacet, CornerFacet],
    facet2: Union[LinearFacet, ArcFacet, CornerFacet],
    n=100,
):
    """
    Compute Hausdorff distance between two facets using numerical approximation.
    
    Args:
        facet1: First facet (LinearFacet, ArcFacet, or CornerFacet)
        facet2: Second facet (LinearFacet, ArcFacet, or CornerFacet)
        n: Number of equispaced points to sample along each facet (default: 100)
        
    Returns:
        float: Hausdorff distance
    """
    
    points1 = facet1.sample(n)
    points2 = facet2.sample(n)
    return hausdorff_points(points1, points2)


def hausdorff_points(points1: Iterable, points2: Iterable) -> float:
    points1 = np.asarray(list(points1), dtype=float)
    points2 = np.asarray(list(points2), dtype=float)
    if points1.size == 0 or points2.size == 0:
        return float("inf")
    if points1.ndim == 1:
        points1 = points1.reshape(1, -1)
    if points2.ndim == 1:
        points2 = points2.reshape(1, -1)

    tree2 = cKDTree(points2)
    d1, _ = tree2.query(points1, k=1)

    tree1 = cKDTree(points1)
    d2, _ = tree1.query(points2, k=1)

    return max(np.max(d1), np.max(d2))


def interface_gap_stats(interface: Interface, mode: str = "euclidean"):
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
            if mode == "normal":
                normal = getNormal(records[i].facet, p_right)
                if normal is None:
                    gap = getDistance(p_right, p_left)
                else:
                    delta = np.asarray(p_right) - np.asarray(p_left)
                    gap = abs(float(np.dot(delta, normal)))
            else:
                gap = getDistance(p_right, p_left)
            gaps.append(gap)

    if not gaps:
        return {"mean": 0.0, "max": 0.0, "p95": 0.0, "count": 0}

    gaps_np = np.asarray(gaps, dtype=float)
    return {
        "mean": float(np.mean(gaps_np)),
        "max": float(np.max(gaps_np)),
        "p95": float(np.quantile(gaps_np, 0.95)),
        "count": int(len(gaps)),
    }


def calculate_facet_gaps(
    mesh,
    reconstructed_facets,
    mode: str = "euclidean",
    infer_missing_neighbors: bool = True,
    return_stats: bool = False,
):
    """Calculate gap metrics between adjacent facets along the interface."""
    interface = Interface.from_merge_mesh(
        mesh,
        reconstructed_facets=reconstructed_facets,
        infer_missing_neighbors=False,
    )
    if infer_missing_neighbors:
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
    stats = interface_gap_stats(interface, mode=mode)
    return stats if return_stats else stats["mean"]


def _normalize(vec: Sequence[float]):
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return None
    return arr / norm


def _flatten_facets(facets: Iterable):
    for facet in facets:
        if isinstance(facet, CornerFacet):
            yield facet.facetLeft
            yield facet.facetRight
        else:
            yield facet


def sample_facets_with_tangents(
    facets: Iterable[Union[LinearFacet, ArcFacet, CornerFacet]],
    n_per_facet: int = 50,
):
    points = []
    tangents = []
    for facet in _flatten_facets(facets):
        if facet is None:
            continue
        samples = facet.sample(max(2, n_per_facet))
        for p in samples:
            tangent = _normalize(facet.getTangent(p))
            if tangent is None:
                continue
            points.append(p)
            tangents.append(tangent)

    if not points:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.asarray(points, dtype=float), np.asarray(tangents, dtype=float)


def tangent_error_to_curve(
    reconstructed_facets: Iterable[Union[LinearFacet, ArcFacet, CornerFacet]],
    true_points: Sequence[Sequence[float]],
    true_tangents: Sequence[Sequence[float]],
    n_per_facet: int = 50,
):
    true_pts = np.asarray(true_points, dtype=float)
    true_tans = []
    for tan in true_tangents:
        normed = _normalize(tan)
        if normed is None:
            true_tans.append(None)
        else:
            true_tans.append(normed)

    recon_pts, recon_tans = sample_facets_with_tangents(
        reconstructed_facets, n_per_facet=n_per_facet
    )
    if recon_pts.size == 0 or true_pts.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}

    tree = cKDTree(recon_pts)
    _, idxs = tree.query(true_pts, k=1)

    angles = []
    for i, idx in enumerate(idxs):
        t_true = true_tans[i]
        if t_true is None:
            continue
        t_recon = recon_tans[idx]
        dot = float(np.clip(np.dot(t_true, t_recon), -1.0, 1.0))
        dot = abs(dot)  # tangent direction is sign-invariant
        angle = math.acos(dot)
        angles.append(angle)

    if not angles:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}

    angles_np = np.asarray(angles, dtype=float)
    return {
        "mean": float(np.mean(angles_np)),
        "median": float(np.median(angles_np)),
        "max": float(np.max(angles_np)),
    }


def polyline_vertex_curvature(points: Sequence[Sequence[float]], closed: bool = True):
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return []

    n = pts.shape[0]
    curvatures = []
    indices = range(n) if closed else range(1, n - 1)
    for i in indices:
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        if not closed and (i == 0 or i == n - 1):
            continue

        v1 = pts[i] - pts[i_prev]
        v2 = pts[i_next] - pts[i]
        len1 = float(np.linalg.norm(v1))
        len2 = float(np.linalg.norm(v2))
        if len1 < 1e-12 or len2 < 1e-12:
            continue

        dot = float(np.dot(v1, v2))
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        angle = abs(math.atan2(cross, dot))
        avg_len = 0.5 * (len1 + len2)
        curvatures.append(angle / avg_len)

    return curvatures


def polyline_average_curvature(points: Sequence[Sequence[float]], closed: bool = True):
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return float("nan")

    segs = pts[1:] - pts[:-1]
    if closed:
        segs = np.vstack([segs, pts[0] - pts[-1]])
    lengths = np.linalg.norm(segs, axis=1)
    total_length = float(np.sum(lengths))
    if total_length < 1e-12:
        return float("nan")

    n = pts.shape[0]
    total_turn = 0.0
    indices = range(n) if closed else range(1, n - 1)
    for i in indices:
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        if not closed and (i == 0 or i == n - 1):
            continue
        v1 = pts[i] - pts[i_prev]
        v2 = pts[i_next] - pts[i]
        if np.linalg.norm(v1) < 1e-12 or np.linalg.norm(v2) < 1e-12:
            continue
        dot = float(np.dot(v1, v2))
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        total_turn += abs(math.atan2(cross, dot))

    return total_turn / total_length


def interface_turning_curvature(interface: Interface):
    curvatures = []
    for component in interface.components:
        records = component.records
        if len(records) < 2:
            continue
        for i in range(len(records)):
            j = (i + 1) % len(records) if component.is_closed else i + 1
            if j >= len(records):
                continue

            rec_i = records[i]
            rec_j = records[j]
            p_joint = rec_i.right_point()

            tan_i = _normalize(rec_i.facet.getTangent(p_joint))
            tan_j = _normalize(rec_j.facet.getTangent(p_joint))
            if tan_i is None or tan_j is None:
                continue

            dot = float(np.clip(np.dot(tan_i, tan_j), -1.0, 1.0))
            cross = float(tan_i[0] * tan_j[1] - tan_i[1] * tan_j[0])
            angle = abs(math.atan2(cross, dot))

            len_i = getDistance(rec_i.left_point(), rec_i.right_point())
            len_j = getDistance(rec_j.left_point(), rec_j.right_point())
            avg_len = 0.5 * (len_i + len_j)
            if avg_len < 1e-12:
                continue
            curvatures.append(angle / avg_len)

    if not curvatures:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}

    curv_np = np.asarray(curvatures, dtype=float)
    return {
        "mean": float(np.mean(curv_np)),
        "median": float(np.median(curv_np)),
        "max": float(np.max(curv_np)),
    }
