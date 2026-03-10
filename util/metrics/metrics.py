import math
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from scipy.spatial import cKDTree

from main.geoms.geoms import getDistance
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.corner_facet import CornerFacet
from main.structs.facets.linear_facet import LinearFacet
from main.structs.interface import Interface
from main.structs.interface_geometry import (
    ArcPrimitive,
    LinePrimitive,
    Primitive,
    iter_primitives_from_facets,
)

FacetLike = Union[LinearFacet, ArcFacet, CornerFacet, LinePrimitive, ArcPrimitive]


def _normalize(vec: Sequence[float]):
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return None
    return arr / norm


def _as_primitive_list(geometry) -> list[Primitive]:
    if geometry is None:
        return []
    if isinstance(geometry, Interface):
        primitives = []
        for component in geometry.components:
            for record in component.records:
                primitives.extend(iter_primitives_from_facets([record.facet]))
        return primitives
    if isinstance(
        geometry,
        (LinearFacet, ArcFacet, CornerFacet, LinePrimitive, ArcPrimitive),
    ):
        return list(iter_primitives_from_facets([geometry]))
    return list(iter_primitives_from_facets(geometry))


def _point_to_line_distance(point, primitive: LinePrimitive) -> float:
    return primitive.distance_to_point(point)


def _point_to_arc_distance(point, primitive: ArcPrimitive) -> float:
    return primitive.distance_to_point(point)


def point_to_primitive_distance(point, primitive: Primitive) -> float:
    if isinstance(primitive, LinePrimitive):
        return _point_to_line_distance(point, primitive)
    return _point_to_arc_distance(point, primitive)


def point_to_primitives_distance(point, primitives: Sequence[Primitive]) -> float:
    if not primitives:
        return float("inf")
    return min(point_to_primitive_distance(point, primitive) for primitive in primitives)


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


def _bbox_diameter_from_primitives(primitives: Sequence[Primitive]) -> float:
    bbox_points = []
    for primitive in primitives:
        bbox_points.extend(primitive.bbox_points())
    if not bbox_points:
        return 0.0
    pts = np.asarray(bbox_points, dtype=float)
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    return float(np.linalg.norm(maxs - mins))


def _sample_primitives_by_spacing(
    primitives: Sequence[Primitive], max_spacing: float
) -> np.ndarray:
    points = []
    prev = None
    for primitive in primitives:
        sample_points = primitive.sample_by_max_spacing(max_spacing)
        for point in sample_points:
            point_arr = np.asarray(point, dtype=float)
            if prev is not None and np.linalg.norm(point_arr - prev) < 1e-14:
                continue
            points.append(point_arr)
            prev = point_arr
    if not points:
        return np.zeros((0, 2))
    return np.asarray(points, dtype=float)


def directed_hausdorff_primitives(
    source_primitives: Sequence[Primitive],
    target_primitives: Sequence[Primitive],
    tol_abs: float,
    ds_min: float,
    initial_samples_per_primitive: int = 8,
) -> float:
    if not source_primitives or not target_primitives:
        return float("inf")

    max_length = max((primitive.length() for primitive in source_primitives), default=0.0)
    if max_length < 1e-12:
        point = source_primitives[0].pLeft
        return point_to_primitives_distance(point, target_primitives)

    initial_samples_per_primitive = max(2, int(initial_samples_per_primitive))
    spacing = max(max_length / (initial_samples_per_primitive - 1), ds_min)
    previous_estimate = None

    while True:
        sample_points = _sample_primitives_by_spacing(source_primitives, spacing)
        if sample_points.size == 0:
            return float("inf")
        estimate = max(
            point_to_primitives_distance(point, target_primitives)
            for point in sample_points
        )

        if previous_estimate is not None and abs(estimate - previous_estimate) <= tol_abs:
            return max(estimate, previous_estimate)
        if spacing <= ds_min * (1.0 + 1e-12):
            return estimate if previous_estimate is None else max(estimate, previous_estimate)

        previous_estimate = estimate
        spacing = max(spacing * 0.5, ds_min)


def hausdorff_interface(
    geometry1,
    geometry2,
    tol_abs: Optional[float] = None,
    ds_min: Optional[float] = None,
    initial_samples_per_primitive: int = 8,
) -> float:
    primitives1 = _as_primitive_list(geometry1)
    primitives2 = _as_primitive_list(geometry2)
    if not primitives1 or not primitives2:
        return float("inf")

    bbox_diameter = _bbox_diameter_from_primitives(primitives1 + primitives2)
    scale = max(bbox_diameter, 1.0)
    tol_abs = 1e-12 * scale if tol_abs is None else tol_abs
    ds_min = 1e-4 * scale if ds_min is None else ds_min

    d12 = directed_hausdorff_primitives(
        primitives1,
        primitives2,
        tol_abs=tol_abs,
        ds_min=ds_min,
        initial_samples_per_primitive=initial_samples_per_primitive,
    )
    d21 = directed_hausdorff_primitives(
        primitives2,
        primitives1,
        tol_abs=tol_abs,
        ds_min=ds_min,
        initial_samples_per_primitive=initial_samples_per_primitive,
    )
    return max(d12, d21)


def hausdorffFacets(
    facet1: FacetLike,
    facet2: FacetLike,
    n=100,
):
    """
    Compute Hausdorff distance between two facets using adaptive sampling over
    their canonical primitive representation.
    """
    return hausdorff_interface(
        [facet1],
        [facet2],
        initial_samples_per_primitive=max(8, int(n)),
    )


def interface_gap_stats(interface: Interface, mode: str = "euclidean"):
    def _euclidean_gap(record_i, record_j):
        endpoints_i = (record_i.facet.pLeft, record_i.facet.pRight)
        endpoints_j = (record_j.facet.pLeft, record_j.facet.pRight)
        return min(
            getDistance(point_i, point_j)
            for point_i in endpoints_i
            for point_j in endpoints_j
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
            record_i = records[i]
            record_j = records[j]
            p_right = record_i.right_point()
            p_left = record_j.left_point()
            if mode == "normal" and record_i.right_joint() != "corner":
                normal = getattr(record_i.facet, "normal_at_point", lambda p: None)(p_right)
                if normal is None:
                    gap = _euclidean_gap(record_i, record_j)
                else:
                    delta = np.asarray(p_right, dtype=float) - np.asarray(p_left, dtype=float)
                    gap = abs(float(np.dot(delta, normal)))
            else:
                gap = _euclidean_gap(record_i, record_j)
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


def sample_facets_with_tangents(
    facets: Iterable[FacetLike],
    n_per_facet: int = 50,
):
    points = []
    tangents = []
    for primitive in iter_primitives_from_facets(facets):
        samples = primitive.sample(max(2, n_per_facet))
        for point in samples:
            tangent = _normalize(primitive.getTangent(point))
            if tangent is None:
                continue
            points.append(point)
            tangents.append(tangent)

    if not points:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.asarray(points, dtype=float), np.asarray(tangents, dtype=float)


def tangent_error_to_curve(
    reconstructed_facets: Iterable[FacetLike],
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
        dot = abs(dot)
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
            if rec_i.right_joint() == "corner":
                continue

            p_joint = rec_i.right_point()
            tan_i = _normalize(rec_i.facet.getTangent(p_joint))
            tan_j = _normalize(rec_j.facet.getTangent(p_joint))
            if tan_i is None or tan_j is None:
                continue

            dot = float(np.clip(np.dot(tan_i, tan_j), -1.0, 1.0))
            cross = float(tan_i[0] * tan_j[1] - tan_i[1] * tan_j[0])
            angle = abs(math.atan2(cross, dot))

            len_i = rec_i.facet.length()
            len_j = rec_j.facet.length()
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
