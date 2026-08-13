"""Metrics for component-aware external baseline reconstructions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from main.algos.baselines.external_geometry import (
    CellIndex,
    ExternalBaselineResult,
    ExternalCellReconstruction,
    ExternalPrimitive,
    ExternalReconstructionStatus,
    Point,
)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _normalize(vector: Sequence[float]) -> Optional[np.ndarray]:
    array = np.asarray(vector, dtype=float)
    magnitude = float(np.linalg.norm(array))
    return None if magnitude == 0.0 else array / magnitude


def _as_primitives(geometry: Any) -> Tuple[ExternalPrimitive, ...]:
    if isinstance(geometry, ExternalBaselineResult):
        records = geometry.cells.values()
        return tuple(
            primitive for record in records for primitive in record.primitives()
        )
    if isinstance(geometry, ExternalCellReconstruction):
        return geometry.primitives()
    if isinstance(geometry, ExternalPrimitive):
        return (geometry,)
    values = tuple(geometry)
    if not values:
        return ()
    if isinstance(values[0], ExternalCellReconstruction):
        return tuple(
            primitive for record in values for primitive in record.primitives()
        )
    return values


def _bbox_scale(primitives: Sequence[ExternalPrimitive]) -> float:
    points = [point for primitive in primitives for point in primitive.bbox_points()]
    if not points:
        return 1.0
    array = np.asarray(points, dtype=float)
    return max(1.0, float(np.linalg.norm(array.max(axis=0) - array.min(axis=0))))


def _sample_by_spacing(
    primitive: ExternalPrimitive, max_spacing: float, minimum_count: int = 2
) -> Tuple[Point, ...]:
    count = max(minimum_count, int(math.ceil(primitive.length() / max_spacing)) + 1)
    return primitive.sample(count)


def directed_hausdorff_external(
    source: Any,
    target: Any,
    *,
    tolerance: Optional[float] = None,
    minimum_spacing: Optional[float] = None,
    initial_samples_per_primitive: int = 8,
) -> float:
    """Estimate one directed supremum using native target-curve distances."""

    source_primitives = _as_primitives(source)
    target_primitives = _as_primitives(target)
    if not source_primitives or not target_primitives:
        return float("inf")
    scale = _bbox_scale(source_primitives + target_primitives)
    tolerance = 1.0e-12 * scale if tolerance is None else float(tolerance)
    minimum_spacing = (
        1.0e-4 * scale if minimum_spacing is None else float(minimum_spacing)
    )
    maximum_length = max(primitive.length() for primitive in source_primitives)
    spacing = max(
        maximum_length / max(1, int(initial_samples_per_primitive) - 1),
        minimum_spacing,
    )
    previous = None
    while True:
        points = [
            point
            for primitive in source_primitives
            for point in _sample_by_spacing(primitive, spacing)
        ]
        estimate = max(
            min(primitive.distance_to_point(point) for primitive in target_primitives)
            for point in points
        )
        if previous is not None and abs(estimate - previous) <= tolerance:
            return max(estimate, previous)
        if spacing <= minimum_spacing * (1.0 + 1.0e-12):
            return estimate if previous is None else max(estimate, previous)
        previous = estimate
        spacing = max(0.5 * spacing, minimum_spacing)


def symmetric_hausdorff_external(source: Any, target: Any, **kwargs: Any) -> float:
    return max(
        directed_hausdorff_external(source, target, **kwargs),
        directed_hausdorff_external(target, source, **kwargs),
    )


def tangent_error_external(
    source: Any,
    target: Any,
    *,
    samples_per_primitive: int = 64,
) -> Mapping[str, float]:
    """Compare native primitive tangents at nearest sampled curve locations."""

    source_primitives = _as_primitives(source)
    target_primitives = _as_primitives(target)
    if not source_primitives or not target_primitives:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}

    target_points = []
    target_tangents = []
    for primitive in target_primitives:
        for index, point in enumerate(primitive.sample(samples_per_primitive)):
            parameter = index / (samples_per_primitive - 1)
            tangent = _normalize(primitive.tangent(parameter))
            if tangent is not None:
                target_points.append(point)
                target_tangents.append(tangent)
    if not target_points:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}
    tree = cKDTree(np.asarray(target_points, dtype=float))

    angles = []
    for primitive in source_primitives:
        for index, point in enumerate(primitive.sample(samples_per_primitive)):
            source_tangent = _normalize(
                primitive.tangent(index / (samples_per_primitive - 1))
            )
            if source_tangent is None:
                continue
            _, target_index = tree.query(np.asarray(point, dtype=float), k=1)
            dot = abs(float(np.dot(source_tangent, target_tangents[int(target_index)])))
            angles.append(math.acos(min(1.0, max(-1.0, dot))))
    if not angles:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}
    values = np.asarray(angles, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _shared_edge(
    polygon_a: Sequence[Point],
    polygon_b: Sequence[Point],
    tolerance: float,
) -> Optional[Tuple[Point, Point]]:
    common = []
    for point_a in polygon_a:
        for point_b in polygon_b:
            if _distance(point_a, point_b) <= tolerance and not any(
                _distance(point_a, existing) <= tolerance for existing in common
            ):
                common.append(point_a)
    if len(common) != 2 or _distance(common[0], common[1]) <= tolerance:
        return None
    return common[0], common[1]


def _deduplicate(points: Iterable[Point], tolerance: float) -> Tuple[Point, ...]:
    unique = []
    for point in points:
        if not any(_distance(point, existing) <= tolerance for existing in unique):
            unique.append(point)
    return tuple(unique)


def _record_crossings(
    record: ExternalCellReconstruction,
    edge: Tuple[Point, Point],
    tolerance: float,
) -> Tuple[Point, ...]:
    return _deduplicate(
        (
            point
            for primitive in record.primitives()
            for point in primitive.edge_crossings(edge, tolerance)
        ),
        tolerance,
    )


@dataclass(frozen=True)
class SharedEdgeGap:
    cells: Tuple[CellIndex, CellIndex]
    edge: Tuple[Point, Point]
    crossings_a: int
    crossings_b: int
    matched_gaps: Tuple[float, ...]
    unmatched_a: int
    unmatched_b: int

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "cells": [list(index) for index in self.cells],
            "edge": [list(point) for point in self.edge],
            "crossings_a": self.crossings_a,
            "crossings_b": self.crossings_b,
            "matched_gaps": list(self.matched_gaps),
            "unmatched_a": self.unmatched_a,
            "unmatched_b": self.unmatched_b,
        }


def shared_edge_gap_metrics(
    result: ExternalBaselineResult,
    *,
    geometry_tolerance: float = 1.0e-10,
) -> Mapping[str, Any]:
    """Match every crossing on each shared Cartesian edge.

    A crossing-count mismatch is retained as an explicit unmatched event; it
    is never hidden by selecting one endpoint pair.
    """

    details = []
    active_statuses = {
        ExternalReconstructionStatus.RECONSTRUCTED,
        ExternalReconstructionStatus.PAPER_FALLBACK,
    }
    for index_a, record_a in sorted(result.cells.items()):
        for delta in ((1, 0), (0, 1)):
            index_b = index_a[0] + delta[0], index_a[1] + delta[1]
            record_b = result.cells.get(index_b)
            if record_b is None:
                continue
            edge = _shared_edge(record_a.polygon, record_b.polygon, geometry_tolerance)
            if edge is None:
                raise ValueError(
                    f"adjacent Cartesian cells {index_a}, {index_b} lack a shared edge"
                )
            if (
                record_a.status not in active_statuses
                or record_b.status not in active_statuses
            ):
                continue
            crossings_a = _record_crossings(record_a, edge, geometry_tolerance)
            crossings_b = _record_crossings(record_b, edge, geometry_tolerance)
            if crossings_a and crossings_b:
                costs = np.asarray(
                    [[_distance(a, b) for b in crossings_b] for a in crossings_a],
                    dtype=float,
                )
                rows, columns = linear_sum_assignment(costs)
                matched = tuple(
                    float(costs[row, column]) for row, column in zip(rows, columns)
                )
            else:
                matched = ()
            details.append(
                SharedEdgeGap(
                    cells=(index_a, index_b),
                    edge=edge,
                    crossings_a=len(crossings_a),
                    crossings_b=len(crossings_b),
                    matched_gaps=matched,
                    unmatched_a=max(0, len(crossings_a) - len(matched)),
                    unmatched_b=max(0, len(crossings_b) - len(matched)),
                )
            )
    gaps = [gap for detail in details for gap in detail.matched_gaps]
    unmatched = sum(detail.unmatched_a + detail.unmatched_b for detail in details)
    return {
        "mean": float(np.mean(gaps)) if gaps else 0.0,
        "max": float(np.max(gaps)) if gaps else 0.0,
        "p95": float(np.quantile(gaps, 0.95)) if gaps else 0.0,
        "matched_count": len(gaps),
        "unmatched_count": unmatched,
        "edge_count": len(details),
        "edges": [detail.to_dict() for detail in details],
    }


def conservation_metrics(result: ExternalBaselineResult) -> Mapping[str, Any]:
    """Evaluate per-cell conservation through each method's exact area callback."""

    rows = []
    unresolved = 0
    for index, record in sorted(result.cells.items()):
        if record.status in (
            ExternalReconstructionStatus.UNSUPPORTED,
            ExternalReconstructionStatus.UNRESOLVED,
        ):
            unresolved += 1
            continue
        if record.target_phase_area is None:
            raise ValueError(f"cell {index} has no target_phase_area")
        reconstructed = record.exact_phase_area()
        residual = reconstructed - record.target_phase_area
        rows.append(
            {
                "cell_index": list(index),
                "target_phase_area": record.target_phase_area,
                "reconstructed_phase_area": reconstructed,
                "absolute_residual": abs(residual),
                "signed_residual": residual,
                "status": record.status.value,
            }
        )
    residuals = [row["absolute_residual"] for row in rows]
    return {
        "mean_absolute_residual": float(np.mean(residuals)) if residuals else 0.0,
        "max_absolute_residual": float(np.max(residuals)) if residuals else 0.0,
        "evaluated_cells": len(rows),
        "unevaluated_cells": unresolved,
        "cells": rows,
    }


__all__ = [
    "SharedEdgeGap",
    "conservation_metrics",
    "directed_hausdorff_external",
    "shared_edge_gap_metrics",
    "symmetric_hausdorff_external",
    "tangent_error_external",
]
