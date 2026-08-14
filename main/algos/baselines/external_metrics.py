"""Metrics for component-aware external baseline reconstructions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.spatial import cKDTree

from main.algos.baselines.external_geometry import (
    CellIndex,
    ExternalBaselineResult,
    ExternalCellReconstruction,
    ExternalArcPrimitive,
    ExternalEllipsePrimitive,
    ExternalLinePrimitive,
    ExternalParabolicPrimitive,
    ExternalPrimitive,
    ExternalQuadraticPrimitive,
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


@dataclass(frozen=True)
class _NativeTargetIndex:
    primitives: Tuple[ExternalPrimitive, ...]
    bbox_minimum: np.ndarray
    bbox_maximum: np.ndarray

    @classmethod
    def build(cls, primitives: Sequence[ExternalPrimitive]) -> "_NativeTargetIndex":
        minimum = []
        maximum = []
        for primitive in primitives:
            points = np.asarray(primitive.bbox_points(), dtype=float)
            minimum.append(points.min(axis=0))
            maximum.append(points.max(axis=0))
        return cls(
            tuple(primitives),
            np.asarray(minimum, dtype=float),
            np.asarray(maximum, dtype=float),
        )

    def distance_to_point(self, point: Sequence[float]) -> float:
        query = np.asarray(point, dtype=float)
        offsets = np.maximum(
            np.maximum(self.bbox_minimum - query, query - self.bbox_maximum),
            0.0,
        )
        lower_bounds = np.linalg.norm(offsets, axis=1)
        best = float("inf")
        for index in np.argsort(lower_bounds):
            if lower_bounds[index] >= best:
                break
            best = min(best, self.primitives[int(index)].distance_to_point(query))
        return best


def _native_supremum_on_primitive(
    source: ExternalPrimitive,
    target: _NativeTargetIndex,
    *,
    tolerance: float,
    minimum_spacing: Optional[float],
    initial_samples: int,
) -> float:
    """Maximize distance to a native target union without point-cloud conversion.

    Projected target endpoints partition the source wherever the nearest target
    primitive can plausibly change. Uniform subdivisions guard long intervals.
    Each resulting interval is optimized independently, making the estimate
    insensitive to an equivalent source or target curve being split into more
    primitives.
    """

    source_length = source.length()
    interval_count = max(4, int(initial_samples))
    if minimum_spacing is not None and minimum_spacing > 0.0:
        interval_count = max(
            interval_count, int(math.ceil(source_length / minimum_spacing))
        )
    breaks = {index / interval_count for index in range(interval_count + 1)}
    for primitive in target.primitives:
        breaks.add(source.closest_parameter(primitive.p_left))
        breaks.add(source.closest_parameter(primitive.p_right))
    ordered = sorted(min(1.0, max(0.0, float(value))) for value in breaks)

    def distance(parameter: float) -> float:
        return target.distance_to_point(source.point(parameter))

    best = max(distance(parameter) for parameter in ordered)
    parameter_tolerance = max(1.0e-13, tolerance / max(source_length, 1.0))
    for lower, upper in zip(ordered[:-1], ordered[1:]):
        if upper - lower <= parameter_tolerance:
            continue
        midpoint = 0.5 * (lower + upper)
        best = max(best, distance(midpoint))
        result = minimize_scalar(
            lambda parameter: -distance(float(parameter)),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": parameter_tolerance},
        )
        if result.success:
            best = max(best, -float(result.fun))
    return best


def directed_hausdorff_external(
    source: Any,
    target: Any,
    *,
    tolerance: Optional[float] = None,
    minimum_spacing: Optional[float] = None,
    initial_samples_per_primitive: int = 16,
) -> float:
    """Estimate one directed supremum using native curves on both sides."""

    source_primitives = _as_primitives(source)
    target_primitives = _as_primitives(target)
    if not source_primitives or not target_primitives:
        return float("inf")
    scale = _bbox_scale(source_primitives + target_primitives)
    tolerance = 1.0e-10 * scale if tolerance is None else float(tolerance)
    target_index = _NativeTargetIndex.build(target_primitives)
    return max(
        _native_supremum_on_primitive(
            primitive,
            target_index,
            tolerance=tolerance,
            minimum_spacing=minimum_spacing,
            initial_samples=initial_samples_per_primitive,
        )
        for primitive in source_primitives
    )


def symmetric_hausdorff_external(source: Any, target: Any, **kwargs: Any) -> float:
    return max(
        directed_hausdorff_external(source, target, **kwargs),
        directed_hausdorff_external(target, source, **kwargs),
    )


def unsigned_curvature_external(
    primitive: ExternalPrimitive, parameter: float
) -> float:
    """Return geometric curvature of any native external primitive."""

    parameter = float(parameter)
    if isinstance(primitive, ExternalLinePrimitive):
        return 0.0
    if isinstance(primitive, ExternalArcPrimitive):
        return 1.0 / primitive.radius
    if isinstance(primitive, ExternalParabolicPrimitive):
        s = primitive.s_start + parameter * (primitive.s_end - primitive.s_start)
        return abs(primitive.curvature) / (1.0 + (primitive.curvature * s) ** 2) ** 1.5
    if isinstance(primitive, ExternalQuadraticPrimitive):
        chord_length = _distance(primitive.p_left, primitive.p_right)
        transverse_speed = 4.0 * primitive.bulge * (1.0 - 2.0 * parameter)
        return (
            abs(8.0 * primitive.bulge * chord_length)
            / (chord_length * chord_length + transverse_speed * transverse_speed) ** 1.5
        )
    if isinstance(primitive, ExternalEllipsePrimitive):
        theta = primitive.start_angle + parameter * primitive.sweep_angle
        denominator = (
            primitive.major_axis**2 * math.sin(theta) ** 2
            + primitive.minor_axis**2 * math.cos(theta) ** 2
        ) ** 1.5
        return primitive.major_axis * primitive.minor_axis / denominator
    raise TypeError(f"unsupported native primitive {type(primitive).__name__}")


def geometric_curvature_error_external(
    reconstruction: Any,
    truth: Any,
    *,
    quadrature_order: int = 16,
) -> Mapping[str, float]:
    """Arc-length-weighted native-curvature error against nearest truth geometry.

    The observable integrates the actual curvature of every reconstructed
    primitive, rather than a method-specific stencil estimate. Its additive
    arc-length quadrature is invariant, to quadrature accuracy, to splitting an
    otherwise identical curve into a different number of facets.
    """

    if quadrature_order < 2:
        raise ValueError("quadrature_order must be at least two")
    source_primitives = _as_primitives(reconstruction)
    truth_primitives = _as_primitives(truth)
    if not source_primitives or not truth_primitives:
        return {
            "mean_absolute_error": float("nan"),
            "rms_error": float("nan"),
            "max_absolute_error": float("nan"),
            "relative_l1_error": float("nan"),
            "reconstructed_length": 0.0,
            "quadrature_samples": 0,
        }

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    absolute_integral = 0.0
    squared_integral = 0.0
    truth_integral = 0.0
    total_length = 0.0
    maximum = 0.0
    sample_count = 0
    for primitive in source_primitives:
        for node, weight in zip(nodes, weights):
            parameter = 0.5 * (float(node) + 1.0)
            point = primitive.point(parameter)
            nearest = min(
                truth_primitives,
                key=lambda candidate: candidate.distance_to_point(point),
            )
            truth_parameter = nearest.closest_parameter(point)
            reconstructed_curvature = unsigned_curvature_external(primitive, parameter)
            truth_curvature = unsigned_curvature_external(nearest, truth_parameter)
            error = abs(reconstructed_curvature - truth_curvature)
            tangent = primitive.tangent(parameter)
            ds = 0.5 * float(weight) * math.hypot(*tangent)
            absolute_integral += ds * error
            squared_integral += ds * error * error
            truth_integral += ds * abs(truth_curvature)
            total_length += ds
            maximum = max(maximum, error)
            sample_count += 1

    return {
        "mean_absolute_error": absolute_integral / total_length,
        "rms_error": math.sqrt(squared_integral / total_length),
        "max_absolute_error": maximum,
        "relative_l1_error": (
            absolute_integral / truth_integral if truth_integral > 0.0 else float("nan")
        ),
        "reconstructed_length": total_length,
        "quadrature_samples": sample_count,
    }


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
    "geometric_curvature_error_external",
    "shared_edge_gap_metrics",
    "symmetric_hausdorff_external",
    "tangent_error_external",
    "unsigned_curvature_external",
]
