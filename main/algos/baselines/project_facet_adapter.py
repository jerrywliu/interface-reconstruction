"""Adapters from project line/arc facets to common native metric geometry."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Tuple

from main.algos.baselines.external_geometry import (
    ExternalArcPrimitive,
    ExternalLinePrimitive,
    ExternalPrimitive,
)
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.linear_facet import LinearFacet


Point = Tuple[float, float]


def _point(value: Sequence[float], name: str) -> Point:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    result = float(value[0]), float(value[1])
    if not all(math.isfinite(coordinate) for coordinate in result):
        raise ValueError(f"{name} coordinates must be finite")
    return result


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(
        float(left[0]) - float(right[0]), float(left[1]) - float(right[1])
    )


def _signed_arc_sweep(
    center: Point,
    p_left: Point,
    p_right: Point,
    midpoint: Point,
) -> float:
    start = math.atan2(p_left[1] - center[1], p_left[0] - center[0])
    middle = math.atan2(midpoint[1] - center[1], midpoint[0] - center[0])
    end = math.atan2(p_right[1] - center[1], p_right[0] - center[0])
    counterclockwise = (end - start) % (2.0 * math.pi)
    middle_counterclockwise = (middle - start) % (2.0 * math.pi)
    if counterclockwise == 0.0:
        raise ValueError("project arc endpoints define a zero angular span")
    if middle_counterclockwise <= counterclockwise + 1.0e-12:
        return counterclockwise
    return counterclockwise - 2.0 * math.pi


def _external_arc(
    *,
    center: Sequence[float],
    radius: float,
    p_left: Sequence[float],
    p_right: Sequence[float],
    sweep_angle: float,
    metadata: Mapping[str, Any],
) -> ExternalArcPrimitive:
    center_point = _point(center, "arc center")
    left_point = _point(p_left, "arc left endpoint")
    right_point = _point(p_right, "arc right endpoint")
    unsigned_radius = abs(float(radius))
    sweep_angle = float(sweep_angle)
    if not math.isfinite(unsigned_radius) or unsigned_radius <= 0.0:
        raise ValueError("project arc radius must be finite and nonzero")
    if not math.isfinite(sweep_angle) or sweep_angle == 0.0:
        raise ValueError("project arc sweep must be finite and nonzero")

    endpoint_tolerance = 1.0e-9 * max(1.0, unsigned_radius)
    for name, point in (("left", left_point), ("right", right_point)):
        radial_error = abs(_distance(center_point, point) - unsigned_radius)
        if radial_error > endpoint_tolerance:
            raise ValueError(
                f"project arc {name} endpoint misses its circle by {radial_error:.3e}"
            )

    start_angle = math.atan2(
        left_point[1] - center_point[1], left_point[0] - center_point[0]
    )
    primitive = ExternalArcPrimitive(
        center_point,
        unsigned_radius,
        start_angle,
        sweep_angle,
        metadata,
    )
    if _distance(primitive.p_left, left_point) > endpoint_tolerance:
        raise ValueError("external arc does not reproduce the project left endpoint")
    if _distance(primitive.p_right, right_point) > endpoint_tolerance:
        raise ValueError("external arc does not reproduce the project right endpoint")
    return primitive


def external_primitive_from_project_facet(
    facet: LinearFacet | ArcFacet,
) -> ExternalPrimitive:
    """Convert a live project facet without reducing its native geometry."""

    metadata = {
        "adapter_source": "project_facet",
        "project_facet_type": type(facet).__name__,
        "project_source_name": str(facet.name),
    }
    if isinstance(facet, LinearFacet):
        return ExternalLinePrimitive(facet.pLeft, facet.pRight, metadata)
    if isinstance(facet, ArcFacet):
        center = _point(facet.center, "arc center")
        p_left = _point(facet.pLeft, "arc left endpoint")
        p_right = _point(facet.pRight, "arc right endpoint")
        midpoint = _point(facet.midpoint, "arc midpoint")
        return _external_arc(
            center=center,
            radius=facet.radius,
            p_left=p_left,
            p_right=p_right,
            sweep_angle=_signed_arc_sweep(center, p_left, p_right, midpoint),
            metadata=metadata,
        )
    raise TypeError(f"unsupported project facet type: {type(facet).__name__}")


def external_primitive_from_facet_metadata(
    record: Mapping[str, Any],
) -> ExternalPrimitive:
    """Convert one exact schema-v2 ``writeFacets`` primitive record."""

    kind = record.get("kind")
    metadata = {
        "adapter_source": "writeFacets_schema_v2",
        "project_source_name": record.get("source_name"),
        "facet_index": record.get("facet_index"),
        "primitive_index": record.get("primitive_index"),
        "serialized_index": record.get("index"),
    }
    if kind == "line":
        return ExternalLinePrimitive(record["p_left"], record["p_right"], metadata)
    if kind == "arc":
        return _external_arc(
            center=record["center"],
            radius=float(record["radius"]),
            p_left=record["p_left"],
            p_right=record["p_right"],
            sweep_angle=float(record["signed_delta"]),
            metadata=metadata,
        )
    raise ValueError(f"unsupported writeFacets primitive kind: {kind!r}")


def external_primitives_from_facet_metadata(
    payload: Mapping[str, Any],
) -> Tuple[ExternalPrimitive, ...]:
    """Convert a complete exact facet-metadata sidecar."""

    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError("expected writeFacets facet metadata schema version 2")
    primitives = tuple(
        external_primitive_from_facet_metadata(record)
        for record in payload.get("primitives", ())
    )
    if not primitives:
        raise ValueError("facet metadata contains no active primitives")
    return primitives


__all__ = [
    "external_primitive_from_facet_metadata",
    "external_primitive_from_project_facet",
    "external_primitives_from_facet_metadata",
]
