"""Geometry-faithful phase-area metrics for reconstructed facets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from main.geoms.circular_facet import getCircleIntersectArea
from main.geoms.corner_facet import getPolyCurvedCornerArea
from main.geoms.geoms import getArea, getPolyLineArea
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.corner_facet import CornerFacet
from main.structs.facets.linear_facet import LinearFacet
from main.structs.interface_geometry import ArcPrimitive, LinePrimitive


class AreaMetricError(RuntimeError):
    """Raised when a reconstruction cannot produce a valid area metric."""


def require_complete_facet_pairs(polygons, facets, *, context: str):
    """Return aligned polygon/facet pairs or fail before metric aggregation."""
    polygon_list = list(polygons)
    facet_list = list(facets)
    if len(polygon_list) != len(facet_list):
        raise AreaMetricError(
            f"{context}: expected one reconstructed facet per polygon, got "
            f"{len(facet_list)} facets for {len(polygon_list)} polygons"
        )

    for facet_index, facet in enumerate(facet_list):
        if facet is None:
            raise AreaMetricError(
                f"{context}: reconstructed facet {facet_index} is missing"
            )

    return list(zip(polygon_list, facet_list))


def facet_from_geometry(record: Mapping[str, Any]):
    """Parse a structured facet record into the canonical facet classes."""
    facet_class = record.get("class")
    if facet_class == "linear":
        return LinearFacet(
            record["p_left"], record["p_right"], record.get("name", "linear")
        )
    if facet_class == "circular":
        return ArcFacet(
            record["center"],
            float(record["radius"]),
            record["p_left"],
            record["p_right"],
        )
    if facet_class in {"linear_corner", "curved_corner", "corner"}:
        left = record.get("left_branch") or {}
        right = record.get("right_branch") or {}
        return CornerFacet(
            centerLeft=left.get("center"),
            centerRight=right.get("center"),
            radiusLeft=(
                float(left["radius"])
                if left.get("radius") is not None
                else None
            ),
            radiusRight=(
                float(right["radius"])
                if right.get("radius") is not None
                else None
            ),
            pLeft=record["p_left"],
            corner=record["corner"],
            pRight=record["p_right"],
        )
    raise ValueError(f"Unsupported facet geometry class: {facet_class!r}")


def facet_area_in_polygon(
    polygon: Sequence[Sequence[float]], facet: Any
) -> float:
    """Return the phase area selected by a live or serialized facet.

    Circular facets represent a signed supporting-circle partition. Their
    finite endpoints delimit the rendered interface segment, but do not change
    which side of the interface is filled inside the reconstruction cell.
    """
    if isinstance(facet, Mapping):
        facet = facet_from_geometry(facet)

    points = [list(point) for point in polygon]
    signed_polygon_area = getArea(points)
    if signed_polygon_area < 0.0:
        points.reverse()
    polygon_area = abs(signed_polygon_area)
    if not isfinite(polygon_area):
        raise ValueError("Polygon area is not finite")

    if isinstance(facet, (LinearFacet, LinePrimitive)):
        area = getPolyLineArea(points, facet.pLeft, facet.pRight)
    elif isinstance(facet, (ArcFacet, ArcPrimitive)):
        area, _ = getCircleIntersectArea(facet.center, facet.radius, points)
    elif isinstance(facet, CornerFacet):
        area = getPolyCurvedCornerArea(
            points,
            facet.pLeft,
            facet.corner,
            facet.pRight,
            facet.radiusLeft,
            facet.radiusRight,
        )
    else:
        raise TypeError(f"Unsupported facet type: {type(facet)!r}")

    if not isfinite(area):
        raise ValueError(f"Facet area is not finite for {type(facet).__name__}")
    tolerance = 1.0e-10 * max(1.0, polygon_area)
    if area < -tolerance or area > polygon_area + tolerance:
        raise ValueError(
            f"Facet area {area} lies outside [0, {polygon_area}] for "
            f"{type(facet).__name__}"
        )
    return min(max(float(area), 0.0), polygon_area)
