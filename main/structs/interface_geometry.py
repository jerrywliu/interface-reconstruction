from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union

from main.geoms.geoms import getDistance, lerp
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.corner_facet import CornerFacet
from main.structs.facets.linear_facet import LinearFacet


@dataclass
class InterfaceJoint:
    point: List[float]
    kind: str = "smooth"


class LinePrimitive:
    name = "linear"

    def __init__(self, pLeft, pRight, source_name: str = "linear"):
        self.pLeft = list(pLeft)
        self.pRight = list(pRight)
        self.source_name = source_name
        self.curvature = 0.0

    def length(self) -> float:
        return getDistance(self.pLeft, self.pRight)

    def sample(self, n: int, mode: str = "arclength"):
        if n <= 1:
            return [self.pLeft]
        points = []
        for i in range(n):
            t = i / (n - 1)
            points.append(lerp(self.pLeft, self.pRight, t))
        return points

    def sample_by_max_spacing(self, max_spacing: float):
        if max_spacing <= 0:
            return self.sample(2)
        n = max(2, int(math.ceil(self.length() / max_spacing)) + 1)
        return self.sample(n)

    def getTangent(self, p):
        return [
            self.pRight[0] - self.pLeft[0],
            self.pRight[1] - self.pLeft[1],
        ]

    def getLeftTangent(self):
        return self.getTangent(self.pLeft)

    def getRightTangent(self):
        return self.getTangent(self.pRight)

    def normal_at_point(self, p):
        tangent = self.getTangent(p)
        tangent_norm = math.hypot(tangent[0], tangent[1])
        if tangent_norm < 1e-12:
            return None
        return [tangent[1] / tangent_norm, -tangent[0] / tangent_norm]

    def distance_to_point(self, point) -> float:
        ax, ay = self.pLeft
        bx, by = self.pRight
        px, py = point
        vx = bx - ax
        vy = by - ay
        vv = vx * vx + vy * vy
        if vv < 1e-24:
            return math.hypot(px - ax, py - ay)
        wx = px - ax
        wy = py - ay
        t = (wx * vx + wy * vy) / vv
        if t <= 0.0:
            dx = px - ax
            dy = py - ay
        elif t >= 1.0:
            dx = px - bx
            dy = py - by
        else:
            projx = ax + t * vx
            projy = ay + t * vy
            dx = px - projx
            dy = py - projy
        return math.hypot(dx, dy)

    def bbox_points(self):
        return [self.pLeft, self.pRight]


class ArcPrimitive:
    name = "arc"

    def __init__(self, center, radius, pLeft, pRight, source_name: str = "arc"):
        self.center = list(center)
        self.radius = float(radius)
        self.pLeft = list(pLeft)
        self.pRight = list(pRight)
        self.source_name = source_name
        self._facet = ArcFacet(self.center, self.radius, self.pLeft, self.pRight)
        self.curvature = self._facet.curvature
        self.midpoint = self._facet.midpoint
        self.is_major_arc = self._facet.is_major_arc

    def _signed_delta(self) -> float:
        def _norm_angle(theta):
            return theta % (2 * math.pi)

        def _in_ccw_range(start, end, angle):
            if start <= end:
                return start <= angle <= end
            return angle >= start or angle <= end

        start_angle = math.atan2(
            self.pLeft[1] - self.center[1], self.pLeft[0] - self.center[0]
        )
        end_angle = math.atan2(
            self.pRight[1] - self.center[1], self.pRight[0] - self.center[0]
        )
        mid_angle = math.atan2(
            self.midpoint[1] - self.center[1], self.midpoint[0] - self.center[0]
        )

        start_n = _norm_angle(start_angle)
        end_n = _norm_angle(end_angle)
        mid_n = _norm_angle(mid_angle)
        delta_ccw = (end_n - start_n) % (2 * math.pi)
        if delta_ccw == 0:
            return 0.0
        if _in_ccw_range(start_n, end_n, mid_n):
            return delta_ccw
        return delta_ccw - 2 * math.pi

    def length(self) -> float:
        return abs(self._signed_delta()) * abs(self.radius)

    def sample(self, n: int, mode: str = "arclength"):
        return self._facet.sample(n, mode=mode)

    def sample_by_max_spacing(self, max_spacing: float):
        if max_spacing <= 0:
            return self.sample(2)
        n = max(2, int(math.ceil(self.length() / max_spacing)) + 1)
        return self.sample(n)

    def getTangent(self, p):
        return self._facet.getTangent(p)

    def getLeftTangent(self):
        return self.getTangent(self.pLeft)

    def getRightTangent(self):
        return self.getTangent(self.pRight)

    def normal_at_point(self, p):
        distance = getDistance(p, self.center)
        if distance < 1e-12:
            return None
        if self.radius > 0:
            return [
                (p[0] - self.center[0]) / distance,
                (p[1] - self.center[1]) / distance,
            ]
        return [
            (self.center[0] - p[0]) / distance,
            (self.center[1] - p[1]) / distance,
        ]

    def distance_to_point(self, point) -> float:
        radial_distance = getDistance(point, self.center)
        circle_radius = abs(self.radius)
        if radial_distance < 1e-12:
            return circle_radius

        projected = [
            self.center[0]
            + circle_radius * (point[0] - self.center[0]) / radial_distance,
            self.center[1]
            + circle_radius * (point[1] - self.center[1]) / radial_distance,
        ]
        if self._facet.pointInArcRange(projected):
            return abs(radial_distance - circle_radius)
        return min(getDistance(point, self.pLeft), getDistance(point, self.pRight))

    def bbox_points(self):
        return [self.pLeft, self.pRight, self.center]


Primitive = Union[LinePrimitive, ArcPrimitive]


@dataclass
class CompositeFacet:
    primitives: List[Primitive]
    joints: List[InterfaceJoint]
    source_name: str
    source_facet: Optional[object] = None

    def sample(self, n: int, mode: str = "arclength"):
        if not self.primitives:
            return []
        if len(self.primitives) == 1:
            return self.primitives[0].sample(n, mode=mode)
        if n <= 1:
            return [self.primitives[0].pLeft]

        lengths = [max(prim.length(), 1e-12) for prim in self.primitives]
        total_length = sum(lengths)
        counts = []
        remaining = n + len(self.primitives) - 1
        for i, length in enumerate(lengths):
            if i == len(lengths) - 1:
                count = max(2, remaining)
            else:
                weight = length / total_length if total_length > 0 else 1 / len(lengths)
                count = max(2, int(round(weight * n)))
            counts.append(count)
            remaining -= count - 1

        samples = []
        for i, (prim, count) in enumerate(zip(self.primitives, counts)):
            pts = prim.sample(count, mode=mode)
            if i > 0 and pts:
                pts = pts[1:]
            samples.extend(pts)
        return samples


def primitive_from_facet(facet, source_name: Optional[str] = None) -> Primitive:
    if isinstance(facet, LinePrimitive) or isinstance(facet, ArcPrimitive):
        return facet
    if isinstance(facet, LinearFacet):
        return LinePrimitive(facet.pLeft, facet.pRight, source_name or facet.name)
    if isinstance(facet, ArcFacet):
        return ArcPrimitive(
            facet.center,
            facet.radius,
            facet.pLeft,
            facet.pRight,
            source_name or facet.name,
        )
    raise TypeError(f"Unsupported facet type for primitive conversion: {type(facet)!r}")


def composite_from_facet(facet) -> CompositeFacet:
    if facet is None:
        return CompositeFacet([], [], "none", source_facet=None)
    if isinstance(facet, CompositeFacet):
        return facet
    if isinstance(facet, CornerFacet):
        left = primitive_from_facet(facet.facetLeft, source_name=facet.name)
        right = primitive_from_facet(facet.facetRight, source_name=facet.name)
        return CompositeFacet(
            primitives=[left, right],
            joints=[InterfaceJoint(list(facet.corner), kind="corner")],
            source_name=facet.name,
            source_facet=facet,
        )
    primitive = primitive_from_facet(facet)
    return CompositeFacet(
        primitives=[primitive],
        joints=[],
        source_name=getattr(facet, "name", primitive.name),
        source_facet=facet,
    )


def iter_facet_primitives(facet) -> Iterable[Primitive]:
    for primitive in composite_from_facet(facet).primitives:
        yield primitive


def iter_primitives_from_facets(facets: Iterable) -> Iterable[Primitive]:
    for facet in facets:
        if facet is None:
            continue
        yield from iter_facet_primitives(facet)


def primitive_type_code(primitive: Primitive) -> int:
    if primitive.source_name == "corner":
        if isinstance(primitive, LinePrimitive):
            return 2
        return 3
    if isinstance(primitive, ArcPrimitive):
        return 1
    return {
        "linear": 0,
        "default_linear": 4,
        "linear_deadend": 5,
        "Youngs": 6,
        "ELVIRA": 7,
        "LVIRA": 8,
    }.get(primitive.source_name, 0)
