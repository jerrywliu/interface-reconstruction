"""Component-aware geometry contract for Cartesian external baselines.

The external methods are deliberately kept outside ``runReconstruction``:
one original cell may own several disconnected interface components, and a
parabolic or quadratic primitive must remain native geometry for metrics and
serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
from scipy.optimize import minimize_scalar


Point = Tuple[float, float]
Polygon = Tuple[Point, ...]
CellIndex = Tuple[int, int]
ExactAreaCallback = Callable[[Sequence[Sequence[float]]], float]


class ExternalReconstructionStatus(str, Enum):
    RECONSTRUCTED = "reconstructed"
    PAPER_FALLBACK = "paper_fallback"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"


def _point(value: Sequence[float]) -> Point:
    if len(value) != 2:
        raise ValueError("points must have exactly two coordinates")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(coordinate) for coordinate in result):
        raise ValueError("point coordinates must be finite")
    return result


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _unit(vector: Sequence[float]) -> Point:
    magnitude = math.hypot(float(vector[0]), float(vector[1]))
    if magnitude == 0.0:
        raise ValueError("zero-length vector")
    return float(vector[0]) / magnitude, float(vector[1]) / magnitude


def _point_on_segment(
    point: Point, edge: Tuple[Point, Point], tolerance: float
) -> bool:
    dx = edge[1][0] - edge[0][0]
    dy = edge[1][1] - edge[0][1]
    length = math.hypot(dx, dy)
    if length == 0.0:
        return False
    cross = dx * (point[1] - edge[0][1]) - dy * (point[0] - edge[0][0])
    if abs(cross) > tolerance * length:
        return False
    projection = (point[0] - edge[0][0]) * dx + (point[1] - edge[0][1]) * dy
    return -tolerance * length <= projection <= length * length + tolerance * length


def _json_value(value: Any, path: str = "metadata") -> Any:
    """Validate and copy metadata into the lossless JSON value domain."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _json_value(float(value), path)
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[]") for item in value]
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            result[key] = _json_value(item, f"{path}.{key}")
        return result
    raise TypeError(f"{path} contains non-JSON value {type(value).__name__}")


@runtime_checkable
class ExternalPrimitive(Protocol):
    """Native primitive interface consumed by external metrics."""

    kind: str

    @property
    def p_left(self) -> Point: ...

    @property
    def p_right(self) -> Point: ...

    def point(self, parameter: float) -> Point: ...

    def tangent(self, parameter: float) -> Point: ...

    def length(self) -> float: ...

    def closest_parameter(self, point: Sequence[float]) -> float: ...

    def distance_to_point(self, point: Sequence[float]) -> float: ...

    def sample(self, count: int) -> Tuple[Point, ...]: ...

    def edge_crossings(
        self, edge: Tuple[Point, Point], tolerance: float = 1.0e-10
    ) -> Tuple[Point, ...]: ...

    def bbox_points(self) -> Tuple[Point, ...]: ...

    def to_dict(self) -> Dict[str, Any]: ...


class _ParametricPrimitiveMixin:
    def sample(self, count: int) -> Tuple[Point, ...]:
        if count < 2:
            raise ValueError("primitive sampling requires at least two points")
        return tuple(self.point(index / (count - 1)) for index in range(count))

    def edge_crossings(
        self, edge: Tuple[Point, Point], tolerance: float = 1.0e-10
    ) -> Tuple[Point, ...]:
        crossings = []
        for candidate in (self.p_left, self.p_right):
            if _point_on_segment(candidate, edge, tolerance) and not any(
                _distance(candidate, existing) <= tolerance for existing in crossings
            ):
                crossings.append(candidate)
        return tuple(crossings)

    def closest_parameter(self, point: Sequence[float]) -> float:
        target = _point(point)

        def squared(parameter: float) -> float:
            candidate = self.point(parameter)
            return (candidate[0] - target[0]) ** 2 + (candidate[1] - target[1]) ** 2

        result = minimize_scalar(
            squared,
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 1.0e-13},
        )
        candidates = [(squared(0.0), 0.0), (squared(1.0), 1.0)]
        if result.success:
            candidates.append((float(result.fun), float(result.x)))
        return min(candidates)[1]

    def distance_to_point(self, point: Sequence[float]) -> float:
        target = _point(point)
        return _distance(target, self.point(self.closest_parameter(target)))

    def length(self) -> float:
        # Native quadrature of the parametric speed; no lower-order conversion.
        nodes, weights = np.polynomial.legendre.leggauss(24)
        total = 0.0
        for node, weight in zip(nodes, weights):
            parameter = 0.5 * (float(node) + 1.0)
            tangent = self.tangent(parameter)
            total += float(weight) * math.hypot(*tangent)
        return 0.5 * total

    def bbox_points(self) -> Tuple[Point, ...]:
        return self.sample(33)


@dataclass(frozen=True)
class ExternalLinePrimitive(_ParametricPrimitiveMixin):
    p_left: Point
    p_right: Point
    metadata: Mapping[str, Any] = field(default_factory=dict)
    kind: str = field(default="line", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "p_left", _point(self.p_left))
        object.__setattr__(self, "p_right", _point(self.p_right))
        if _distance(self.p_left, self.p_right) == 0.0:
            raise ValueError("line endpoints must be distinct")
        object.__setattr__(self, "metadata", _json_value(self.metadata))

    def point(self, parameter: float) -> Point:
        return (
            self.p_left[0] + float(parameter) * (self.p_right[0] - self.p_left[0]),
            self.p_left[1] + float(parameter) * (self.p_right[1] - self.p_left[1]),
        )

    def tangent(self, parameter: float) -> Point:
        return self.p_right[0] - self.p_left[0], self.p_right[1] - self.p_left[1]

    def length(self) -> float:
        return _distance(self.p_left, self.p_right)

    def closest_parameter(self, point: Sequence[float]) -> float:
        point = _point(point)
        dx = self.p_right[0] - self.p_left[0]
        dy = self.p_right[1] - self.p_left[1]
        denominator = dx * dx + dy * dy
        parameter = (
            (point[0] - self.p_left[0]) * dx + (point[1] - self.p_left[1]) * dy
        ) / denominator
        return min(1.0, max(0.0, parameter))

    def bbox_points(self) -> Tuple[Point, ...]:
        return self.p_left, self.p_right

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "p_left": list(self.p_left),
            "p_right": list(self.p_right),
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True)
class ExternalArcPrimitive(_ParametricPrimitiveMixin):
    center: Point
    radius: float
    start_angle: float
    sweep_angle: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    kind: str = field(default="arc", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _point(self.center))
        for name in ("radius", "start_angle", "sweep_angle"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.radius <= 0.0 or self.sweep_angle == 0.0:
            raise ValueError("arc radius and sweep magnitude must be positive")
        if abs(self.sweep_angle) > 2.0 * math.pi + 1.0e-12:
            raise ValueError("one arc primitive cannot sweep more than one revolution")
        object.__setattr__(self, "metadata", _json_value(self.metadata))

    @property
    def p_left(self) -> Point:
        return self.point(0.0)

    @property
    def p_right(self) -> Point:
        return self.point(1.0)

    def point(self, parameter: float) -> Point:
        angle = self.start_angle + float(parameter) * self.sweep_angle
        return (
            self.center[0] + self.radius * math.cos(angle),
            self.center[1] + self.radius * math.sin(angle),
        )

    def tangent(self, parameter: float) -> Point:
        angle = self.start_angle + float(parameter) * self.sweep_angle
        return (
            -self.radius * self.sweep_angle * math.sin(angle),
            self.radius * self.sweep_angle * math.cos(angle),
        )

    def length(self) -> float:
        return self.radius * abs(self.sweep_angle)

    def closest_parameter(self, point: Sequence[float]) -> float:
        point = _point(point)
        angle = math.atan2(point[1] - self.center[1], point[0] - self.center[0])
        candidates = [0.0, 1.0]
        for turns in range(-2, 3):
            candidate = (
                angle + 2.0 * math.pi * turns - self.start_angle
            ) / self.sweep_angle
            if 0.0 <= candidate <= 1.0:
                candidates.append(candidate)
        return min(
            candidates, key=lambda parameter: _distance(point, self.point(parameter))
        )

    def bbox_points(self) -> Tuple[Point, ...]:
        parameters = [0.0, 1.0]
        for critical in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            for turns in range(-2, 3):
                parameter = (
                    critical + 2.0 * math.pi * turns - self.start_angle
                ) / self.sweep_angle
                if 0.0 < parameter < 1.0:
                    parameters.append(parameter)
        return tuple(self.point(parameter) for parameter in sorted(set(parameters)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "center": list(self.center),
            "radius": self.radius,
            "start_angle": self.start_angle,
            "sweep_angle": self.sweep_angle,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True)
class ExternalParabolicPrimitive(_ParametricPrimitiveMixin):
    """PLVIRA parabola over a retained local-tangent interval."""

    center: Point
    angle: float
    curvature: float
    shift: float
    s_start: float
    s_end: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    kind: str = field(default="parabola", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _point(self.center))
        for name in ("angle", "curvature", "shift", "s_start", "s_end"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.s_start == self.s_end:
            raise ValueError("parabolic interval must have positive extent")
        object.__setattr__(self, "metadata", _json_value(self.metadata))

    @property
    def p_left(self) -> Point:
        return self.point(0.0)

    @property
    def p_right(self) -> Point:
        return self.point(1.0)

    def _frame(self) -> Tuple[Point, Point]:
        normal = (math.cos(self.angle), math.sin(self.angle))
        tangent = (-normal[1], normal[0])
        return normal, tangent

    def point(self, parameter: float) -> Point:
        s = self.s_start + float(parameter) * (self.s_end - self.s_start)
        n = self.shift - 0.5 * self.curvature * s * s
        normal, tangent = self._frame()
        return (
            self.center[0] + tangent[0] * s + normal[0] * n,
            self.center[1] + tangent[1] * s + normal[1] * n,
        )

    def tangent(self, parameter: float) -> Point:
        s = self.s_start + float(parameter) * (self.s_end - self.s_start)
        ds = self.s_end - self.s_start
        normal, tangent = self._frame()
        return (
            ds * (tangent[0] - self.curvature * s * normal[0]),
            ds * (tangent[1] - self.curvature * s * normal[1]),
        )

    def closest_parameter(self, point: Sequence[float]) -> float:
        target = _point(point)
        normal, tangent = self._frame()
        displacement = (
            target[0] - self.center[0],
            target[1] - self.center[1],
        )
        target_s = displacement[0] * tangent[0] + displacement[1] * tangent[1]
        target_n = displacement[0] * normal[0] + displacement[1] * normal[1]
        normal_offset = self.shift - target_n
        coefficients = (
            0.5 * self.curvature * self.curvature,
            0.0,
            1.0 - self.curvature * normal_offset,
            -target_s,
        )
        candidates = [self.s_start, self.s_end]
        lower, upper = sorted((self.s_start, self.s_end))
        for root in np.roots(coefficients):
            if abs(float(root.imag)) <= 1.0e-10 * max(1.0, abs(float(root.real))):
                value = float(root.real)
                if lower <= value <= upper:
                    candidates.append(value)

        parameters = [
            (candidate - self.s_start) / (self.s_end - self.s_start)
            for candidate in candidates
        ]
        return min(
            parameters,
            key=lambda parameter: _distance(target, self.point(parameter)),
        )

    def bbox_points(self) -> Tuple[Point, ...]:
        parameters = [0.0, 1.0]
        tangent_start = self.tangent(0.0)
        tangent_end = self.tangent(1.0)
        for coordinate in (0, 1):
            change = tangent_end[coordinate] - tangent_start[coordinate]
            if change != 0.0:
                parameter = -tangent_start[coordinate] / change
                if 0.0 < parameter < 1.0:
                    parameters.append(parameter)
        return tuple(self.point(parameter) for parameter in sorted(set(parameters)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "center": list(self.center),
            "angle": self.angle,
            "curvature": self.curvature,
            "shift": self.shift,
            "s_start": self.s_start,
            "s_end": self.s_end,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True)
class ExternalQuadraticPrimitive(_ParametricPrimitiveMixin):
    """QUASI-style quadratic in its oriented chord frame."""

    p_left: Point
    p_right: Point
    bulge: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    kind: str = field(default="quadratic", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "p_left", _point(self.p_left))
        object.__setattr__(self, "p_right", _point(self.p_right))
        if _distance(self.p_left, self.p_right) == 0.0:
            raise ValueError("quadratic endpoints must be distinct")
        if not math.isfinite(float(self.bulge)):
            raise ValueError("bulge must be finite")
        object.__setattr__(self, "metadata", _json_value(self.metadata))

    def _frame(self) -> Tuple[Point, Point, float]:
        chord = (self.p_right[0] - self.p_left[0], self.p_right[1] - self.p_left[1])
        length = math.hypot(*chord)
        normal = (-chord[1] / length, chord[0] / length)
        return chord, normal, length

    def point(self, parameter: float) -> Point:
        parameter = float(parameter)
        chord, normal, _ = self._frame()
        displacement = 4.0 * self.bulge * parameter * (1.0 - parameter)
        return (
            self.p_left[0] + parameter * chord[0] + displacement * normal[0],
            self.p_left[1] + parameter * chord[1] + displacement * normal[1],
        )

    def tangent(self, parameter: float) -> Point:
        chord, normal, _ = self._frame()
        derivative = 4.0 * self.bulge * (1.0 - 2.0 * float(parameter))
        return chord[0] + derivative * normal[0], chord[1] + derivative * normal[1]

    def closest_parameter(self, point: Sequence[float]) -> float:
        target = _point(point)
        chord, normal, length = self._frame()
        tangent = (chord[0] / length, chord[1] / length)
        displacement = (
            target[0] - self.p_left[0],
            target[1] - self.p_left[1],
        )
        target_x = displacement[0] * tangent[0] + displacement[1] * tangent[1]
        target_y = displacement[0] * normal[0] + displacement[1] * normal[1]
        quadratic = -4.0 * self.bulge
        linear = 4.0 * self.bulge
        coefficients = (
            2.0 * quadratic * quadratic,
            3.0 * quadratic * linear,
            linear * linear - 2.0 * quadratic * target_y + length * length,
            -linear * target_y - length * target_x,
        )
        candidates = [0.0, 1.0]
        for root in np.roots(coefficients):
            if abs(float(root.imag)) <= 1.0e-10 * max(1.0, abs(float(root.real))):
                parameter = float(root.real)
                if 0.0 <= parameter <= 1.0:
                    candidates.append(parameter)
        return min(
            candidates, key=lambda parameter: _distance(target, self.point(parameter))
        )

    def bbox_points(self) -> Tuple[Point, ...]:
        parameters = [0.0, 1.0]
        tangent_start = self.tangent(0.0)
        tangent_end = self.tangent(1.0)
        for coordinate in (0, 1):
            change = tangent_end[coordinate] - tangent_start[coordinate]
            if change != 0.0:
                parameter = -tangent_start[coordinate] / change
                if 0.0 < parameter < 1.0:
                    parameters.append(parameter)
        return tuple(self.point(parameter) for parameter in sorted(set(parameters)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "p_left": list(self.p_left),
            "p_right": list(self.p_right),
            "bulge": self.bulge,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True)
class ExternalEllipsePrimitive(_ParametricPrimitiveMixin):
    """Analytic ellipse truth primitive, optionally restricted to an arc."""

    center: Point
    major_axis: float
    minor_axis: float
    angle: float = 0.0
    start_angle: float = 0.0
    sweep_angle: float = 2.0 * math.pi
    metadata: Mapping[str, Any] = field(default_factory=dict)
    kind: str = field(default="ellipse", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _point(self.center))
        for name in (
            "major_axis",
            "minor_axis",
            "angle",
            "start_angle",
            "sweep_angle",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.major_axis <= 0.0 or self.minor_axis <= 0.0:
            raise ValueError("ellipse semiaxes must be positive")
        if self.sweep_angle == 0.0 or abs(self.sweep_angle) > 2.0 * math.pi + 1.0e-12:
            raise ValueError("ellipse sweep must span at most one revolution")
        object.__setattr__(self, "metadata", _json_value(self.metadata))

    def _world_point(self, theta: float) -> Point:
        cosine, sine = math.cos(self.angle), math.sin(self.angle)
        local_x = self.major_axis * math.cos(theta)
        local_y = self.minor_axis * math.sin(theta)
        return (
            self.center[0] + cosine * local_x - sine * local_y,
            self.center[1] + sine * local_x + cosine * local_y,
        )

    @property
    def p_left(self) -> Point:
        return self.point(0.0)

    @property
    def p_right(self) -> Point:
        return self.point(1.0)

    def point(self, parameter: float) -> Point:
        return self._world_point(self.start_angle + float(parameter) * self.sweep_angle)

    def tangent(self, parameter: float) -> Point:
        theta = self.start_angle + float(parameter) * self.sweep_angle
        cosine, sine = math.cos(self.angle), math.sin(self.angle)
        local_x = -self.major_axis * math.sin(theta) * self.sweep_angle
        local_y = self.minor_axis * math.cos(theta) * self.sweep_angle
        return (
            cosine * local_x - sine * local_y,
            sine * local_x + cosine * local_y,
        )

    def closest_parameter(self, point: Sequence[float]) -> float:
        target = _point(point)
        dx = target[0] - self.center[0]
        dy = target[1] - self.center[1]
        cosine, sine = math.cos(self.angle), math.sin(self.angle)
        target_x = cosine * dx + sine * dy
        target_y = -sine * dx + cosine * dy
        a, b = self.major_axis, self.minor_axis
        delta = b * b - a * a
        coefficients = (
            b * target_y,
            -2.0 * delta + 2.0 * a * target_x,
            0.0,
            2.0 * delta + 2.0 * a * target_x,
            -b * target_y,
        )
        angles = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]
        for root in np.roots(coefficients):
            if abs(float(root.imag)) <= 1.0e-10 * max(1.0, abs(float(root.real))):
                angles.append(2.0 * math.atan(float(root.real)))

        candidates = [0.0, 1.0]
        for theta in angles:
            for turns in range(-2, 3):
                parameter = (
                    theta + 2.0 * math.pi * turns - self.start_angle
                ) / self.sweep_angle
                if 0.0 <= parameter <= 1.0:
                    candidates.append(parameter)
        return min(
            candidates,
            key=lambda parameter: _distance(target, self.point(parameter)),
        )

    def bbox_points(self) -> Tuple[Point, ...]:
        cosine, sine = math.cos(self.angle), math.sin(self.angle)
        critical_angles = (
            math.atan2(-self.minor_axis * sine, self.major_axis * cosine),
            math.atan2(-self.minor_axis * sine, self.major_axis * cosine) + math.pi,
            math.atan2(self.minor_axis * cosine, self.major_axis * sine),
            math.atan2(self.minor_axis * cosine, self.major_axis * sine) + math.pi,
        )
        parameters = [0.0, 1.0]
        for theta in critical_angles:
            for turns in range(-2, 3):
                parameter = (
                    theta + 2.0 * math.pi * turns - self.start_angle
                ) / self.sweep_angle
                if 0.0 < parameter < 1.0:
                    parameters.append(parameter)
        return tuple(self.point(parameter) for parameter in sorted(set(parameters)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "center": list(self.center),
            "major_axis": self.major_axis,
            "minor_axis": self.minor_axis,
            "angle": self.angle,
            "start_angle": self.start_angle,
            "sweep_angle": self.sweep_angle,
            "metadata": _json_value(self.metadata),
        }


PrimitiveTypes = (
    ExternalLinePrimitive,
    ExternalArcPrimitive,
    ExternalParabolicPrimitive,
    ExternalQuadraticPrimitive,
    ExternalEllipsePrimitive,
)


def adapt_linear_facet(
    facet: Any, metadata: Optional[Mapping[str, Any]] = None
) -> ExternalLinePrimitive:
    """Adapt a line-like object without retaining its project facet type."""

    p_left = getattr(facet, "pLeft", getattr(facet, "p_left", None))
    p_right = getattr(facet, "pRight", getattr(facet, "p_right", None))
    if p_left is None or p_right is None:
        raise TypeError("line adapter requires pLeft/pRight or p_left/p_right")
    return ExternalLinePrimitive(p_left, p_right, metadata or {})


def adapt_quadratic_facet(
    facet: Any, metadata: Optional[Mapping[str, Any]] = None
) -> ExternalQuadraticPrimitive:
    """Adapt a QUASI-like object while preserving its native bulge."""

    if not hasattr(facet, "bulge"):
        raise TypeError("quadratic adapter requires a bulge attribute")
    line = adapt_linear_facet(facet)
    return ExternalQuadraticPrimitive(
        line.p_left, line.p_right, float(facet.bulge), metadata or {}
    )


def adapt_parabolic_interval(
    interface: Any,
    s_start: float,
    s_end: float,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ExternalParabolicPrimitive:
    """Adapt one clipped interval from a PLVIRA-like interface."""

    required = ("center", "angle", "curvature", "shift")
    missing = [name for name in required if not hasattr(interface, name)]
    if missing:
        raise TypeError(
            f"parabolic adapter is missing attributes: {', '.join(missing)}"
        )
    return ExternalParabolicPrimitive(
        interface.center,
        float(interface.angle),
        float(interface.curvature),
        float(interface.shift),
        float(s_start),
        float(s_end),
        metadata or {},
    )


def primitive_from_dict(payload: Mapping[str, Any]) -> ExternalPrimitive:
    kind = payload.get("kind")
    metadata = payload.get("metadata", {})
    if kind == "line":
        return ExternalLinePrimitive(payload["p_left"], payload["p_right"], metadata)
    if kind == "arc":
        return ExternalArcPrimitive(
            payload["center"],
            float(payload["radius"]),
            float(payload["start_angle"]),
            float(payload["sweep_angle"]),
            metadata,
        )
    if kind == "parabola":
        return ExternalParabolicPrimitive(
            payload["center"],
            float(payload["angle"]),
            float(payload["curvature"]),
            float(payload["shift"]),
            float(payload["s_start"]),
            float(payload["s_end"]),
            metadata,
        )
    if kind == "quadratic":
        return ExternalQuadraticPrimitive(
            payload["p_left"],
            payload["p_right"],
            float(payload["bulge"]),
            metadata,
        )
    if kind == "ellipse":
        return ExternalEllipsePrimitive(
            payload["center"],
            float(payload["major_axis"]),
            float(payload["minor_axis"]),
            float(payload.get("angle", 0.0)),
            float(payload.get("start_angle", 0.0)),
            float(payload.get("sweep_angle", 2.0 * math.pi)),
            metadata,
        )
    raise ValueError(f"unknown external primitive kind: {kind!r}")


@dataclass(frozen=True)
class ExternalInterfaceComponent:
    primitives: Tuple[ExternalPrimitive, ...]
    closed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        primitives = tuple(self.primitives)
        if not primitives:
            raise ValueError("an interface component must contain a primitive")
        for primitive in primitives:
            if not isinstance(primitive, ExternalPrimitive):
                raise TypeError("component primitives must satisfy ExternalPrimitive")
        object.__setattr__(self, "primitives", primitives)
        object.__setattr__(self, "metadata", _json_value(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "closed": self.closed,
            "primitives": [primitive.to_dict() for primitive in self.primitives],
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalInterfaceComponent":
        return cls(
            tuple(primitive_from_dict(item) for item in payload["primitives"]),
            bool(payload.get("closed", False)),
            payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class ExternalCellReconstruction:
    cell_index: CellIndex
    polygon: Polygon
    components: Tuple[ExternalInterfaceComponent, ...]
    source_method: str
    source_variant: str
    status: ExternalReconstructionStatus
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    target_phase_area: Optional[float] = None
    stored_exact_phase_area: Optional[float] = None
    exact_area_callback: Optional[ExactAreaCallback] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cell_index", (int(self.cell_index[0]), int(self.cell_index[1]))
        )
        polygon = tuple(_point(point) for point in self.polygon)
        if len(polygon) < 3:
            raise ValueError("cell polygons require at least three vertices")
        object.__setattr__(self, "polygon", polygon)
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "status", ExternalReconstructionStatus(self.status))
        object.__setattr__(self, "diagnostics", _json_value(self.diagnostics))
        if (
            self.status
            in (
                ExternalReconstructionStatus.RECONSTRUCTED,
                ExternalReconstructionStatus.PAPER_FALLBACK,
            )
            and not self.components
        ):
            raise ValueError(f"status {self.status.value} requires geometry")
        if (
            self.status
            in (
                ExternalReconstructionStatus.UNSUPPORTED,
                ExternalReconstructionStatus.UNRESOLVED,
            )
            and self.components
        ):
            raise ValueError(f"status {self.status.value} cannot carry geometry")
        for name in ("target_phase_area", "stored_exact_phase_area"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or value < 0.0):
                raise ValueError(f"{name} must be a finite nonnegative value")

    def exact_phase_area(
        self, polygon: Optional[Sequence[Sequence[float]]] = None
    ) -> float:
        target_polygon = self.polygon if polygon is None else polygon
        if self.exact_area_callback is not None:
            value = float(self.exact_area_callback(target_polygon))
        elif polygon is None and self.stored_exact_phase_area is not None:
            value = float(self.stored_exact_phase_area)
        else:
            raise RuntimeError(
                "exact area callback is unavailable for this polygon; serialized "
                "records retain only the exact area evaluated in their owning cell"
            )
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("exact area callback returned an invalid area")
        return value

    def primitives(self) -> Tuple[ExternalPrimitive, ...]:
        return tuple(
            primitive
            for component in self.components
            for primitive in component.primitives
        )

    def to_dict(self) -> Dict[str, Any]:
        exact_area = self.stored_exact_phase_area
        if exact_area is None and self.exact_area_callback is not None:
            exact_area = self.exact_phase_area()
        return {
            "cell_index": list(self.cell_index),
            "polygon": [list(point) for point in self.polygon],
            "components": [component.to_dict() for component in self.components],
            "source_method": self.source_method,
            "source_variant": self.source_variant,
            "status": self.status.value,
            "diagnostics": _json_value(self.diagnostics),
            "target_phase_area": self.target_phase_area,
            "exact_phase_area": exact_area,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalCellReconstruction":
        return cls(
            cell_index=tuple(payload["cell_index"]),
            polygon=tuple(tuple(point) for point in payload["polygon"]),
            components=tuple(
                ExternalInterfaceComponent.from_dict(item)
                for item in payload.get("components", [])
            ),
            source_method=str(payload["source_method"]),
            source_variant=str(payload["source_variant"]),
            status=ExternalReconstructionStatus(payload["status"]),
            diagnostics=payload.get("diagnostics", {}),
            target_phase_area=payload.get("target_phase_area"),
            stored_exact_phase_area=payload.get("exact_phase_area"),
        )


@dataclass(frozen=True)
class ExternalBaselineResult:
    source_method: str
    source_variant: str
    cells: Mapping[CellIndex, ExternalCellReconstruction]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        normalized = {}
        for index, record in self.cells.items():
            key = (int(index[0]), int(index[1]))
            if key != record.cell_index:
                raise ValueError("cell mapping key must equal record.cell_index")
            if record.source_method != self.source_method:
                raise ValueError("cell source_method differs from run source_method")
            if record.source_variant != self.source_variant:
                raise ValueError("cell source_variant differs from run source_variant")
            normalized[key] = record
        object.__setattr__(self, "cells", normalized)
        object.__setattr__(self, "metadata", _json_value(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_method": self.source_method,
            "source_variant": self.source_variant,
            "metadata": _json_value(self.metadata),
            "cells": [self.cells[index].to_dict() for index in sorted(self.cells)],
        }

    def to_json(self, path: Optional[Path] = None, *, indent: int = 2) -> str:
        serialized = json.dumps(
            self.to_dict(), indent=indent, sort_keys=True, allow_nan=False
        )
        if path is not None:
            Path(path).write_text(serialized + "\n", encoding="utf-8")
        return serialized

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalBaselineResult":
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("unsupported external baseline schema version")
        cells = [
            ExternalCellReconstruction.from_dict(item) for item in payload["cells"]
        ]
        if len({cell.cell_index for cell in cells}) != len(cells):
            raise ValueError("external baseline JSON contains duplicate cell indices")
        return cls(
            source_method=str(payload["source_method"]),
            source_variant=str(payload["source_variant"]),
            cells={cell.cell_index: cell for cell in cells},
            metadata=payload.get("metadata", {}),
            schema_version=1,
        )

    @classmethod
    def from_json(cls, source: Any) -> "ExternalBaselineResult":
        if isinstance(source, Path):
            payload = json.loads(source.read_text(encoding="utf-8"))
        else:
            payload = json.loads(str(source))
        return cls.from_dict(payload)


def flatten_external_primitives(
    geometry: Iterable[ExternalCellReconstruction],
) -> Tuple[ExternalPrimitive, ...]:
    return tuple(primitive for cell in geometry for primitive in cell.primitives())


__all__ = [
    "CellIndex",
    "ExactAreaCallback",
    "ExternalArcPrimitive",
    "ExternalBaselineResult",
    "ExternalCellReconstruction",
    "ExternalInterfaceComponent",
    "ExternalLinePrimitive",
    "ExternalParabolicPrimitive",
    "ExternalPrimitive",
    "ExternalQuadraticPrimitive",
    "ExternalReconstructionStatus",
    "Point",
    "Polygon",
    "flatten_external_primitives",
    "adapt_linear_facet",
    "adapt_parabolic_interval",
    "adapt_quadratic_facet",
    "primitive_from_dict",
]
