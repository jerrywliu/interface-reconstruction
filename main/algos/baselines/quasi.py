"""Static, Cartesian prototype of the 2009 QUASI reconstruction method.

The implementation follows Sections 2.1--2.4 of Diwakar, Das, and
Sundararajan (JCP 228, 2009, 9107--9130).  The curvature correction from
Section 2.5 is deliberately reported as unresolved: the paper does not define
how the neighboring mixed cell used as the curvature target is selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from main.geoms.geoms import getDistance, getPolyLineArea


Point = List[float]
CellIndex = Tuple[int, int]


class QuasiTopologyError(RuntimeError):
    """Raised when the prototype reaches QUASI's underspecified correction."""


@dataclass(frozen=True)
class QuadraticFacet:
    """Area-preserving parabolic facet in the chord-aligned frame.

    ``bulge`` is the signed midpoint displacement in the left normal direction
    of the oriented chord.  This is Eq. (10) written in a frame where the two
    endpoints have ordinates zero.
    """

    pLeft: Point
    pRight: Point
    bulge: float
    name: str = "QUASI"

    @property
    def _chord(self) -> np.ndarray:
        return np.asarray(self.pRight, dtype=float) - np.asarray(
            self.pLeft, dtype=float
        )

    @property
    def chord_length(self) -> float:
        return float(np.linalg.norm(self._chord))

    @property
    def _left_normal(self) -> np.ndarray:
        chord = self._chord
        length = self.chord_length
        if length == 0.0:
            raise ValueError("QUASI facet endpoints must be distinct")
        return np.asarray([-chord[1], chord[0]]) / length

    @property
    def midpoint(self) -> Point:
        return self.point(0.5)

    def point(self, parameter: float) -> Point:
        """Evaluate the quadratic facet for ``parameter`` in ``[0, 1]``."""

        p_left = np.asarray(self.pLeft, dtype=float)
        point = (
            p_left
            + parameter * self._chord
            + 4.0 * self.bulge * parameter * (1.0 - parameter) * self._left_normal
        )
        return point.tolist()

    def tangent(self, parameter: float) -> Point:
        tangent = self._chord + (
            4.0 * self.bulge * (1.0 - 2.0 * parameter) * self._left_normal
        )
        return tangent.tolist()

    def getLeftTangent(self) -> Point:
        return self.tangent(0.0)

    def getRightTangent(self) -> Point:
        return self.tangent(1.0)

    def curvature(self, parameter: float = 0.5) -> float:
        tangent = np.asarray(self.tangent(parameter), dtype=float)
        second = -8.0 * self.bulge * self._left_normal
        numerator = tangent[0] * second[1] - tangent[1] * second[0]
        denominator = float(np.linalg.norm(tangent) ** 3)
        return 0.0 if denominator == 0.0 else float(numerator / denominator)

    def sample(self, n: int, mode: str = "parameter") -> List[Point]:
        if n <= 1:
            return [list(self.pLeft)]
        return [self.point(i / (n - 1)) for i in range(n)]

    def represented_area(self, polygon: Sequence[Point]) -> float:
        """Return the analytic left-side area represented in ``polygon``."""

        chord_area = getPolyLineArea(polygon, self.pLeft, self.pRight)
        return chord_area - (2.0 / 3.0) * self.bulge * self.chord_length


@dataclass(frozen=True)
class QuasiJoin:
    cells: Tuple[CellIndex, CellIndex]
    slots: Tuple[int, int]
    edge: Tuple[Point, Point]


@dataclass
class QuasiResult:
    facets: Dict[CellIndex, QuadraticFacet]
    joins: List[QuasiJoin]
    unresolved: List[str] = field(default_factory=list)
    c1_updates: int = 0
    c1_misses: int = 0

    def facet_grid(self, shape: Tuple[int, int]) -> List[List[List[QuadraticFacet]]]:
        grid: List[List[List[QuadraticFacet]]] = [
            [[] for _ in range(shape[1])] for _ in range(shape[0])
        ]
        for (x, y), facet in self.facets.items():
            grid[x][y] = [facet]
        return grid


@dataclass
class _CellState:
    index: CellIndex
    polygon: Any
    endpoints: List[Point]
    facet: Optional[QuadraticFacet] = None


def _cross(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _subtract(a: Sequence[float], b: Sequence[float]) -> Point:
    return [float(a[0] - b[0]), float(a[1] - b[1])]


def _unit_right_normal(p_left: Sequence[float], p_right: Sequence[float]) -> Point:
    chord = _subtract(p_right, p_left)
    length = math.hypot(chord[0], chord[1])
    if length == 0.0:
        return [0.0, 0.0]
    return [chord[1] / length, -chord[0] / length]


def _point_on_segment(
    point: Sequence[float],
    edge: Tuple[Point, Point],
    tolerance: float,
) -> bool:
    edge_vector = _subtract(edge[1], edge[0])
    point_vector = _subtract(point, edge[0])
    edge_length = math.hypot(edge_vector[0], edge_vector[1])
    if edge_length == 0.0:
        return False
    if abs(_cross(edge_vector, point_vector)) > tolerance * edge_length:
        return False
    projection = point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]
    return -tolerance <= projection <= edge_length * edge_length + tolerance


def _shared_edge(
    points_a: Sequence[Point],
    points_b: Sequence[Point],
    tolerance: float,
) -> Optional[Tuple[Point, Point]]:
    shared: List[Point] = []
    for point_a in points_a:
        for point_b in points_b:
            if getDistance(point_a, point_b) <= tolerance:
                if not any(
                    getDistance(point_a, point) <= tolerance for point in shared
                ):
                    shared.append(list(point_a))
    if len(shared) != 2:
        return None
    return (shared[0], shared[1])


def _is_axis_aligned_rectangle(points: Sequence[Point], tolerance: float) -> bool:
    if len(points) != 4:
        return False
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        dx = abs(next_point[0] - point[0])
        dy = abs(next_point[1] - point[1])
        if dx > tolerance and dy > tolerance:
            return False
        if dx <= tolerance and dy <= tolerance:
            return False
    return True


def _make_quadratic(state: _CellState) -> QuadraticFacet:
    p_left, p_right = state.endpoints
    length = getDistance(p_left, p_right)
    if length == 0.0:
        raise ValueError(f"Degenerate QUASI chord in cell {state.index}")
    chord_area = getPolyLineArea(state.polygon.points, p_left, p_right)
    target_area = state.polygon.getArea()
    # Eqs. (7)--(10): integral(q) = chord_area - target_area and
    # integral(q) = 2 * midpoint_ordinate * chord_length / 3.
    bulge = 1.5 * (chord_area - target_area) / length
    return QuadraticFacet(list(p_left), list(p_right), float(bulge))


def _phase_label(
    vertex: Sequence[float], endpoints: Sequence[Point], tolerance: float
) -> bool:
    chord = _subtract(endpoints[1], endpoints[0])
    offset = _subtract(vertex, endpoints[0])
    return _cross(chord, offset) >= -tolerance


def _select_movable_endpoint(
    fixed_state: _CellState,
    moving_state: _CellState,
    fixed_point: Point,
    edge: Tuple[Point, Point],
    tolerance: float,
) -> int:
    """Apply the two criteria described in Section 2.2.1 of the paper."""

    fixed_labels = [
        _phase_label(vertex, fixed_state.endpoints, tolerance) for vertex in edge
    ]
    original_normal = _unit_right_normal(*moving_state.endpoints)
    candidates = []
    for slot in (0, 1):
        endpoints = [list(point) for point in moving_state.endpoints]
        endpoints[slot] = list(fixed_point)
        if getDistance(endpoints[0], endpoints[1]) <= tolerance:
            continue
        labels = [_phase_label(vertex, endpoints, tolerance) for vertex in edge]
        consistent_vertices = sum(a == b for a, b in zip(fixed_labels, labels))
        normal = _unit_right_normal(*endpoints)
        normal_dot = sum(a * b for a, b in zip(original_normal, normal))
        candidates.append((consistent_vertices, normal_dot, -slot, slot))
    if not candidates:
        raise QuasiTopologyError(
            f"No nondegenerate C0 endpoint candidate for cell {moving_state.index}"
        )
    return max(candidates)[-1]


def _register_join(
    joins: List[QuasiJoin],
    occupied: Dict[Tuple[CellIndex, int], QuasiJoin],
    join: QuasiJoin,
    unresolved: List[str],
) -> None:
    keys = ((join.cells[0], join.slots[0]), (join.cells[1], join.slots[1]))
    if any(key in occupied for key in keys):
        unresolved.append(
            "C0 endpoint is claimed by multiple neighbors; this requires the "
            f"Section 2.5 vertex-jump path: {join.cells}"
        )
        return
    joins.append(join)
    for key in keys:
        occupied[key] = join


def _establish_c0(
    states: Dict[CellIndex, _CellState], tolerance: float
) -> Tuple[List[QuasiJoin], List[str]]:
    joins: List[QuasiJoin] = []
    unresolved: List[str] = []
    occupied: Dict[Tuple[CellIndex, int], QuasiJoin] = {}
    directions = ((1, 0), (0, 1))

    for index_a, state_a in states.items():
        for dx, dy in directions:
            index_b = (index_a[0] + dx, index_a[1] + dy)
            state_b = states.get(index_b)
            if state_b is None:
                continue
            edge = _shared_edge(
                state_a.polygon.points, state_b.polygon.points, tolerance
            )
            if edge is None:
                continue
            slots_a = [
                slot
                for slot, point in enumerate(state_a.endpoints)
                if _point_on_segment(point, edge, tolerance)
            ]
            slots_b = [
                slot
                for slot, point in enumerate(state_b.endpoints)
                if _point_on_segment(point, edge, tolerance)
            ]

            if slots_a and slots_b:
                slot_a, slot_b = min(
                    ((a, b) for a in slots_a for b in slots_b),
                    key=lambda pair: getDistance(
                        state_a.endpoints[pair[0]], state_b.endpoints[pair[1]]
                    ),
                )
                target = [
                    0.5
                    * (
                        state_a.endpoints[slot_a][coordinate]
                        + state_b.endpoints[slot_b][coordinate]
                    )
                    for coordinate in (0, 1)
                ]
            elif slots_a:
                slot_a = slots_a[0]
                target = list(state_a.endpoints[slot_a])
                slot_b = _select_movable_endpoint(
                    state_a, state_b, target, edge, tolerance
                )
            elif slots_b:
                slot_b = slots_b[0]
                target = list(state_b.endpoints[slot_b])
                slot_a = _select_movable_endpoint(
                    state_b, state_a, target, edge, tolerance
                )
            else:
                unresolved.append(
                    "Adjacent mixed cells have no PLIC endpoint on their common "
                    f"edge and require Section 2.5 curvature correction: {index_a}, {index_b}"
                )
                continue

            join = QuasiJoin((index_a, index_b), (slot_a, slot_b), edge)
            before_count = len(joins)
            _register_join(joins, occupied, join, unresolved)
            if len(joins) != before_count:
                state_a.endpoints[slot_a] = list(target)
                state_b.endpoints[slot_b] = list(target)

    # Fig. 3(c): diagonally adjacent mixed cells cannot be joined without the
    # Section 2.5 curvature/vertex-jump correction. Report each pair once.
    for index_a in states:
        for dx, dy in ((1, 1), (1, -1)):
            index_b = (index_a[0] + dx, index_a[1] + dy)
            if index_b not in states:
                continue
            bridge_a = (index_a[0] + dx, index_a[1])
            bridge_b = (index_a[0], index_a[1] + dy)
            if bridge_a not in states and bridge_b not in states:
                unresolved.append(
                    "Diagonal mixed cells require Section 2.5 curvature correction: "
                    f"{index_a}, {index_b}"
                )

    return joins, unresolved


def _edge_parameter(point: Sequence[float], edge: Tuple[Point, Point]) -> float:
    edge_vector = np.asarray(edge[1], dtype=float) - np.asarray(edge[0], dtype=float)
    denominator = float(np.dot(edge_vector, edge_vector))
    if denominator == 0.0:
        return 0.0
    return float(
        np.dot(
            np.asarray(point, dtype=float) - np.asarray(edge[0], dtype=float),
            edge_vector,
        )
        / denominator
    )


def _join_mismatch(
    alpha: float,
    join: QuasiJoin,
    states: Dict[CellIndex, _CellState],
) -> float:
    edge_point = [
        join.edge[0][coordinate]
        + alpha * (join.edge[1][coordinate] - join.edge[0][coordinate])
        for coordinate in (0, 1)
    ]
    facets = []
    tangents = []
    for cell, slot in zip(join.cells, join.slots):
        state = states[cell]
        endpoints = [list(point) for point in state.endpoints]
        endpoints[slot] = list(edge_point)
        proposal = _CellState(state.index, state.polygon, endpoints)
        facet = _make_quadratic(proposal)
        tangent = facet.getLeftTangent() if slot == 0 else facet.getRightTangent()
        facets.append(facet)
        tangents.append(np.asarray(tangent, dtype=float))
    denominator = float(np.linalg.norm(tangents[0]) * np.linalg.norm(tangents[1]))
    if denominator == 0.0:
        return float("nan")
    return _cross(tangents[0], tangents[1]) / denominator


def _candidate_c1_roots(
    join: QuasiJoin,
    states: Dict[CellIndex, _CellState],
    root_tolerance: float,
) -> List[float]:
    # Eq. (16) is a cubic. Evaluating the equivalent slope mismatch and
    # bracketing all roots avoids copying its frame-dependent expanded
    # coefficients while preserving the same C1 condition.
    samples = np.linspace(0.0, 1.0, 65)
    values = [_join_mismatch(float(alpha), join, states) for alpha in samples]
    roots: List[float] = []
    for alpha, value in zip(samples, values):
        if math.isfinite(value) and abs(value) <= root_tolerance:
            roots.append(float(alpha))
    for left, right, f_left, f_right in zip(
        samples[:-1], samples[1:], values[:-1], values[1:]
    ):
        if not (math.isfinite(f_left) and math.isfinite(f_right)):
            continue
        if f_left * f_right < 0.0:
            roots.append(
                float(
                    brentq(
                        lambda alpha: _join_mismatch(alpha, join, states),
                        float(left),
                        float(right),
                        xtol=root_tolerance,
                    )
                )
            )
    unique: List[float] = []
    for root in roots:
        if not any(
            abs(root - existing) <= 10.0 * root_tolerance for existing in unique
        ):
            unique.append(root)
    return unique


def _apply_c1_join(
    join: QuasiJoin,
    states: Dict[CellIndex, _CellState],
    root_tolerance: float,
) -> bool:
    current_point = states[join.cells[0]].endpoints[join.slots[0]]
    current_alpha = min(1.0, max(0.0, _edge_parameter(current_point, join.edge)))
    roots = _candidate_c1_roots(join, states, root_tolerance)
    if not roots:
        return False
    # The paper does not state how multiple admissible cubic roots are chosen.
    # Retaining the root nearest the predictor is the least-displacing choice.
    alpha = min(roots, key=lambda candidate: abs(candidate - current_alpha))
    target = [
        join.edge[0][coordinate]
        + alpha * (join.edge[1][coordinate] - join.edge[0][coordinate])
        for coordinate in (0, 1)
    ]
    for cell, slot in zip(join.cells, join.slots):
        states[cell].endpoints[slot] = list(target)
        states[cell].facet = _make_quadratic(states[cell])
    return True


def reconstruct_quasi(
    mesh: Any,
    *,
    iterations: int = 10,
    threshold: float = 1.0e-6,
    geometry_tolerance: float = 1.0e-10,
    root_tolerance: float = 1.0e-12,
    strict: bool = True,
) -> QuasiResult:
    """Reconstruct static mixed cells using the connected QUASI core.

    The source method and this prototype are restricted to an axis-aligned
    Cartesian mesh. ``strict=True`` prevents an incomplete result from being
    mistaken for full QUASI when Section 2.5 curvature correction is needed.
    """

    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    states: Dict[CellIndex, _CellState] = {}
    for x, column in enumerate(mesh.polys):
        for y, polygon in enumerate(column):
            if not _is_axis_aligned_rectangle(polygon.points, geometry_tolerance):
                raise ValueError(
                    "QUASI baseline is defined here only for Cartesian cells"
                )
            if not polygon.isMixed(tolerance=threshold):
                continue
            polygon.set3x3Stencil(mesh.get3x3Stencil(x, y))
            plic = polygon.runYoungs(ret=True)
            states[(x, y)] = _CellState(
                (x, y), polygon, [list(plic.pLeft), list(plic.pRight)]
            )

    joins, unresolved = _establish_c0(states, geometry_tolerance)
    for state in states.values():
        state.facet = _make_quadratic(state)

    if unresolved and strict:
        raise QuasiTopologyError("; ".join(unresolved))

    c1_updates = 0
    c1_misses = 0
    for _ in range(iterations):
        for join in joins:
            if _apply_c1_join(join, states, root_tolerance):
                c1_updates += 1
            else:
                c1_misses += 1

    return QuasiResult(
        facets={index: state.facet for index, state in states.items()},
        joins=joins,
        unresolved=unresolved,
        c1_updates=c1_updates,
        c1_misses=c1_misses,
    )


__all__ = [
    "QuadraticFacet",
    "QuasiJoin",
    "QuasiResult",
    "QuasiTopologyError",
    "reconstruct_quasi",
]
