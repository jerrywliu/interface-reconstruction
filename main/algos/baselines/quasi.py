"""Static Cartesian port of the 2009 QUASI reconstruction method.

The implementation follows Sections 2.1--2.5 of Diwakar, Das, and
Sundararajan (JCP 228, 2009, 9107--9130).  Choices left open by the article are
collected in :class:`QuasiPolicy` so benchmark runs can retain them verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from main.geoms.geoms import getDistance, getPolyLineArea
from main.algos.baselines.quasi_roots import enumerate_real_polynomial_roots


Point = List[float]
CellIndex = Tuple[int, int]


class QuasiTopologyError(RuntimeError):
    """Raised when strict execution cannot complete a QUASI correction."""


@dataclass(frozen=True)
class QuasiPolicy:
    """Frozen porting choices not prescribed completely by the article."""

    max_sweeps: int = 10
    convergence_tolerance: float = 1.0e-11
    target_fraction_lower: float = 0.02
    target_fraction_upper: float = 0.98
    fallback: str = "retain-area-preserving-quadratic"

    def __post_init__(self) -> None:
        if self.max_sweeps < 0:
            raise ValueError("max_sweeps must be nonnegative")
        if self.convergence_tolerance <= 0.0:
            raise ValueError("convergence_tolerance must be positive")
        if not 0.0 <= self.target_fraction_lower < self.target_fraction_upper <= 1.0:
            raise ValueError("target fraction bounds must lie in [0, 1]")
        if self.fallback != "retain-area-preserving-quadratic":
            raise ValueError("unsupported QUASI fallback policy")


DEFAULT_QUASI_POLICY = QuasiPolicy()


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
    kind: str = "edge"


@dataclass
class QuasiResult:
    facets: Dict[CellIndex, QuadraticFacet]
    joins: List[QuasiJoin]
    unresolved: List[str] = field(default_factory=list)
    c1_updates: int = 0
    c1_misses: int = 0
    curvature_updates: int = 0
    vertex_jumps: int = 0
    sweeps_completed: int = 0
    converged: bool = False
    sweep_diagnostics: Tuple["QuasiSweepDiagnostic", ...] = ()
    policy: Dict[str, Any] = field(default_factory=dict)

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


@dataclass(frozen=True)
class _CorrectionRequest:
    cells: Tuple[CellIndex, CellIndex]
    reason: str


@dataclass(frozen=True)
class QuasiSweepDiagnostic:
    sweep: int
    updates: int
    misses: int
    multiple_root_updates: int
    root_branch_switches: int
    max_update_displacement: float
    mean_update_displacement: float
    max_net_endpoint_displacement: float
    max_two_sweep_endpoint_displacement: float
    mean_c1_mismatch_before: float
    max_c1_mismatch_before: float
    mean_c1_mismatch_after: float
    max_c1_mismatch_after: float
    max_area_residual: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sweep": self.sweep,
            "updates": self.updates,
            "misses": self.misses,
            "multiple_root_updates": self.multiple_root_updates,
            "root_branch_switches": self.root_branch_switches,
            "max_update_displacement": self.max_update_displacement,
            "mean_update_displacement": self.mean_update_displacement,
            "max_net_endpoint_displacement": self.max_net_endpoint_displacement,
            "max_two_sweep_endpoint_displacement": (
                self.max_two_sweep_endpoint_displacement
            ),
            "mean_c1_mismatch_before": self.mean_c1_mismatch_before,
            "max_c1_mismatch_before": self.max_c1_mismatch_before,
            "mean_c1_mismatch_after": self.mean_c1_mismatch_after,
            "max_c1_mismatch_after": self.max_c1_mismatch_after,
            "max_area_residual": self.max_area_residual,
        }


@dataclass(frozen=True)
class _C1UpdateResult:
    updated: bool
    displacement: float
    root_count: int = 0
    selected_root_ordinal: Optional[int] = None


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
    correction_requests: List[_CorrectionRequest],
) -> None:
    keys = ((join.cells[0], join.slots[0]), (join.cells[1], join.slots[1]))
    if any(key in occupied for key in keys):
        unresolved.append(
            "C0 endpoint is claimed by multiple neighbors; this requires the "
            f"Section 2.5 vertex-jump path: {join.cells}"
        )
        correction_requests.append(
            _CorrectionRequest(tuple(sorted(join.cells)), "multiply-claimed-endpoint")
        )
        return
    joins.append(join)
    for key in keys:
        occupied[key] = join


def _establish_c0(
    states: Dict[CellIndex, _CellState], tolerance: float
) -> Tuple[List[QuasiJoin], List[str], List[_CorrectionRequest]]:
    joins: List[QuasiJoin] = []
    unresolved: List[str] = []
    correction_requests: List[_CorrectionRequest] = []
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
                correction_requests.append(
                    _CorrectionRequest(
                        tuple(sorted((index_a, index_b))), "no-common-edge-endpoint"
                    )
                )
                continue

            join = QuasiJoin((index_a, index_b), (slot_a, slot_b), edge)
            before_count = len(joins)
            _register_join(joins, occupied, join, unresolved, correction_requests)
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
                correction_requests.append(
                    _CorrectionRequest(
                        tuple(sorted((index_a, index_b))), "diagonal-vertex-jump"
                    )
                )

    unique_requests = sorted(
        set(correction_requests), key=lambda request: (request.cells, request.reason)
    )
    return joins, unresolved, unique_requests


def _polygon_area(points: Sequence[Point]) -> float:
    return abs(
        0.5
        * sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[index][1] * points[(index + 1) % len(points)][0]
            for index in range(len(points))
        )
    )


def _volume_fraction(state: _CellState) -> float:
    if hasattr(state.polygon, "getFraction"):
        return float(state.polygon.getFraction())
    cell_area = _polygon_area(state.polygon.points)
    return 0.0 if cell_area == 0.0 else float(state.polygon.getArea()) / cell_area


def _point_at_parameter(edge: Tuple[Point, Point], alpha: float) -> Point:
    return [
        edge[0][coordinate] + alpha * (edge[1][coordinate] - edge[0][coordinate])
        for coordinate in (0, 1)
    ]


def _select_curvature_target(
    source: _CellState,
    preferred: _CellState,
    states: Dict[CellIndex, _CellState],
    policy: QuasiPolicy,
) -> Optional[_CellState]:
    """Select the frozen Section 2.5 target without result-driven tuning."""

    source_midpoint = np.mean(np.asarray(source.endpoints, dtype=float), axis=0)
    source_tangent = np.asarray(source.facet.tangent(0.5), dtype=float)
    source_norm = float(np.linalg.norm(source_tangent))
    candidates = []
    for index, candidate in states.items():
        if index == source.index:
            continue
        if max(abs(index[0] - source.index[0]), abs(index[1] - source.index[1])) > 1:
            continue
        fraction = _volume_fraction(candidate)
        if not policy.target_fraction_lower < fraction < policy.target_fraction_upper:
            continue
        candidate_midpoint = np.asarray(candidate.facet.midpoint, dtype=float)
        candidate_tangent = np.asarray(candidate.facet.tangent(0.5), dtype=float)
        candidate_norm = float(np.linalg.norm(candidate_tangent))
        alignment = 0.0
        if source_norm > 0.0 and candidate_norm > 0.0:
            alignment = abs(
                float(np.dot(source_tangent, candidate_tangent))
                / (source_norm * candidate_norm)
            )
        candidates.append(
            (
                0 if index == preferred.index else 1,
                float(np.linalg.norm(candidate_midpoint - source_midpoint)),
                -alignment,
                index,
                candidate,
            )
        )
    return min(candidates)[-1] if candidates else None


def _poly_trim(coefficients: Sequence[float], tolerance: float) -> np.ndarray:
    values = np.asarray(coefficients, dtype=float)
    scale = max(1.0, float(np.max(np.abs(values))))
    while values.size > 1 and abs(values[-1]) <= tolerance * scale:
        values = values[:-1]
    return values


def _verified_algebraic_roots(
    coefficients: Sequence[float],
    residual,
    current_parameter: float,
    tolerance: float,
) -> List[float]:
    root_set = enumerate_real_polynomial_roots(
        _poly_trim(coefficients, tolerance),
        lower=0.0,
        upper=1.0,
        tolerance=tolerance,
    )
    if root_set.identically_zero:
        return [current_parameter]
    residual_tolerance = max(1.0e-8, 256.0 * tolerance)
    return [
        root.value
        for root in root_set.roots
        if math.isfinite(residual(root.value))
        and abs(residual(root.value)) <= residual_tolerance
    ]


def _moving_endpoint_polynomials(
    state: _CellState,
    slot: int,
    edge: Tuple[Point, Point],
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """Return chord, squared length, and area residual polynomials."""

    moving = (
        np.asarray([edge[0][0], edge[1][0] - edge[0][0]], dtype=float),
        np.asarray([edge[0][1], edge[1][1] - edge[0][1]], dtype=float),
    )
    fixed_point = state.endpoints[1 - slot]
    fixed = (
        np.asarray([fixed_point[0]], dtype=float),
        np.asarray([fixed_point[1]], dtype=float),
    )
    if slot == 0:
        chord = (
            np.polynomial.polynomial.polysub(fixed[0], moving[0]),
            np.polynomial.polynomial.polysub(fixed[1], moving[1]),
        )
    else:
        chord = (
            np.polynomial.polynomial.polysub(moving[0], fixed[0]),
            np.polynomial.polynomial.polysub(moving[1], fixed[1]),
        )
    squared_length = np.polynomial.polynomial.polyadd(
        np.polynomial.polynomial.polymul(chord[0], chord[0]),
        np.polynomial.polynomial.polymul(chord[1], chord[1]),
    )

    areas = []
    for alpha in (0.0, 1.0):
        endpoints = [list(point) for point in state.endpoints]
        endpoints[slot] = _point_at_parameter(edge, alpha)
        areas.append(getPolyLineArea(state.polygon.points, *endpoints))
    area_residual = np.asarray(
        [areas[0] - state.polygon.getArea(), areas[1] - areas[0]],
        dtype=float,
    )
    return chord, squared_length, area_residual


def _midpoint_curvature_polynomial(
    state: _CellState,
    slot: int,
    edge: Tuple[Point, Point],
    target_curvature: float,
) -> np.ndarray:
    """Cross-multiplied midpoint-curvature relation from Eqs. (17)--(21)."""

    _, squared_length, area_residual = _moving_endpoint_polynomials(state, slot, edge)
    residual_squared = np.polynomial.polynomial.polymul(area_residual, area_residual)
    length_sixth = np.polynomial.polynomial.polymul(
        np.polynomial.polynomial.polymul(squared_length, squared_length),
        squared_length,
    )
    return np.polynomial.polynomial.polysub(
        144.0 * residual_squared,
        target_curvature * target_curvature * length_sixth,
    )


def _endpoint_distance_to_edge(
    point: Sequence[float], edge: Tuple[Point, Point]
) -> Tuple[float, float]:
    alpha = min(1.0, max(0.0, _edge_parameter(point, edge)))
    return getDistance(point, _point_at_parameter(edge, alpha)), alpha


def _shared_vertex(
    first: _CellState, second: _CellState, tolerance: float
) -> Optional[Point]:
    shared = []
    for point_a in first.polygon.points:
        for point_b in second.polygon.points:
            if getDistance(point_a, point_b) <= tolerance:
                shared.append(list(point_a))
    if len(shared) != 1:
        return None
    return shared[0]


def _apply_curvature_correction(
    source: _CellState,
    target: _CellState,
    edge: Tuple[Point, Point],
    occupied: set[Tuple[CellIndex, int]],
    geometry_tolerance: float,
    root_tolerance: float,
) -> Tuple[Optional[QuasiJoin], bool]:
    target_curvature = target.facet.curvature(0.5)
    slot = min(
        (0, 1),
        key=lambda candidate: (
            _endpoint_distance_to_edge(source.endpoints[candidate], edge)[0],
            candidate,
        ),
    )
    target_slot = min(
        (0, 1),
        key=lambda candidate: (
            _endpoint_distance_to_edge(target.endpoints[candidate], edge)[0],
            candidate,
        ),
    )
    if (source.index, slot) in occupied or (target.index, target_slot) in occupied:
        return None, False
    source_alpha = _endpoint_distance_to_edge(source.endpoints[slot], edge)[1]
    target_alpha = _endpoint_distance_to_edge(target.endpoints[target_slot], edge)[1]
    current_alpha = 0.5 * (source_alpha + target_alpha)

    polynomial = _midpoint_curvature_polynomial(source, slot, edge, target_curvature)

    def mismatch(alpha: float) -> float:
        endpoints = [list(point) for point in source.endpoints]
        endpoints[slot] = _point_at_parameter(edge, alpha)
        if getDistance(endpoints[0], endpoints[1]) <= geometry_tolerance:
            return float("nan")
        proposal = _CellState(source.index, source.polygon, endpoints)
        return _make_quadratic(proposal).curvature(0.5) - target_curvature

    roots = _verified_algebraic_roots(
        polynomial, mismatch, current_alpha, root_tolerance
    )
    jumped = False
    if roots:
        alpha = min(roots, key=lambda value: (abs(value - current_alpha), value))
    else:
        current_residual = abs(mismatch(current_alpha))
        vertex_candidates = [
            (abs(mismatch(alpha)), abs(alpha - current_alpha), alpha)
            for alpha in (0.0, 1.0)
            if math.isfinite(mismatch(alpha))
        ]
        if not vertex_candidates:
            return None, False
        residual, _, alpha = min(vertex_candidates)
        if not residual < current_residual:
            return None, False
        jumped = True

    common_point = _point_at_parameter(edge, alpha)
    source.endpoints[slot] = list(common_point)
    target.endpoints[target_slot] = list(common_point)
    source.facet = _make_quadratic(source)
    target.facet = _make_quadratic(target)
    return (
        QuasiJoin((source.index, target.index), (slot, target_slot), edge, "edge"),
        jumped,
    )


def _apply_vertex_jump(
    first: _CellState,
    second: _CellState,
    vertex: Point,
    occupied: set[Tuple[CellIndex, int]],
    geometry_tolerance: float,
) -> Optional[QuasiJoin]:
    slots = tuple(
        min(
            (0, 1),
            key=lambda slot: (getDistance(state.endpoints[slot], vertex), slot),
        )
        for state in (first, second)
    )
    if any(
        (state.index, slot) in occupied for state, slot in zip((first, second), slots)
    ):
        return None
    for state, slot in zip((first, second), slots):
        if getDistance(state.endpoints[1 - slot], vertex) <= geometry_tolerance:
            return None
    for state, slot in zip((first, second), slots):
        state.endpoints[slot] = list(vertex)
        state.facet = _make_quadratic(state)
    return QuasiJoin(
        (first.index, second.index), slots, (list(vertex), list(vertex)), "vertex-jump"
    )


def _resolve_curvature_requests(
    requests: Sequence[_CorrectionRequest],
    states: Dict[CellIndex, _CellState],
    joins: List[QuasiJoin],
    policy: QuasiPolicy,
    geometry_tolerance: float,
    root_tolerance: float,
) -> Tuple[List[str], int, int]:
    unresolved: List[str] = []
    updates = 0
    jumps = 0
    occupied = {
        (cell, slot) for join in joins for cell, slot in zip(join.cells, join.slots)
    }
    for request in sorted(requests, key=lambda item: (item.cells, item.reason)):
        first, second = (states[index] for index in request.cells)
        if request.reason == "multiply-claimed-endpoint":
            unresolved.append(
                "Section 2.5 fallback retained a conservative local quadratic "
                f"for multiply claimed endpoint {request.cells}"
            )
            continue

        eligible = [
            policy.target_fraction_lower
            < _volume_fraction(state)
            < policy.target_fraction_upper
            for state in (first, second)
        ]
        if eligible == [True, False]:
            source, preferred = second, first
        elif eligible == [False, True]:
            source, preferred = first, second
        else:
            source, preferred = second, first
        target = _select_curvature_target(source, preferred, states, policy)
        if target is None:
            unresolved.append(
                f"No eligible Section 2.5 target for {source.index} ({request.reason})"
            )
            continue

        if request.reason == "diagonal-vertex-jump":
            vertex = _shared_vertex(source, target, geometry_tolerance)
            join = (
                None
                if vertex is None
                else _apply_vertex_jump(
                    source, target, vertex, occupied, geometry_tolerance
                )
            )
            jumped = join is not None
        else:
            edge = _shared_edge(
                source.polygon.points, target.polygon.points, geometry_tolerance
            )
            if edge is None:
                join, jumped = None, False
            else:
                join, jumped = _apply_curvature_correction(
                    source,
                    target,
                    edge,
                    occupied,
                    geometry_tolerance,
                    root_tolerance,
                )

        if join is not None and not any(
            (cell, slot) in occupied for cell, slot in zip(join.cells, join.slots)
        ):
            joins.append(join)
            occupied.update(zip(join.cells, join.slots))
            updates += 1
            jumps += int(jumped)
        else:
            unresolved.append(
                "Section 2.5 correction retained the area-preserving local "
                f"quadratic in {source.index} ({request.reason})"
            )
    return unresolved, updates, jumps


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
    if join.kind != "edge" or getDistance(*join.edge) == 0.0:
        return []

    tangent_numerators = []
    for cell, slot in zip(join.cells, join.slots):
        state = states[cell]
        chord, squared_length, area_residual = _moving_endpoint_polynomials(
            state, slot, join.edge
        )
        sign = 1.0 if slot == 0 else -1.0
        tangent_x = np.polynomial.polynomial.polyadd(
            np.polynomial.polynomial.polymul(squared_length, chord[0]),
            sign * 6.0 * np.polynomial.polynomial.polymul(area_residual, -chord[1]),
        )
        tangent_y = np.polynomial.polynomial.polyadd(
            np.polynomial.polynomial.polymul(squared_length, chord[1]),
            sign * 6.0 * np.polynomial.polynomial.polymul(area_residual, chord[0]),
        )
        tangent_numerators.append((tangent_x, tangent_y))

    polynomial = np.polynomial.polynomial.polysub(
        np.polynomial.polynomial.polymul(
            tangent_numerators[0][0], tangent_numerators[1][1]
        ),
        np.polynomial.polynomial.polymul(
            tangent_numerators[0][1], tangent_numerators[1][0]
        ),
    )
    current_point = states[join.cells[0]].endpoints[join.slots[0]]
    current_alpha = min(1.0, max(0.0, _edge_parameter(current_point, join.edge)))
    return _verified_algebraic_roots(
        polynomial,
        lambda alpha: _join_mismatch(alpha, join, states),
        current_alpha,
        root_tolerance,
    )


def _apply_c1_join(
    join: QuasiJoin,
    states: Dict[CellIndex, _CellState],
    root_tolerance: float,
) -> _C1UpdateResult:
    current_point = states[join.cells[0]].endpoints[join.slots[0]]
    current_alpha = min(1.0, max(0.0, _edge_parameter(current_point, join.edge)))
    roots = _candidate_c1_roots(join, states, root_tolerance)
    if not roots:
        return _C1UpdateResult(False, 0.0)
    # The paper does not state how multiple admissible cubic roots are chosen.
    # Retaining the root nearest the predictor is the least-displacing choice.
    selected_root_ordinal, alpha = min(
        enumerate(roots),
        key=lambda item: (abs(item[1] - current_alpha), item[1]),
    )
    target = [
        join.edge[0][coordinate]
        + alpha * (join.edge[1][coordinate] - join.edge[0][coordinate])
        for coordinate in (0, 1)
    ]
    for cell, slot in zip(join.cells, join.slots):
        states[cell].endpoints[slot] = list(target)
        states[cell].facet = _make_quadratic(states[cell])
    return _C1UpdateResult(
        True,
        abs(alpha - current_alpha) * getDistance(*join.edge),
        len(roots),
        selected_root_ordinal,
    )


def _endpoint_snapshot(
    states: Dict[CellIndex, _CellState],
) -> Dict[Tuple[CellIndex, int], Tuple[float, float]]:
    return {
        (index, slot): (float(point[0]), float(point[1]))
        for index, state in states.items()
        for slot, point in enumerate(state.endpoints)
    }


def _max_snapshot_displacement(
    first: Mapping[Tuple[CellIndex, int], Tuple[float, float]],
    second: Mapping[Tuple[CellIndex, int], Tuple[float, float]],
) -> float:
    return max((getDistance(first[key], second[key]) for key in first), default=0.0)


def _current_join_mismatches(
    joins: Sequence[QuasiJoin], states: Dict[CellIndex, _CellState]
) -> Tuple[float, ...]:
    values = []
    for join in joins:
        point = states[join.cells[0]].endpoints[join.slots[0]]
        alpha = min(1.0, max(0.0, _edge_parameter(point, join.edge)))
        value = _join_mismatch(alpha, join, states)
        if math.isfinite(value):
            values.append(abs(value))
    return tuple(values)


def _max_area_residual(states: Dict[CellIndex, _CellState]) -> float:
    return max(
        (
            abs(
                state.facet.represented_area(state.polygon.points)
                - state.polygon.getArea()
            )
            for state in states.values()
        ),
        default=0.0,
    )


def reconstruct_quasi(
    mesh: Any,
    *,
    policy: QuasiPolicy = DEFAULT_QUASI_POLICY,
    iterations: Optional[int] = None,
    threshold: float = 1.0e-6,
    geometry_tolerance: float = 1.0e-10,
    root_tolerance: float = 1.0e-12,
    strict: bool = False,
    trace_sweeps: bool = False,
) -> QuasiResult:
    """Reconstruct static mixed cells with the frozen QUASI port.

    The method is restricted to an axis-aligned Cartesian mesh. Physical-domain
    boundary endpoints remain open because the article's static reconstruction
    does not prescribe ghost-cell geometry. ``strict=True`` raises after the
    documented conservative fallback instead of returning fallback diagnostics.
    """

    sweep_limit = policy.max_sweeps if iterations is None else iterations
    if sweep_limit < 0:
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

    joins, _, correction_requests = _establish_c0(states, geometry_tolerance)
    for state in states.values():
        state.facet = _make_quadratic(state)

    unresolved, curvature_updates, vertex_jumps = _resolve_curvature_requests(
        correction_requests,
        states,
        joins,
        policy,
        geometry_tolerance,
        root_tolerance,
    )
    if unresolved and strict:
        raise QuasiTopologyError("; ".join(unresolved))

    c1_updates = 0
    c1_misses = 0
    sweeps_completed = 0
    converged = not any(join.kind == "edge" for join in joins)
    ordered_joins = sorted(
        (join for join in joins if join.kind == "edge"),
        key=lambda join: (join.cells, join.slots),
    )
    sweep_diagnostics: List[QuasiSweepDiagnostic] = []
    endpoint_history = [_endpoint_snapshot(states)] if trace_sweeps else []
    previous_root_ordinals: Dict[
        Tuple[Tuple[CellIndex, CellIndex], Tuple[int, int]], int
    ] = {}
    if ordered_joins:
        for sweep_index in range(sweep_limit):
            max_displacement = 0.0
            displacements = []
            sweep_updates = 0
            sweep_misses = 0
            multiple_root_updates = 0
            root_branch_switches = 0
            mismatch_before = (
                _current_join_mismatches(ordered_joins, states) if trace_sweeps else ()
            )
            for join in ordered_joins:
                update = _apply_c1_join(join, states, root_tolerance)
                if update.updated:
                    c1_updates += 1
                    sweep_updates += 1
                    displacements.append(update.displacement)
                    max_displacement = max(max_displacement, update.displacement)
                    multiple_root_updates += int(update.root_count > 1)
                    if update.selected_root_ordinal is not None:
                        join_key = (join.cells, join.slots)
                        previous = previous_root_ordinals.get(join_key)
                        root_branch_switches += int(
                            previous is not None
                            and previous != update.selected_root_ordinal
                        )
                        previous_root_ordinals[join_key] = update.selected_root_ordinal
                else:
                    c1_misses += 1
                    sweep_misses += 1
            sweeps_completed += 1
            if trace_sweeps:
                current_snapshot = _endpoint_snapshot(states)
                mismatch_after = _current_join_mismatches(ordered_joins, states)
                sweep_diagnostics.append(
                    QuasiSweepDiagnostic(
                        sweep=sweep_index + 1,
                        updates=sweep_updates,
                        misses=sweep_misses,
                        multiple_root_updates=multiple_root_updates,
                        root_branch_switches=root_branch_switches,
                        max_update_displacement=max_displacement,
                        mean_update_displacement=(
                            float(np.mean(displacements)) if displacements else 0.0
                        ),
                        max_net_endpoint_displacement=_max_snapshot_displacement(
                            endpoint_history[-1], current_snapshot
                        ),
                        max_two_sweep_endpoint_displacement=(
                            _max_snapshot_displacement(
                                endpoint_history[-2], current_snapshot
                            )
                            if len(endpoint_history) >= 2
                            else 0.0
                        ),
                        mean_c1_mismatch_before=(
                            float(np.mean(mismatch_before)) if mismatch_before else 0.0
                        ),
                        max_c1_mismatch_before=max(mismatch_before, default=0.0),
                        mean_c1_mismatch_after=(
                            float(np.mean(mismatch_after)) if mismatch_after else 0.0
                        ),
                        max_c1_mismatch_after=max(mismatch_after, default=0.0),
                        max_area_residual=_max_area_residual(states),
                    )
                )
                endpoint_history.append(current_snapshot)
            if max_displacement <= policy.convergence_tolerance:
                converged = True
                break

    return QuasiResult(
        facets={index: state.facet for index, state in states.items()},
        joins=joins,
        unresolved=unresolved,
        c1_updates=c1_updates,
        c1_misses=c1_misses,
        curvature_updates=curvature_updates,
        vertex_jumps=vertex_jumps,
        sweeps_completed=sweeps_completed,
        converged=converged,
        sweep_diagnostics=tuple(sweep_diagnostics),
        policy={
            "target_neighbor": (
                "triggering eligible neighbor, then nearest eligible 8-neighbor; "
                "tangent alignment and lexicographic index break ties"
            ),
            "multiple_root": "least displacement from the current predictor",
            "update_order": "lexicographic Gauss-Seidel",
            "max_sweeps": sweep_limit,
            "convergence_tolerance": policy.convergence_tolerance,
            "boundary": "retain open physical-boundary endpoints",
            "fallback": policy.fallback,
            "trace_sweeps": trace_sweeps,
        },
    )


__all__ = [
    "QuadraticFacet",
    "DEFAULT_QUASI_POLICY",
    "QuasiPolicy",
    "QuasiJoin",
    "QuasiResult",
    "QuasiSweepDiagnostic",
    "QuasiTopologyError",
    "reconstruct_quasi",
]
