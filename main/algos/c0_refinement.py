"""Conservative joint refinement of incompatible line and arc facet joins."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize, root

from main.geoms.circular_facet import getCenter
from main.geoms.geoms import pointInPoly
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.linear_facet import LinearFacet


DEFAULT_GAP_TOLERANCE = 1.0e-8
DEFAULT_CONSERVATION_TOLERANCE = 1.0e-10
DEFAULT_MAX_FUNCTION_EVALUATIONS = 500


@dataclass(frozen=True)
class Join:
    first_id: int
    first_side: str
    second_id: int
    second_side: str
    edge_start: np.ndarray
    edge_end: np.ndarray
    first_endpoint: np.ndarray
    second_endpoint: np.ndarray


@dataclass
class ComponentSolution:
    facets: dict[int, Any]
    shared_points: dict[int, np.ndarray]
    normalized_area_residuals: dict[int, float]
    tangent_angles: dict[int, float]
    solution_kind: str
    score: float
    function_evaluations: int


@dataclass(frozen=True)
class JointC0Assignment:
    merge_id: int
    facet: Any
    component_index: int
    solution_kind: str


@dataclass(frozen=True)
class JointC0ComponentRecord:
    component_index: int
    merge_ids: tuple[int, ...]
    num_facets: int
    num_bad_joins: int
    solved: bool
    solution_kind: str
    max_relative_area_residual: float | None
    max_tangent_angle_radians: float | None
    function_evaluations: int | None


@dataclass(frozen=True)
class JointC0Report:
    mode: str
    eligible_joins: int
    bad_joins_before: int
    bad_joins_after: int
    mean_gap_before: float
    mean_gap_after: float
    max_gap_before: float
    max_gap_after: float
    components: tuple[JointC0ComponentRecord, ...]
    max_relative_area_residual_after: float

    @property
    def components_solved(self) -> int:
        return sum(component.solved for component in self.components)

    @property
    def components_failed(self) -> int:
        return len(self.components) - self.components_solved

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "eligible_joins": self.eligible_joins,
            "bad_joins_before": self.bad_joins_before,
            "bad_joins_after": self.bad_joins_after,
            "mean_gap_before": self.mean_gap_before,
            "mean_gap_after": self.mean_gap_after,
            "max_gap_before": self.max_gap_before,
            "max_gap_after": self.max_gap_after,
            "components_solved": self.components_solved,
            "components_failed": self.components_failed,
            "max_relative_area_residual_after": (self.max_relative_area_residual_after),
            "components": [asdict(component) for component in self.components],
        }


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return float(
        np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float))
    )


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _endpoint(facet: Any, side: str) -> np.ndarray:
    return np.asarray(facet.pLeft if side == "left" else facet.pRight, dtype=float)


def _eligible_facet(facet: Any) -> bool:
    return isinstance(facet, (LinearFacet, ArcFacet))


def _oriented_tangent(facet: Any, point: np.ndarray) -> np.ndarray:
    tangent = np.asarray(facet.getTangent(point.tolist()), dtype=float)
    if isinstance(facet, ArcFacet) and facet.radius < 0.0:
        tangent *= -1.0
    norm = float(np.linalg.norm(tangent))
    if norm == 0.0:
        raise ValueError("Degenerate facet tangent")
    return tangent / norm


def _point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator == 0.0:
        return float(np.linalg.norm(point - start))
    coordinate = float(np.dot(point - start, direction) / denominator)
    coordinate = min(1.0, max(0.0, coordinate))
    return float(np.linalg.norm(point - (start + coordinate * direction)))


def _overlap_segment(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    *,
    tolerance: float = 1.0e-9,
) -> tuple[np.ndarray, np.ndarray] | None:
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    first_length = float(np.linalg.norm(first_direction))
    second_length = float(np.linalg.norm(second_direction))
    if first_length <= tolerance or second_length <= tolerance:
        return None
    if (
        abs(_cross(first_direction, second_direction))
        > tolerance * first_length * second_length
    ):
        return None
    if (
        abs(_cross(second_start - first_start, first_direction))
        > tolerance * first_length
    ):
        return None
    denominator = float(np.dot(first_direction, first_direction))
    second_coordinates = [
        float(np.dot(point - first_start, first_direction) / denominator)
        for point in (second_start, second_end)
    ]
    lower = max(0.0, min(second_coordinates))
    upper = min(1.0, max(second_coordinates))
    if upper - lower <= tolerance:
        return None
    return (
        first_start + lower * first_direction,
        first_start + upper * first_direction,
    )


def _shared_boundary_segment(
    first_poly: Any,
    second_poly: Any,
    first_endpoint: np.ndarray,
    second_endpoint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first_points = np.asarray(first_poly.points, dtype=float)
    second_points = np.asarray(second_poly.points, dtype=float)
    midpoint = 0.5 * (first_endpoint + second_endpoint)
    candidates = []
    for first_start, first_end in zip(first_points, np.roll(first_points, -1, axis=0)):
        for second_start, second_end in zip(
            second_points, np.roll(second_points, -1, axis=0)
        ):
            overlap = _overlap_segment(first_start, first_end, second_start, second_end)
            if overlap is None:
                continue
            candidates.append(
                (_point_segment_distance(midpoint, overlap[0], overlap[1]), overlap)
            )
    if not candidates:
        raise RuntimeError("Neighboring reconstruction cells have no shared edge")
    return min(candidates, key=lambda item: item[0])[1]


def _join_key(
    first_id: int, first_side: str, second_id: int, second_side: str
) -> tuple[tuple[int, str], tuple[int, str]]:
    return tuple(sorted(((first_id, first_side), (second_id, second_side))))


def _collect_joins(
    mesh: Any, merged_polys: Sequence[Any]
) -> tuple[dict[int, Any], list[Join]]:
    mesh_polys = getattr(mesh, "merged_polys", {})
    merge_id_by_object = {
        id(poly): int(merge_id) for merge_id, poly in mesh_polys.items()
    }
    if not merge_id_by_object:
        merge_id_by_object = {
            id(poly): int(getattr(poly, "_merge_id", index))
            for index, poly in enumerate(merged_polys)
        }
    polys = {
        merge_id_by_object[id(poly)]: poly
        for poly in merged_polys
        if id(poly) in merge_id_by_object and _eligible_facet(poly.getFacet())
    }
    joins = []
    seen = set()
    for merge_id, poly in polys.items():
        candidates = (
            ("left", poly.getLeftNeighbor(), "right"),
            ("right", poly.getRightNeighbor(), "left"),
        )
        for side, neighbor, neighbor_side in candidates:
            neighbor_id = merge_id_by_object.get(id(neighbor))
            if neighbor_id not in polys:
                continue
            key = _join_key(merge_id, side, neighbor_id, neighbor_side)
            if key in seen:
                continue
            seen.add(key)
            first_endpoint = _endpoint(poly.getFacet(), side)
            second_endpoint = _endpoint(neighbor.getFacet(), neighbor_side)
            edge_start, edge_end = _shared_boundary_segment(
                poly, neighbor, first_endpoint, second_endpoint
            )
            joins.append(
                Join(
                    first_id=merge_id,
                    first_side=side,
                    second_id=neighbor_id,
                    second_side=neighbor_side,
                    edge_start=edge_start,
                    edge_end=edge_end,
                    first_endpoint=first_endpoint,
                    second_endpoint=second_endpoint,
                )
            )
    return polys, joins


def _join_gap(join: Join, facets: Mapping[int, Any]) -> float:
    return _distance(
        _endpoint(facets[join.first_id], join.first_side),
        _endpoint(facets[join.second_id], join.second_side),
    )


def _join_summary(
    joins: Sequence[Join], facets: Mapping[int, Any], gap_tolerance: float
) -> dict[str, Any]:
    gaps = [_join_gap(join, facets) for join in joins]
    return {
        "eligible_joins": len(gaps),
        "bad_joins": sum(gap > gap_tolerance for gap in gaps),
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "max_gap": max(gaps, default=0.0),
    }


def _bad_components(
    joins: Sequence[Join], facets: Mapping[int, Any], gap_tolerance: float
) -> list[list[int]]:
    bad_indices = [
        index
        for index, join in enumerate(joins)
        if _join_gap(join, facets) > gap_tolerance
    ]
    incident: dict[int, set[int]] = defaultdict(set)
    for index in bad_indices:
        join = joins[index]
        incident[join.first_id].add(index)
        incident[join.second_id].add(index)
    components = []
    remaining = set(bad_indices)
    while remaining:
        seed = remaining.pop()
        component = {seed}
        node_stack = [joins[seed].first_id, joins[seed].second_id]
        while node_stack:
            node = node_stack.pop()
            for edge_index in incident[node]:
                if edge_index in component:
                    continue
                component.add(edge_index)
                remaining.discard(edge_index)
                edge = joins[edge_index]
                node_stack.extend((edge.first_id, edge.second_id))
        components.append(sorted(component))
    return components


def _edge_coordinate(join: Join, point: np.ndarray) -> float:
    direction = join.edge_end - join.edge_start
    return float(
        np.dot(point - join.edge_start, direction) / np.dot(direction, direction)
    )


def _edge_point(join: Join, coordinate: float) -> np.ndarray:
    return join.edge_start + coordinate * (join.edge_end - join.edge_start)


def _component_nodes(component: Sequence[int], joins: Sequence[Join]) -> list[int]:
    return sorted(
        {
            merge_id
            for index in component
            for merge_id in (joins[index].first_id, joins[index].second_id)
        }
    )


def _curvature_from_latent(
    latent: float, p_left: np.ndarray, p_right: np.ndarray
) -> float:
    chord = float(np.linalg.norm(p_right - p_left))
    if chord == 0.0:
        raise ValueError("Cannot fit curvature to a zero-length chord")
    maximum = 2.0 / chord * (1.0 - 1.0e-8)
    return maximum * math.tanh(latent)


def _latent_from_curvature(
    curvature: float, p_left: np.ndarray, p_right: np.ndarray
) -> float:
    chord = float(np.linalg.norm(p_right - p_left))
    if chord == 0.0:
        raise ValueError("Cannot parameterize curvature on a zero-length chord")
    ratio = curvature * chord / (2.0 * (1.0 - 1.0e-8))
    ratio = min(1.0 - 1.0e-8, max(-1.0 + 1.0e-8, ratio))
    return float(np.arctanh(ratio))


def _facet_from_latent(p_left: np.ndarray, p_right: np.ndarray, latent: float) -> Any:
    curvature = _curvature_from_latent(latent, p_left, p_right)
    if abs(curvature) < 1.0e-11:
        return LinearFacet(p_left.tolist(), p_right.tolist())
    radius = 1.0 / curvature
    center = getCenter(p_left.tolist(), p_right.tolist(), radius)
    return ArcFacet(center, radius, p_left.tolist(), p_right.tolist())


def _component_state(
    values: np.ndarray,
    component: Sequence[int],
    joins: Sequence[Join],
    node_ids: Sequence[int],
    base_facets: Mapping[int, Any],
) -> tuple[dict[int, Any], dict[int, np.ndarray]]:
    shared_points = {
        edge_index: _edge_point(joins[edge_index], float(values[position]))
        for position, edge_index in enumerate(component)
    }
    endpoint_overrides: dict[tuple[int, str], np.ndarray] = {}
    for edge_index in component:
        join = joins[edge_index]
        point = shared_points[edge_index]
        endpoint_overrides[(join.first_id, join.first_side)] = point
        endpoint_overrides[(join.second_id, join.second_side)] = point

    facets = {}
    offset = len(component)
    for node_position, node_id in enumerate(node_ids):
        base = base_facets[node_id]
        p_left = endpoint_overrides.get(
            (node_id, "left"), np.asarray(base.pLeft, dtype=float)
        )
        p_right = endpoint_overrides.get(
            (node_id, "right"), np.asarray(base.pRight, dtype=float)
        )
        facets[node_id] = _facet_from_latent(
            p_left, p_right, float(values[offset + node_position])
        )
    return facets, shared_points


def _facet_is_local(poly: Any, facet: Any) -> bool:
    if isinstance(facet, ArcFacet) and facet.is_major_arc:
        return False
    points = np.asarray(facet.sample(31), dtype=float)[1:-1]
    polygon = np.asarray(poly.points, dtype=float)
    lower = np.min(polygon, axis=0) - 1.0e-7
    upper = np.max(polygon, axis=0) + 1.0e-7
    if np.any(points < lower) or np.any(points > upper):
        return False
    return all(pointInPoly(point.tolist(), poly.points) for point in points)


def _component_residual(
    values: np.ndarray,
    component: Sequence[int],
    joins: Sequence[Join],
    node_ids: Sequence[int],
    polys: Mapping[int, Any],
    base_facets: Mapping[int, Any],
) -> np.ndarray:
    try:
        facets, shared_points = _component_state(
            values, component, joins, node_ids, base_facets
        )
        residuals = []
        for node_id in node_ids:
            poly = polys[node_id]
            residuals.append(
                (poly._facet_phase_area(facets[node_id]) - poly.getArea())
                / poly.getMaxArea()
            )
        for edge_index in component:
            join = joins[edge_index]
            point = shared_points[edge_index]
            first_tangent = _oriented_tangent(facets[join.first_id], point)
            second_tangent = _oriented_tangent(facets[join.second_id], point)
            tangent_dot = float(
                np.clip(np.dot(first_tangent, second_tangent), -1.0, 1.0)
            )
            residuals.append(
                math.atan2(_cross(first_tangent, second_tangent), tangent_dot)
            )
        return np.asarray(residuals, dtype=float)
    except Exception:
        return np.full(len(node_ids) + len(component), 1.0e3, dtype=float)


def _initial_values(
    alpha: float,
    component: Sequence[int],
    joins: Sequence[Join],
    node_ids: Sequence[int],
    base_facets: Mapping[int, Any],
) -> np.ndarray:
    coordinates = []
    endpoint_overrides: dict[tuple[int, str], np.ndarray] = {}
    for edge_index in component:
        join = joins[edge_index]
        point = (1.0 - alpha) * join.first_endpoint + alpha * join.second_endpoint
        coordinate = min(1.0, max(0.0, _edge_coordinate(join, point)))
        coordinates.append(coordinate)
        shared_point = _edge_point(join, coordinate)
        endpoint_overrides[(join.first_id, join.first_side)] = shared_point
        endpoint_overrides[(join.second_id, join.second_side)] = shared_point

    latents = []
    for node_id in node_ids:
        facet = base_facets[node_id]
        p_left = endpoint_overrides.get(
            (node_id, "left"), np.asarray(facet.pLeft, dtype=float)
        )
        p_right = endpoint_overrides.get(
            (node_id, "right"), np.asarray(facet.pRight, dtype=float)
        )
        latents.append(_latent_from_curvature(float(facet.curvature), p_left, p_right))
    return np.asarray(coordinates + latents, dtype=float)


def _solution_score(
    solution: ComponentSolution,
    component: Sequence[int],
    joins: Sequence[Join],
    base_facets: Mapping[int, Any],
) -> float:
    movement = 0.0
    for edge_index in component:
        join = joins[edge_index]
        point = solution.shared_points[edge_index]
        scale = float(np.linalg.norm(join.edge_end - join.edge_start))
        movement += (
            _distance(point, join.first_endpoint) ** 2
            + _distance(point, join.second_endpoint) ** 2
        ) / scale**2
    curvature_change = 0.0
    for node_id, facet in solution.facets.items():
        base = base_facets[node_id]
        chord = _distance(facet.pLeft, facet.pRight)
        curvature_change += ((facet.curvature - base.curvature) * chord) ** 2
    return movement + 1.0e-3 * curvature_change


def _solve_component_c0_fallback(
    component: Sequence[int],
    joins: Sequence[Join],
    node_ids: Sequence[int],
    polys: Mapping[int, Any],
    base_facets: Mapping[int, Any],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    max_nfev: int,
    conservation_tolerance: float,
) -> ComponentSolution | None:
    def residual(values: np.ndarray) -> np.ndarray:
        return _component_residual(
            values, component, joins, node_ids, polys, base_facets
        )

    def area_constraints(values: np.ndarray) -> np.ndarray:
        return residual(values)[: len(node_ids)]

    def objective(values: np.ndarray) -> float:
        full_residual = residual(values)
        tangent_angles = full_residual[len(node_ids) :]
        if np.any(~np.isfinite(tangent_angles)):
            return 1.0e12
        facets, shared_points = _component_state(
            values, component, joins, node_ids, base_facets
        )
        provisional = ComponentSolution(
            facets=facets,
            shared_points=shared_points,
            normalized_area_residuals={},
            tangent_angles={},
            solution_kind="c0_min_tangent",
            score=0.0,
            function_evaluations=0,
        )
        return float(np.dot(tangent_angles, tangent_angles)) + 1.0e-8 * _solution_score(
            provisional, component, joins, base_facets
        )

    seeds = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        try:
            seeds.append(
                _initial_values(alpha, component, joins, node_ids, base_facets)
            )
        except ValueError as error:
            if "zero-length chord" not in str(error):
                raise
    if not seeds:
        return None
    center = seeds[len(seeds) // 2]
    seed = 7919 + sum(node_ids) + 1009 * sum(component)
    rng = np.random.default_rng(seed)
    for _ in range(20):
        trial = center.copy()
        trial[: len(component)] = np.clip(
            center[: len(component)] + rng.normal(0.0, 0.2, len(component)),
            0.0,
            1.0,
        )
        trial[len(component) :] = np.clip(
            center[len(component) :] + rng.normal(0.0, 0.7, len(node_ids)),
            -6.5,
            6.5,
        )
        seeds.append(trial)

    candidates = []
    for initial in seeds:
        try:
            solved = minimize(
                objective,
                initial,
                method="SLSQP",
                bounds=list(zip(lower_bounds, upper_bounds)),
                constraints={"type": "eq", "fun": area_constraints},
                options={"ftol": 1.0e-12, "maxiter": max_nfev, "disp": False},
            )
            values = np.asarray(solved.x, dtype=float)
            full_residual = residual(values)
            area_values = full_residual[: len(node_ids)]
            if (
                np.any(~np.isfinite(full_residual))
                or np.max(np.abs(area_values)) > conservation_tolerance
            ):
                continue
            facets, shared_points = _component_state(
                values, component, joins, node_ids, base_facets
            )
            if not all(
                _facet_is_local(polys[node_id], facets[node_id]) for node_id in node_ids
            ):
                continue
            candidates.append(
                ComponentSolution(
                    facets=facets,
                    shared_points=shared_points,
                    normalized_area_residuals={
                        node_id: abs(float(area_values[position]))
                        for position, node_id in enumerate(node_ids)
                    },
                    tangent_angles={
                        edge_index: abs(float(full_residual[len(node_ids) + position]))
                        for position, edge_index in enumerate(component)
                    },
                    solution_kind="c0_min_tangent",
                    score=objective(values),
                    function_evaluations=int(solved.nfev),
                )
            )
        except Exception:
            continue
    return (
        min(candidates, key=lambda candidate: candidate.score) if candidates else None
    )


def _solve_component(
    component: Sequence[int],
    joins: Sequence[Join],
    polys: Mapping[int, Any],
    base_facets: Mapping[int, Any],
    *,
    max_nfev: int,
    conservation_tolerance: float,
) -> ComponentSolution | None:
    node_ids = _component_nodes(component, joins)
    variable_count = len(component) + len(node_ids)
    lower_bounds = np.asarray([0.0] * len(component) + [-7.0] * len(node_ids))
    upper_bounds = np.asarray([1.0] * len(component) + [7.0] * len(node_ids))
    candidates = []

    def try_initial(initial: np.ndarray) -> None:
        solved = least_squares(
            _component_residual,
            initial,
            args=(component, joins, node_ids, polys, base_facets),
            bounds=(lower_bounds, upper_bounds),
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=max_nfev,
            x_scale="jac",
        )
        refined = root(
            _component_residual,
            solved.x,
            args=(component, joins, node_ids, polys, base_facets),
            method="hybr",
            tol=1.0e-11,
        )
        trial_values = [solved.x]
        if refined.success:
            trial_values.insert(0, refined.x)
        for values in trial_values:
            if len(values) != variable_count:
                continue
            if np.any(values[: len(component)] < -1.0e-9) or np.any(
                values[: len(component)] > 1.0 + 1.0e-9
            ):
                continue
            residual = _component_residual(
                values, component, joins, node_ids, polys, base_facets
            )
            if np.max(np.abs(residual)) > conservation_tolerance:
                continue
            facets, shared_points = _component_state(
                values, component, joins, node_ids, base_facets
            )
            if not all(
                _facet_is_local(polys[node_id], facets[node_id]) for node_id in node_ids
            ):
                continue
            candidate = ComponentSolution(
                facets=facets,
                shared_points=shared_points,
                normalized_area_residuals={
                    node_id: abs(float(residual[position]))
                    for position, node_id in enumerate(node_ids)
                },
                tangent_angles={
                    edge_index: abs(float(residual[len(node_ids) + position]))
                    for position, edge_index in enumerate(component)
                },
                solution_kind="exact_c1",
                score=0.0,
                function_evaluations=int(solved.nfev + getattr(refined, "nfev", 0)),
            )
            candidate.score = _solution_score(candidate, component, joins, base_facets)
            candidates.append(candidate)

    initial_values = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        try:
            initial = _initial_values(
                alpha, component, joins, node_ids, base_facets
            )
        except ValueError as error:
            if "zero-length chord" not in str(error):
                raise
            continue
        initial_values.append(initial)
        try_initial(initial)

    if not candidates and initial_values:
        center = initial_values[len(initial_values) // 2]
        seed = sum(node_ids) + 1009 * sum(component)
        rng = np.random.default_rng(seed)
        for trial in range(30):
            initial = center.copy()
            if trial % 5 == 4:
                initial[: len(component)] = rng.uniform(0.0, 1.0, len(component))
            else:
                initial[: len(component)] = np.clip(
                    center[: len(component)] + rng.normal(0.0, 0.18, len(component)),
                    0.0,
                    1.0,
                )
            initial[len(component) :] = np.clip(
                center[len(component) :] + rng.normal(0.0, 0.65, len(node_ids)),
                -6.5,
                6.5,
            )
            try_initial(initial)
            if candidates:
                break
    if candidates:
        return min(candidates, key=lambda candidate: candidate.score)
    return _solve_component_c0_fallback(
        component,
        joins,
        node_ids,
        polys,
        base_facets,
        lower_bounds,
        upper_bounds,
        max_nfev=max_nfev,
        conservation_tolerance=conservation_tolerance,
    )


def plan_joint_c0_refinement(
    mesh: Any,
    merged_polys: Sequence[Any],
    *,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
    conservation_tolerance: float = DEFAULT_CONSERVATION_TOLERANCE,
    max_nfev: int = DEFAULT_MAX_FUNCTION_EVALUATIONS,
) -> tuple[list[JointC0Assignment], JointC0Report]:
    """Plan conservative joint repairs for all remaining incompatible joins."""

    polys, joins = _collect_joins(mesh, merged_polys)
    base_facets = {merge_id: poly.getFacet() for merge_id, poly in polys.items()}
    before = _join_summary(joins, base_facets, gap_tolerance)
    components = _bad_components(joins, base_facets, gap_tolerance)
    assignments = []
    component_records = []
    final_facets = dict(base_facets)

    for component_index, component in enumerate(components):
        node_ids = _component_nodes(component, joins)
        solution = _solve_component(
            component,
            joins,
            polys,
            base_facets,
            max_nfev=max_nfev,
            conservation_tolerance=conservation_tolerance,
        )
        solved = solution is not None
        if solved:
            for merge_id, facet in solution.facets.items():
                final_facets[merge_id] = facet
                assignments.append(
                    JointC0Assignment(
                        merge_id=merge_id,
                        facet=facet,
                        component_index=component_index,
                        solution_kind=solution.solution_kind,
                    )
                )
        component_records.append(
            JointC0ComponentRecord(
                component_index=component_index,
                merge_ids=tuple(node_ids),
                num_facets=len(node_ids),
                num_bad_joins=len(component),
                solved=solved,
                solution_kind=solution.solution_kind if solved else "failed",
                max_relative_area_residual=(
                    max(solution.normalized_area_residuals.values(), default=0.0)
                    if solved
                    else None
                ),
                max_tangent_angle_radians=(
                    max(solution.tangent_angles.values(), default=0.0)
                    if solved
                    else None
                ),
                function_evaluations=solution.function_evaluations if solved else None,
            )
        )

    after = _join_summary(joins, final_facets, gap_tolerance)
    area_residuals = [
        abs(poly._facet_phase_area(final_facets[merge_id]) - poly.getArea())
        / poly.getMaxArea()
        for merge_id, poly in polys.items()
    ]
    report = JointC0Report(
        mode="joint",
        eligible_joins=before["eligible_joins"],
        bad_joins_before=before["bad_joins"],
        bad_joins_after=after["bad_joins"],
        mean_gap_before=before["mean_gap"],
        mean_gap_after=after["mean_gap"],
        max_gap_before=before["max_gap"],
        max_gap_after=after["max_gap"],
        components=tuple(component_records),
        max_relative_area_residual_after=max(area_residuals, default=0.0),
    )
    return assignments, report


__all__ = [
    "DEFAULT_CONSERVATION_TOLERANCE",
    "DEFAULT_GAP_TOLERANCE",
    "DEFAULT_MAX_FUNCTION_EVALUATIONS",
    "JointC0Assignment",
    "JointC0ComponentRecord",
    "JointC0Report",
    "plan_joint_c0_refinement",
]
