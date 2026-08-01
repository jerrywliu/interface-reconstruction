"""Apply joint conservative endpoint repair to rejected ellipse C0 components."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize, root


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.static import ellipses
from experiments.static import generate_section6_maintext_figures as figures
from experiments.submission.optimize_case10_joint_c0 import (
    _clone_facet,
    _endpoint,
    _facet_points,
    _oriented_tangent,
)
from main.geoms.circular_facet import getCenter
from main.geoms.geoms import pointInPoly
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.linear_facet import LinearFacet
from main.structs.meshes.merge_mesh import MergeMesh


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "submission"
    / "ellipse_joint_c0_posthoc_n32_cartesian_final_20260801"
)
DEFAULT_RUN_NAME = "ellipse_joint_c0_posthoc_n32_cartesian_final_20260801"
DEFAULT_BASELINE_ROOT = (
    REPO_ROOT / "plots" / "audit_c0_full_ellipse_linear_n32_w000_20260801"
)
GAP_TOLERANCE = 1.0e-8
CONSERVATION_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class Join:
    first_id: str
    first_side: str
    second_id: str
    second_side: str
    edge_start: np.ndarray
    edge_end: np.ndarray
    first_endpoint: np.ndarray
    second_endpoint: np.ndarray


@dataclass
class ComponentSolution:
    facets: dict[str, Any]
    shared_points: dict[int, np.ndarray]
    normalized_area_residuals: dict[str, float]
    tangent_angles: dict[int, float]
    solution_kind: str
    score: float
    function_evaluations: int


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second)))


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


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
    if abs(_cross(first_direction, second_direction)) > (
        tolerance * first_length * second_length
    ):
        return None
    if abs(_cross(second_start - first_start, first_direction)) > (
        tolerance * first_length
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
                (
                    _point_segment_distance(midpoint, overlap[0], overlap[1]),
                    overlap,
                )
            )
    if not candidates:
        raise RuntimeError("Neighboring reconstruction zones have no shared edge")
    return min(candidates, key=lambda item: item[0])[1]


def _join_key(
    first_id: str, first_side: str, second_id: str, second_side: str
) -> tuple[tuple[str, str], tuple[str, str]]:
    return tuple(sorted(((first_id, first_side), (second_id, second_side))))


def _collect_joins(
    mesh: MergeMesh, merged_polys: Sequence[Any]
) -> tuple[dict[str, Any], list[Join]]:
    merge_id_by_object = {
        id(poly): str(merge_id) for merge_id, poly in mesh.merged_polys.items()
    }
    polys = {
        merge_id_by_object[id(poly)]: poly
        for poly in merged_polys
        if id(poly) in merge_id_by_object
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


def _join_gap(join: Join, facets: Mapping[str, Any]) -> float:
    return _distance(
        _endpoint(facets[join.first_id], join.first_side),
        _endpoint(facets[join.second_id], join.second_side),
    )


def _bad_components(
    joins: Sequence[Join], facets: Mapping[str, Any]
) -> list[list[int]]:
    bad_indices = [
        index
        for index, join in enumerate(joins)
        if _join_gap(join, facets) > GAP_TOLERANCE
    ]
    incident: dict[str, set[int]] = defaultdict(set)
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


def _component_nodes(component: Sequence[int], joins: Sequence[Join]) -> list[str]:
    return sorted(
        {
            merge_id
            for index in component
            for merge_id in (joins[index].first_id, joins[index].second_id)
        },
        key=int,
    )


def _curvature_from_latent(
    latent: float, p_left: np.ndarray, p_right: np.ndarray
) -> float:
    chord = float(np.linalg.norm(p_right - p_left))
    maximum = 2.0 / chord * (1.0 - 1.0e-8)
    return maximum * math.tanh(latent)


def _latent_from_curvature(
    curvature: float, p_left: np.ndarray, p_right: np.ndarray
) -> float:
    chord = float(np.linalg.norm(p_right - p_left))
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
    node_ids: Sequence[str],
    base_facets: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    shared_points = {
        edge_index: _edge_point(join, float(values[position]))
        for position, (edge_index, join) in enumerate(
            (index, joins[index]) for index in component
        )
    }
    endpoint_overrides: dict[tuple[str, str], np.ndarray] = {}
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
    points = _facet_points(facet, 31)[1:-1]
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
    node_ids: Sequence[str],
    polys: Mapping[str, Any],
    base_facets: Mapping[str, Any],
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
            tangent_cross = _cross(first_tangent, second_tangent)
            residuals.append(math.atan2(tangent_cross, tangent_dot))
        return np.asarray(residuals, dtype=float)
    except Exception:
        return np.full(len(node_ids) + len(component), 1.0e3, dtype=float)


def _initial_values(
    alpha: float,
    component: Sequence[int],
    joins: Sequence[Join],
    node_ids: Sequence[str],
    base_facets: Mapping[str, Any],
) -> np.ndarray:
    coordinates = []
    endpoint_overrides: dict[tuple[str, str], np.ndarray] = {}
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
    base_facets: Mapping[str, Any],
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


def _solve_component(
    component: Sequence[int],
    joins: Sequence[Join],
    polys: Mapping[str, Any],
    base_facets: Mapping[str, Any],
    *,
    max_nfev: int,
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
            if np.max(np.abs(residual)) > CONSERVATION_TOLERANCE:
                continue
            facets, shared_points = _component_state(
                values, component, joins, node_ids, base_facets
            )
            if not all(
                _facet_is_local(polys[node_id], facets[node_id]) for node_id in node_ids
            ):
                continue
            tangent_angles = {
                edge_index: abs(float(residual[len(node_ids) + position]))
                for position, edge_index in enumerate(component)
            }
            area_residuals = {
                node_id: abs(float(residual[position]))
                for position, node_id in enumerate(node_ids)
            }
            candidate = ComponentSolution(
                facets=facets,
                shared_points=shared_points,
                normalized_area_residuals=area_residuals,
                tangent_angles=tangent_angles,
                solution_kind="exact_c1",
                score=0.0,
                function_evaluations=int(solved.nfev + getattr(refined, "nfev", 0)),
            )
            candidate.score = _solution_score(candidate, component, joins, base_facets)
            candidates.append(candidate)

    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        try_initial(_initial_values(alpha, component, joins, node_ids, base_facets))

    if not candidates:
        center = _initial_values(0.5, component, joins, node_ids, base_facets)
        seed = sum(int(node_id) for node_id in node_ids) + 1009 * sum(component)
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
    )


def _solve_component_c0_fallback(
    component: Sequence[int],
    joins: Sequence[Join],
    node_ids: Sequence[str],
    polys: Mapping[str, Any],
    base_facets: Mapping[str, Any],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    max_nfev: int,
) -> ComponentSolution | None:
    """Enforce conservative C0 joins while minimizing residual tangent mismatch."""

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
        return float(np.dot(tangent_angles, tangent_angles)) + 1.0e-8 * (
            _solution_score(provisional, component, joins, base_facets)
        )

    seeds = [
        _initial_values(alpha, component, joins, node_ids, base_facets)
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    center = seeds[2]
    seed = 7919 + sum(int(node_id) for node_id in node_ids) + 1009 * sum(component)
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
                or np.max(np.abs(area_values)) > CONSERVATION_TOLERANCE
            ):
                continue
            facets, shared_points = _component_state(
                values, component, joins, node_ids, base_facets
            )
            if not all(
                _facet_is_local(polys[node_id], facets[node_id]) for node_id in node_ids
            ):
                continue
            tangent_angles = {
                edge_index: abs(float(full_residual[len(node_ids) + position]))
                for position, edge_index in enumerate(component)
            }
            area_residuals = {
                node_id: abs(float(area_values[position]))
                for position, node_id in enumerate(node_ids)
            }
            candidate = ComponentSolution(
                facets=facets,
                shared_points=shared_points,
                normalized_area_residuals=area_residuals,
                tangent_angles=tangent_angles,
                solution_kind="c0_min_tangent",
                score=objective(values),
                function_evaluations=int(solved.nfev),
            )
            candidates.append(candidate)
        except Exception:
            continue
    return (
        min(candidates, key=lambda candidate: candidate.score) if candidates else None
    )


def _join_summary(joins: Sequence[Join], facets: Mapping[str, Any]) -> dict[str, Any]:
    gaps = [_join_gap(join, facets) for join in joins]
    return {
        "eligible_joins": len(gaps),
        "bad_joins": sum(gap > GAP_TOLERANCE for gap in gaps),
        "mean_gap": statistics.fmean(gaps) if gaps else 0.0,
        "max_gap": max(gaps, default=0.0),
        "gaps": gaps,
    }


def _apply_posthoc(
    mesh: MergeMesh,
    merged_polys: Sequence[Any],
    *,
    case_index: int,
    max_nfev: int,
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    polys, joins = _collect_joins(mesh, merged_polys)
    base_facets = {
        merge_id: _clone_facet(poly.getFacet()) for merge_id, poly in polys.items()
    }
    before = _join_summary(joins, base_facets)
    components = _bad_components(joins, base_facets)
    component_rows = []
    solved_components = 0
    solved_nodes = set()
    for component_index, component in enumerate(components):
        node_ids = _component_nodes(component, joins)
        solution = _solve_component(
            component,
            joins,
            polys,
            base_facets,
            max_nfev=max_nfev,
        )
        solved = solution is not None
        if solved:
            solved_components += 1
            solved_nodes.update(node_ids)
            for node_id, facet in solution.facets.items():
                polys[node_id].setFacet(facet)
        component_rows.append(
            {
                "case_index": case_index,
                "component_index": component_index,
                "num_nodes": len(node_ids),
                "num_bad_joins": len(component),
                "solved": int(solved),
                "solution_kind": solution.solution_kind if solved else "failed",
                "max_relative_area_residual": (
                    max(solution.normalized_area_residuals.values()) if solved else None
                ),
                "max_tangent_angle_radians": (
                    max(solution.tangent_angles.values()) if solved else None
                ),
                "function_evaluations": (
                    solution.function_evaluations if solved else None
                ),
            }
        )
    final_facets = {merge_id: poly.getFacet() for merge_id, poly in polys.items()}
    after = _join_summary(joins, final_facets)
    area_residuals = [
        abs(poly._facet_phase_area(final_facets[merge_id]) - poly.getArea())
        / poly.getMaxArea()
        for merge_id, poly in polys.items()
    ]
    case_row = {
        "case_index": case_index,
        "eligible_joins": before["eligible_joins"],
        "bad_joins_before": before["bad_joins"],
        "bad_joins_after": after["bad_joins"],
        "repaired_bad_joins": before["bad_joins"] - after["bad_joins"],
        "components": len(components),
        "components_solved": solved_components,
        "components_failed": len(components) - solved_components,
        "nodes_changed": len(solved_nodes),
        "mean_gap_before": before["mean_gap"],
        "mean_gap_after": after["mean_gap"],
        "max_gap_before": before["max_gap"],
        "max_gap_after": after["max_gap"],
        "max_relative_area_residual_after": max(area_residuals, default=0.0),
        "continuous_after": int(after["bad_joins"] == 0),
    }
    geometry = {
        "before": base_facets,
        "after": {
            merge_id: _clone_facet(facet) for merge_id, facet in final_facets.items()
        },
        "polys": polys,
    }
    return (
        list(merged_polys),
        case_row,
        {"components": component_rows, "geometry": geometry},
    )


def _read_case_metrics(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as stream:
        return {int(row["case_index"]): row for row in csv.DictReader(stream)}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _safe_log_values(values: Iterable[float]) -> np.ndarray:
    return np.maximum(np.asarray(list(values), dtype=float), 1.0e-16)


def _generate_summary_figure(
    output_pdf: Path,
    rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    resolution_count: int,
) -> Path:
    mpl.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    case_indices = np.asarray([int(row["case_index"]) for row in rows])
    before_bad = np.asarray([int(row["bad_joins_before"]) for row in rows])
    after_bad = np.asarray([int(row["bad_joins_after"]) for row in rows])
    width = 0.38
    axes[0, 0].bar(
        case_indices - width / 2, before_bad, width, color="#94a3b8", label="Guarded"
    )
    axes[0, 0].bar(
        case_indices + width / 2,
        after_bad,
        width,
        color="#059669",
        label="Joint postprocess",
    )
    axes[0, 0].set_ylabel("Joins above $10^{-8}$")
    axes[0, 0].set_xlabel("Ellipse case")
    axes[0, 0].legend(frameon=False)

    before_max = _safe_log_values(row["max_gap_before"] for row in rows)
    after_max = _safe_log_values(row["max_gap_after"] for row in rows)
    axes[0, 1].scatter(before_max, after_max, color="#059669", s=26)
    lower = min(float(np.min(before_max)), float(np.min(after_max)))
    upper = max(float(np.max(before_max)), float(np.max(after_max)))
    axes[0, 1].plot([lower, upper], [lower, upper], color="#64748b", linestyle="--")
    axes[0, 1].axhline(GAP_TOLERANCE, color="#dc2626", linewidth=0.8, linestyle=":")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Guarded maximum join gap")
    axes[0, 1].set_ylabel("Postprocessed maximum join gap")

    before_h = _safe_log_values(float(row["hausdorff_before"]) for row in rows)
    after_h = _safe_log_values(float(row["hausdorff_after"]) for row in rows)
    axes[1, 0].scatter(before_h, after_h, color="#2563eb", s=26)
    lower = min(float(np.min(before_h)), float(np.min(after_h)))
    upper = max(float(np.max(before_h)), float(np.max(after_h)))
    axes[1, 0].plot([lower, upper], [lower, upper], color="#64748b", linestyle="--")
    axes[1, 0].locator_params(axis="both", nbins=5)
    axes[1, 0].set_xlabel("Guarded Hausdorff")
    axes[1, 0].set_ylabel("Postprocessed Hausdorff")

    sizes = sorted({int(row["num_nodes"]) for row in component_rows})
    solved = [
        sum(
            int(row["solved"])
            for row in component_rows
            if int(row["num_nodes"]) == size
        )
        for size in sizes
    ]
    failed = [
        sum(
            1 - int(row["solved"])
            for row in component_rows
            if int(row["num_nodes"]) == size
        )
        for size in sizes
    ]
    axes[1, 1].bar(sizes, solved, color="#059669", label="Solved")
    axes[1, 1].bar(sizes, failed, bottom=solved, color="#dc2626", label="Failed")
    axes[1, 1].set_xlabel("Facets in rejected-join component")
    axes[1, 1].set_ylabel("Components")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(True, color="#e2e8f0", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
    figure.suptitle(
        f"Ellipse joint conservative postprocessing: Cartesian $N={resolution_count}$",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_pdf, bbox_inches="tight")
    output_png = output_pdf.with_suffix(".png")
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_png


def _draw_facets(axis: Any, facets: Mapping[str, Any], color: str) -> None:
    for facet in facets.values():
        points = _facet_points(facet, 200)
        axis.plot(points[:, 0], points[:, 1], color=color, linewidth=1.45)


def _generate_representative_figure(
    output_pdf: Path,
    geometries: Mapping[int, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    mesh_path: Path,
) -> Path:
    selected = [index for index in (10, 12, 23) if index in geometries]
    if not selected:
        selected = [int(row["case_index"]) for row in rows[:3]]
    by_case = {int(row["case_index"]): row for row in rows}
    figure, axes = plt.subplots(2, len(selected), figsize=(4.0 * len(selected), 7.1))
    if len(selected) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    mesh_segments = figures._mesh_segments(mesh_path)
    for column, case_index in enumerate(selected):
        geometry = geometries[case_index]
        truth = figures._ellipse_true_segments(case_index)
        for row_index, (label, facet_key, color) in enumerate(
            (
                ("Guarded", "before", "#2563eb"),
                ("Joint postprocess", "after", "#059669"),
            )
        ):
            axis = axes[row_index, column]
            figures._add_segments(
                axis, mesh_segments, color="#cbd5e1", linewidth=0.32, alpha=0.65
            )
            figures._add_segments(
                axis, truth, color="#111827", linewidth=1.0, linestyle="--"
            )
            _draw_facets(axis, geometry[facet_key], color)
            xmin, xmax, ymin, ymax = figures._compute_view_bounds(
                truth, min_span=66.0, margin_frac=0.08
            )
            axis.set_xlim(xmin, xmax)
            axis.set_ylim(ymin, ymax)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"Case {case_index}", fontsize=11, fontweight="bold")
            if column == 0:
                axis.set_ylabel(label, fontsize=10, fontweight="bold")
            metrics = by_case[case_index]
            gap_key = "max_gap_before" if row_index == 0 else "max_gap_after"
            bad_key = "bad_joins_before" if row_index == 0 else "bad_joins_after"
            axis.text(
                0.02,
                0.02,
                f"bad joins: {metrics[bad_key]}\nmax gap: {float(metrics[gap_key]):.3e}",
                transform=axis.transAxes,
                fontsize=8,
                va="bottom",
                bbox={
                    "boxstyle": "square,pad=0.25",
                    "facecolor": "white",
                    "alpha": 0.92,
                },
            )
    figure.suptitle("Representative ellipse repairs", fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_pdf, bbox_inches="tight")
    output_png = output_pdf.with_suffix(".png")
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_png


def _write_readme(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
) -> None:
    total_bad_before = sum(int(row["bad_joins_before"]) for row in rows)
    total_bad_after = sum(int(row["bad_joins_after"]) for row in rows)
    solved_components = sum(int(row["solved"]) for row in component_rows)
    exact_c1_components = sum(
        row["solution_kind"] == "exact_c1" for row in component_rows
    )
    c0_fallback_components = sum(
        row["solution_kind"] == "c0_min_tangent" for row in component_rows
    )
    continuous = sum(int(row["continuous_after"]) for row in rows)
    fallback_angles = [
        float(row["max_tangent_angle_radians"])
        for row in component_rows
        if row["solution_kind"] == "c0_min_tangent"
    ]
    text = f"""# Ellipse Joint-C0 Postprocessing

This diagnostic applies a connected-component consensus solve after the guarded
`C0` pass. Every bad join receives one shared edge coordinate and every facet
in the component receives one signed curvature. The primary solve enforces all
zone areas and endpoint tangent matches simultaneously while holding boundary
endpoints of the component fixed. If that system has no admissible root, a
fallback enforces the shared endpoints and zone areas while minimizing the
remaining tangent mismatch.

- cases: `{len(rows)}`
- rejected-join components solved: `{solved_components}/{len(component_rows)}`
- exact-C1 components: `{exact_c1_components}`
- conservative-C0 fallback components: `{c0_fallback_components}`
- maximum fallback tangent mismatch (radians): `{max(fallback_angles, default=0.0):.6e}`
- joins above `1e-8`: `{total_bad_before} -> {total_bad_after}`
- globally continuous cases: `{continuous}/{len(rows)}`
- median maximum gap: `{statistics.median(float(row['max_gap_before']) for row in rows):.6e} -> {statistics.median(float(row['max_gap_after']) for row in rows):.6e}`
- median Hausdorff: `{statistics.median(float(row['hausdorff_before']) for row in rows):.6e} -> {statistics.median(float(row['hausdorff_after']) for row in rows):.6e}`

Production reconstruction is unchanged. Failed or inadmissible components are
left exactly as produced by the guarded pass.
"""
    path.write_text(text)


def run(
    *,
    run_name: str,
    output_dir: Path,
    baseline_root: Path,
    resolution: float,
    case_indices: Sequence[int] | None,
    max_nfev: int,
) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = list(case_indices) if case_indices is not None else list(range(25))
    case_iterator = iter(cases)
    case_rows = []
    component_rows = []
    geometries = {}
    original_make_c0 = MergeMesh.makeC0

    def instrumented_make_c0(mesh: MergeMesh, merged_polys: Sequence[Any]):
        adjusted = original_make_c0(mesh, merged_polys)
        case_index = next(case_iterator)
        previous_stage = getattr(mesh, "_provenance_stage", None)
        if hasattr(mesh, "_provenance_stage"):
            mesh._provenance_stage = "joint_c0_posthoc"
        try:
            adjusted, case_row, details = _apply_posthoc(
                mesh,
                adjusted,
                case_index=case_index,
                max_nfev=max_nfev,
            )
        finally:
            if hasattr(mesh, "_provenance_stage"):
                mesh._provenance_stage = previous_stage
        case_rows.append(case_row)
        component_rows.extend(details["components"])
        geometries[case_index] = details["geometry"]
        return adjusted

    MergeMesh.makeC0 = instrumented_make_c0
    try:
        ellipses.main(
            config_setting="static/ellipse",
            resolution=resolution,
            facet_algo="linear",
            save_name=run_name,
            num_ellipses=25,
            case_indices=cases,
            mesh_type="perturbed_quads",
            perturb_wiggle=0.0,
            perturb_seed=0,
            perturb_fix_boundary=True,
            do_c0=True,
            plic_fallback="LVIRA",
            corner_behavior_profile="pre_f8_corner",
        )
    finally:
        MergeMesh.makeC0 = original_make_c0

    baseline_metrics = _read_case_metrics(
        baseline_root / "metrics" / "case_metrics.csv"
    )
    after_metrics = _read_case_metrics(
        REPO_ROOT / "plots" / run_name / "metrics" / "case_metrics.csv"
    )
    for row in case_rows:
        case_index = int(row["case_index"])
        row["hausdorff_before"] = baseline_metrics[case_index]["hausdorff"]
        row["hausdorff_after"] = after_metrics[case_index]["hausdorff"]
        row["facet_gap_metric_before"] = baseline_metrics[case_index]["facet_gap"]
        row["facet_gap_metric_after"] = after_metrics[case_index]["facet_gap"]
    case_rows.sort(key=lambda row: int(row["case_index"]))
    component_rows.sort(
        key=lambda row: (int(row["case_index"]), int(row["component_index"]))
    )

    case_csv = output_dir / "case_summary.csv"
    component_csv = output_dir / "component_summary.csv"
    _write_csv(case_csv, case_rows)
    _write_csv(component_csv, component_rows)
    summary_pdf = output_dir / "ellipse_joint_c0_summary.pdf"
    summary_png = _generate_summary_figure(
        summary_pdf, case_rows, component_rows, int(round(100 * resolution))
    )
    representative_pdf = output_dir / "ellipse_joint_c0_representatives.pdf"
    representative_png = _generate_representative_figure(
        representative_pdf,
        geometries,
        case_rows,
        REPO_ROOT / "plots" / run_name / "vtk" / "mesh.vtk",
    )
    readme = output_dir / "README.md"
    _write_readme(readme, case_rows, component_rows)
    return {
        "case_csv": case_csv,
        "component_csv": component_csv,
        "summary_pdf": summary_pdf,
        "summary_png": summary_png,
        "representative_pdf": representative_pdf,
        "representative_png": representative_png,
        "readme": readme,
    }


def _parse_case_indices(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return sorted({int(item.strip()) for item in raw.split(",") if item.strip()})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--resolution", type=float, default=0.32)
    parser.add_argument("--case-indices", default=None)
    parser.add_argument("--max-nfev", type=int, default=3000)
    args = parser.parse_args(argv)
    outputs = run(
        run_name=args.run_name,
        output_dir=args.output_dir,
        baseline_root=args.baseline_root.resolve(),
        resolution=args.resolution,
        case_indices=_parse_case_indices(args.case_indices),
        max_nfev=args.max_nfev,
    )
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
