"""Joint shared-endpoint and curvature proof of concept for ellipse case 10."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import brentq, root
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.static import ellipses
from experiments.static import generate_section6_maintext_figures as figures
from main.geoms.circular_facet import getCenter
from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.linear_facet import LinearFacet
from main.structs.meshes.merge_mesh import MergeMesh


CASE_INDEX = 10
PAIR_IDS = ("7", "8")
DEFAULT_RUN_NAME = "joint_c0_case10_poc_source_final_v2_20260801"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "submission" / "c0_joint_case10_poc_20260801"
)


@dataclass
class PairMember:
    merge_id: str
    poly: Any
    shared_side: str
    original_facet: Any
    guarded_facet: Any


@dataclass
class JointSolution:
    shared_point: np.ndarray
    facets: tuple[Any, Any]
    normalized_residuals: np.ndarray
    tangent_dot: float
    alternating_seed: np.ndarray
    root_iterations: int


def _clone_facet(facet: Any) -> Any:
    if isinstance(facet, LinearFacet):
        return LinearFacet(list(facet.pLeft), list(facet.pRight), name=facet.name)
    if isinstance(facet, ArcFacet):
        return ArcFacet(
            list(facet.center),
            float(facet.radius),
            list(facet.pLeft),
            list(facet.pRight),
        )
    raise TypeError(f"Unsupported facet type: {type(facet).__name__}")


def _endpoint(facet: Any, side: str) -> np.ndarray:
    return np.asarray(facet.pLeft if side == "left" else facet.pRight, dtype=float)


def _other_endpoint(facet: Any, side: str) -> np.ndarray:
    return np.asarray(facet.pRight if side == "left" else facet.pLeft, dtype=float)


def _facet_with_shared_endpoint(
    member: PairMember, shared_point: np.ndarray, curvature: float
) -> Any:
    outer = _other_endpoint(member.guarded_facet, member.shared_side)
    if member.shared_side == "left":
        p_left, p_right = shared_point, outer
    else:
        p_left, p_right = outer, shared_point
    chord = float(np.linalg.norm(p_right - p_left))
    if abs(curvature) < 1.0e-12:
        return LinearFacet(p_left.tolist(), p_right.tolist())
    radius = 1.0 / curvature
    if abs(radius) <= 0.5 * chord:
        raise ValueError("Curvature produces an inadmissible circular chord")
    center = getCenter(p_left.tolist(), p_right.tolist(), radius)
    return ArcFacet(center, radius, p_left.tolist(), p_right.tolist())


def _oriented_tangent(facet: Any, point: np.ndarray) -> np.ndarray:
    tangent = np.asarray(facet.getTangent(point.tolist()), dtype=float)
    if isinstance(facet, ArcFacet) and facet.radius < 0.0:
        tangent *= -1.0
    norm = float(np.linalg.norm(tangent))
    if norm == 0.0:
        raise ValueError("Degenerate tangent")
    return tangent / norm


def _tangent_metrics(first: Any, second: Any, point: np.ndarray) -> tuple[float, float]:
    first_tangent = _oriented_tangent(first, point)
    second_tangent = _oriented_tangent(second, point)
    dot = float(np.clip(np.dot(first_tangent, second_tangent), -1.0, 1.0))
    cross = float(
        first_tangent[0] * second_tangent[1] - first_tangent[1] * second_tangent[0]
    )
    return dot, cross


def _shared_edge(
    first: PairMember, second: PairMember
) -> tuple[np.ndarray, np.ndarray]:
    first_points = np.asarray(first.poly.points, dtype=float)
    second_points = np.asarray(second.poly.points, dtype=float)
    shared = []
    for point in first_points:
        if np.min(np.linalg.norm(second_points - point, axis=1)) <= 1.0e-10:
            shared.append(point)
    if len(shared) != 2:
        raise RuntimeError(
            f"Expected two shared cell-edge vertices, found {len(shared)}"
        )
    edge_start, edge_end = sorted(shared, key=lambda point: (point[0], point[1]))
    return np.asarray(edge_start), np.asarray(edge_end)


def _shared_point(edge: tuple[np.ndarray, np.ndarray], coordinate: float) -> np.ndarray:
    return edge[0] + coordinate * (edge[1] - edge[0])


def _edge_coordinate(edge: tuple[np.ndarray, np.ndarray], point: np.ndarray) -> float:
    direction = edge[1] - edge[0]
    return float(np.dot(point - edge[0], direction) / np.dot(direction, direction))


def _area_residual(member: PairMember, facet: Any) -> float:
    return float(member.poly._facet_phase_area(facet) - member.poly.getArea())


def _curvature_roots(
    member: PairMember,
    shared_point: np.ndarray,
    *,
    sample_count: int = 501,
) -> list[float]:
    outer = _other_endpoint(member.guarded_facet, member.shared_side)
    chord = float(np.linalg.norm(outer - shared_point))
    maximum = 2.0 / chord * (1.0 - 1.0e-7)
    linear = np.linspace(-maximum, maximum, sample_count)
    concentrated = np.sign(linear) * maximum * (np.abs(linear) / maximum) ** 3
    curvatures = np.unique(np.concatenate((linear, concentrated, [0.0])))

    values = []
    for curvature in curvatures:
        try:
            facet = _facet_with_shared_endpoint(member, shared_point, float(curvature))
            values.append(_area_residual(member, facet))
        except Exception:
            values.append(float("nan"))

    roots = []
    for left, right, left_value, right_value in zip(
        curvatures[:-1], curvatures[1:], values[:-1], values[1:]
    ):
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            continue
        if left_value == 0.0:
            candidate = float(left)
        elif left_value * right_value > 0.0:
            continue
        else:
            try:
                candidate = brentq(
                    lambda curvature: _area_residual(
                        member,
                        _facet_with_shared_endpoint(
                            member, shared_point, float(curvature)
                        ),
                    ),
                    float(left),
                    float(right),
                    xtol=1.0e-13,
                )
            except Exception:
                continue
        if all(abs(candidate - existing) > 1.0e-7 for existing in roots):
            roots.append(candidate)
    return roots


def _alternating_seed(
    members: tuple[PairMember, PairMember],
    edge: tuple[np.ndarray, np.ndarray],
    midpoint: np.ndarray,
) -> np.ndarray:
    midpoint_coordinate = _edge_coordinate(edge, midpoint)
    search_half_width = 0.22
    lower = max(0.0, midpoint_coordinate - search_half_width)
    upper = min(1.0, midpoint_coordinate + search_half_width)
    best: tuple[float, np.ndarray] | None = None
    for coordinate in np.linspace(lower, upper, 91):
        point = _shared_point(edge, float(coordinate))
        first_roots = _curvature_roots(members[0], point)
        second_roots = _curvature_roots(members[1], point)
        for first_curvature in first_roots:
            first_facet = _facet_with_shared_endpoint(
                members[0], point, first_curvature
            )
            if isinstance(first_facet, ArcFacet) and first_facet.is_major_arc:
                continue
            for second_curvature in second_roots:
                second_facet = _facet_with_shared_endpoint(
                    members[1], point, second_curvature
                )
                if isinstance(second_facet, ArcFacet) and second_facet.is_major_arc:
                    continue
                tangent_dot, tangent_cross = _tangent_metrics(
                    first_facet, second_facet, point
                )
                if tangent_dot <= 0.0:
                    continue
                displacement = coordinate - midpoint_coordinate
                score = (
                    tangent_cross**2
                    + 1.0e-6 * displacement**2
                    + 1.0e-8 * (first_curvature**2 + second_curvature**2)
                )
                row = np.asarray(
                    [coordinate, first_curvature, second_curvature], dtype=float
                )
                if best is None or score < best[0]:
                    best = (score, row)
    if best is None:
        raise RuntimeError("Alternating scan found no jointly conservative seed")
    return best[1]


def _solve_joint(
    members: tuple[PairMember, PairMember],
    edge: tuple[np.ndarray, np.ndarray],
    midpoint: np.ndarray,
) -> JointSolution:
    seed = _alternating_seed(members, edge, midpoint)

    def residual(values: np.ndarray) -> np.ndarray:
        coordinate, first_curvature, second_curvature = map(float, values)
        point = _shared_point(edge, coordinate)
        try:
            first_facet = _facet_with_shared_endpoint(
                members[0], point, first_curvature
            )
            second_facet = _facet_with_shared_endpoint(
                members[1], point, second_curvature
            )
            _, tangent_cross = _tangent_metrics(first_facet, second_facet, point)
            return np.asarray(
                [
                    _area_residual(members[0], first_facet)
                    / members[0].poly.getMaxArea(),
                    _area_residual(members[1], second_facet)
                    / members[1].poly.getMaxArea(),
                    tangent_cross,
                ],
                dtype=float,
            )
        except Exception:
            return np.asarray([1.0e3, 1.0e3, 1.0e3], dtype=float)

    solved = root(residual, seed, method="hybr", tol=1.0e-11)
    point = _shared_point(edge, float(solved.x[0]))
    joint_facets = (
        _facet_with_shared_endpoint(members[0], point, float(solved.x[1])),
        _facet_with_shared_endpoint(members[1], point, float(solved.x[2])),
    )
    normalized_residuals = residual(solved.x)
    tangent_dot, _ = _tangent_metrics(*joint_facets, point)
    valid = (
        solved.success
        and -1.0e-10 <= solved.x[0] <= 1.0 + 1.0e-10
        and np.max(np.abs(normalized_residuals)) <= 1.0e-9
        and tangent_dot >= 1.0 - 1.0e-8
        and all(
            not isinstance(facet, ArcFacet) or not facet.is_major_arc
            for facet in joint_facets
        )
    )
    if not valid:
        raise RuntimeError(
            "Joint solve did not produce an admissible conservative tangent pair: "
            f"success={solved.success}, x={solved.x}, residual={normalized_residuals}, "
            f"tangent_dot={tangent_dot}"
        )
    return JointSolution(
        shared_point=point,
        facets=joint_facets,
        normalized_residuals=normalized_residuals,
        tangent_dot=tangent_dot,
        alternating_seed=seed,
        root_iterations=int(getattr(solved, "nfev", 0)),
    )


def _facet_points(facet: Any, count: int = 600) -> np.ndarray:
    return np.asarray(facet.sample(count), dtype=float)


def _ellipse_points(count: int = 40000) -> np.ndarray:
    params = figures._ellipse_case_params(CASE_INDEX)
    ts = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    local = np.column_stack(
        (params["major_axis"] * np.cos(ts), params["minor_axis"] * np.sin(ts))
    )
    cosine = math.cos(params["theta"])
    sine = math.sin(params["theta"])
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + params["center"]


def _ellipse_edge_intersections(
    edge: tuple[np.ndarray, np.ndarray],
) -> list[np.ndarray]:
    params = figures._ellipse_case_params(CASE_INDEX)
    cosine = math.cos(params["theta"])
    sine = math.sin(params["theta"])
    inverse_rotation = np.asarray([[cosine, sine], [-sine, cosine]])
    origin = inverse_rotation @ (edge[0] - params["center"])
    direction = inverse_rotation @ (edge[1] - edge[0])
    inverse_axes_squared = np.asarray(
        [1.0 / params["major_axis"] ** 2, 1.0 / params["minor_axis"] ** 2]
    )
    coefficients = (
        float(np.sum(direction * direction * inverse_axes_squared)),
        float(2.0 * np.sum(origin * direction * inverse_axes_squared)),
        float(np.sum(origin * origin * inverse_axes_squared) - 1.0),
    )
    intersections = []
    for coordinate in np.roots(coefficients[:2] + (coefficients[2],)):
        if abs(float(np.imag(coordinate))) > 1.0e-10:
            continue
        coordinate = float(np.real(coordinate))
        if -1.0e-10 <= coordinate <= 1.0 + 1.0e-10:
            intersections.append(_shared_point(edge, coordinate))
    return intersections


def _strategy_metrics(
    name: str,
    members: tuple[PairMember, PairMember],
    facets: tuple[Any, Any],
    shared_sides: tuple[str, str],
    truth_tree: cKDTree,
) -> dict[str, Any]:
    endpoints = tuple(
        _endpoint(facet, side) for facet, side in zip(facets, shared_sides)
    )
    gap = float(np.linalg.norm(endpoints[0] - endpoints[1]))
    area_residuals = [
        abs(_area_residual(member, facet)) for member, facet in zip(members, facets)
    ]
    samples = np.concatenate([_facet_points(facet) for facet in facets], axis=0)
    truth_distances = truth_tree.query(samples)[0]
    first_tangent = _oriented_tangent(facets[0], endpoints[0])
    second_tangent = _oriented_tangent(facets[1], endpoints[1])
    tangent_dot = float(np.clip(np.dot(first_tangent, second_tangent), -1.0, 1.0))
    tangent_angle = math.degrees(math.acos(float(np.clip(tangent_dot, -1.0, 1.0))))
    return {
        "strategy": name,
        "join_gap": gap,
        "max_absolute_area_residual": max(area_residuals),
        "max_relative_zone_area_residual": max(
            residual / member.poly.getMaxArea()
            for residual, member in zip(area_residuals, members)
        ),
        "mean_reconstruction_to_truth_distance": float(np.mean(truth_distances)),
        "max_reconstruction_to_truth_distance": float(np.max(truth_distances)),
        "tangent_angle_degrees": tangent_angle,
        "first_curvature": float(facets[0].curvature),
        "second_curvature": float(facets[1].curvature),
        "first_shared_x": float(endpoints[0][0]),
        "first_shared_y": float(endpoints[0][1]),
        "second_shared_x": float(endpoints[1][0]),
        "second_shared_y": float(endpoints[1][1]),
    }


def _draw_facet(axis: Any, facet: Any, color: str, *, linewidth: float = 2.3) -> None:
    points = _facet_points(facet)
    axis.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth, zorder=4)


def _draw_pair(
    axis: Any,
    members: tuple[PairMember, PairMember],
    facets: tuple[Any, Any],
    colors: tuple[str, str],
    truth: np.ndarray,
    *,
    zoom: bool,
) -> None:
    for member in members:
        polygon = np.asarray(member.poly.points, dtype=float)
        polygon = np.vstack((polygon, polygon[0]))
        axis.plot(polygon[:, 0], polygon[:, 1], color="#cbd5e1", linewidth=0.75)
    axis.plot(
        truth[:, 0],
        truth[:, 1],
        color="#111827",
        linestyle="--",
        linewidth=1.1,
        zorder=2,
    )
    for facet, color in zip(facets, colors):
        _draw_facet(axis, facet, color)
        endpoints = np.asarray([facet.pLeft, facet.pRight], dtype=float)
        axis.scatter(
            endpoints[:, 0],
            endpoints[:, 1],
            s=24,
            facecolors="white",
            edgecolors=color,
            linewidths=1.25,
            zorder=6,
        )
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#64748b")
    if zoom:
        axis.set_xlim(27.68, 28.18)
        axis.set_ylim(49.82, 50.48)
    else:
        axis.set_xlim(25.55, 28.32)
        axis.set_ylim(46.65, 50.58)


def _write_summary(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(
    path: Path,
    rows: Sequence[dict[str, Any]],
    solution: JointSolution,
    true_shared_point: np.ndarray,
) -> None:
    by_name = {row["strategy"]: row for row in rows}
    joint = by_name["joint consensus"]
    guarded = by_name["guarded midpoint"]
    text = f"""# Case-10 Joint C0 Proof of Concept

This diagnostic changes no production reconstruction. It isolates merge IDs
`7` and `8` in Cartesian ellipse case `10` at `N=32` and holds each facet's
outer endpoint fixed at the guarded-pass result.

The local solve uses one shared edge coordinate and two signed curvatures. An
alternating scan first solves each cell's conservation equation independently
for fixed shared coordinates. A three-variable root refinement then enforces
both normalized area residuals and tangent agreement simultaneously.

- shared point: `{solution.shared_point.tolist()}`
- analytic crossing: `{true_shared_point.tolist()}`
- shared-point error: `{np.linalg.norm(solution.shared_point - true_shared_point):.6e}`
- curvatures: `{joint['first_curvature']:.12g}`, `{joint['second_curvature']:.12g}`
- joint gap: `{joint['join_gap']:.6e}`
- joint maximum relative area residual: `{joint['max_relative_zone_area_residual']:.6e}`
- joint tangent mismatch: `{joint['tangent_angle_degrees']:.6e}` degrees
- guarded gap: `{guarded['join_gap']:.6e}`
- guarded maximum reconstruction-to-truth distance: `{guarded['max_reconstruction_to_truth_distance']:.6e}`
- joint maximum reconstruction-to-truth distance: `{joint['max_reconstruction_to_truth_distance']:.6e}`

The result establishes local feasibility for this join. It does not yet show
that the solve is unique, robust on arbitrary meshes, or suitable as a global
production pass.
"""
    path.write_text(text)


def _generate_figure(
    output_pdf: Path,
    members: tuple[PairMember, PairMember],
    strategies: Sequence[tuple[str, tuple[Any, Any], tuple[str, str]]],
    metrics: Sequence[dict[str, Any]],
    truth: np.ndarray,
    joint_solution: JointSolution,
) -> Path:
    mpl.rcParams.update(
        {
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 6.9))
    for column, ((name, facets, colors), row) in enumerate(zip(strategies, metrics)):
        _draw_pair(axes[0, column], members, facets, colors, truth, zoom=False)
        _draw_pair(axes[1, column], members, facets, colors, truth, zoom=True)
        axes[0, column].set_title(name, fontsize=10.5, fontweight="bold", pad=7)
        axes[0, column].text(
            0.02,
            0.02,
            f"gap: {row['join_gap']:.3e}\n"
            f"max rel. area residual: {row['max_relative_zone_area_residual']:.3e}\n"
            f"max truth distance: {row['max_reconstruction_to_truth_distance']:.3e}\n"
            f"tangent mismatch: {row['tangent_angle_degrees']:.3e} deg",
            transform=axes[0, column].transAxes,
            ha="left",
            va="bottom",
            fontsize=7.4,
            bbox={
                "boxstyle": "square,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#94a3b8",
                "linewidth": 0.55,
                "alpha": 0.94,
            },
            zorder=8,
        )
    axes[1, 2].scatter(
        [joint_solution.shared_point[0]],
        [joint_solution.shared_point[1]],
        s=38,
        marker="x",
        color="#047857",
        linewidths=1.8,
        zorder=8,
    )
    figure.suptitle(
        "Ellipse case 10: joint shared-endpoint and two-curvature proof of concept",
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )
    handles = [
        Line2D([0], [0], color="#111827", linestyle="--", label="Analytic ellipse"),
        Line2D([0], [0], color="#64748b", linewidth=2.3, label="Original facets"),
        Line2D(
            [0], [0], color="#2563eb", linewidth=2.3, label="Accepted guarded refit"
        ),
        Line2D(
            [0],
            [0],
            color="#dc2626",
            linewidth=2.3,
            label="Retained conservative facet",
        ),
        Line2D(
            [0], [0], color="#059669", linewidth=2.3, label="Joint conservative facets"
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=8.4,
    )
    figure.text(
        0.5,
        0.065,
        "The joint solve preserves both local areas, uses one shared endpoint, and matches the two endpoint tangents.",
        ha="center",
        va="center",
        fontsize=9,
        color="#334155",
    )
    figure.subplots_adjust(
        left=0.035, right=0.985, top=0.90, bottom=0.13, wspace=0.20, hspace=0.16
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    output_png = output_pdf.with_suffix(".png")
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_png


def run(run_name: str, output_dir: Path) -> dict[str, Path]:
    capture: dict[str, Any] = {}
    original_make_c0 = MergeMesh.makeC0

    def instrumented_make_c0(mesh: MergeMesh, merged_polys: Sequence[Any]):
        merge_ids = {
            id(poly): str(merge_id) for merge_id, poly in mesh.merged_polys.items()
        }
        selected = {
            merge_ids[id(poly)]: poly
            for poly in merged_polys
            if merge_ids.get(id(poly)) in PAIR_IDS
        }
        if set(selected) != set(PAIR_IDS):
            raise RuntimeError(f"Could not find target merge pair {PAIR_IDS}")

        first_poly, second_poly = (selected[merge_id] for merge_id in PAIR_IDS)
        first_original = _clone_facet(first_poly.getFacet())
        second_original = _clone_facet(second_poly.getFacet())
        endpoint_pairs = []
        for first_side in ("left", "right"):
            for second_side in ("left", "right"):
                endpoint_pairs.append(
                    (
                        float(
                            np.linalg.norm(
                                _endpoint(first_original, first_side)
                                - _endpoint(second_original, second_side)
                            )
                        ),
                        first_side,
                        second_side,
                    )
                )
        _, first_side, second_side = min(endpoint_pairs)
        midpoint = 0.5 * (
            _endpoint(first_original, first_side)
            + _endpoint(second_original, second_side)
        )

        adjusted = original_make_c0(mesh, merged_polys)
        members = (
            PairMember(
                merge_id=PAIR_IDS[0],
                poly=first_poly,
                shared_side=first_side,
                original_facet=first_original,
                guarded_facet=_clone_facet(first_poly.getFacet()),
            ),
            PairMember(
                merge_id=PAIR_IDS[1],
                poly=second_poly,
                shared_side=second_side,
                original_facet=second_original,
                guarded_facet=_clone_facet(second_poly.getFacet()),
            ),
        )
        edge = _shared_edge(*members)
        solution = _solve_joint(members, edge, midpoint)
        capture.update(
            {
                "members": members,
                "midpoint": midpoint,
                "edge": edge,
                "solution": solution,
            }
        )
        return adjusted

    MergeMesh.makeC0 = instrumented_make_c0
    try:
        ellipses.main(
            config_setting="static/ellipse",
            resolution=0.32,
            facet_algo="linear",
            save_name=run_name,
            num_ellipses=25,
            case_indices=[CASE_INDEX],
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

    if not capture:
        raise RuntimeError("The case-10 C0 hook did not run")
    members: tuple[PairMember, PairMember] = capture["members"]
    solution: JointSolution = capture["solution"]
    original_facets = tuple(member.original_facet for member in members)
    guarded_facets = tuple(member.guarded_facet for member in members)
    shared_sides = tuple(member.shared_side for member in members)
    truth = _ellipse_points()
    truth_tree = cKDTree(truth)
    strategy_specs = [
        ("original conservative", original_facets, ("#64748b", "#64748b")),
        ("guarded midpoint", guarded_facets, ("#2563eb", "#dc2626")),
        ("joint consensus", solution.facets, ("#059669", "#059669")),
    ]
    metric_rows = [
        _strategy_metrics(
            name,
            members,
            facets,
            shared_sides,
            truth_tree,
        )
        for name, facets, _ in strategy_specs
    ]

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / "case10_joint_c0_comparison.pdf"
    output_png = _generate_figure(
        output_pdf,
        members,
        strategy_specs,
        metric_rows,
        truth,
        solution,
    )
    output_csv = output_dir / "case10_joint_c0_metrics.csv"
    _write_summary(output_csv, metric_rows)
    output_json = output_dir / "case10_joint_c0_solution.json"
    true_edge_points = _ellipse_edge_intersections(capture["edge"])
    if not true_edge_points:
        raise RuntimeError(
            "Analytic ellipse does not intersect the selected shared edge"
        )
    true_shared_point = min(
        true_edge_points,
        key=lambda point: float(np.linalg.norm(point - solution.shared_point)),
    )
    output_json.write_text(
        json.dumps(
            {
                "case_index": CASE_INDEX,
                "merge_ids": PAIR_IDS,
                "source_run": run_name,
                "midpoint": capture["midpoint"].tolist(),
                "shared_edge": [point.tolist() for point in capture["edge"]],
                "shared_point": solution.shared_point.tolist(),
                "analytic_shared_point": true_shared_point.tolist(),
                "shared_point_truth_error": float(
                    np.linalg.norm(solution.shared_point - true_shared_point)
                ),
                "curvatures": [float(facet.curvature) for facet in solution.facets],
                "radii": [float(facet.radius) for facet in solution.facets],
                "normalized_residuals": solution.normalized_residuals.tolist(),
                "tangent_dot": solution.tangent_dot,
                "alternating_seed": solution.alternating_seed.tolist(),
                "root_function_evaluations": solution.root_iterations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    output_readme = output_dir / "README.md"
    _write_readme(output_readme, metric_rows, solution, true_shared_point)
    return {
        "pdf": output_pdf,
        "png": output_png,
        "csv": output_csv,
        "json": output_json,
        "readme": output_readme,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    outputs = run(args.run_name, args.output_dir)
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
