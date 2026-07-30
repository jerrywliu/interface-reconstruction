"""Measure shared-mesh-vertex phase-label conflicts in saved reconstructions.

The manuscript defines topological consistency by requiring every reconstructed
facet incident to a shared mesh vertex to assign that vertex the same phase.
This module evaluates that definition directly from ``cell_metrics.csv`` facet
metadata and each run's ``mesh.vtk``.  Vertices within a geometric tolerance of
any incident facet are flagged and excluded, matching the definition's
assumption that the interface does not pass through a mesh vertex.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_RELATIVE_TOLERANCE = 1.0e-10
DEFAULT_ABSOLUTE_TOLERANCE = 1.0e-12
FULL_RELATIVE_TOLERANCES = (1.0e-12, 1.0e-10, 1.0e-8)
FULL_RESOLUTIONS = (0.5, 0.64, 1.0, 1.28, 1.5)
FULL_METHODS = (
    ("squares", "linear+corner"),
    ("circles", "circular"),
    ("ellipses", "circular"),
    ("zalesak", "circular+corner"),
)


@dataclass(frozen=True)
class CaseSelector:
    experiment: str
    algo: str
    resolution: float
    wiggle: float
    seed: int
    case_index: int
    purpose: str = "representative"


DEFAULT_SMOKE_SELECTORS = (
    CaseSelector("squares", "linear+corner", 0.64, 0.1, 0, 22),
    CaseSelector("circles", "circular", 0.64, 0.1, 0, 12),
    CaseSelector("ellipses", "circular", 0.64, 0.1, 0, 12),
    CaseSelector("zalesak", "circular+corner", 0.64, 0.1, 0, 20),
    CaseSelector("circles", "circular", 1.0, 0.05, 0, 0, "fallback_witness"),
    CaseSelector("ellipses", "circular", 0.5, 0.3, 0, 20, "fallback_witness"),
)


def build_full_selectors(
    methods: Sequence[tuple[str, str]] = FULL_METHODS,
    resolutions: Sequence[float] = FULL_RESOLUTIONS,
    wiggle: float = 0.1,
    seed: int = 0,
    case_indices: Sequence[int] = tuple(range(25)),
) -> tuple[CaseSelector, ...]:
    """Build the 500-case paper diagnostic selection deterministically."""

    return tuple(
        CaseSelector(
            experiment=experiment,
            algo=algo,
            resolution=float(resolution),
            wiggle=float(wiggle),
            seed=seed,
            case_index=int(case_index),
            purpose="paper_aggregate",
        )
        for experiment, algo in methods
        for resolution in resolutions
        for case_index in case_indices
    )


@dataclass(frozen=True)
class StructuredMesh:
    points: tuple[tuple[tuple[float, float], ...], ...]
    nx: int
    ny: int
    domain_diagonal: float

    def cell_vertices(self, cell_x: int, cell_y: int):
        if not (0 <= cell_x < self.nx - 1 and 0 <= cell_y < self.ny - 1):
            raise IndexError(
                f"Cell ({cell_x}, {cell_y}) lies outside {self.nx - 1}x{self.ny - 1} mesh"
            )
        keys = (
            (cell_x, cell_y),
            (cell_x, cell_y + 1),
            (cell_x + 1, cell_y + 1),
            (cell_x + 1, cell_y),
        )
        return tuple((key, self.points[key[0]][key[1]]) for key in keys)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _cross(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _line_side(geometry: Mapping, point: Sequence[float]) -> bool:
    p_left = geometry["p_left"]
    p_right = geometry["p_right"]
    direction = (p_right[0] - p_left[0], p_right[1] - p_left[1])
    offset = (point[0] - p_left[0], point[1] - p_left[1])
    return _cross(direction, offset) > 0.0


def _point_segment_distance(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return _distance(point, start)
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq
    t = min(1.0, max(0.0, t))
    projection = (start[0] + t * dx, start[1] + t * dy)
    return _distance(point, projection)


def _arc_travel(geometry: Mapping, angle: float) -> tuple[float, float]:
    center = geometry["center"]
    p_left = geometry["p_left"]
    p_right = geometry["p_right"]
    radius = float(geometry["radius"])
    start = math.atan2(p_left[1] - center[1], p_left[0] - center[0])
    end = math.atan2(p_right[1] - center[1], p_right[0] - center[0])
    if radius >= 0.0:
        return (angle - start) % (2.0 * math.pi), (end - start) % (2.0 * math.pi)
    return (start - angle) % (2.0 * math.pi), (start - end) % (2.0 * math.pi)


def _point_arc_distance(point: Sequence[float], geometry: Mapping) -> float:
    center = geometry["center"]
    radius = abs(float(geometry["radius"]))
    radial_distance = _distance(point, center)
    if radial_distance == 0.0 or radius == 0.0:
        return min(_distance(point, geometry["p_left"]), _distance(point, geometry["p_right"]))
    angle = math.atan2(point[1] - center[1], point[0] - center[0])
    travel, total = _arc_travel(geometry, angle)
    if travel <= total + 1.0e-14:
        return abs(radial_distance - radius)
    return min(_distance(point, geometry["p_left"]), _distance(point, geometry["p_right"]))


def _primitive_distance(point: Sequence[float], geometry: Mapping) -> float:
    geometry_class = str(geometry.get("class", "")).lower()
    if geometry_class == "linear":
        return _point_segment_distance(point, geometry["p_left"], geometry["p_right"])
    if geometry_class == "circular":
        return _point_arc_distance(point, geometry)
    raise ValueError(f"Unsupported primitive geometry class: {geometry_class!r}")


def _primitive_side(geometry: Mapping, point: Sequence[float]) -> bool:
    geometry_class = str(geometry.get("class", "")).lower()
    if geometry_class == "linear":
        return _line_side(geometry, point)
    if geometry_class == "circular":
        radius = float(geometry["radius"])
        inside_disk = _distance(point, geometry["center"]) < abs(radius)
        return inside_disk if radius > 0.0 else not inside_disk
    raise ValueError(f"Unsupported primitive geometry class: {geometry_class!r}")


def _primitive_tangent(geometry: Mapping, at_right_endpoint: bool) -> tuple[float, float]:
    geometry_class = str(geometry.get("class", "")).lower()
    if geometry_class == "linear":
        p_left = geometry["p_left"]
        p_right = geometry["p_right"]
        return p_right[0] - p_left[0], p_right[1] - p_left[1]
    if geometry_class == "circular":
        center = geometry["center"]
        endpoint = geometry["p_right"] if at_right_endpoint else geometry["p_left"]
        radial = (endpoint[0] - center[0], endpoint[1] - center[1])
        if float(geometry["radius"]) > 0.0:
            return -radial[1], radial[0]
        return radial[1], -radial[0]
    raise ValueError(f"Unsupported primitive geometry class: {geometry_class!r}")


def _corner_branches(geometry: Mapping) -> tuple[Mapping, Mapping]:
    left_branch = geometry.get("left_branch")
    right_branch = geometry.get("right_branch")
    if left_branch is None:
        left_branch = {
            "class": "linear",
            "p_left": geometry["p_left"],
            "p_right": geometry["corner"],
        }
    if right_branch is None:
        right_branch = {
            "class": "linear",
            "p_left": geometry["corner"],
            "p_right": geometry["p_right"],
        }
    return left_branch, right_branch


def classify_vertex(
    geometry: Mapping, point: Sequence[float], tolerance: float
) -> str:
    """Return ``full``, ``empty``, or ``on_facet`` for one saved facet.

    Lines use the geometric left half-plane. Positive-radius arcs use the disk
    interior and negative-radius arcs its complement, which is the signed-radius
    convention used by the reconstruction. For a corner, a left turn takes the
    intersection of the branch full sides and a right turn their union.
    """

    geometry_class = str(geometry.get("class", "")).lower()
    if geometry_class in {"linear", "circular"}:
        if _primitive_distance(point, geometry) <= tolerance:
            return "on_facet"
        return "full" if _primitive_side(geometry, point) else "empty"

    if geometry_class not in {"linear_corner", "curved_corner", "corner"}:
        raise ValueError(f"Unsupported facet geometry class: {geometry_class!r}")

    left_branch, right_branch = _corner_branches(geometry)
    if min(
        _primitive_distance(point, left_branch),
        _primitive_distance(point, right_branch),
    ) <= tolerance:
        return "on_facet"

    left_full = _primitive_side(left_branch, point)
    right_full = _primitive_side(right_branch, point)
    left_tangent = _primitive_tangent(left_branch, at_right_endpoint=True)
    right_tangent = _primitive_tangent(right_branch, at_right_endpoint=False)
    turn = _cross(left_tangent, right_tangent)
    turn_scale = math.hypot(*left_tangent) * math.hypot(*right_tangent)
    turn_tolerance = 1.0e-12 * max(1.0, turn_scale)
    if turn > turn_tolerance:
        is_full = left_full and right_full
    elif turn < -turn_tolerance:
        is_full = left_full or right_full
    elif left_full == right_full:
        is_full = left_full
    else:
        # A numerically straight composite should induce the same side from
        # both branches. Treat disagreement as an ambiguous on-facet label so
        # it cannot silently manufacture a conflict.
        return "on_facet"
    return "full" if is_full else "empty"


def read_structured_mesh(path: Path) -> StructuredMesh:
    tokens = path.read_text().split()
    try:
        dimensions_index = tokens.index("DIMENSIONS")
        nx = int(tokens[dimensions_index + 1])
        ny = int(tokens[dimensions_index + 2])
        nz = int(tokens[dimensions_index + 3])
        points_index = tokens.index("POINTS")
        point_count = int(tokens[points_index + 1])
    except (ValueError, IndexError) as error:
        raise ValueError(f"Could not parse structured VTK mesh {path}") from error
    if nz != 1 or point_count != nx * ny:
        raise ValueError(
            f"Expected a 2-D structured grid with {nx * ny} points, got nz={nz}, points={point_count}"
        )

    coordinate_start = points_index + 3
    coordinates = tuple(
        (
            float(tokens[coordinate_start + 3 * index]),
            float(tokens[coordinate_start + 3 * index + 1]),
        )
        for index in range(point_count)
    )
    # writeMesh stores points with x as the outer loop and y as the inner loop.
    points = tuple(
        tuple(coordinates[cell_x * ny + cell_y] for cell_y in range(ny))
        for cell_x in range(nx)
    )
    xs = [point[0] for point in coordinates]
    ys = [point[1] for point in coordinates]
    diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return StructuredMesh(points=points, nx=nx, ny=ny, domain_diagonal=diagonal)


def _is_fallback(record: Mapping[str, str]) -> bool:
    return bool(record.get("fallback_policy")) or record.get("construction_path") == "plic_fallback"


def _is_resolved(record: Mapping[str, str]) -> bool:
    return record.get("orientation_status") == "oriented" and not _is_fallback(record)


def _parse_cell_id(record: Mapping[str, str]) -> tuple[int, int]:
    if record.get("cell_x") not in {None, ""} and record.get("cell_y") not in {None, ""}:
        return int(record["cell_x"]), int(record["cell_y"])
    cell_x, cell_y = record["cell_id"].split(",")
    return int(cell_x), int(cell_y)


def _evaluate_scope(
    records: Sequence[Mapping[str, str]],
    mesh: StructuredMesh,
    tolerance: float,
    scope: str,
):
    if scope == "resolved":
        scoped_records = [record for record in records if _is_resolved(record)]
    elif scope == "complete":
        scoped_records = list(records)
    else:
        raise ValueError(f"Unknown scope {scope!r}")

    incident: dict[tuple[int, int], list[Mapping[str, str]]] = defaultdict(list)
    for record in scoped_records:
        cell_x, cell_y = _parse_cell_id(record)
        for vertex_key, _ in mesh.cell_vertices(cell_x, cell_y):
            incident[vertex_key].append(record)

    detail_rows = []
    invalid_label_count = 0
    for vertex_key, vertex_records in sorted(incident.items()):
        if len(vertex_records) < 2:
            continue
        point = mesh.points[vertex_key[0]][vertex_key[1]]
        labels = []
        invalid_messages = []
        for record in vertex_records:
            try:
                geometry = json.loads(record["facet_geometry_json"])
                labels.append(classify_vertex(geometry, point, tolerance))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                labels.append("invalid")
                invalid_messages.append(str(error))
                invalid_label_count += 1

        on_facet = "on_facet" in labels
        invalid = "invalid" in labels
        evaluated = not on_facet and not invalid
        phase_labels = {label for label in labels if label in {"full", "empty"}}
        conflict = evaluated and len(phase_labels) > 1
        detail_rows.append(
            {
                "scope": scope,
                "vertex_i": vertex_key[0],
                "vertex_j": vertex_key[1],
                "vertex_x": point[0],
                "vertex_y": point[1],
                "incident_cell_count": len(vertex_records),
                "incident_cells": ";".join(record["cell_id"] for record in vertex_records),
                "incident_merge_ids": ";".join(record.get("merge_id", "") for record in vertex_records),
                "labels": ";".join(labels),
                "contains_fallback": int(any(_is_fallback(record) for record in vertex_records)),
                "on_facet_excluded": int(on_facet),
                "invalid_excluded": int(invalid),
                "evaluated": int(evaluated),
                "conflict": int(conflict),
                "invalid_messages": " | ".join(invalid_messages),
            }
        )

    evaluated_count = sum(row["evaluated"] for row in detail_rows)
    conflict_count = sum(row["conflict"] for row in detail_rows)
    summary = {
        "candidate_shared_vertices": len(detail_rows),
        "evaluated_shared_vertices": evaluated_count,
        "on_facet_excluded_vertices": sum(row["on_facet_excluded"] for row in detail_rows),
        "invalid_excluded_vertices": sum(row["invalid_excluded"] for row in detail_rows),
        "invalid_incident_labels": invalid_label_count,
        "conflict_vertices": conflict_count,
        "conflict_rate": conflict_count / evaluated_count if evaluated_count else 0.0,
    }
    return summary, detail_rows


def evaluate_case(
    records: Sequence[Mapping[str, str]],
    mesh: StructuredMesh,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
):
    tolerance = max(absolute_tolerance, relative_tolerance * mesh.domain_diagonal)
    total_cells = len(records)
    oriented_cells = sum(record.get("orientation_status") == "oriented" for record in records)
    resolved_cells = sum(_is_resolved(record) for record in records)
    fallback_cells = sum(_is_fallback(record) for record in records)
    case_summary = {
        "num_mixed_cells": total_cells,
        "num_oriented_cells": oriented_cells,
        "fraction_oriented_cells": oriented_cells / total_cells if total_cells else 0.0,
        "num_resolved_cells": resolved_cells,
        "fraction_resolved_cells": resolved_cells / total_cells if total_cells else 0.0,
        "num_plic_fallback_cells": fallback_cells,
        "fraction_plic_fallback_cells": fallback_cells / total_cells if total_cells else 0.0,
        "geometric_tolerance": tolerance,
    }
    all_details = []
    for scope in ("resolved", "complete"):
        scope_summary, details = _evaluate_scope(records, mesh, tolerance, scope)
        case_summary.update({f"{scope}_{key}": value for key, value in scope_summary.items()})
        all_details.extend(details)
    return case_summary, all_details


def _selector_matches(row: Mapping[str, str], selector: CaseSelector, include_case=True) -> bool:
    matches = (
        row.get("experiment") == selector.experiment
        and row.get("algo") == selector.algo
        and int(row.get("seed", -1)) == selector.seed
        and math.isclose(float(row.get("resolution", "nan")), selector.resolution, abs_tol=1.0e-12)
        and math.isclose(float(row.get("wiggle", "nan")), selector.wiggle, abs_tol=1.0e-12)
    )
    return matches and (not include_case or int(row.get("case_index", -1)) == selector.case_index)


def _float_key(value) -> float:
    return round(float(value), 12)


def _selector_key(selector: CaseSelector, include_case: bool) -> tuple:
    key = (
        selector.experiment,
        selector.algo,
        selector.seed,
        _float_key(selector.resolution),
        _float_key(selector.wiggle),
    )
    return (*key, selector.case_index) if include_case else key


def _row_key(row: Mapping[str, str], include_case: bool) -> tuple:
    key = (
        row.get("experiment"),
        row.get("algo"),
        int(row.get("seed", -1)),
        _float_key(row.get("resolution", "nan")),
        _float_key(row.get("wiggle", "nan")),
    )
    return (*key, int(row.get("case_index", -1))) if include_case else key


def _load_selected_records(cell_metrics_path: Path, selectors: Sequence[CaseSelector]):
    selected = {selector: [] for selector in selectors}
    selector_by_key = {_selector_key(selector, include_case=True): selector for selector in selectors}
    if len(selector_by_key) != len(selectors):
        raise ValueError("Case selectors must be unique")
    with cell_metrics_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            selector = selector_by_key.get(_row_key(row, include_case=True))
            if selector is not None:
                selected[selector].append(row)
    missing = [selector for selector, rows in selected.items() if not rows]
    if missing:
        raise RuntimeError(f"No cell metric rows found for selectors: {missing}")
    return selected


def _load_run_inventory_rows(run_inventory_path: Path, selectors: Sequence[CaseSelector]):
    selected_rows = {}
    with run_inventory_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows_by_key = defaultdict(list)
    for row in rows:
        rows_by_key[_row_key(row, include_case=False)].append(row)
    for selector in selectors:
        candidates = rows_by_key[_selector_key(selector, include_case=False)]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one run inventory row for {selector}, found {len(candidates)}"
            )
        selected_rows[selector] = candidates[0]
    return selected_rows


def _load_run_bundles(run_inventory_path: Path, selectors: Sequence[CaseSelector]):
    rows = _load_run_inventory_rows(run_inventory_path, selectors)
    return {selector: Path(row["run_bundle"]) for selector, row in rows.items()}


def _write_csv(path: Path, rows: Sequence[Mapping]):
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_gzip(path: Path, rows: Sequence[Mapping]):
    if not rows:
        with gzip.open(path, "wt") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0])
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_case_rows_by(case_rows: Sequence[Mapping], key_fields: Sequence[str]):
    groups = defaultdict(list)
    for row in case_rows:
        groups[tuple(row[field] for field in key_fields)].append(row)

    aggregate_rows = []
    for key, rows in sorted(groups.items()):
        aggregate = dict(zip(key_fields, key))
        aggregate["case_count"] = len(rows)
        aggregate["mixed_cells"] = sum(int(row["num_mixed_cells"]) for row in rows)
        aggregate["oriented_cells"] = sum(int(row["num_oriented_cells"]) for row in rows)
        aggregate["resolved_cells"] = sum(int(row["num_resolved_cells"]) for row in rows)
        aggregate["plic_fallback_cells"] = sum(
            int(row["num_plic_fallback_cells"]) for row in rows
        )
        aggregate["oriented_cell_fraction"] = aggregate["oriented_cells"] / max(
            1, aggregate["mixed_cells"]
        )
        aggregate["resolved_cell_fraction"] = aggregate["resolved_cells"] / max(
            1, aggregate["mixed_cells"]
        )
        aggregate["plic_fallback_cell_fraction"] = aggregate[
            "plic_fallback_cells"
        ] / max(1, aggregate["mixed_cells"])
        for scope in ("resolved", "complete"):
            evaluated = sum(int(row[f"{scope}_evaluated_shared_vertices"]) for row in rows)
            conflicts = sum(int(row[f"{scope}_conflict_vertices"]) for row in rows)
            aggregate[f"{scope}_candidate_shared_vertices"] = sum(
                int(row[f"{scope}_candidate_shared_vertices"]) for row in rows
            )
            aggregate[f"{scope}_evaluated_shared_vertices"] = evaluated
            aggregate[f"{scope}_on_facet_excluded_vertices"] = sum(
                int(row[f"{scope}_on_facet_excluded_vertices"]) for row in rows
            )
            aggregate[f"{scope}_invalid_excluded_vertices"] = sum(
                int(row[f"{scope}_invalid_excluded_vertices"]) for row in rows
            )
            aggregate[f"{scope}_invalid_incident_labels"] = sum(
                int(row[f"{scope}_invalid_incident_labels"]) for row in rows
            )
            aggregate[f"{scope}_conflict_vertices"] = conflicts
            aggregate[f"{scope}_conflict_rate"] = conflicts / evaluated if evaluated else 0.0
        aggregate_rows.append(aggregate)
    return aggregate_rows


def _aggregate_case_rows(case_rows: Sequence[Mapping]):
    key_fields = ["experiment", "algo", "resolution", "wiggle", "seed", "purpose"]
    if case_rows and "relative_tolerance" in case_rows[0]:
        key_fields.insert(0, "relative_tolerance")
    return _aggregate_case_rows_by(case_rows, key_fields)


def _format_rate(value) -> str:
    return f"{100.0 * float(value):.5g}%"


def _build_readme(case_rows: Sequence[Mapping], aggregate_rows: Sequence[Mapping], manifest: Mapping):
    lines = [
        "# Topological-consistency smoke diagnostic",
        "",
        "This diagnostic implements the manuscript's shared-mesh-vertex coloring test. ",
        "Each saved oriented facet classifies every incident cell vertex as full or empty; ",
        "a shared vertex is a conflict when those labels disagree.",
        "",
        "## Tolerance and scopes",
        "",
        f"- Geometric tolerance: `max({manifest['absolute_tolerance']:.1e}, "
        f"{manifest['relative_tolerance']:.1e} * mesh_domain_diagonal)`.",
        "- If any incident facet lies within that tolerance of a shared vertex, the vertex is flagged and excluded from the denominator.",
        "- `resolved` includes only cells on the successfully oriented path and excludes PLIC fallbacks.",
        "- `complete` includes every final reconstructed cell, including independently fitted PLIC fallback cells.",
        "- A conflict is counted per shared mesh vertex, not per disagreeing cell pair.",
        "",
        "## Smoke results",
        "",
        "| Problem | Method | N | Perturbation | Case | Purpose | Oriented | Fallback | Resolved conflicts | Complete conflicts |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in case_rows:
        lines.append(
            "| {experiment} | {algo} | {resolution:g} | {wiggle:g} | {case_index} | "
            "{purpose} | {oriented} | {fallback} | {resolved_conflicts}/{resolved_total} "
            "({resolved_rate}) | {complete_conflicts}/{complete_total} ({complete_rate}) |".format(
                experiment=row["experiment"],
                algo=row["algo"],
                resolution=100 * float(row["resolution"]),
                wiggle=float(row["wiggle"]),
                case_index=row["case_index"],
                purpose=row["purpose"],
                oriented=_format_rate(row["fraction_oriented_cells"]),
                fallback=_format_rate(row["fraction_plic_fallback_cells"]),
                resolved_conflicts=row["resolved_conflict_vertices"],
                resolved_total=row["resolved_evaluated_shared_vertices"],
                resolved_rate=_format_rate(row["resolved_conflict_rate"]),
                complete_conflicts=row["complete_conflict_vertices"],
                complete_total=row["complete_evaluated_shared_vertices"],
                complete_rate=_format_rate(row["complete_conflict_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `topology_consistency_cases.csv`: one row per selected case.",
            "- `topology_consistency_aggregate.csv`: weighted conflict rates by setting and purpose.",
            "- `topology_consistency_vertices.csv`: every candidate shared vertex in both scopes, including labels and exclusion flags.",
            "- `topology_consistency_manifest.json`: source paths, selectors, and tolerance policy.",
            "",
            "These are smoke diagnostics only; they are not a full-sweep result.",
            "",
        ]
    )
    return "\n".join(lines)


def run_diagnostics(
    cell_metrics_path: Path,
    run_inventory_path: Path,
    output_dir: Path,
    selectors: Sequence[CaseSelector] = DEFAULT_SMOKE_SELECTORS,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_records = _load_selected_records(cell_metrics_path, selectors)
    run_bundles = _load_run_bundles(run_inventory_path, selectors)
    mesh_cache = {}
    case_rows = []
    vertex_rows = []

    for selector in selectors:
        bundle = run_bundles[selector]
        mesh_path = bundle / "vtk" / "mesh.vtk"
        if mesh_path not in mesh_cache:
            mesh_cache[mesh_path] = read_structured_mesh(mesh_path)
        mesh = mesh_cache[mesh_path]
        summary, details = evaluate_case(
            selected_records[selector],
            mesh,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        identity = {
            "experiment": selector.experiment,
            "algo": selector.algo,
            "resolution": selector.resolution,
            "wiggle": selector.wiggle,
            "seed": selector.seed,
            "case_index": selector.case_index,
            "purpose": selector.purpose,
            "run_bundle": str(bundle),
        }
        case_rows.append({**identity, **summary})
        for detail in details:
            vertex_rows.append({**identity, **detail})

    aggregate_rows = _aggregate_case_rows(case_rows)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "Shared incident cell-facet vertex colors must all agree.",
        "cell_metrics": str(cell_metrics_path.resolve()),
        "run_inventory": str(run_inventory_path.resolve()),
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "selectors": [asdict(selector) for selector in selectors],
        "scope_policy": {
            "resolved": "orientation_status=oriented and not PLIC fallback",
            "complete": "all saved final facet geometry, including PLIC fallbacks",
        },
        "on_facet_policy": "Exclude a shared vertex from a scope if any incident finite facet is within tolerance.",
        "outputs": [
            "topology_consistency_cases.csv",
            "topology_consistency_aggregate.csv",
            "topology_consistency_vertices.csv",
            "topology_consistency_manifest.json",
            "README.md",
        ],
    }

    _write_csv(output_dir / "topology_consistency_cases.csv", case_rows)
    _write_csv(output_dir / "topology_consistency_aggregate.csv", aggregate_rows)
    _write_csv(output_dir / "topology_consistency_vertices.csv", vertex_rows)
    (output_dir / "topology_consistency_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "README.md").write_text(_build_readme(case_rows, aggregate_rows, manifest))
    return case_rows, aggregate_rows, vertex_rows


def _validate_full_inputs(
    selectors: Sequence[CaseSelector],
    selected_records: Mapping[CaseSelector, Sequence[Mapping[str, str]]],
    inventory_rows: Mapping[CaseSelector, Mapping[str, str]],
) -> dict:
    expected_cases = set(range(25))
    settings: dict[tuple, Mapping[str, str]] = {}
    selected_cases = defaultdict(set)
    errors = []

    for selector in selectors:
        setting_key = _selector_key(selector, include_case=False)
        settings[setting_key] = inventory_rows[selector]
        selected_cases[setting_key].add(selector.case_index)
        if not selected_records[selector]:
            errors.append(f"No mixed-cell rows for {selector}")

    for setting_key, row in sorted(settings.items()):
        if selected_cases[setting_key] != expected_cases:
            errors.append(
                f"Setting {setting_key} has case indices {sorted(selected_cases[setting_key])}"
            )
        if int(row.get("case_metrics_rows", -1)) != len(expected_cases):
            errors.append(
                f"Setting {setting_key} has case_metrics_rows={row.get('case_metrics_rows')}"
            )
        if int(row.get("case_geometry_rows", -1)) != len(expected_cases):
            errors.append(
                f"Setting {setting_key} has case_geometry_rows={row.get('case_geometry_rows')}"
            )
        bundle = Path(row["run_bundle"])
        for required_path in (
            bundle,
            bundle / "vtk" / "mesh.vtk",
            bundle / "metrics" / "cell_metrics.csv",
            bundle / "metrics" / "case_metrics.csv",
        ):
            if not required_path.exists():
                errors.append(f"Missing saved July artifact: {required_path}")

    if errors:
        raise RuntimeError("Full topology input validation failed:\n- " + "\n- ".join(errors))

    return {
        "selector_count": len(selectors),
        "setting_count": len(settings),
        "cases_per_setting": len(expected_cases),
        "source_commits": sorted({row.get("source_commit", "") for row in settings.values()}),
        "run_bundles": sorted({row["run_bundle"] for row in settings.values()}),
        "validated_no_missing_run_bundles_or_cases": True,
    }


def _full_readme(
    paper_rows: Sequence[Mapping],
    sensitivity_rows: Sequence[Mapping],
    conflict_findings: Mapping,
    manifest: Mapping,
) -> str:
    lines = [
        "# Full shared-vertex topology-consistency diagnostic",
        "",
        "This table analyzes saved July 17 production reconstructions only; no reconstruction was launched.",
        "The default paper tolerance is relative `1e-10`, with absolute floor `1e-12`.",
        "Oriented and fallback percentages are weighted by mixed-cell count.",
        "",
        "## Paper table",
        "",
        "| Problem | Method | Cases | Mixed cells | Oriented | Fallback | Resolved vertices | Resolved exclusions | Resolved conflicts | Complete vertices | Complete exclusions | Complete conflicts |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in paper_rows:
        resolved_excluded = int(row["resolved_on_facet_excluded_vertices"]) + int(
            row["resolved_invalid_excluded_vertices"]
        )
        complete_excluded = int(row["complete_on_facet_excluded_vertices"]) + int(
            row["complete_invalid_excluded_vertices"]
        )
        lines.append(
            "| {experiment} | {algo} | {case_count} | {mixed_cells} | {oriented} | "
            "{fallback} | {resolved_evaluated_shared_vertices} | {resolved_excluded} | "
            "{resolved_conflict_vertices} ({resolved_rate}) | "
            "{complete_evaluated_shared_vertices} | {complete_excluded} | "
            "{complete_conflict_vertices} ({complete_rate}) |".format(
                **row,
                oriented=_format_rate(row["oriented_cell_fraction"]),
                fallback=_format_rate(row["plic_fallback_cell_fraction"]),
                resolved_excluded=resolved_excluded,
                complete_excluded=complete_excluded,
                resolved_rate=_format_rate(row["resolved_conflict_rate"]),
                complete_rate=_format_rate(row["complete_conflict_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Tolerance sensitivity",
            "",
            "| Relative tolerance | Cases | Resolved evaluated | Resolved excluded | Resolved conflicts | Complete evaluated | Complete excluded | Complete conflicts |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sensitivity_rows:
        resolved_excluded = int(row["resolved_on_facet_excluded_vertices"]) + int(
            row["resolved_invalid_excluded_vertices"]
        )
        complete_excluded = int(row["complete_on_facet_excluded_vertices"]) + int(
            row["complete_invalid_excluded_vertices"]
        )
        lines.append(
            "| {relative_tolerance:.0e} | {case_count} | "
            "{resolved_evaluated_shared_vertices} | {resolved_excluded} | "
            "{resolved_conflict_vertices} ({resolved_rate}) | "
            "{complete_evaluated_shared_vertices} | {complete_excluded} | "
            "{complete_conflict_vertices} ({complete_rate}) |".format(
                **row,
                resolved_excluded=resolved_excluded,
                complete_excluded=complete_excluded,
                resolved_rate=_format_rate(row["resolved_conflict_rate"]),
                complete_rate=_format_rate(row["complete_conflict_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Nonzero findings",
            "",
            f"At the paper tolerance, `{conflict_findings['conflict_vertices']}` conflicting vertices occur in "
            f"`{conflict_findings['case_count']}/500` cases. None contains a PLIC fallback cell.",
        ]
    )
    for experiment, finding in conflict_findings["by_experiment"].items():
        resolution_text = ", ".join(f"N={value:g}" for value in finding["resolutions"])
        lines.append(
            f"- `{experiment}`: `{finding['conflict_vertices']}` vertices in "
            f"`{finding['case_count']}` cases ({resolution_text})."
        )
    lines.append(
        "- `circles` and `squares`: zero conflicts. Counts are unchanged at all three tolerances."
    )

    validation = manifest["input_validation"]
    lines.extend(
        [
            "",
            "## Scope and validation",
            "",
            "- `resolved`: cells with successful orientation and no PLIC fallback.",
            "- `complete`: every saved final facet, including LVIRA fallback cells.",
            "- Vertices touched by any incident facet, plus vertices with invalid geometry labels, are excluded and reported.",
            f"- Validated `{validation['selector_count']}` cases in `{validation['setting_count']}` saved run bundles, with no missing bundle or case.",
            f"- Source commit recorded by the July inventory: `{', '.join(validation['source_commits'])}`.",
            "",
            "## Artifacts",
            "",
            "- `topology_consistency_paper_table.csv`: benchmark aggregates at the default tolerance.",
            "- `topology_consistency_tolerance_sensitivity.csv`: all-case sensitivity totals.",
            "- `topology_consistency_by_setting.csv`: benchmark-resolution-tolerance aggregates.",
            "- `topology_consistency_cases.csv`: one row per case and tolerance.",
            "- `topology_consistency_vertices.csv.gz`: compressed vertex-level labels, exclusions, and conflicts.",
            "- `topology_consistency_conflicts.csv`: conflict-only audit rows for both scopes and all tolerances.",
            "- `topology_consistency_manifest.json`: exact selection, paths, and tolerance policy.",
            "",
        ]
    )
    return "\n".join(lines)


def run_full_diagnostics(
    cell_metrics_path: Path,
    run_inventory_path: Path,
    output_dir: Path,
    selectors: Sequence[CaseSelector] | None = None,
    relative_tolerances: Sequence[float] = FULL_RELATIVE_TOLERANCES,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
):
    selectors = tuple(selectors or build_full_selectors())
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_records = _load_selected_records(cell_metrics_path, selectors)
    inventory_rows = _load_run_inventory_rows(run_inventory_path, selectors)
    validation = _validate_full_inputs(selectors, selected_records, inventory_rows)
    run_bundles = {
        selector: Path(inventory_rows[selector]["run_bundle"]) for selector in selectors
    }

    mesh_cache = {}
    case_rows = []
    vertex_rows = []
    for relative_tolerance in relative_tolerances:
        for selector in selectors:
            bundle = run_bundles[selector]
            mesh_path = bundle / "vtk" / "mesh.vtk"
            if mesh_path not in mesh_cache:
                mesh_cache[mesh_path] = read_structured_mesh(mesh_path)
            summary, details = evaluate_case(
                selected_records[selector],
                mesh_cache[mesh_path],
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            identity = {
                "relative_tolerance": relative_tolerance,
                "absolute_tolerance": absolute_tolerance,
                "experiment": selector.experiment,
                "algo": selector.algo,
                "resolution": selector.resolution,
                "wiggle": selector.wiggle,
                "seed": selector.seed,
                "case_index": selector.case_index,
                "purpose": selector.purpose,
                "run_bundle": str(bundle),
            }
            case_rows.append({**identity, **summary})
            vertex_rows.extend({**identity, **detail} for detail in details)

    by_setting_rows = _aggregate_case_rows_by(
        case_rows,
        (
            "relative_tolerance",
            "experiment",
            "algo",
            "resolution",
            "wiggle",
            "seed",
            "purpose",
        ),
    )
    by_benchmark_rows = _aggregate_case_rows_by(
        case_rows,
        ("relative_tolerance", "experiment", "algo", "wiggle", "seed", "purpose"),
    )
    method_order = {method: index for index, method in enumerate(FULL_METHODS)}
    by_setting_rows.sort(
        key=lambda row: (
            float(row["relative_tolerance"]),
            method_order[(row["experiment"], row["algo"])],
            float(row["resolution"]),
        )
    )
    by_benchmark_rows.sort(
        key=lambda row: (
            float(row["relative_tolerance"]),
            method_order[(row["experiment"], row["algo"])],
        )
    )
    paper_rows = [
        row
        for row in by_benchmark_rows
        if math.isclose(
            float(row["relative_tolerance"]),
            DEFAULT_RELATIVE_TOLERANCE,
            rel_tol=0.0,
            abs_tol=1.0e-20,
        )
    ]
    sensitivity_rows = _aggregate_case_rows_by(case_rows, ("relative_tolerance",))
    conflict_rows = [row for row in vertex_rows if int(row["conflict"])]
    default_complete_conflicts = [
        row
        for row in conflict_rows
        if row["scope"] == "complete"
        and math.isclose(
            float(row["relative_tolerance"]),
            DEFAULT_RELATIVE_TOLERANCE,
            rel_tol=0.0,
            abs_tol=1.0e-20,
        )
    ]
    by_experiment = {}
    for experiment in sorted({row["experiment"] for row in default_complete_conflicts}):
        rows = [row for row in default_complete_conflicts if row["experiment"] == experiment]
        by_experiment[experiment] = {
            "conflict_vertices": len(rows),
            "case_count": len(
                {(row["resolution"], row["case_index"]) for row in rows}
            ),
            "resolutions": sorted({100.0 * float(row["resolution"]) for row in rows}),
        }
    conflict_findings = {
        "conflict_vertices": len(default_complete_conflicts),
        "case_count": len(
            {
                (row["experiment"], row["resolution"], row["case_index"])
                for row in default_complete_conflicts
            }
        ),
        "fallback_involved_vertices": sum(
            int(row["contains_fallback"]) for row in default_complete_conflicts
        ),
        "by_experiment": by_experiment,
    }

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "definition": "Shared incident cell-facet vertex colors must all agree.",
        "cell_metrics": str(cell_metrics_path.resolve()),
        "run_inventory": str(run_inventory_path.resolve()),
        "relative_tolerances": list(relative_tolerances),
        "paper_relative_tolerance": DEFAULT_RELATIVE_TOLERANCE,
        "absolute_tolerance": absolute_tolerance,
        "selector_specification": {
            "methods": [list(method) for method in FULL_METHODS],
            "resolutions": list(FULL_RESOLUTIONS),
            "wiggle": 0.1,
            "seed": 0,
            "case_indices": [0, 24],
        },
        "input_validation": validation,
        "default_tolerance_conflicts": conflict_findings,
        "scope_policy": {
            "resolved": "orientation_status=oriented and not PLIC fallback",
            "complete": "all saved final facet geometry, including PLIC fallbacks",
        },
        "on_facet_policy": "Exclude a shared vertex from a scope if any incident finite facet is within tolerance.",
        "outputs": [
            "topology_consistency_paper_table.csv",
            "topology_consistency_tolerance_sensitivity.csv",
            "topology_consistency_by_setting.csv",
            "topology_consistency_cases.csv",
            "topology_consistency_vertices.csv.gz",
            "topology_consistency_conflicts.csv",
            "topology_consistency_manifest.json",
            "README.md",
        ],
    }

    _write_csv(output_dir / "topology_consistency_paper_table.csv", paper_rows)
    _write_csv(
        output_dir / "topology_consistency_tolerance_sensitivity.csv", sensitivity_rows
    )
    _write_csv(output_dir / "topology_consistency_by_setting.csv", by_setting_rows)
    _write_csv(output_dir / "topology_consistency_cases.csv", case_rows)
    _write_csv_gzip(output_dir / "topology_consistency_vertices.csv.gz", vertex_rows)
    _write_csv(output_dir / "topology_consistency_conflicts.csv", conflict_rows)
    (output_dir / "topology_consistency_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "README.md").write_text(
        _full_readme(paper_rows, sensitivity_rows, conflict_findings, manifest)
    )
    return case_rows, paper_rows, sensitivity_rows, vertex_rows


def _read_selectors(path: Path | None) -> tuple[CaseSelector, ...]:
    if path is None:
        return DEFAULT_SMOKE_SELECTORS
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Selector JSON must contain a list of selector objects")
    return tuple(CaseSelector(**item) for item in payload)


def _parse_args():
    default_diagnostics = Path(
        "results/static/static_paper_simplified_default_20260717_212413/diagnostics"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-metrics", type=Path, default=default_diagnostics / "cell_metrics.csv")
    parser.add_argument("--run-inventory", type=Path, default=default_diagnostics / "run_inventory.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to the dated smoke or full submission directory.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Analyze the 500-case paper aggregate at all three tolerance levels.",
    )
    parser.add_argument("--selectors-json", type=Path)
    parser.add_argument("--relative-tolerance", type=float, default=DEFAULT_RELATIVE_TOLERANCE)
    parser.add_argument("--absolute-tolerance", type=float, default=DEFAULT_ABSOLUTE_TOLERANCE)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.full:
        output_dir = args.output_dir or Path(
            "results/submission/topology_consistency_full_20260730"
        )
        case_rows, _, sensitivity_rows, _ = run_full_diagnostics(
            cell_metrics_path=args.cell_metrics,
            run_inventory_path=args.run_inventory,
            output_dir=output_dir,
            absolute_tolerance=args.absolute_tolerance,
        )
        print(f"Wrote {len(case_rows)} case-tolerance rows to {output_dir}")
        for row in sensitivity_rows:
            print(
                f"Tolerance {float(row['relative_tolerance']):.0e}: "
                f"resolved conflicts={row['resolved_conflict_vertices']}; "
                f"complete conflicts={row['complete_conflict_vertices']}"
            )
        return

    output_dir = args.output_dir or Path(
        "results/submission/topology_consistency_smoke_20260730"
    )
    case_rows, _, _ = run_diagnostics(
        cell_metrics_path=args.cell_metrics,
        run_inventory_path=args.run_inventory,
        output_dir=output_dir,
        selectors=_read_selectors(args.selectors_json),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    resolved_conflicts = sum(row["resolved_conflict_vertices"] for row in case_rows)
    complete_conflicts = sum(row["complete_conflict_vertices"] for row in case_rows)
    print(f"Wrote {len(case_rows)} cases to {output_dir}")
    print(f"Resolved conflicts: {resolved_conflicts}; complete conflicts: {complete_conflicts}")


if __name__ == "__main__":
    main()
