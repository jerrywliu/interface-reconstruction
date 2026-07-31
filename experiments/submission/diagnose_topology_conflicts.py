"""Diagnose shared-vertex conflicts from a saved full topology audit.

The full topology audit originally evaluated saved facet metadata at vertices
read from ``mesh.vtk``. This follow-up compares those rounded coordinates with
the exact seeded mesh regenerated from each run manifest, verifies the saved
metadata against the per-run files, and records direct phase-side and area
checks for every incident facet.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.submission.conservation_analyzer import load_run_grid
from experiments.submission.topology_consistency_diagnostics import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_RELATIVE_TOLERANCE,
    StructuredMesh,
    _corner_branches,
    _cross,
    _primitive_distance,
    _primitive_tangent,
    classify_vertex,
    read_structured_mesh,
)
from main.geoms.geoms import getArea
from util.metrics.area_metrics import facet_area_in_polygon


REPO_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = {
    "a": "a_real_reconstructed_phase_label_inconsistency",
    "b": "b_diagnostic_classification_bug",
    "c": "c_rounded_mesh_vtk_artifact",
    "d": "d_stale_saved_facet_metadata_mismatch",
    "e": "e_numerical_near_interface_ambiguity",
}
METADATA_FIELDS = (
    "merge_id",
    "merge_component_size",
    "orientation_status",
    "final_facet_class",
    "final_facet_name",
    "construction_path",
    "fallback_policy",
    "event_count",
    "facet_geometry_json",
)


def _float_key(value: Any) -> float:
    return round(float(value), 12)


def _record_key(identity: Mapping[str, Any], cell_id: str) -> tuple:
    return (
        identity["experiment"],
        identity["algo"],
        _float_key(identity["resolution"]),
        _float_key(identity["wiggle"]),
        int(identity["seed"]),
        int(identity["case_index"]),
        cell_id,
    )


def _mesh_from_points(points: Sequence[Sequence[Sequence[float]]]) -> StructuredMesh:
    normalized = tuple(
        tuple((float(point[0]), float(point[1])) for point in column)
        for column in points
    )
    flat = [point for column in normalized for point in column]
    diagonal = math.hypot(
        max(point[0] for point in flat) - min(point[0] for point in flat),
        max(point[1] for point in flat) - min(point[1] for point in flat),
    )
    return StructuredMesh(
        points=normalized,
        nx=len(normalized),
        ny=len(normalized[0]),
        domain_diagonal=diagonal,
    )


def load_exact_mesh(run_bundle: Path) -> tuple[StructuredMesh, Mapping[str, Any]]:
    grid = load_run_grid(run_bundle, repo_root=REPO_ROOT)
    return _mesh_from_points(grid.points), {
        "mesh_source": grid.source,
        "vtk_max_point_delta": grid.vtk_max_point_delta,
        "mesh_note": grid.note,
    }


def _primitive_margin(geometry: Mapping[str, Any], point: Sequence[float]) -> float:
    """Signed geometric distance proxy; positive denotes the full phase."""

    geometry_class = str(geometry.get("class", "")).lower()
    if geometry_class == "linear":
        left = geometry["p_left"]
        right = geometry["p_right"]
        direction = (right[0] - left[0], right[1] - left[1])
        length = math.hypot(*direction)
        if length == 0.0:
            return float("nan")
        offset = (point[0] - left[0], point[1] - left[1])
        return _cross(direction, offset) / length
    if geometry_class == "circular":
        radius = float(geometry["radius"])
        radial_distance = math.dist(point, geometry["center"])
        return math.copysign(1.0, radius) * (abs(radius) - radial_distance)
    raise ValueError(f"Unsupported primitive geometry class: {geometry_class!r}")


def direct_phase_test(
    geometry: Mapping[str, Any], point: Sequence[float], tolerance: float
) -> tuple[str, float]:
    """Evaluate phase membership directly from oriented line/circle sides."""

    geometry_class = str(geometry.get("class", "")).lower()
    if geometry_class in {"linear", "circular"}:
        if _primitive_distance(point, geometry) <= tolerance:
            return "on_facet", 0.0
        margin = _primitive_margin(geometry, point)
        return ("full" if margin > 0.0 else "empty"), margin

    if geometry_class not in {"linear_corner", "curved_corner", "corner"}:
        raise ValueError(f"Unsupported facet geometry class: {geometry_class!r}")
    left, right = _corner_branches(geometry)
    if (
        min(_primitive_distance(point, left), _primitive_distance(point, right))
        <= tolerance
    ):
        return "on_facet", 0.0

    left_margin = _primitive_margin(left, point)
    right_margin = _primitive_margin(right, point)
    left_tangent = _primitive_tangent(left, at_right_endpoint=True)
    right_tangent = _primitive_tangent(right, at_right_endpoint=False)
    turn = _cross(left_tangent, right_tangent)
    turn_tolerance = 1.0e-12 * max(
        1.0, math.hypot(*left_tangent) * math.hypot(*right_tangent)
    )
    if turn > turn_tolerance:
        margin = min(left_margin, right_margin)
    elif turn < -turn_tolerance:
        margin = max(left_margin, right_margin)
    elif (left_margin > 0.0) == (right_margin > 0.0):
        margin = 0.5 * (left_margin + right_margin)
    else:
        return "on_facet", 0.0
    return ("full" if margin > 0.0 else "empty"), margin


def assign_taxonomy(
    *,
    vtk_conflict: bool,
    exact_conflict: bool,
    metadata_mismatch: bool,
    classification_mismatch: bool,
    min_abs_exact_margin: float,
    ambiguity_tolerance: float,
) -> str:
    if metadata_mismatch:
        return TAXONOMY["d"]
    if classification_mismatch:
        return TAXONOMY["b"]
    if vtk_conflict and not exact_conflict:
        return TAXONOMY["c"]
    if exact_conflict and min_abs_exact_margin <= ambiguity_tolerance:
        return TAXONOMY["e"]
    if exact_conflict:
        return TAXONOMY["a"]
    raise ValueError("Flagged VTK conflict has no supported diagnosis")


def _read_default_conflicts(source_dir: Path) -> list[dict[str, str]]:
    path = source_dir / "topology_consistency_conflicts.csv"
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["scope"] == "complete"
            and math.isclose(
                float(row["relative_tolerance"]),
                DEFAULT_RELATIVE_TOLERANCE,
                rel_tol=0.0,
                abs_tol=1.0e-20,
            )
        ]
    if len(rows) != 22:
        raise RuntimeError(f"Expected 22 default-scope conflicts, found {len(rows)}")
    return rows


def _read_case_records(run_bundle: Path, case_index: int) -> dict[str, dict[str, str]]:
    path = run_bundle / "metrics/cell_metrics.csv"
    with path.open(newline="") as handle:
        return {
            row["cell_id"]: row
            for row in csv.DictReader(handle)
            if int(row["case_index"]) == case_index
        }


def _read_case_events(
    run_bundle: Path, case_index: int
) -> dict[str, list[dict[str, str]]]:
    path = run_bundle / "metrics/merge_events.csv"
    events: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["case_index"]) == case_index:
                events[row["merge_id"]].append(row)
    return events


def _load_consolidated_records(
    path: Path, target_keys: set[tuple]
) -> dict[tuple, dict[str, str]]:
    selected: dict[tuple, dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = _record_key(row, row["cell_id"])
            if key in target_keys:
                selected[key] = row
    return selected


def _metadata_matches(
    local: Mapping[str, str], consolidated: Mapping[str, str] | None
) -> bool:
    if consolidated is None:
        return False
    for field in METADATA_FIELDS:
        if field == "facet_geometry_json":
            if json.loads(local[field]) != json.loads(consolidated[field]):
                return False
        elif local.get(field, "") != consolidated.get(field, ""):
            return False
    return True


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    return abs(getArea([list(point) for point in points]))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _facet_points(geometry: Mapping[str, Any], count: int = 120):
    geometry_class = str(geometry.get("class", "")).lower()
    if geometry_class == "linear":
        return [geometry["p_left"], geometry["p_right"]]
    if geometry_class == "circular":
        center = geometry["center"]
        radius = float(geometry["radius"])
        start = math.atan2(
            geometry["p_left"][1] - center[1], geometry["p_left"][0] - center[0]
        )
        end = math.atan2(
            geometry["p_right"][1] - center[1], geometry["p_right"][0] - center[0]
        )
        if radius >= 0.0:
            travel = (end - start) % (2.0 * math.pi)
            angles = [start + travel * index / (count - 1) for index in range(count)]
        else:
            travel = (start - end) % (2.0 * math.pi)
            angles = [start - travel * index / (count - 1) for index in range(count)]
        return [
            [
                center[0] + abs(radius) * math.cos(angle),
                center[1] + abs(radius) * math.sin(angle),
            ]
            for angle in angles
        ]
    points = []
    for branch in _corner_branches(geometry):
        branch_points = _facet_points(branch, max(2, count // 2))
        points.extend(branch_points if not points else branch_points[1:])
    return points


def _plot_examples(
    output_dir: Path,
    examples: Sequence[Mapping[str, Any]],
    detail_lookup: Mapping[tuple, Sequence[Mapping[str, Any]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    figure, axes = plt.subplots(len(examples), 2, figsize=(8.2, 6.4), squeeze=False)
    phase_colors = {"full": "#167D73", "empty": "#C44E52", "on_facet": "#777777"}
    for row_index, example in enumerate(examples):
        for column_index, mesh_kind in enumerate(("vtk", "exact")):
            axis = axes[row_index][column_index]
            mesh = example[f"{mesh_kind}_mesh"]
            vertex_key = example["vertex_key"]
            point = mesh.points[vertex_key[0]][vertex_key[1]]
            details = detail_lookup[example["lookup_key"]]
            for detail in details:
                cell_x, cell_y = (int(value) for value in detail["cell_id"].split(","))
                polygon = [item[1] for item in mesh.cell_vertices(cell_x, cell_y)]
                closed = polygon + [polygon[0]]
                axis.plot(
                    [item[0] for item in closed],
                    [item[1] for item in closed],
                    color="#B7B7B7",
                    linewidth=0.8,
                    zorder=1,
                )
                geometry = detail["geometry"]
                facet = _facet_points(geometry)
                label = detail[f"{mesh_kind}_label"]
                axis.plot(
                    [item[0] for item in facet],
                    [item[1] for item in facet],
                    color=phase_colors[label],
                    linewidth=1.8,
                    zorder=2,
                )
                centroid = (
                    sum(item[0] for item in polygon) / len(polygon),
                    sum(item[1] for item in polygon) / len(polygon),
                )
                axis.text(
                    centroid[0],
                    centroid[1],
                    f"{detail['cell_id']}\n{label}",
                    color=phase_colors[label],
                    fontsize=7,
                    ha="center",
                    va="center",
                )
            axis.scatter(
                [point[0]], [point[1]], marker="x", color="black", s=42, zorder=4
            )
            axis.set_aspect("equal", adjustable="box")
            axis.margins(0.08)
            axis.set_title(
                f"{example['experiment'].title()} N={100 * float(example['resolution']):g}, "
                f"case {example['case_index']}\n"
                f"{'rounded VTK' if mesh_kind == 'vtk' else 'exact regenerated'}: conflict",
                fontsize=9,
            )
            axis.tick_params(labelsize=7)
    figure.suptitle(
        "Shared-vertex conflicts persist on exact seeded meshes", fontsize=11
    )
    figure.tight_layout()
    figure.savefig(output_dir / "topology_conflict_examples.pdf", bbox_inches="tight")
    figure.savefig(
        output_dir / "topology_conflict_examples.png", dpi=220, bbox_inches="tight"
    )
    plt.close(figure)


def _read_source_audit_totals(
    source_dir: Path, source_manifest: Mapping[str, Any]
) -> dict[str, int]:
    tolerance = float(source_manifest["paper_relative_tolerance"])
    path = source_dir / "topology_consistency_tolerance_sensitivity.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        matches = [
            row
            for row in csv.DictReader(stream)
            if math.isclose(
                float(row["relative_tolerance"]),
                tolerance,
                rel_tol=0.0,
                abs_tol=1.0e-20,
            )
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one topology sensitivity row at tolerance {tolerance}, "
            f"found {len(matches)}"
        )
    row = matches[0]
    return {
        "audited_case_count": int(row["case_count"]),
        "complete_evaluated_shared_vertices": int(
            row["complete_evaluated_shared_vertices"]
        ),
    }


def _build_readme(
    taxonomy_rows: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
    source_totals: Mapping[str, int],
) -> str:
    total_conflicts = len(taxonomy_rows)
    exact_conflicts = sum(int(row["exact_conflict"]) for row in taxonomy_rows)
    exact_vtk_matches = sum(
        row["vtk_labels"] == row["exact_labels"] for row in taxonomy_rows
    )
    metadata_matches = sum(not int(row["metadata_mismatch"]) for row in taxonomy_rows)
    classification_matches = sum(
        not int(row["classification_mismatch"]) for row in taxonomy_rows
    )
    fallback_vertices = sum(int(row["contains_fallback"]) for row in taxonomy_rows)
    taxonomy_counts = Counter(row["taxonomy"] for row in taxonomy_rows)
    experiment_counts = Counter(row["experiment"] for row in taxonomy_rows)

    readme = [
        "# Exact-mesh topology conflict diagnosis",
        "",
        f"All {total_conflicts} conflicts from the source full audit were replayed on exact seeded meshes,",
        "checked against the per-run and consolidated facet metadata, and independently",
        "evaluated using oriented line/circle phase-side tests.",
        "",
        "## Result",
        "",
        f"- Exact replay preserves `{exact_conflicts}/{total_conflicts}` conflicts.",
        f"- Exact and VTK phase labels agree for `{exact_vtk_matches}/{total_conflicts}` vertices.",
        f"- Per-run and consolidated facet metadata agree for `{metadata_matches}/{total_conflicts}` vertices.",
        f"- Independent phase-side labels agree with the diagnostic for `{classification_matches}/{total_conflicts}` vertices.",
    ]
    if taxonomy_rows:
        minimum_margin = min(
            float(row["min_abs_exact_phase_margin"]) for row in taxonomy_rows
        )
        maximum_ambiguity = max(
            float(row["ambiguity_tolerance"]) for row in taxonomy_rows
        )
        readme.append(
            f"- The smallest exact signed phase margin is `{minimum_margin:.6g}`; "
            f"the largest ambiguity threshold is `{maximum_ambiguity:.6g}`."
        )
    readme.extend(
        [
            "",
            "| Taxonomy | Vertices |",
            "| --- | ---: |",
        ]
    )
    for name in TAXONOMY.values():
        readme.append(f"| `{name}` | {taxonomy_counts.get(name, 0)} |")

    incidence = total_conflicts / max(
        1, int(source_totals["complete_evaluated_shared_vertices"])
    )
    affected_cases = len(case_rows)
    audited_cases = int(source_totals["audited_case_count"])
    experiment_text = (
        ", ".join(
            f"`{experiment}` ({count})"
            for experiment, count in sorted(experiment_counts.items())
        )
        or "none"
    )
    readme.extend(
        [
            "",
            "## Interpretation",
            "",
            "The conflicts are genuine saved-reconstruction phase-label disagreements when exact replay, metadata, and independent phase-side checks agree; any exceptions are enumerated in the taxonomy table rather than assumed away.",
            "",
            f"The measured incidence is `{total_conflicts}/{source_totals['complete_evaluated_shared_vertices']}` evaluated vertices (`{100.0 * incidence:.6g}%`) in `{affected_cases}/{audited_cases}` audited cases. "
            f"`{fallback_vertices}` flagged vertices use a PLIC fallback. Conflict counts by benchmark are {experiment_text}.",
            "",
            "The paper should therefore qualify any exact topological-consistency statement. A narrowly scoped algorithmic repair should be attempted only if exact consistency is essential to the claim; otherwise report the measured incidence and retain this as a limitation.",
            "",
            "## Files",
            "",
            "- `topology_conflict_taxonomy.csv`: one classified row per flagged vertex.",
            f"- `topology_conflict_case_counts.csv`: exact counts for all {affected_cases} affected cases.",
            "- `topology_conflict_incident_facets.csv`: facet metadata, phase margins, area checks, and merge-event provenance.",
            "- `topology_conflict_examples.pdf`: vector comparison for representative conflicts when available.",
            "- `topology_conflict_examples.png`: review rendering of the same comparison when available.",
            "",
        ]
    )
    return "\n".join(readme)


def diagnose(
    source_dir: Path, output_dir: Path
) -> tuple[list[dict], list[dict], list[dict]]:
    conflicts = _read_default_conflicts(source_dir)
    source_manifest = json.loads(
        (source_dir / "topology_consistency_manifest.json").read_text(encoding="utf-8")
    )
    cell_metrics_path = Path(source_manifest["cell_metrics"]).resolve()
    if cell_metrics_path.parent.name == "diagnostics":
        release_root = cell_metrics_path.parent.parent
        try:
            output_dir.resolve().relative_to(release_root)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"Validation output must be outside immutable release {release_root}: "
                f"{output_dir.resolve()}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    case_cache: dict[tuple[Path, int], dict[str, dict[str, str]]] = {}
    event_cache: dict[tuple[Path, int], dict[str, list[dict[str, str]]]] = {}
    mesh_cache: dict[Path, tuple[StructuredMesh, StructuredMesh, Mapping[str, Any]]] = (
        {}
    )
    target_keys = {
        _record_key(conflict, cell_id)
        for conflict in conflicts
        for cell_id in conflict["incident_cells"].split(";")
    }
    consolidated = _load_consolidated_records(
        Path(source_manifest["cell_metrics"]), target_keys
    )

    taxonomy_rows: list[dict[str, Any]] = []
    incident_rows: list[dict[str, Any]] = []
    visual_examples: list[dict[str, Any]] = []
    detail_lookup: dict[tuple, list[dict[str, Any]]] = {}

    for conflict in conflicts:
        bundle = Path(conflict["run_bundle"])
        case_index = int(conflict["case_index"])
        cache_key = (bundle, case_index)
        if cache_key not in case_cache:
            case_cache[cache_key] = _read_case_records(bundle, case_index)
            event_cache[cache_key] = _read_case_events(bundle, case_index)
        if bundle not in mesh_cache:
            vtk_mesh = read_structured_mesh(bundle / "vtk/mesh.vtk")
            exact_mesh, mesh_audit = load_exact_mesh(bundle)
            mesh_cache[bundle] = vtk_mesh, exact_mesh, mesh_audit
        vtk_mesh, exact_mesh, mesh_audit = mesh_cache[bundle]

        vertex_key = (int(conflict["vertex_i"]), int(conflict["vertex_j"]))
        vtk_point = vtk_mesh.points[vertex_key[0]][vertex_key[1]]
        exact_point = exact_mesh.points[vertex_key[0]][vertex_key[1]]
        vtk_tolerance = max(
            DEFAULT_ABSOLUTE_TOLERANCE,
            DEFAULT_RELATIVE_TOLERANCE * vtk_mesh.domain_diagonal,
        )
        exact_tolerance = max(
            DEFAULT_ABSOLUTE_TOLERANCE,
            DEFAULT_RELATIVE_TOLERANCE * exact_mesh.domain_diagonal,
        )
        ambiguity_tolerance = max(
            100.0 * exact_tolerance,
            1.0e-8 * exact_mesh.domain_diagonal,
        )

        vtk_labels = []
        exact_labels = []
        direct_labels = []
        exact_margins = []
        metadata_mismatch = False
        details = []
        for cell_id in conflict["incident_cells"].split(";"):
            record = case_cache[cache_key][cell_id]
            geometry = json.loads(record["facet_geometry_json"])
            vtk_label = classify_vertex(geometry, vtk_point, vtk_tolerance)
            exact_label = classify_vertex(geometry, exact_point, exact_tolerance)
            direct_label, exact_margin = direct_phase_test(
                geometry, exact_point, exact_tolerance
            )
            vtk_labels.append(vtk_label)
            exact_labels.append(exact_label)
            direct_labels.append(direct_label)
            exact_margins.append(exact_margin)

            consolidated_record = consolidated.get(_record_key(conflict, cell_id))
            metadata_match = _metadata_matches(record, consolidated_record)
            metadata_mismatch = metadata_mismatch or not metadata_match
            cell_x, cell_y = (int(value) for value in cell_id.split(","))
            polygon = [item[1] for item in exact_mesh.cell_vertices(cell_x, cell_y)]
            area = _polygon_area(polygon)
            reconstructed_fraction = facet_area_in_polygon(polygon, geometry) / area
            merge_events = event_cache[cache_key].get(record["merge_id"], [])
            event_summary = [
                {
                    "event_order": event["event_order"],
                    "stage": event["stage"],
                    "event_kind": event["event_kind"],
                    "fallback_policy": event["fallback_policy"],
                    "previous_facet_class": event["previous_facet_class"],
                    "facet_class": event["facet_class"],
                    "facet_name": event["facet_name"],
                }
                for event in merge_events
            ]
            incident = {
                "experiment": conflict["experiment"],
                "algo": conflict["algo"],
                "resolution": conflict["resolution"],
                "wiggle": conflict["wiggle"],
                "seed": conflict["seed"],
                "case_index": conflict["case_index"],
                "vertex_i": conflict["vertex_i"],
                "vertex_j": conflict["vertex_j"],
                "cell_id": cell_id,
                "merge_id": record["merge_id"],
                "merge_component_size": record["merge_component_size"],
                "orientation_status": record["orientation_status"],
                "construction_path": record["construction_path"],
                "fallback_policy": record["fallback_policy"],
                "facet_class": geometry["class"],
                "vtk_label": vtk_label,
                "exact_label": exact_label,
                "direct_phase_label": direct_label,
                "exact_signed_phase_margin": exact_margin,
                "exact_distance_to_finite_facet": (
                    _primitive_distance(exact_point, geometry)
                    if geometry["class"] in {"linear", "circular"}
                    else min(
                        _primitive_distance(exact_point, branch)
                        for branch in _corner_branches(geometry)
                    )
                ),
                "cell_fraction": record["cell_fraction"],
                "direct_reconstructed_fraction": reconstructed_fraction,
                "constituent_fraction_residual": reconstructed_fraction
                - float(record["cell_fraction"]),
                "metadata_matches_consolidated": int(metadata_match),
                "merge_event_count": len(merge_events),
                "merge_event_provenance_json": json.dumps(
                    event_summary, separators=(",", ":")
                ),
                "facet_geometry_json": record["facet_geometry_json"],
                "geometry": geometry,
            }
            incident_rows.append(
                {key: value for key, value in incident.items() if key != "geometry"}
            )
            details.append(incident)

        vtk_conflict = "on_facet" not in vtk_labels and len(set(vtk_labels)) > 1
        exact_conflict = "on_facet" not in exact_labels and len(set(exact_labels)) > 1
        classification_mismatch = exact_labels != direct_labels
        finite_margins = [abs(value) for value in exact_margins if math.isfinite(value)]
        min_abs_margin = min(finite_margins) if finite_margins else float("nan")
        taxonomy = assign_taxonomy(
            vtk_conflict=vtk_conflict,
            exact_conflict=exact_conflict,
            metadata_mismatch=metadata_mismatch,
            classification_mismatch=classification_mismatch,
            min_abs_exact_margin=min_abs_margin,
            ambiguity_tolerance=ambiguity_tolerance,
        )
        taxonomy_row = {
            "experiment": conflict["experiment"],
            "algo": conflict["algo"],
            "resolution": conflict["resolution"],
            "wiggle": conflict["wiggle"],
            "seed": conflict["seed"],
            "case_index": conflict["case_index"],
            "vertex_i": conflict["vertex_i"],
            "vertex_j": conflict["vertex_j"],
            "incident_cells": conflict["incident_cells"],
            "incident_merge_ids": conflict["incident_merge_ids"],
            "facet_classes": ";".join(detail["facet_class"] for detail in details),
            "construction_paths": ";".join(
                detail["construction_path"] for detail in details
            ),
            "contains_fallback": conflict["contains_fallback"],
            "vtk_labels": ";".join(vtk_labels),
            "exact_labels": ";".join(exact_labels),
            "direct_phase_labels": ";".join(direct_labels),
            "vtk_conflict": int(vtk_conflict),
            "exact_conflict": int(exact_conflict),
            "classification_mismatch": int(classification_mismatch),
            "metadata_mismatch": int(metadata_mismatch),
            "mesh_source": mesh_audit["mesh_source"],
            "vtk_max_point_delta": mesh_audit["vtk_max_point_delta"],
            "conflict_vertex_coordinate_shift": math.dist(vtk_point, exact_point),
            "exact_geometric_tolerance": exact_tolerance,
            "ambiguity_tolerance": ambiguity_tolerance,
            "min_abs_exact_phase_margin": min_abs_margin,
            "taxonomy": taxonomy,
        }
        taxonomy_rows.append(taxonomy_row)
        lookup_key = (
            conflict["experiment"],
            conflict["resolution"],
            conflict["case_index"],
            conflict["vertex_i"],
            conflict["vertex_j"],
        )
        detail_lookup[lookup_key] = details
        if len(visual_examples) < 2 and not any(
            item["experiment"] == conflict["experiment"] for item in visual_examples
        ):
            visual_examples.append(
                {
                    "lookup_key": lookup_key,
                    "experiment": conflict["experiment"],
                    "resolution": conflict["resolution"],
                    "case_index": conflict["case_index"],
                    "vertex_key": vertex_key,
                    "vtk_mesh": vtk_mesh,
                    "exact_mesh": exact_mesh,
                }
            )

    case_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in taxonomy_rows:
        key = (
            row["experiment"],
            row["algo"],
            row["resolution"],
            row["wiggle"],
            row["seed"],
            row["case_index"],
        )
        case_groups[key].append(row)
    case_rows = []
    for key, rows in sorted(case_groups.items()):
        counts = Counter(row["taxonomy"] for row in rows)
        case_rows.append(
            {
                "experiment": key[0],
                "algo": key[1],
                "resolution": key[2],
                "wiggle": key[3],
                "seed": key[4],
                "case_index": key[5],
                "flagged_vertices": len(rows),
                "exact_conflicts": sum(row["exact_conflict"] for row in rows),
                **{name: counts.get(name, 0) for name in TAXONOMY.values()},
            }
        )

    _write_csv(output_dir / "topology_conflict_taxonomy.csv", taxonomy_rows)
    _write_csv(output_dir / "topology_conflict_case_counts.csv", case_rows)
    _write_csv(output_dir / "topology_conflict_incident_facets.csv", incident_rows)
    if visual_examples:
        _plot_examples(output_dir, visual_examples, detail_lookup)

    taxonomy_counts = Counter(row["taxonomy"] for row in taxonomy_rows)
    source_totals = _read_source_audit_totals(source_dir, source_manifest)
    (output_dir / "README.md").write_text(
        _build_readme(taxonomy_rows, case_rows, source_totals), encoding="utf-8"
    )
    manifest = {
        "source_diagnostic": str(source_dir.resolve()),
        "source_manifest": source_manifest,
        "taxonomy_counts": dict(taxonomy_counts),
        "conflict_count": len(taxonomy_rows),
        "affected_case_count": len(case_rows),
        "audited_case_count": source_totals["audited_case_count"],
        "audited_complete_evaluated_shared_vertices": source_totals[
            "complete_evaluated_shared_vertices"
        ],
        "recommendation": "qualify_manuscript_claim; no diagnostic correction; consider a narrow algorithm fix only if exact consistency is required",
        "outputs": [
            "README.md",
            "topology_conflict_taxonomy.csv",
            "topology_conflict_case_counts.csv",
            "topology_conflict_incident_facets.csv",
            "topology_conflict_examples.pdf",
            "topology_conflict_examples.png",
        ],
    }
    (output_dir / "topology_conflict_diagnosis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return taxonomy_rows, case_rows, incident_rows


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    taxonomy_rows, case_rows, _ = diagnose(args.source_dir, args.output_dir)
    counts = Counter(row["taxonomy"] for row in taxonomy_rows)
    print(f"Diagnosed {len(taxonomy_rows)} conflicts in {len(case_rows)} cases")
    for taxonomy, count in sorted(counts.items()):
        print(f"  {taxonomy}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
