import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path

from main.structs.facets.circular_facet import ArcFacet
from main.structs.facets.corner_facet import CornerFacet
from main.structs.facets.linear_facet import LinearFacet


CELL_FIELDNAMES = [
    "case_index",
    "cell_id",
    "cell_x",
    "cell_y",
    "cell_fraction",
    "merge_id",
    "merge_component_size",
    "merge_fraction",
    "is_merged",
    "orientation_status",
    "has_3x3_stencil",
    "final_facet_class",
    "final_facet_name",
    "construction_path",
    "fallback_policy",
    "used_circular",
    "used_linear_corner",
    "used_curved_corner",
    "used_curved_corner_rescue",
    "event_count",
    "facet_geometry_json",
]

EVENT_FIELDNAMES = [
    "case_index",
    "event_order",
    "merge_id",
    "member_cells_json",
    "stage",
    "event_kind",
    "fallback_policy",
    "fallback_reason",
    "previous_facet_class",
    "previous_facet_name",
    "facet_class",
    "facet_name",
]

CASE_FIELDNAMES = [
    "case_index",
    "num_mixed_cells",
    "num_merge_components",
    "num_merged_cells",
    "num_merged_components",
    "num_plic_fallback_cells",
    "num_early_orientation_hints",
    "num_late_orientation_hints",
    "num_orientation_retry_passes",
    "num_orientation_retry_candidates",
    "num_orientation_retry_degree_0",
    "num_orientation_retry_degree_1",
    "num_orientation_retry_degree_2",
    "num_orientation_retry_degree_3plus",
    "num_orientation_retry_unoriented",
    "num_orientation_retry_half_oriented",
    "num_used_circular_cells",
    "num_used_linear_corner_cells",
    "num_used_curved_corner_cells",
    "num_used_curved_corner_rescue_cells",
    "num_final_linear_cells",
    "num_final_circular_cells",
    "num_final_linear_corner_cells",
    "num_final_curved_corner_cells",
    "num_final_missing_cells",
    "fraction_merged_cells",
    "fraction_plic_fallback_cells",
    "fraction_used_circular_cells",
    "fraction_used_linear_corner_cells",
    "fraction_used_curved_corner_cells",
    "fraction_used_curved_corner_rescue_cells",
    "fraction_final_linear_cells",
    "fraction_final_circular_cells",
    "fraction_final_linear_corner_cells",
    "fraction_final_curved_corner_cells",
    "hausdorff",
    "facet_gap",
    "area_error",
    "curvature_error",
    "tangent_error",
    "curvature_proxy_error",
]


def _source_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _source_branch():
    try:
        return subprocess.run(
            ["git", "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def write_run_manifest(output_dirs, experiment, parameters):
    """Write the immutable settings needed to reproduce a static run bundle."""
    manifest = {
        "schema_version": 1,
        "experiment": experiment,
        "source_commit": _source_commit(),
        "source_branch": _source_branch(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "argv": list(sys.argv),
        "parameters": parameters,
        "artifacts": {
            "mesh": "vtk/mesh.vtk",
            "case_geometry": "metrics/case_geometry.jsonl",
            "case_metrics": "metrics/case_metrics.csv",
            "cell_metrics": "metrics/cell_metrics.csv",
            "merge_events": "metrics/merge_events.csv",
            "fallback_events": "metrics/unresolved_plic_fallbacks.csv",
        },
    }
    path = Path(output_dirs["base"]) / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def append_case_geometry(output_dirs, case_index, geometry):
    """Append exact case parameters and the saved truth-geometry sidecars."""
    record = {"case_index": case_index, **geometry}
    path = Path(output_dirs["metrics"]) / "case_geometry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_json_dumps(record) + "\n")


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _json_dumps(value):
    return json.dumps(value, default=_json_default, separators=(",", ":"))


def _normalise_merge_coords(raw_coords):
    """Accept both ``[[x, y], ...]`` and legacy singleton ``[x, y]`` entries."""
    if raw_coords is None:
        return []
    values = list(raw_coords)
    if len(values) == 2 and all(
        not hasattr(value, "__iter__") or isinstance(value, (str, bytes))
        for value in values
    ):
        return [values]
    return [list(coord) for coord in values]


def _append_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def facet_geometry_class(facet):
    if facet is None:
        return "missing"
    if isinstance(facet, CornerFacet):
        if isinstance(facet.facetLeft, LinearFacet) and isinstance(
            facet.facetRight, LinearFacet
        ):
            return "linear_corner"
        if isinstance(facet.facetLeft, ArcFacet) or isinstance(
            facet.facetRight, ArcFacet
        ):
            return "curved_corner"
        return "corner"
    if isinstance(facet, ArcFacet):
        return "circular"
    if isinstance(facet, LinearFacet):
        return "linear"
    return type(facet).__name__


def facet_geometry_record(facet):
    if facet is None:
        return None
    if isinstance(facet, LinearFacet):
        return {
            "class": "linear",
            "name": getattr(facet, "name", ""),
            "p_left": facet.pLeft,
            "p_right": facet.pRight,
        }
    if isinstance(facet, ArcFacet):
        return {
            "class": "circular",
            "name": getattr(facet, "name", ""),
            "center": facet.center,
            "radius": facet.radius,
            "p_left": facet.pLeft,
            "p_right": facet.pRight,
        }
    if isinstance(facet, CornerFacet):
        return {
            "class": facet_geometry_class(facet),
            "name": getattr(facet, "name", ""),
            "p_left": facet.pLeft,
            "corner": facet.corner,
            "p_right": facet.pRight,
            "left_branch": facet_geometry_record(facet.facetLeft),
            "right_branch": facet_geometry_record(facet.facetRight),
        }
    return {"class": facet_geometry_class(facet), "name": getattr(facet, "name", "")}


def _event_flags(events):
    classes = {event.get("facet_class", "") for event in events}
    stages = {event.get("stage", "") for event in events}
    return {
        "used_circular": "circular" in classes,
        "used_linear_corner": "linear_corner" in classes,
        "used_curved_corner": "curved_corner" in classes,
        "used_curved_corner_rescue": any(
            "curved_corner" in stage and "rescue" in stage for stage in stages
        ),
    }


def _component_rows(mesh, case_index):
    events_by_merge_id = defaultdict(list)
    for event in getattr(mesh, "facet_provenance_events", []):
        events_by_merge_id[event.get("merge_id")].append(event)

    fallback_by_merge_id = defaultdict(set)
    for record in getattr(mesh, "plic_fallback_records", []):
        fallback_by_merge_id[record.get("merge_id")].add(record.get("policy", ""))

    rows = []
    components = []
    for merge_id, poly in mesh.merged_polys.items():
        if merge_id >= len(mesh.merge_ids_to_coords):
            continue
        coords = _normalise_merge_coords(mesh.merge_ids_to_coords[merge_id])
        active_merge_ids = getattr(mesh, "coords_to_merge_id", None)
        if active_merge_ids is not None:
            coords = [
                (x, y)
                for x, y in coords
                if active_merge_ids[x][y] == merge_id
            ]
        if not coords:
            continue
        components.append((merge_id, coords))
        events = events_by_merge_id.get(merge_id, [])
        flags = _event_flags(events)
        fallback_policies = sorted(policy for policy in fallback_by_merge_id[merge_id] if policy)
        final_facet = poly.getFacet()
        final_class = facet_geometry_class(final_facet)
        final_name = str(getattr(final_facet, "name", "") or "")
        component_size = len(coords)
        if fallback_policies:
            construction_path = "plic_fallback"
        elif component_size > 1:
            construction_path = "merged"
        else:
            construction_path = "direct_fit"
        orientation_status = (
            "oriented"
            if poly.fullyOriented()
            else "unresolved_or_deadend"
        )
        for x, y in coords:
            cell = mesh.polys[x][y]
            rows.append(
                {
                    "case_index": case_index,
                    "cell_id": f"{x},{y}",
                    "cell_x": x,
                    "cell_y": y,
                    "cell_fraction": cell.getFraction(),
                    "merge_id": merge_id,
                    "merge_component_size": component_size,
                    "merge_fraction": poly.getFraction(),
                    "is_merged": int(component_size > 1),
                    "orientation_status": orientation_status,
                    "has_3x3_stencil": int(poly.has3x3Stencil()),
                    "final_facet_class": final_class,
                    "final_facet_name": final_name,
                    "construction_path": construction_path,
                    "fallback_policy": "|".join(fallback_policies),
                    "used_circular": int(flags["used_circular"]),
                    "used_linear_corner": int(flags["used_linear_corner"]),
                    "used_curved_corner": int(flags["used_curved_corner"]),
                    "used_curved_corner_rescue": int(
                        flags["used_curved_corner_rescue"]
                    ),
                    "event_count": len(events),
                    "facet_geometry_json": _json_dumps(
                        facet_geometry_record(final_facet)
                    ),
                }
            )
    return rows, components, events_by_merge_id


def _case_summary(rows, components):
    num_cells = len(rows)
    num_components = len(components)
    num_merged_cells = sum(int(row["is_merged"]) for row in rows)
    num_merged_components = sum(len(coords) > 1 for _, coords in components)
    class_counts = Counter(row["final_facet_class"] for row in rows)
    plic_count = sum(row["construction_path"] == "plic_fallback" for row in rows)

    def flag_count(field):
        return sum(int(row[field]) for row in rows)

    def fraction(value):
        return value / num_cells if num_cells else 0.0

    return {
        "num_mixed_cells": num_cells,
        "num_merge_components": num_components,
        "num_merged_cells": num_merged_cells,
        "num_merged_components": num_merged_components,
        "num_plic_fallback_cells": plic_count,
        "num_used_circular_cells": flag_count("used_circular"),
        "num_used_linear_corner_cells": flag_count("used_linear_corner"),
        "num_used_curved_corner_cells": flag_count("used_curved_corner"),
        "num_used_curved_corner_rescue_cells": flag_count(
            "used_curved_corner_rescue"
        ),
        "num_final_linear_cells": class_counts["linear"],
        "num_final_circular_cells": class_counts["circular"],
        "num_final_linear_corner_cells": class_counts["linear_corner"],
        "num_final_curved_corner_cells": class_counts["curved_corner"],
        "num_final_missing_cells": class_counts["missing"],
        "fraction_merged_cells": fraction(num_merged_cells),
        "fraction_plic_fallback_cells": fraction(plic_count),
        "fraction_used_circular_cells": fraction(flag_count("used_circular")),
        "fraction_used_linear_corner_cells": fraction(
            flag_count("used_linear_corner")
        ),
        "fraction_used_curved_corner_cells": fraction(
            flag_count("used_curved_corner")
        ),
        "fraction_used_curved_corner_rescue_cells": fraction(
            flag_count("used_curved_corner_rescue")
        ),
        "fraction_final_linear_cells": fraction(class_counts["linear"]),
        "fraction_final_circular_cells": fraction(class_counts["circular"]),
        "fraction_final_linear_corner_cells": fraction(class_counts["linear_corner"]),
        "fraction_final_curved_corner_cells": fraction(class_counts["curved_corner"]),
    }


def write_reconstruction_diagnostics(mesh, case_index, output_dirs):
    rows, components, events_by_merge_id = _component_rows(mesh, case_index)
    summary = _case_summary(rows, components)
    hint_records = getattr(mesh, "orientation_hint_records", [])
    summary["num_early_orientation_hints"] = sum(
        record.get("phase") == "early" for record in hint_records
    )
    summary["num_late_orientation_hints"] = sum(
        record.get("phase") == "late" for record in hint_records
    )
    summary["num_orientation_retry_passes"] = getattr(
        mesh, "orientation_retry_passes", 0
    )
    retry_records = getattr(mesh, "orientation_retry_records", [])
    for field in (
        "queue_size",
        "degree_0",
        "degree_1",
        "degree_2",
        "degree_3plus",
        "unoriented",
        "half_oriented",
    ):
        summary[f"num_orientation_retry_{field.replace('queue_size', 'candidates')}"] = sum(
            record.get(field, 0) for record in retry_records
        )
    mesh.reconstruction_diagnostic_summary = summary

    event_rows = []
    for merge_id, events in events_by_merge_id.items():
        if merge_id is None or merge_id >= len(mesh.merge_ids_to_coords):
            member_cells = []
        else:
            member_cells = _normalise_merge_coords(mesh.merge_ids_to_coords[merge_id])
        for event in events:
            event_rows.append(
                {
                    "case_index": case_index,
                    "event_order": event.get("event_order"),
                    "merge_id": merge_id,
                    "member_cells_json": _json_dumps(member_cells),
                    "stage": event.get("stage", ""),
                    "event_kind": event.get("event_kind", ""),
                    "fallback_policy": event.get("fallback_policy", ""),
                    "fallback_reason": event.get("fallback_reason", ""),
                    "previous_facet_class": event.get("previous_facet_class", ""),
                    "previous_facet_name": event.get("previous_facet_name", ""),
                    "facet_class": event.get("facet_class", ""),
                    "facet_name": event.get("facet_name", ""),
                }
            )

    metrics_dir = output_dirs["metrics"]
    _append_csv(Path(metrics_dir) / "cell_metrics.csv", CELL_FIELDNAMES, rows)
    _append_csv(
        Path(metrics_dir) / "merge_events.csv",
        EVENT_FIELDNAMES,
        sorted(event_rows, key=lambda row: row["event_order"]),
    )
    return summary


def append_case_metrics(output_dirs, case_index, metrics, summary=None):
    summary = dict(summary or {})
    row = {field: "" for field in CASE_FIELDNAMES}
    row["case_index"] = case_index
    row.update(summary)
    row.update(metrics)
    _append_csv(Path(output_dirs["metrics"]) / "case_metrics.csv", CASE_FIELDNAMES, [row])
