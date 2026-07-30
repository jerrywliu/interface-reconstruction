"""Conservation diagnostics for saved interface-reconstruction artifacts.

The production merge path assigns a merged polygon the sum of the prescribed
fluid areas of its constituent cells.  A single final facet is then fitted to
that merged polygon.  Consequently, the fit constrains the merged-zone total;
it does not independently constrain each original cell.  This module measures
both levels by clipping the final facet back into every constituent cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from main.geoms.circular_facet import getCircleIntersectArea
from main.geoms.geoms import getArea, getPolyIntersectArea, getPolyLineArea
from util.config import read_yaml
from util.initialize.mesh_factory import apply_mesh_overrides, make_points_from_config
from util.metrics.area_metrics import facet_area_in_polygon


@dataclass(frozen=True)
class GridRecord:
    points: list[list[list[float]]]
    source: str
    vtk_max_point_delta: float | None = None
    note: str = ""

    @property
    def nx(self) -> int:
        return len(self.points)

    @property
    def ny(self) -> int:
        return len(self.points[0])

    def cell_polygon(self, x: int, y: int) -> list[list[float]]:
        return [
            self.points[x][y],
            self.points[x + 1][y],
            self.points[x + 1][y + 1],
            self.points[x][y + 1],
        ]


@dataclass(frozen=True)
class ConservationAnalysis:
    summary: dict[str, Any]
    zone_rows: list[dict[str, Any]]
    cell_rows: list[dict[str, Any]]


IDENTIFIER_FIELDS = (
    "experiment",
    "algo",
    "resolution",
    "wiggle",
    "seed",
    "save_name",
    "source_commit",
    "source_branch",
    "plic_fallback",
    "rescue_profile",
    "corner_behavior_profile",
    "case_index",
)


def _float(value: Any) -> float:
    return float(value)


def _int(value: Any) -> int:
    return int(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def parse_legacy_structured_grid(path: str | Path) -> GridRecord:
    """Read the ASCII legacy structured grids written by ``writeMesh``."""
    path = Path(path)
    tokens = path.read_text(encoding="utf-8").split()
    try:
        dim_index = tokens.index("DIMENSIONS")
        nx, ny, nz = map(int, tokens[dim_index + 1 : dim_index + 4])
        point_index = tokens.index("POINTS")
        point_count = int(tokens[point_index + 1])
    except (ValueError, IndexError) as error:
        raise ValueError(f"Unsupported legacy VTK grid: {path}") from error
    if nz != 1 or point_count != nx * ny:
        raise ValueError(
            f"Expected a 2-D structured grid, got dimensions {(nx, ny, nz)} "
            f"and {point_count} points"
        )

    value_start = point_index + 3
    flat = list(map(float, tokens[value_start : value_start + 3 * point_count]))
    if len(flat) != 3 * point_count:
        raise ValueError(f"Truncated point data in {path}")
    xyz = [flat[index : index + 3] for index in range(0, len(flat), 3)]
    points = [
        [xyz[x * ny + y][:2] for y in range(ny)]
        for x in range(nx)
    ]
    return GridRecord(
        points=points,
        source="legacy_vtk",
        note="Coordinates use the precision retained by the legacy VTK writer.",
    )


def _regenerate_manifest_grid(
    run_root: Path, manifest: Mapping[str, Any], repo_root: Path
) -> list[list[list[float]]]:
    parameters = manifest["parameters"]
    config_name = parameters["config"]
    config = read_yaml(str(repo_root / "config" / f"{config_name}.yaml"))
    mesh_config = apply_mesh_overrides(
        config["MESH"],
        resolution=parameters.get("resolution"),
        mesh_type=parameters.get("mesh_type"),
        perturb_wiggle=parameters.get("perturb_wiggle"),
        perturb_seed=parameters.get("perturb_seed"),
        perturb_fix_boundary=parameters.get("perturb_fix_boundary"),
        perturb_max_tries=parameters.get("perturb_max_tries"),
        perturb_type=parameters.get("perturb_type"),
    )
    return make_points_from_config(mesh_config)


def _max_point_delta(
    first: Sequence[Sequence[Sequence[float]]],
    second: Sequence[Sequence[Sequence[float]]],
) -> float:
    if len(first) != len(second) or len(first[0]) != len(second[0]):
        return float("inf")
    return max(
        math.dist(first[x][y], second[x][y])
        for x in range(len(first))
        for y in range(len(first[0]))
    )


def load_run_grid(
    run_root: str | Path,
    *,
    repo_root: str | Path | None = None,
    regeneration_tolerance: float = 1.0e-3,
) -> GridRecord:
    """Load a run grid, preferring exact deterministic regeneration.

    The legacy ``mesh.vtk`` files round perturbed coordinates enough to pollute
    residuals near the solver tolerance.  The immutable run manifest contains
    the mesh seed and perturbation parameters, so exact regeneration is used
    only after it agrees with the saved VTK grid within a conservative audit
    tolerance.
    """
    run_root = Path(run_root)
    vtk_grid = parse_legacy_structured_grid(run_root / "vtk" / "mesh.vtk")
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        return vtk_grid

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        regenerated = _regenerate_manifest_grid(run_root, manifest, repo_root)
        delta = _max_point_delta(regenerated, vtk_grid.points)
    except Exception as error:
        return GridRecord(
            points=vtk_grid.points,
            source="legacy_vtk",
            note=f"Manifest regeneration unavailable: {error}",
        )
    if delta > regeneration_tolerance:
        return GridRecord(
            points=vtk_grid.points,
            source="legacy_vtk",
            vtk_max_point_delta=delta,
            note=(
                "Manifest regeneration disagreed with the saved grid; retained "
                "the rounded VTK coordinates."
            ),
        )
    return GridRecord(
        points=regenerated,
        source="manifest_regenerated",
        vtk_max_point_delta=delta,
        note="Exact seeded grid regenerated and audited against mesh.vtk.",
    )


def _ellipse_area_in_polygon(
    polygon: Sequence[Sequence[float]], geometry: Mapping[str, Any]
) -> float:
    major = _float(geometry["major_axis"])
    minor = _float(geometry["minor_axis"])
    theta = _float(geometry["theta"])
    center = geometry["center"]
    transform = np.array(
        [
            [
                major * math.cos(theta) ** 2 + minor * math.sin(theta) ** 2,
                (major - minor) * math.cos(theta) * math.sin(theta),
            ],
            [
                (major - minor) * math.cos(theta) * math.sin(theta),
                major * math.sin(theta) ** 2 + minor * math.cos(theta) ** 2,
            ],
        ]
    )
    inverse = np.linalg.inv(transform)
    transformed = []
    for point in polygon:
        centered = np.array([point[0] - center[0], point[1] - center[1]])
        transformed.append((inverse @ centered).tolist())
    unit_area, _ = getCircleIntersectArea([0.0, 0.0], 1.0, transformed)
    return min(unit_area * major * minor, abs(getArea(polygon)))


def truth_area_in_polygon(
    polygon: Sequence[Sequence[float]], geometry: Mapping[str, Any]
) -> float:
    geometry_type = geometry["geometry_type"]
    if geometry_type == "line":
        return getPolyLineArea(polygon, geometry["p_left"], geometry["p_right"])
    if geometry_type == "circle":
        return getCircleIntersectArea(
            geometry["center"], _float(geometry["radius"]), polygon
        )[0]
    if geometry_type == "ellipse":
        return _ellipse_area_in_polygon(polygon, geometry)
    if geometry_type == "square":
        return sum(
            abs(getArea(part))
            for part in getPolyIntersectArea(geometry["vertices"], polygon)
        )
    if geometry_type == "zalesak":
        circle_area, _ = getCircleIntersectArea(
            geometry["center"], _float(geometry["radius"]), polygon
        )
        removed_area = 0.0
        for intersection in getPolyIntersectArea(geometry["slot_vertices"], polygon):
            overlap, _ = getCircleIntersectArea(
                geometry["center"], _float(geometry["radius"]), intersection
            )
            removed_area += min(max(overlap, 0.0), abs(getArea(intersection)))
        return max(0.0, circle_area - removed_area)
    raise ValueError(f"Unsupported truth geometry: {geometry_type!r}")


def prescribed_phase_area(
    grid: GridRecord, geometry: Mapping[str, Any]
) -> float:
    return sum(
        truth_area_in_polygon(grid.cell_polygon(x, y), geometry)
        for x in range(grid.nx - 1)
        for y in range(grid.ny - 1)
    )


def _row_identifiers(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in IDENTIFIER_FIELDS}


def analyze_case_records(
    grid: GridRecord,
    rows: Iterable[Mapping[str, Any]],
    *,
    total_prescribed_phase_area: float | None,
    stage: str = "final",
    geometry: Mapping[str, Any] | None = None,
) -> ConservationAnalysis:
    """Analyze one case from saved-style cell rows or equivalent in-memory rows."""
    rows = list(rows)
    if not rows:
        raise ValueError("No cell records supplied")
    identifiers = _row_identifiers(rows[0])
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if _row_identifiers(row) != identifiers:
            raise ValueError("All rows must describe the same case")
        groups[str(row["merge_id"])].append(row)

    cell_rows: list[dict[str, Any]] = []
    zone_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    signed_global_delta = 0.0
    mixed_truth_mismatches: list[float] = []

    for merge_id, member_rows in groups.items():
        geometry_records = {
            json.dumps(_json_value(row["facet_geometry_json"]), sort_keys=True)
            for row in member_rows
        }
        if len(geometry_records) != 1:
            raise ValueError(f"Merge zone {merge_id} has inconsistent final facets")
        facet_geometry = json.loads(next(iter(geometry_records)))
        if facet_geometry is None:
            failures.append(f"merge_id={merge_id}: missing final facet")
            continue

        zone_target = 0.0
        zone_reconstructed = 0.0
        for row in member_rows:
            x = _int(row.get("cell_x", str(row["cell_id"]).split(",")[0]))
            y = _int(row.get("cell_y", str(row["cell_id"]).split(",")[1]))
            polygon = grid.cell_polygon(x, y)
            cell_area = abs(getArea(polygon))
            target = _float(row["cell_fraction"]) * cell_area
            try:
                reconstructed = facet_area_in_polygon(polygon, facet_geometry)
            except Exception as error:
                failures.append(f"merge_id={merge_id}, cell={x},{y}: {error}")
                continue
            signed_residual = reconstructed - target
            absolute_residual = abs(signed_residual)
            zone_target += target
            zone_reconstructed += reconstructed
            signed_global_delta += signed_residual
            if geometry is not None:
                truth_target = truth_area_in_polygon(polygon, geometry)
                mixed_truth_mismatches.append(abs(truth_target - target))
            cell_rows.append(
                {
                    **identifiers,
                    "stage": stage,
                    "merge_id": merge_id,
                    "merge_component_size": len(member_rows),
                    "cell_id": f"{x},{y}",
                    "cell_x": x,
                    "cell_y": y,
                    "facet_class": facet_geometry["class"],
                    "cell_area": cell_area,
                    "prescribed_phase_area": target,
                    "reconstructed_phase_area": reconstructed,
                    "signed_residual": signed_residual,
                    "absolute_residual": absolute_residual,
                    "relative_residual": absolute_residual / max(abs(target), 1.0e-300),
                    "cell_area_relative_residual": absolute_residual / cell_area,
                }
            )

        signed_zone_residual = zone_reconstructed - zone_target
        zone_rows.append(
            {
                **identifiers,
                "stage": stage,
                "merge_id": merge_id,
                "merge_component_size": len(member_rows),
                "is_merged": int(len(member_rows) > 1),
                "facet_class": facet_geometry["class"],
                "prescribed_phase_area": zone_target,
                "reconstructed_phase_area": zone_reconstructed,
                "signed_residual": signed_zone_residual,
                "absolute_residual": abs(signed_zone_residual),
                "relative_residual": abs(signed_zone_residual)
                / max(abs(zone_target), 1.0e-300),
            }
        )

    complete = not failures and len(cell_rows) == len(rows)
    global_relative_error = None
    global_reconstructed_area = None
    if complete and total_prescribed_phase_area is not None:
        global_reconstructed_area = total_prescribed_phase_area + signed_global_delta
        global_relative_error = abs(signed_global_delta) / max(
            abs(total_prescribed_phase_area), 1.0e-300
        )

    def maximum(items: Sequence[Mapping[str, Any]], field: str) -> float | None:
        return max((float(item[field]) for item in items), default=None)

    merged_zone_rows = [row for row in zone_rows if row["is_merged"]]
    merged_cell_rows = [
        row for row in cell_rows if int(row["merge_component_size"]) > 1
    ]
    summary = {
        **identifiers,
        "stage": stage,
        "complete": complete,
        "num_cells": len(rows),
        "num_zones": len(groups),
        "num_merged_zones": sum(len(group) > 1 for group in groups.values()),
        "num_merged_cells": sum(len(group) for group in groups.values() if len(group) > 1),
        "grid_source": grid.source,
        "grid_vtk_max_point_delta": grid.vtk_max_point_delta,
        "grid_note": grid.note,
        "prescribed_phase_area": total_prescribed_phase_area,
        "reconstructed_phase_area": global_reconstructed_area,
        "signed_global_phase_area_residual": signed_global_delta if complete else None,
        "global_relative_phase_area_error": global_relative_error,
        "max_zone_absolute_residual": maximum(zone_rows, "absolute_residual"),
        "max_zone_relative_residual": maximum(zone_rows, "relative_residual"),
        "max_merged_zone_absolute_residual": maximum(
            merged_zone_rows, "absolute_residual"
        ),
        "max_merged_zone_relative_residual": maximum(
            merged_zone_rows, "relative_residual"
        ),
        "max_cell_absolute_residual": maximum(cell_rows, "absolute_residual"),
        "max_cell_relative_residual": maximum(cell_rows, "relative_residual"),
        "max_cell_area_relative_residual": maximum(
            cell_rows, "cell_area_relative_residual"
        ),
        "max_merged_cell_absolute_residual": maximum(
            merged_cell_rows, "absolute_residual"
        ),
        "max_merged_cell_relative_residual": maximum(
            merged_cell_rows, "relative_residual"
        ),
        "max_merged_cell_area_relative_residual": maximum(
            merged_cell_rows, "cell_area_relative_residual"
        ),
        "max_prescribed_mixed_truth_mismatch": max(
            mixed_truth_mismatches, default=None
        ),
        "failure_count": len(failures),
        "failures": failures,
    }
    return ConservationAnalysis(summary, zone_rows, cell_rows)


def _read_case_rows(path: Path, case_index: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if _int(row["case_index"]) == case_index
        ]


def _read_case_geometry(path: Path, case_index: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if _int(record["case_index"]) == case_index:
                return record
    return None


def _read_reported_area_error(path: Path, case_index: int) -> float | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if _int(row["case_index"]) != case_index:
                continue
            value = row.get("area_error")
            return None if value in (None, "") else _float(value)
    return None


def analyze_saved_case(
    run_root: str | Path,
    case_index: int,
    *,
    stage: str | None = None,
    repo_root: str | Path | None = None,
) -> ConservationAnalysis:
    run_root = Path(run_root)
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    if stage is None:
        stage = "after_c0" if manifest["parameters"].get("do_c0") else "before_c0"
    grid = load_run_grid(run_root, repo_root=repo_root)
    rows = _read_case_rows(run_root / "metrics" / "cell_metrics.csv", case_index)
    parameters = manifest.get("parameters", {})
    manifest_identifiers = {
        "experiment": manifest.get("experiment", ""),
        "algo": parameters.get("facet_algo", ""),
        "resolution": parameters.get("resolution", ""),
        "wiggle": parameters.get("perturb_wiggle", ""),
        "seed": parameters.get("perturb_seed", ""),
        "save_name": run_root.name,
        "source_commit": manifest.get("source_commit", ""),
        "source_branch": manifest.get("source_branch", ""),
        "plic_fallback": parameters.get("plic_fallback", ""),
        "rescue_profile": parameters.get("rescue_profile", ""),
        "corner_behavior_profile": parameters.get("corner_behavior_profile", ""),
        "case_index": case_index,
    }
    rows = [
        {
            **manifest_identifiers,
            **{key: value for key, value in row.items() if value not in (None, "")},
        }
        for row in rows
    ]
    geometry = _read_case_geometry(
        run_root / "metrics" / "case_geometry.jsonl", case_index
    )
    total = prescribed_phase_area(grid, geometry) if geometry is not None else None
    analysis = analyze_case_records(
        grid,
        rows,
        total_prescribed_phase_area=total,
        stage=stage,
        geometry=geometry,
    )
    analysis.summary["legacy_reported_area_error"] = _read_reported_area_error(
        run_root / "metrics" / "case_metrics.csv", case_index
    )
    analysis.summary["legacy_area_error_used_by_analyzer"] = False
    return analysis


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields and field != "failures":
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_analysis_bundle(
    analyses: Sequence[ConservationAnalysis], output_dir: str | Path
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [analysis.summary for analysis in analyses]
    zones = [row for analysis in analyses for row in analysis.zone_rows]
    cells = [row for analysis in analyses for row in analysis.cell_rows]
    _write_csv(output_dir / "conservation_case_summary.csv", summaries)
    _write_csv(output_dir / "conservation_zone_residuals.csv", zones)
    _write_csv(output_dir / "conservation_cell_residuals.csv", cells)
    c0_comparisons = compare_c0_stages(summaries)
    _write_csv(output_dir / "conservation_c0_comparison.csv", c0_comparisons)
    payload = {
        "schema_version": 1,
        "production_constraint": "merged_zone_total_only",
        "c0_pairing": (
            "Supported when separate before/after cell records are supplied; a single "
            "saved final stage cannot be split retrospectively."
        ),
        "cases": summaries,
        "c0_comparisons": c0_comparisons,
    }
    (output_dir / "conservation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def compare_c0_stages(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join eligible before/after C0 summaries without inventing a missing stage."""
    comparison_fields = (
        "global_relative_phase_area_error",
        "max_merged_zone_absolute_residual",
        "max_merged_cell_absolute_residual",
    )
    key_fields = tuple(
        field for field in IDENTIFIER_FIELDS if field not in {"algo", "save_name"}
    )
    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for summary in summaries:
        stage = str(summary.get("stage", ""))
        if stage in {"before_c0", "after_c0"}:
            grouped[tuple(summary.get(field, "") for field in key_fields)][stage] = summary

    rows = []
    for key, stages in grouped.items():
        if set(stages) != {"before_c0", "after_c0"}:
            continue
        before = stages["before_c0"]
        after = stages["after_c0"]
        row = dict(zip(key_fields, key))
        row["algo_before_c0"] = before.get("algo", "")
        row["algo_after_c0"] = after.get("algo", "")
        for field in comparison_fields:
            before_value = before.get(field)
            after_value = after.get(field)
            row[f"{field}_before_c0"] = before_value
            row[f"{field}_after_c0"] = after_value
            row[f"{field}_delta_after_minus_before"] = (
                None
                if before_value is None or after_value is None
                else float(after_value) - float(before_value)
            )
        rows.append(row)
    return rows


def _selection_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args(argv)

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    analyses = []
    for item in selection["cases"]:
        run_root = _selection_path(args.repo_root, item["run_root"])
        analyses.append(
            analyze_saved_case(
                run_root,
                _int(item["case_index"]),
                stage=item.get("stage"),
                repo_root=args.repo_root,
            )
        )
    write_analysis_bundle(analyses, args.output)
    print(f"Wrote {len(analyses)} conservation case analyses to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
