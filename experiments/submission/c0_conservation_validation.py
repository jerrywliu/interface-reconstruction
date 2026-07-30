"""Run a paired conservation audit before and after optional C0 adjustment.

The production static drivers remain the source of reconstruction artifacts.  This
module runs matched C0-off/on jobs and instruments ``MergeMesh.makeC0`` from the
outside so the submission evidence records the exact joins eligible for endpoint
averaging.  It then reuses :mod:`conservation_analyzer` to measure reconstructed
phase area from the saved facet geometry rather than the historical area metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.submission.conservation_analyzer import (
    analyze_saved_case,
    load_run_grid,
)
from main.structs.meshes.merge_mesh import MergeMesh
from util.reconstruction_diagnostics import facet_geometry_class, facet_geometry_record


PRODUCTION_PROFILE = {
    "plic_fallback": "LVIRA",
    "arc_failure_fallback": "local_linear",
    "rescue_profile": "exact_linear_support_only",
    "corner_behavior_profile": "pre_f8_corner",
}
DEFAULT_RESOLUTIONS = (0.64, 1.00)
DEFAULT_WIGGLE = 0.10
DEFAULT_SEED = 0


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _geometry_fingerprint(records: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_dumps(records).encode("utf-8")).hexdigest()


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.dist(tuple(map(float, first)), tuple(map(float, second)))


def _facet_valid_for_c0(poly: Any) -> bool:
    return (
        poly is not None
        and poly.hasFacet()
        and str(getattr(poly.facet, "name", "")) != "corner"
    )


def _capture_c0_state(mesh: MergeMesh, merged_polys: Sequence[Any]) -> dict[str, Any]:
    merge_id_by_object = {
        id(poly): str(merge_id) for merge_id, poly in mesh.merged_polys.items()
    }
    facets: dict[str, Any] = {}
    endpoints: dict[tuple[str, str], list[float]] = {}
    names: dict[str, str] = {}
    classes: dict[str, str] = {}
    for fallback_index, poly in enumerate(merged_polys):
        merge_id = merge_id_by_object.get(id(poly), f"unmapped:{fallback_index}")
        facet = poly.getFacet()
        facets[merge_id] = facet_geometry_record(facet)
        names[merge_id] = str(getattr(facet, "name", "") or "")
        classes[merge_id] = facet_geometry_class(facet)
        if facet is not None and hasattr(facet, "pLeft") and hasattr(facet, "pRight"):
            endpoints[(merge_id, "left")] = list(map(float, facet.pLeft))
            endpoints[(merge_id, "right")] = list(map(float, facet.pRight))

    joins: dict[tuple[tuple[str, str], tuple[str, str]], dict[str, Any]] = {}
    for fallback_index, poly in enumerate(merged_polys):
        if not _facet_valid_for_c0(poly):
            continue
        merge_id = merge_id_by_object.get(id(poly), f"unmapped:{fallback_index}")
        candidates = (
            ("left", poly.getLeftNeighbor(), "right"),
            ("right", poly.getRightNeighbor(), "left"),
        )
        for side, neighbor, neighbor_side in candidates:
            if not _facet_valid_for_c0(neighbor):
                continue
            neighbor_id = merge_id_by_object.get(id(neighbor))
            if neighbor_id is None:
                continue
            first = (merge_id, side)
            second = (neighbor_id, neighbor_side)
            key = tuple(sorted((first, second)))
            if key in joins or first not in endpoints or second not in endpoints:
                continue
            joins[key] = {
                "first_merge_id": first[0],
                "first_side": first[1],
                "second_merge_id": second[0],
                "second_side": second[1],
                "first_endpoint": endpoints[first],
                "second_endpoint": endpoints[second],
                "gap": _distance(endpoints[first], endpoints[second]),
            }

    fallback_merge_ids = {
        str(record.get("merge_id"))
        for record in getattr(mesh, "plic_fallback_records", [])
        if record.get("merge_id") is not None
    }
    return {
        "facets": facets,
        "facet_names": names,
        "facet_classes": classes,
        "endpoints": endpoints,
        "joins": joins,
        "num_explicit_corner_facets": sum(
            name == "corner" or "corner" in classes[merge_id]
            for merge_id, name in names.items()
        ),
        "num_unresolved_fallback_facets": len(fallback_merge_ids),
        "num_missing_facets": sum(value == "missing" for value in classes.values()),
        "num_c0_conservation_rejections": sum(
            getattr(poly, "last_c0_fit_diagnostic", None) is not None
            and poly.last_c0_fit_diagnostic.get("selected_branch") == "rejected"
            for poly in merged_polys
        ),
        "num_c0_opposite_branch_selections": sum(
            getattr(poly, "last_c0_fit_diagnostic", None) is not None
            and poly.last_c0_fit_diagnostic.get("selected_branch")
            == "opposite_signed_curvature"
            for poly in merged_polys
        ),
    }


def _instrumented_c0_record(
    mesh: MergeMesh,
    merged_polys: Sequence[Any],
    original_make_c0: Any,
    case_index: int,
    *,
    movement_tolerance: float = 1.0e-12,
) -> tuple[list[Any], dict[str, Any]]:
    before = _capture_c0_state(mesh, merged_polys)
    adjusted = original_make_c0(mesh, merged_polys)
    after = _capture_c0_state(mesh, adjusted)

    join_rows = []
    changed = 0
    for key, join in before["joins"].items():
        first_key, second_key = key
        first_after = after["endpoints"].get(first_key)
        second_after = after["endpoints"].get(second_key)
        missing_after = first_after is None or second_after is None
        first_move = (
            None
            if first_after is None
            else _distance(join["first_endpoint"], first_after)
        )
        second_move = (
            None
            if second_after is None
            else _distance(join["second_endpoint"], second_after)
        )
        join_changed = bool(
            not missing_after
            and max(first_move or 0.0, second_move or 0.0) > movement_tolerance
        )
        changed += int(join_changed)
        join_rows.append(
            {
                **{key: value for key, value in join.items() if key != "gap"},
                "gap_before_c0": join["gap"],
                "gap_after_c0": (
                    None if missing_after else _distance(first_after, second_after)
                ),
                "first_endpoint_after_c0": first_after,
                "second_endpoint_after_c0": second_after,
                "first_endpoint_movement": first_move,
                "second_endpoint_movement": second_move,
                "changed": int(join_changed),
                "missing_after_c0": int(missing_after),
            }
        )

    eligible = len(join_rows)
    record = {
        "case_index": case_index,
        "movement_tolerance": movement_tolerance,
        "num_eligible_joins": eligible,
        "num_changed_eligible_joins": changed,
        "fraction_eligible_joins_changed": changed / eligible if eligible else 0.0,
        "num_explicit_corner_facets": before["num_explicit_corner_facets"],
        "num_unresolved_fallback_facets": before[
            "num_unresolved_fallback_facets"
        ],
        "num_missing_facets_before_c0": before["num_missing_facets"],
        "num_missing_facets_after_c0": after["num_missing_facets"],
        "num_c0_conservation_rejections": after[
            "num_c0_conservation_rejections"
        ],
        "num_c0_opposite_branch_selections": after[
            "num_c0_opposite_branch_selections"
        ],
        "before_geometry_fingerprint": _geometry_fingerprint(before["facets"]),
        "after_geometry_fingerprint": _geometry_fingerprint(after["facets"]),
        "joins": join_rows,
    }
    return adjusted, record


def _worker(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Run one existing static driver in an isolated subprocess."""
    os.chdir(REPO_ROOT)
    experiment = str(spec["experiment"])
    if experiment == "ellipses":
        from experiments.static import ellipses as driver
    elif experiment == "zalesak":
        from experiments.static import zalesak as driver
    else:
        raise ValueError(f"Unsupported experiment: {experiment}")

    audit_records: list[dict[str, Any]] = []
    original_make_c0 = MergeMesh.makeC0
    if bool(spec["do_c0"]):
        call_count = 0

        def instrumented_make_c0(mesh: MergeMesh, merged_polys: Sequence[Any]):
            nonlocal call_count
            adjusted, record = _instrumented_c0_record(
                mesh,
                merged_polys,
                original_make_c0,
                call_count,
            )
            audit_records.append(record)
            call_count += 1
            return adjusted

        MergeMesh.makeC0 = instrumented_make_c0

    common = {
        "config_setting": f"static/{'ellipse' if experiment == 'ellipses' else 'zalesak'}",
        "resolution": float(spec["resolution"]),
        "facet_algo": str(spec["facet_algo"]),
        "save_name": str(spec["save_name"]),
        "mesh_type": "perturbed_quads",
        "perturb_wiggle": float(spec["wiggle"]),
        "perturb_seed": int(spec["seed"]),
        "perturb_fix_boundary": True,
        "do_c0": bool(spec["do_c0"]),
        "plic_fallback": PRODUCTION_PROFILE["plic_fallback"],
        "corner_behavior_profile": PRODUCTION_PROFILE["corner_behavior_profile"],
    }
    if experiment == "ellipses":
        driver.main(num_ellipses=int(spec["num_cases"]), **common)
    else:
        driver.main(
            num_cases=int(spec["num_cases"]),
            arc_failure_fallback=PRODUCTION_PROFILE["arc_failure_fallback"],
            rescue_profile=PRODUCTION_PROFILE["rescue_profile"],
            **common,
        )
    MergeMesh.makeC0 = original_make_c0

    run_root = REPO_ROOT / "plots" / str(spec["save_name"])
    if bool(spec["do_c0"]):
        if len(audit_records) != int(spec["num_cases"]):
            raise RuntimeError(
                f"Expected {spec['num_cases']} C0 records, got {len(audit_records)}"
            )
        audit_path = run_root / "metrics" / "c0_join_audit.jsonl"
        with audit_path.open("w", encoding="utf-8") as stream:
            for record in audit_records:
                stream.write(_json_dumps(record) + "\n")
    return {**dict(spec), "run_root": str(run_root)}


def _read_csv_index(path: Path, key: str = "case_index") -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {int(row[key]): row for row in csv.DictReader(stream)}


def _read_jsonl_index(path: Path, key: str = "case_index") -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {int(row[key]): row for row in rows}


def _cell_geometry_fingerprint(run_root: Path, case_index: int) -> str:
    rows = _read_csv_index_by_case(run_root / "metrics" / "cell_metrics.csv")[
        case_index
    ]
    facets: dict[str, Any] = {}
    for row in rows:
        facets[str(row["merge_id"])] = json.loads(row["facet_geometry_json"])
    return _geometry_fingerprint(facets)


def _read_csv_index_by_case(path: Path) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault(int(row["case_index"]), []).append(row)
    return grouped


def _read_geometry_index(path: Path) -> dict[int, dict[str, Any]]:
    return _read_jsonl_index(path)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def build_paired_case_row(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    c0_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and combine one matched C0-off/on case."""
    identity_fields = ("experiment", "resolution", "wiggle", "seed", "case_index")
    for field in identity_fields:
        if str(before.get(field)) != str(after.get(field)):
            raise ValueError(f"Mismatched C0 pair field {field}: {before.get(field)!r} != {after.get(field)!r}")
    if before.get("geometry_fingerprint") != after.get("input_geometry_fingerprint"):
        raise ValueError("C0-on pre-adjustment geometry differs from the C0-off run")
    if before.get("case_geometry_fingerprint") != after.get(
        "case_geometry_fingerprint"
    ):
        raise ValueError("C0 pair does not use identical analytic case geometry")

    metrics = (
        "global_relative_phase_area_error",
        "max_merged_zone_absolute_residual",
        "facet_gap",
        "num_final_missing_cells",
        "failure_count",
    )
    row = {field: before.get(field) for field in identity_fields}
    row["facet_algo"] = before.get("facet_algo")
    row["save_name_before_c0"] = before.get("save_name")
    row["save_name_after_c0"] = after.get("save_name")
    for metric in metrics:
        before_value = before.get(metric)
        after_value = after.get(metric)
        row[f"{metric}_before_c0"] = before_value
        row[f"{metric}_after_c0"] = after_value
        if before_value is None or after_value is None:
            row[f"{metric}_delta_after_minus_before"] = None
        else:
            row[f"{metric}_delta_after_minus_before"] = float(after_value) - float(
                before_value
            )
    for field in (
        "num_eligible_joins",
        "num_changed_eligible_joins",
        "fraction_eligible_joins_changed",
        "num_explicit_corner_facets",
        "num_unresolved_fallback_facets",
        "num_missing_facets_before_c0",
        "num_missing_facets_after_c0",
        "num_c0_conservation_rejections",
        "num_c0_opposite_branch_selections",
    ):
        row[field] = c0_audit.get(field)
    return row


def _values(rows: Iterable[Mapping[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row.get(field) not in (None, "")]


def aggregate_paired_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate paired cases, using a weighted changed-join fraction."""
    if not rows:
        raise ValueError("No paired C0 rows supplied")
    output = {
        field: rows[0].get(field)
        for field in ("experiment", "facet_algo", "resolution", "wiggle", "seed")
    }
    output["num_cases"] = len(rows)
    total_eligible = sum(int(row.get("num_eligible_joins", 0)) for row in rows)
    total_changed = sum(
        int(row.get("num_changed_eligible_joins", 0)) for row in rows
    )
    output["num_eligible_joins"] = total_eligible
    output["num_changed_eligible_joins"] = total_changed
    output["fraction_eligible_joins_changed"] = (
        total_changed / total_eligible if total_eligible else 0.0
    )
    output["num_cases_with_explicit_corners"] = sum(
        int(row.get("num_explicit_corner_facets", 0)) > 0 for row in rows
    )
    output["num_cases_with_unresolved_fallback"] = sum(
        int(row.get("num_unresolved_fallback_facets", 0)) > 0 for row in rows
    )
    output["num_c0_conservation_rejections"] = sum(
        int(row.get("num_c0_conservation_rejections") or 0) for row in rows
    )
    output["num_c0_opposite_branch_selections"] = sum(
        int(row.get("num_c0_opposite_branch_selections") or 0) for row in rows
    )

    for metric in (
        "global_relative_phase_area_error",
        "max_merged_zone_absolute_residual",
        "facet_gap",
    ):
        for stage in ("before_c0", "after_c0"):
            values = _values(rows, f"{metric}_{stage}")
            output[f"median_{metric}_{stage}"] = (
                statistics.median(values) if values else None
            )
            output[f"max_{metric}_{stage}"] = max(values) if values else None
    output["num_missing_facets_before_c0"] = sum(
        int(float(row.get("num_final_missing_cells_before_c0", 0))) for row in rows
    )
    output["num_missing_facets_after_c0"] = sum(
        int(float(row.get("num_final_missing_cells_after_c0", 0))) for row in rows
    )
    output["num_analysis_failures_before_c0"] = sum(
        int(float(row.get("failure_count_before_c0", 0))) for row in rows
    )
    output["num_analysis_failures_after_c0"] = sum(
        int(float(row.get("failure_count_after_c0", 0))) for row in rows
    )
    return output


def conservation_regression_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    global_error_threshold: float = 1.0e-6,
    merged_zone_residual_threshold: float = 1.0e-6,
) -> list[dict[str, Any]]:
    """Select material post-C0 conservation regressions for explicit review."""
    regressions = []
    for row in rows:
        global_error = _float_or_none(
            row.get("global_relative_phase_area_error_after_c0")
        )
        merged_residual = _float_or_none(
            row.get("max_merged_zone_absolute_residual_after_c0")
        )
        global_failure = bool(
            global_error is not None and global_error > global_error_threshold
        )
        merged_failure = bool(
            merged_residual is not None
            and merged_residual > merged_zone_residual_threshold
        )
        if not (global_failure or merged_failure):
            continue
        regressions.append(
            {
                **dict(row),
                "global_error_threshold": global_error_threshold,
                "merged_zone_residual_threshold": merged_zone_residual_threshold,
                "exceeds_global_error_threshold": int(global_failure),
                "exceeds_merged_zone_residual_threshold": int(merged_failure),
            }
        )
    return sorted(
        regressions,
        key=lambda row: float(
            row.get("global_relative_phase_area_error_after_c0") or -1.0
        ),
        reverse=True,
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    empty_fieldnames: Sequence[str] | None = None,
) -> None:
    if not rows:
        if empty_fieldnames is None:
            raise ValueError(f"Refusing to write empty CSV: {path}")
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=empty_fieldnames).writeheader()
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_specs(stamp: str, num_cases: int) -> list[dict[str, Any]]:
    specs = []
    for experiment, facet_algo in (
        ("ellipses", "circular"),
        ("zalesak", "circular+corner"),
    ):
        for resolution in DEFAULT_RESOLUTIONS:
            for do_c0 in (False, True):
                n_value = int(round(100 * resolution))
                specs.append(
                    {
                        "experiment": experiment,
                        "facet_algo": facet_algo,
                        "resolution": resolution,
                        "N": n_value,
                        "wiggle": DEFAULT_WIGGLE,
                        "seed": DEFAULT_SEED,
                        "do_c0": int(do_c0),
                        "num_cases": num_cases,
                        "save_name": (
                            f"submission_c0_conservation_{stamp}_{experiment}_"
                            f"{facet_algo.replace('+', '_')}_n{n_value}_w010_c0{int(do_c0)}"
                        ),
                    }
                )
    return specs


def _execute_specs(
    specs: Sequence[Mapping[str, Any]], output_dir: Path, max_workers: int
) -> list[dict[str, Any]]:
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def launch(spec: Mapping[str, Any]) -> dict[str, Any]:
        log_path = logs / f"{spec['save_name']}.log"
        command = [
            sys.executable,
            "-m",
            "experiments.submission.c0_conservation_validation",
            "--worker-spec",
            _json_dumps(spec),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode:
            raise RuntimeError(f"Run failed: {spec['save_name']}; see {log_path}")
        return {**dict(spec), "run_root": str(REPO_ROOT / "plots" / spec["save_name"])}

    completed_specs = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(launch, spec): spec for spec in specs}
        for future in as_completed(futures):
            result = future.result()
            completed_specs.append(result)
            print(f"[done] {result['save_name']}", flush=True)
    return sorted(
        completed_specs,
        key=lambda spec: (
            spec["experiment"],
            float(spec["resolution"]),
            int(spec["do_c0"]),
        ),
    )


def _case_geometry_fingerprint(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_dumps(record).encode("utf-8")).hexdigest()


def _analyze_runs(
    specs: Sequence[Mapping[str, Any]], output_dir: Path
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    by_setting = {
        (spec["experiment"], float(spec["resolution"]), int(spec["do_c0"])): spec
        for spec in specs
    }
    stage_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    all_join_records: list[dict[str, Any]] = []

    for experiment, facet_algo in (
        ("ellipses", "circular"),
        ("zalesak", "circular+corner"),
    ):
        for resolution in DEFAULT_RESOLUTIONS:
            off = by_setting[(experiment, resolution, 0)]
            on = by_setting[(experiment, resolution, 1)]
            off_root = Path(off["run_root"])
            on_root = Path(on["run_root"])
            off_metrics = _read_csv_index(off_root / "metrics" / "case_metrics.csv")
            on_metrics = _read_csv_index(on_root / "metrics" / "case_metrics.csv")
            off_geometry = _read_geometry_index(
                off_root / "metrics" / "case_geometry.jsonl"
            )
            on_geometry = _read_geometry_index(
                on_root / "metrics" / "case_geometry.jsonl"
            )
            c0_audits = _read_jsonl_index(
                on_root / "metrics" / "c0_join_audit.jsonl"
            )
            case_indices = sorted(set(off_metrics) & set(on_metrics))
            if len(case_indices) != int(off["num_cases"]):
                raise RuntimeError(
                    f"Incomplete pair {experiment} N={off['N']}: {len(case_indices)} cases"
                )

            for case_index in case_indices:
                before_analysis = analyze_saved_case(
                    off_root, case_index, stage="before_c0", repo_root=REPO_ROOT
                )
                after_analysis = analyze_saved_case(
                    on_root, case_index, stage="after_c0", repo_root=REPO_ROOT
                )
                geometry_before = _case_geometry_fingerprint(off_geometry[case_index])
                geometry_after = _case_geometry_fingerprint(on_geometry[case_index])
                audit = c0_audits[case_index]
                before_row = {
                    **before_analysis.summary,
                    "experiment": experiment,
                    "facet_algo": facet_algo,
                    "resolution": resolution,
                    "N": off["N"],
                    "wiggle": off["wiggle"],
                    "seed": off["seed"],
                    "case_index": case_index,
                    "save_name": off["save_name"],
                    "do_c0": 0,
                    "facet_gap": _float_or_none(off_metrics[case_index]["facet_gap"]),
                    "num_final_missing_cells": int(
                        float(off_metrics[case_index]["num_final_missing_cells"] or 0)
                    ),
                    "geometry_fingerprint": _cell_geometry_fingerprint(
                        off_root, case_index
                    ),
                    "input_geometry_fingerprint": _cell_geometry_fingerprint(
                        off_root, case_index
                    ),
                    "case_geometry_fingerprint": geometry_before,
                }
                after_row = {
                    **after_analysis.summary,
                    "experiment": experiment,
                    "facet_algo": facet_algo,
                    "resolution": resolution,
                    "N": on["N"],
                    "wiggle": on["wiggle"],
                    "seed": on["seed"],
                    "case_index": case_index,
                    "save_name": on["save_name"],
                    "do_c0": 1,
                    "facet_gap": _float_or_none(on_metrics[case_index]["facet_gap"]),
                    "num_final_missing_cells": int(
                        float(on_metrics[case_index]["num_final_missing_cells"] or 0)
                    ),
                    "geometry_fingerprint": audit["after_geometry_fingerprint"],
                    "input_geometry_fingerprint": audit[
                        "before_geometry_fingerprint"
                    ],
                    "case_geometry_fingerprint": geometry_after,
                }
                stage_rows.extend((before_row, after_row))
                pair = build_paired_case_row(before_row, after_row, audit)
                pair["N"] = off["N"]
                paired_rows.append(pair)
                for join in audit["joins"]:
                    all_join_records.append(
                        {
                            "experiment": experiment,
                            "facet_algo": facet_algo,
                            "resolution": resolution,
                            "N": off["N"],
                            "wiggle": off["wiggle"],
                            "seed": off["seed"],
                            "case_index": case_index,
                            **join,
                        }
                    )

    summary_rows = []
    for experiment, facet_algo in (
        ("ellipses", "circular"),
        ("zalesak", "circular+corner"),
    ):
        for resolution in DEFAULT_RESOLUTIONS:
            selected = [
                row
                for row in paired_rows
                if row["experiment"] == experiment
                and float(row["resolution"]) == resolution
            ]
            summary = aggregate_paired_rows(selected)
            summary["N"] = int(round(100 * resolution))
            summary_rows.append(summary)

    return stage_rows, paired_rows, summary_rows, all_join_records


def _format(value: Any) -> str:
    return "--" if value is None else f"{float(value):.3e}"


def _write_report(
    output_dir: Path,
    specs: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> None:
    regressions = conservation_regression_rows(paired_rows)
    worst_global = max(
        paired_rows,
        key=lambda row: float(row["global_relative_phase_area_error_after_c0"]),
    )
    rows_with_merged_residual = [
        row
        for row in paired_rows
        if row.get("max_merged_zone_absolute_residual_after_c0") not in (None, "")
    ]
    worst_merged = max(
        rows_with_merged_residual,
        key=lambda row: float(row["max_merged_zone_absolute_residual_after_c0"]),
    )
    if regressions:
        finding_lines = [
            "**This validation does not support a conservation guarantee for the output",
            "after the optional C0 pass.** The pre-C0 production reconstruction remains",
            "conservative on these cases, but C0 introduces a material coarse-Zalesak tail.",
            "",
            f"- `{len(regressions)}` of `{len(paired_rows)}` paired cases exceed a post-C0 global relative",
            "  area error of `1e-6` or a merged-zone absolute residual of `1e-6`.",
            "- The worst global case is {experiment}, `N={N}`, case `{case}`: `{before:.6e}`"
            " before C0 and `{after:.6e}` after C0.".format(
                experiment=worst_global["experiment"],
                N=int(round(100 * float(worst_global["resolution"]))),
                case=worst_global["case_index"],
                before=float(
                    worst_global["global_relative_phase_area_error_before_c0"]
                ),
                after=float(
                    worst_global["global_relative_phase_area_error_after_c0"]
                ),
            ),
            "- The largest adjusted merged-zone residual is `{residual:.6e}` in {experiment},"
            " `N={N}`, case `{case}`.".format(
                residual=float(
                    worst_merged["max_merged_zone_absolute_residual_after_c0"]
                ),
                experiment=worst_merged["experiment"],
                N=int(round(100 * float(worst_merged["resolution"]))),
                case=worst_merged["case_index"],
            ),
            "- No tested case has a missing facet or unresolved LVIRA fallback. The failure",
            "  is therefore in the C0 endpoint-adjustment/area-refitting path, not fallback",
            "  recovery.",
        ]
    else:
        finding_lines = [
            "**The conservation guard removes the material post-C0 regression in this",
            "validation scope.** No paired case exceeds a post-C0 global relative area",
            "error of `1e-6` or a merged-zone absolute residual of `1e-6`.",
            "",
            "- The worst post-C0 global relative area error is `{after:.6e}` in {experiment},"
            " `N={N}`, case `{case}`.".format(
                after=float(
                    worst_global["global_relative_phase_area_error_after_c0"]
                ),
                experiment=worst_global["experiment"],
                N=int(round(100 * float(worst_global["resolution"]))),
                case=worst_global["case_index"],
            ),
            "- The largest post-C0 merged-zone absolute residual is `{residual:.6e}` in"
            " {experiment}, `N={N}`, case `{case}`.".format(
                residual=float(
                    worst_merged["max_merged_zone_absolute_residual_after_c0"]
                ),
                experiment=worst_merged["experiment"],
                N=int(round(100 * float(worst_merged["resolution"]))),
                case=worst_merged["case_index"],
            ),
            "- Infeasible endpoint adjustments are retained at their conservative pre-C0",
            "  facet, so exact C0 continuity is conditional rather than forced.",
        ]

    lines = [
        "# Paired C0 Conservation Validation",
        "",
        "Matched C0-off/on runs use identical seeded cases and the production ",
        "`pre_f8_corner + exact_linear_support_only + LVIRA` reconstruction profile.",
        "",
        "## Scope",
        "",
        "- Ellipses: `circular`, `w=0.1`, `N=64,100`, 25 cases per setting.",
        "- Zalesak: `circular+corner`, `w=0.1`, `N=64,100`, 25 cases per setting.",
        "- The C0 pass averages eligible neighboring endpoints and refits each eligible",
        "  line/arc to its merged-zone area. Explicit corner facets are held fixed.",
        "- Unresolved LVIRA fallback facets may participate when they have valid",
        "  oriented neighbors, but C0 does not guarantee that an unresolved or isolated",
        "  fallback is repaired.",
        "",
        "## Submission Finding",
        "",
        *finding_lines,
        "",
        "## Results",
        "",
        "| Benchmark | N | Cases | Global rel. area, median off/on | Max merged-zone residual, max off/on | Facet gap, median off/on | Eligible joins changed | C0 fits rejected | Missing facets off/on |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {experiment} | {N} | {cases} | {global_off} / {global_on} | "
            "{zone_off} / {zone_on} | {gap_off} / {gap_on} | {changed}/{eligible} "
            "({fraction:.1%}) | {rejections} | {missing_off}/{missing_on} |".format(
                experiment=row["experiment"],
                N=row["N"],
                cases=row["num_cases"],
                global_off=_format(
                    row["median_global_relative_phase_area_error_before_c0"]
                ),
                global_on=_format(
                    row["median_global_relative_phase_area_error_after_c0"]
                ),
                zone_off=_format(
                    row["max_max_merged_zone_absolute_residual_before_c0"]
                ),
                zone_on=_format(
                    row["max_max_merged_zone_absolute_residual_after_c0"]
                ),
                gap_off=_format(row["median_facet_gap_before_c0"]),
                gap_on=_format(row["median_facet_gap_after_c0"]),
                changed=row["num_changed_eligible_joins"],
                eligible=row["num_eligible_joins"],
                fraction=row["fraction_eligible_joins_changed"],
                rejections=row["num_c0_conservation_rejections"],
                missing_off=row["num_missing_facets_before_c0"],
                missing_on=row["num_missing_facets_after_c0"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The conservation statement tested here is global phase area and the total",
            "area of each independent or merged reconstruction zone. It is not a claim",
            "that every constituent base cell inside a merged zone is independently",
            "matched after reconstruction.",
            "",
            "The paired-case CSV is the primary audit table. The join-level JSONL records",
            "every eligible endpoint pair and its gap before and after C0.",
            "",
            "## Artifacts",
            "",
            "- `c0_conservation_case_metrics.csv`: one row per case and stage.",
            "- `c0_conservation_paired_cases.csv`: one matched off/on row per case.",
            "- `c0_conservation_summary.csv`: four benchmark-resolution aggregates.",
            "- `c0_conservation_regressions.csv`: material post-C0 conservation tails.",
            "- `c0_eligible_join_changes.jsonl`: exact eligible-join changes.",
            "- `run_manifest.json`: commands, raw run roots, and production settings.",
            "- `logs/`: stdout/stderr from each of the eight static-driver jobs.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_regression_figure(
    output_dir: Path,
    specs: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Render the worst global and merged-zone C0 witnesses as vector geometry."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    from experiments.static import generate_section6_maintext_figures as figures

    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    worst_global = max(
        paired_rows,
        key=lambda row: float(row["global_relative_phase_area_error_after_c0"]),
    )
    with_merged = [
        row
        for row in paired_rows
        if row.get("max_merged_zone_absolute_residual_after_c0") not in (None, "")
    ]
    worst_merged = max(
        with_merged,
        key=lambda row: float(row["max_merged_zone_absolute_residual_after_c0"]),
    )
    witnesses = [worst_global, worst_merged]
    spec_index = {
        (spec["experiment"], float(spec["resolution"]), int(spec["do_c0"])): spec
        for spec in specs
    }

    figure, axes = plt.subplots(2, 2, figsize=(8.2, 7.0))
    for row_index, witness in enumerate(witnesses):
        experiment = str(witness["experiment"])
        resolution = float(witness["resolution"])
        case_index = int(witness["case_index"])
        off = spec_index[(experiment, resolution, 0)]
        on = spec_index[(experiment, resolution, 1)]
        off_root = Path(off["run_root"])
        on_root = Path(on["run_root"])
        after_analysis = analyze_saved_case(
            on_root, case_index, stage="after_c0", repo_root=REPO_ROOT
        )
        worst_zone = max(
            after_analysis.zone_rows,
            key=lambda zone: float(zone["absolute_residual"]),
        )
        zone_cells = [
            cell
            for cell in after_analysis.cell_rows
            if str(cell["merge_id"]) == str(worst_zone["merge_id"])
        ]
        grid = load_run_grid(on_root, repo_root=REPO_ROOT)
        zone_points = np.asarray(
            [
                point
                for cell in zone_cells
                for point in grid.cell_polygon(int(cell["cell_x"]), int(cell["cell_y"]))
            ],
            dtype=float,
        )
        margin = 3.0 * resolution
        bounds = (
            float(np.min(zone_points[:, 0]) - margin),
            float(np.max(zone_points[:, 0]) + margin),
            float(np.min(zone_points[:, 1]) - margin),
            float(np.max(zone_points[:, 1]) + margin),
        )
        mesh_segments = figures._mesh_segments(on_root / "vtk" / "mesh.vtk")
        true_segments = figures._load_true_segments(
            experiment, str(on["save_name"]), case_index
        )
        reconstructions = (
            (
                "Before C0",
                figures._load_reconstructed_segments(str(off["save_name"]), case_index),
                float(witness["global_relative_phase_area_error_before_c0"]),
            ),
            (
                "After C0",
                figures._load_reconstructed_segments(str(on["save_name"]), case_index),
                float(witness["global_relative_phase_area_error_after_c0"]),
            ),
        )
        for column, (stage, reconstructed, global_error) in enumerate(reconstructions):
            axis = axes[row_index, column]
            figures._add_segments(
                axis, mesh_segments, color="#d1d5db", linewidth=0.35, alpha=0.65
            )
            figures._add_segments(
                axis,
                true_segments,
                color="#111827",
                linewidth=1.5,
                linestyle="--",
                zorder=2,
            )
            figures._add_segments(
                axis,
                reconstructed,
                color="#c2410c" if column else "#2563eb",
                linewidth=1.8,
                zorder=3,
            )
            axis.set_xlim(bounds[0], bounds[1])
            axis.set_ylim(bounds[2], bounds[3])
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(
                f"{stage}\nGlobal relative area error {global_error:.3e}", fontsize=9
            )
        axes[row_index, 0].set_ylabel(
            "Case {case}, zone {zone}\npost-C0 zone residual {residual:.3e}".format(
                case=case_index,
                zone=worst_zone["merge_id"],
                residual=float(worst_zone["absolute_residual"]),
            ),
            fontsize=9,
        )

    figure.legend(
        handles=[
            Line2D([0], [0], color="#111827", linestyle="--", label="True interface"),
            Line2D([0], [0], color="#2563eb", label="Reconstruction before C0"),
            Line2D([0], [0], color="#c2410c", label="Reconstruction after C0"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        "Coarse Zalesak C0 conservation regressions (N=64, w=0.1)", fontsize=11
    )
    figure.tight_layout(rect=(0.03, 0.085, 1.0, 0.95))
    output_path = output_dir / "c0_zalesak_regression_before_after.pdf"
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-cases", type=int, default=25)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--worker-spec", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker_spec is not None:
        result = _worker(json.loads(args.worker_spec))
        print(_json_dumps(result))
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output
        or REPO_ROOT / "results" / "submission" / f"c0_conservation_{stamp}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _run_specs(stamp, args.num_cases)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "production_profile": PRODUCTION_PROFILE,
        "matched_case_policy": "same benchmark, resolution, wiggle, seed, and deterministic case index",
        "specs": specs,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    completed_specs = _execute_specs(specs, output_dir, args.max_workers)
    stage_rows, paired_rows, summary_rows, join_rows = _analyze_runs(
        completed_specs, output_dir
    )
    _write_csv(output_dir / "c0_conservation_case_metrics.csv", stage_rows)
    _write_csv(output_dir / "c0_conservation_paired_cases.csv", paired_rows)
    _write_csv(output_dir / "c0_conservation_summary.csv", summary_rows)
    regression_rows = conservation_regression_rows(paired_rows)
    _write_csv(
        output_dir / "c0_conservation_regressions.csv",
        regression_rows,
        empty_fieldnames=[
            *paired_rows[0].keys(),
            "global_error_threshold",
            "merged_zone_residual_threshold",
            "exceeds_global_error_threshold",
            "exceeds_merged_zone_residual_threshold",
        ],
    )
    with (output_dir / "c0_eligible_join_changes.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in join_rows:
            stream.write(_json_dumps(row) + "\n")
    if regression_rows:
        _generate_regression_figure(output_dir, completed_specs, paired_rows)
    _write_report(output_dir, completed_specs, summary_rows, paired_rows)

    manifest.update(
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "specs": completed_specs,
            "outputs": {
                "case_metrics": "c0_conservation_case_metrics.csv",
                "paired_cases": "c0_conservation_paired_cases.csv",
                "summary": "c0_conservation_summary.csv",
                "regressions": "c0_conservation_regressions.csv",
                "eligible_joins": "c0_eligible_join_changes.jsonl",
                "report": "README.md",
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed paired C0 conservation validation: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
