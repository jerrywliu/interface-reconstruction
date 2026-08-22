#!/usr/bin/env python3
"""Run isolated high-resolution circle and ellipse convergence smoke tests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from experiments.submission.conservation_analyzer import analyze_saved_case
from main.structs.meshes.merge_mesh import MergeMesh


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CELLS_PER_SIDE = (256, 300, 512)
DEFAULT_WIGGLES = (0.0, 0.2)
DEFAULT_CASE_INDICES = (0, 1, 2, 3, 4)
LINEAR_METHODS = ("Youngs", "ELVIRA", "LVIRA", "safe_linear", "linear")
ELLIPSE_METHODS = ("circular",)
EXPERIMENTS = ("circles", "ellipses")
PLIC_FALLBACK = "LVIRA"
FIT_TOLERANCE = 1.0e-10
LOG_DIAGNOSTIC_PATTERNS = {
    "arc_fit_error_messages": "Error in getArcFacet",
    "easy_orientation_error_messages": "Error in easy orientation",
    "final_failed_orientation_messages": "Final failed orientations:",
    "traceback_messages": "Traceback (most recent call last)",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_csv(value: str, cast: type = float) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip()


def _source_state() -> dict[str, Any]:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "status_porcelain": _git("status", "--short"),
    }


def _environment() -> dict[str, Any]:
    return {
        "captured_utc": _utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def _tag(value: Any) -> str:
    return str(value).lower().replace(".", "p").replace("+", "plus")


def _build_specs(
    output_dir: Path,
    cells_per_side: Sequence[int],
    wiggles: Sequence[float],
    case_indices: Sequence[int],
    experiments: Sequence[str] = EXPERIMENTS,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    cases = ",".join(map(str, case_indices))
    families = (
        ("circles", "static/circle", "--num_circles", LINEAR_METHODS),
        ("ellipses", "static/ellipse", "--num_ellipses", ELLIPSE_METHODS),
    )
    selected = set(experiments)
    for experiment, config, count_arg, methods in families:
        if experiment not in selected:
            continue
        for n in cells_per_side:
            resolution = n / 100.0
            for wiggle in wiggles:
                for method in methods:
                    save_name = (
                        f"{output_dir.name}_{experiment}_{_tag(method)}_"
                        f"n{n}_w{_tag(wiggle)}_s0"
                    )
                    command = [
                        sys.executable,
                        "-m",
                        f"experiments.static.{experiment}",
                        "--config",
                        config,
                        "--resolution",
                        str(resolution),
                        "--facet_algo",
                        method,
                        "--save_name",
                        save_name,
                        count_arg,
                        str(max(case_indices) + 1),
                        "--case_indices",
                        cases,
                        "--mesh_type",
                        "perturbed_quads",
                        "--perturb_wiggle",
                        str(wiggle),
                        "--perturb_seed",
                        "0",
                        "--perturb_fix_boundary",
                        "1",
                        "--plic_fallback",
                        PLIC_FALLBACK,
                        "--corner_behavior_profile",
                        MergeMesh.default_corner_behavior_profile,
                    ]
                    specs.append(
                        {
                            "experiment": experiment,
                            "method": method,
                            "cells_per_side": n,
                            "resolution": resolution,
                            "wiggle": wiggle,
                            "seed": 0,
                            "case_indices": list(case_indices),
                            "save_name": save_name,
                            "command": command,
                            "command_text": " ".join(command),
                            "temporary_plot_dir": str(REPO_ROOT / "plots" / save_name),
                            "raw_run_dir": str(output_dir / "raw_runs" / save_name),
                            "log": str(output_dir / "logs" / f"{save_name}.log"),
                            "expected_fit_floor": FIT_TOLERANCE / resolution,
                        }
                    )
    return specs


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _log_diagnostics(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    return {
        field: text.count(pattern)
        for field, pattern in LOG_DIAGNOSTIC_PATTERNS.items()
    }


def _float(row: Mapping[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _execute(spec: Mapping[str, Any]) -> dict[str, Any]:
    log_path = Path(spec["log"])
    temporary_run = Path(spec["temporary_plot_dir"])
    raw_run = Path(spec["raw_run_dir"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    raw_run.parent.mkdir(parents=True, exist_ok=True)
    if temporary_run.exists() or raw_run.exists():
        return {
            **spec,
            "status": "failed_preflight",
            "returncode": None,
            "wall_time_seconds": 0.0,
            "error": f"refusing to overwrite {temporary_run if temporary_run.exists() else raw_run}",
        }

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"started_utc={_utc_now()}\ncommand={spec['command_text']}\n\n")
        stream.flush()
        result = subprocess.run(
            spec["command"],
            cwd=REPO_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    status = "succeeded" if result.returncode == 0 else "failed"
    error = ""
    if result.returncode == 0 and temporary_run.is_dir():
        shutil.move(str(temporary_run), str(raw_run))
    elif result.returncode == 0:
        status = "failed_missing_output"
        error = f"driver returned success without {temporary_run}"
    return {
        **spec,
        **_log_diagnostics(log_path),
        "status": status,
        "returncode": result.returncode,
        "wall_time_seconds": elapsed,
        "completed_utc": _utc_now(),
        "error": error,
    }


def _collect_case_rows(result: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    run_root = Path(result["raw_run_dir"])
    metric_rows = _read_csv(run_root / "metrics" / "case_metrics.csv")
    by_case = {int(row["case_index"]): row for row in metric_rows}
    output: list[dict[str, Any]] = []
    failures: list[str] = []
    for case_index in result["case_indices"]:
        metric_row = by_case.get(case_index)
        if metric_row is None:
            failures.append(f"case {case_index}: missing case_metrics.csv row")
            continue
        try:
            conservation = analyze_saved_case(
                run_root, case_index, repo_root=REPO_ROOT
            ).summary
        except Exception as exc:  # Preserve successful reconstruction for diagnosis.
            conservation = {}
            failures.append(f"case {case_index}: conservation analysis failed: {exc}")
        hausdorff = _float(metric_row, "hausdorff")
        facet_gap = _float(metric_row, "facet_gap")
        floor = float(result["expected_fit_floor"])
        output.append(
            {
                "experiment": result["experiment"],
                "method": result["method"],
                "cells_per_side": result["cells_per_side"],
                "resolution": result["resolution"],
                "wiggle": result["wiggle"],
                "seed": result["seed"],
                "case_index": case_index,
                "hausdorff": hausdorff,
                "facet_gap": facet_gap,
                "expected_fit_floor": floor,
                "hausdorff_to_expected_floor": (
                    hausdorff / floor if hausdorff is not None else None
                ),
                "facet_gap_to_expected_floor": (
                    facet_gap / floor if facet_gap is not None else None
                ),
                "num_mixed_cells": metric_row.get("num_mixed_cells", ""),
                "num_missing_facets": metric_row.get("num_final_missing_cells", ""),
                "num_plic_fallback_cells": metric_row.get(
                    "num_plic_fallback_cells", ""
                ),
                "conservation_complete": conservation.get("complete"),
                "conservation_failure_count": conservation.get("failure_count"),
                "global_relative_phase_area_error": conservation.get(
                    "global_relative_phase_area_error"
                ),
                "signed_global_phase_area_residual": conservation.get(
                    "signed_global_phase_area_residual"
                ),
                "max_fitted_component_absolute_residual": conservation.get(
                    "max_zone_absolute_residual"
                ),
                "max_merged_component_absolute_residual": conservation.get(
                    "max_merged_zone_absolute_residual"
                ),
                "max_cell_area_relative_residual": conservation.get(
                    "max_cell_area_relative_residual"
                ),
                "grid_source": conservation.get("grid_source", ""),
                "run_wall_time_seconds": result["wall_time_seconds"],
                "save_name": result["save_name"],
            }
        )
    return output, failures


def _finite(rows: Iterable[Mapping[str, Any]], field: str) -> list[float]:
    return [value for row in rows if (value := _float(row, field)) is not None]


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for a finite value sequence."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must lie in [0, 1]")
    ordered = sorted(values)
    position = percentile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _summaries(case_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in case_rows:
        key = (
            row["experiment"],
            row["method"],
            row["cells_per_side"],
            row["wiggle"],
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        experiment, method, n, wiggle = key
        record: dict[str, Any] = {
            "experiment": experiment,
            "method": method,
            "cells_per_side": n,
            "wiggle": wiggle,
            "num_cases": len(rows),
            "failed_conservation_cases": sum(
                not _is_true(row.get("conservation_complete")) for row in rows
            ),
        }
        for field in (
            "hausdorff",
            "facet_gap",
            "global_relative_phase_area_error",
            "max_fitted_component_absolute_residual",
            "max_cell_area_relative_residual",
        ):
            values = _finite(rows, field)
            record[f"{field}_median"] = median(values) if values else None
            record[f"{field}_q1"] = _percentile(values, 0.25) if values else None
            record[f"{field}_q3"] = _percentile(values, 0.75) if values else None
            record[f"{field}_iqr"] = (
                record[f"{field}_q3"] - record[f"{field}_q1"]
                if values
                else None
            )
            record[f"{field}_max"] = max(values) if values else None
        summaries.append(record)
    return summaries


def _estimate_full_cases(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [
        float(row["wall_time_seconds"])
        for row in results
        if row.get("status") == "succeeded"
    ]
    total = sum(successful)
    return {
        "basis": "five-case subprocess wall times scaled linearly to 25 cases",
        "observed_successful_runs": len(successful),
        "observed_five_case_wall_seconds": total,
        "estimated_25_case_serial_seconds": total * 5.0,
        "estimated_25_case_serial_hours": total * 5.0 / 3600.0,
        "caveat": "Mesh construction is paid once per run, so linear scaling is conservative.",
    }


def _checkpoint(
    output_dir: Path,
    manifest: dict[str, Any],
    results: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> None:
    manifest["updated_utc"] = _utc_now()
    manifest["completed_runs"] = len(results)
    manifest["successful_runs"] = sum(row.get("status") == "succeeded" for row in results)
    manifest["failed_runs"] = sum(row.get("status") != "succeeded" for row in results)
    manifest["solver_diagnostic_totals"] = {
        field: sum(int(row.get(field, 0) or 0) for row in results)
        for field in LOG_DIAGNOSTIC_PATTERNS
    }
    manifest["status"] = (
        "complete" if len(results) == len(manifest["runs"]) else "running"
    )
    if manifest["status"] == "complete":
        manifest["full_25_case_estimate"] = _estimate_full_cases(results)
    _write_json_atomic(output_dir / "manifest.json", manifest)
    _write_csv(output_dir / "run_status.csv", results)
    _write_csv(output_dir / "case_metrics.csv", case_rows)
    _write_csv(output_dir / "summary_metrics.csv", _summaries(case_rows))
    _write_csv(output_dir / "failures.csv", failures)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="new output directory (default: timestamped results/static directory)",
    )
    parser.add_argument("--cells-per-side", default="256,300,512")
    parser.add_argument("--wiggles", default="0,0.2")
    parser.add_argument("--case-indices", default="0,1,2,3,4")
    parser.add_argument(
        "--experiments",
        default=",".join(EXPERIMENTS),
        help="comma-separated subset of circles,ellipses",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        REPO_ROOT / "results" / "static" / f"extended_convergence_smoke_{_run_id()}"
    )
    output_dir = output_dir.resolve()
    expected_parent = (REPO_ROOT / "results" / "static").resolve()
    if output_dir.parent != expected_parent or not output_dir.name.startswith(
        "extended_convergence_smoke_"
    ):
        raise SystemExit(
            "output must be a new results/static/extended_convergence_smoke_<timestamp> directory"
        )
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_dir}")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    cells_per_side = _parse_csv(args.cells_per_side, int)
    wiggles = _parse_csv(args.wiggles, float)
    case_indices = _parse_csv(args.case_indices, int)
    experiments = _parse_csv(args.experiments, str)
    if not cells_per_side or not wiggles or not case_indices:
        raise SystemExit("cells-per-side, wiggles, and case-indices must be nonempty")
    unknown_experiments = sorted(set(experiments) - set(EXPERIMENTS))
    if not experiments or unknown_experiments:
        raise SystemExit(
            "experiments must be a nonempty subset of circles,ellipses; "
            f"unknown: {','.join(unknown_experiments)}"
        )
    output_dir.mkdir(parents=True)
    (output_dir / "logs").mkdir()
    (output_dir / "raw_runs").mkdir()
    specs = _build_specs(
        output_dir, cells_per_side, wiggles, case_indices, experiments
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "extended_convergence_smoke",
        "status": "planned" if args.dry_run else "running",
        "created_utc": _utc_now(),
        "output_dir": str(output_dir),
        "source": _source_state(),
        "environment": _environment(),
        "settings": {
            "cells_per_side": list(cells_per_side),
            "resolutions": [n / 100.0 for n in cells_per_side],
            "wiggles": list(wiggles),
            "case_indices": list(case_indices),
            "experiments": list(experiments),
            "circle_methods": list(LINEAR_METHODS),
            "ellipse_methods": list(ELLIPSE_METHODS),
            "plic_fallback": PLIC_FALLBACK,
            "corner_behavior_profile": MergeMesh.default_corner_behavior_profile,
            "workers": args.workers,
            "fit_tolerance": FIT_TOLERANCE,
        },
        "runs": specs,
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    if args.dry_run:
        print(output_dir)
        print(f"planned {len(specs)} runs / {len(specs) * len(case_indices)} cases")
        return 0

    results: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_execute, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    **spec,
                    "status": "runner_exception",
                    "returncode": None,
                    "wall_time_seconds": 0.0,
                    "completed_utc": _utc_now(),
                    "error": str(exc),
                }
            results.append(result)
            if result["status"] == "succeeded":
                rows, diagnostic_failures = _collect_case_rows(result)
                case_rows.extend(rows)
                failures.extend(
                    {"save_name": result["save_name"], "failure": failure}
                    for failure in diagnostic_failures
                )
            else:
                failures.append(
                    {
                        "save_name": result["save_name"],
                        "failure": result.get("error") or f"return code {result.get('returncode')}",
                    }
                )
            _checkpoint(output_dir, manifest, results, case_rows, failures)
            print(
                f"[{len(results)}/{len(specs)}] {result['status']}: "
                f"{result['save_name']} ({result['wall_time_seconds']:.1f}s)",
                flush=True,
            )

    manifest["status"] = "complete_with_failures" if failures else "complete"
    manifest["completed_utc"] = _utc_now()
    _checkpoint(output_dir, manifest, results, case_rows, failures)
    manifest["status"] = "complete_with_failures" if failures else "complete"
    _write_json_atomic(output_dir / "manifest.json", manifest)
    print(output_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
