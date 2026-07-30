#!/usr/bin/env python3
"""Compare a completed submission release with a completed reference release."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Optional, Sequence


SCHEMA_VERSION = 1
RUN_FIELDS = ("experiment", "algo", "resolution", "wiggle", "seed")
CASE_FIELDS = RUN_FIELDS + ("case_index",)
REQUIRED_CASE_COLUMNS = set(CASE_FIELDS) | {"hausdorff", "facet_gap"}
SCIENTIFIC_METRICS = (
    "hausdorff",
    "facet_gap",
    "area_error",
    "curvature_error",
    "tangent_error",
    "curvature_proxy_error",
)
PERFECT_RECONSTRUCTION_METHODS = {
    "squares": "linear+corner",
    "zalesak": "circular+corner",
}
DEFAULT_BASELINE_GLOB = "static_paper_simplified_default_*"
DEFAULT_CANDIDATE_GLOB = "submission_static_*"

RUN_COVERAGE_FIELDS = (
    *RUN_FIELDS,
    "cells_per_side",
    "baseline_present",
    "candidate_present",
    "baseline_case_count",
    "candidate_case_count",
    "matched_case_count",
    "status",
)
STAT_FIELDS = (
    "paired_case_count",
    "baseline_value_count",
    "candidate_value_count",
    "baseline_median",
    "candidate_median",
    "delta_median",
    "candidate_to_baseline_median_ratio",
    "baseline_p95",
    "candidate_p95",
    "delta_p95",
    "baseline_max",
    "candidate_max",
    "delta_max",
    "material_improved_cases",
    "material_worsened_cases",
    "material_stable_cases",
    "baseline_above_tail_count",
    "candidate_above_tail_count",
    "fixed_tail_cases",
    "introduced_tail_cases",
)
METHOD_METRIC_FIELDS = ("experiment", "algo", "metric", *STAT_FIELDS)
SETTING_METRIC_FIELDS = (
    *RUN_FIELDS,
    "cells_per_side",
    "metric",
    *STAT_FIELDS,
)
CASE_METRIC_FIELDS = (
    *CASE_FIELDS,
    "cells_per_side",
    "metric",
    "baseline_value",
    "candidate_value",
    "delta",
    "candidate_to_baseline_ratio",
    "outcome",
    "value_status",
)
TAIL_CASE_FIELDS = (
    *CASE_FIELDS,
    "cells_per_side",
    "baseline_hausdorff",
    "candidate_hausdorff",
    "delta_hausdorff",
    "candidate_to_baseline_ratio",
    "outcome",
    "baseline_plic_fallback_cells",
    "candidate_plic_fallback_cells",
    "reasons",
)
PERFECT_FIELDS = (
    *RUN_FIELDS,
    "cells_per_side",
    "case_count",
    "threshold",
    "baseline_hausdorff_median",
    "candidate_hausdorff_median",
    "baseline_hausdorff_p95",
    "candidate_hausdorff_p95",
    "baseline_hausdorff_max",
    "candidate_hausdorff_max",
    "baseline_facet_gap_median",
    "candidate_facet_gap_median",
    "baseline_facet_gap_max",
    "candidate_facet_gap_max",
    "baseline_hausdorff_median_below_threshold",
    "candidate_hausdorff_median_below_threshold",
    "baseline_joint_median_below_threshold",
    "candidate_joint_median_below_threshold",
    "baseline_joint_floor_case_count",
    "candidate_joint_floor_case_count",
    "baseline_all_cases_joint_floor",
    "candidate_all_cases_joint_floor",
    "threshold_outcome",
    "high_resolution",
)


class ComparisonError(RuntimeError):
    """Raised when a release cannot support a trustworthy comparison."""


@dataclass(frozen=True, order=True)
class RunKey:
    experiment: str
    algo: str
    resolution: Decimal
    wiggle: Decimal
    seed: int

    def csv_values(self) -> dict:
        return {
            "experiment": self.experiment,
            "algo": self.algo,
            "resolution": _decimal_text(self.resolution),
            "wiggle": _decimal_text(self.wiggle),
            "seed": self.seed,
        }

    @property
    def cells_per_side(self) -> int:
        return int(round(float(self.resolution) * 100.0))


@dataclass(frozen=True, order=True)
class CaseKey:
    run: RunKey
    case_index: int

    def csv_values(self) -> dict:
        return {**self.run.csv_values(), "case_index": self.case_index}


@dataclass
class CaseRecord:
    key: CaseKey
    metrics: dict[str, Optional[float]]
    plic_fallback_cells: Optional[int]


@dataclass
class ReleaseData:
    root: Path
    manifest: dict
    cases: dict[CaseKey, CaseRecord]
    cases_by_run: dict[RunKey, dict[int, CaseRecord]]
    source_state: dict

    @property
    def run_keys(self) -> set[RunKey]:
        return set(self.cases_by_run)


def _decimal_text(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return "0" if text in {"-0", ""} else text


def _parse_decimal(value: object, field: str, path: Path, line_number: int) -> Decimal:
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise ComparisonError(
            f"{path}:{line_number}: invalid {field} value {value!r}"
        ) from None
    if not parsed.is_finite():
        raise ComparisonError(
            f"{path}:{line_number}: nonfinite {field} value {value!r}"
        )
    return parsed.normalize()


def _parse_int(value: object, field: str, path: Path, line_number: int) -> int:
    try:
        return int(str(value).strip())
    except ValueError:
        raise ComparisonError(
            f"{path}:{line_number}: invalid {field} value {value!r}"
        ) from None


def _parse_optional_float(
    value: object, field: str, path: Path, line_number: int
) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        raise ComparisonError(
            f"{path}:{line_number}: invalid {field} value {value!r}"
        ) from None
    if not math.isfinite(parsed):
        raise ComparisonError(
            f"{path}:{line_number}: nonfinite {field} value {value!r}"
        )
    if parsed < 0:
        raise ComparisonError(
            f"{path}:{line_number}: negative error metric {field}={parsed}"
        )
    return parsed


def _parse_optional_int(
    value: object, field: str, path: Path, line_number: int
) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    return _parse_int(value, field, path, line_number)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ComparisonError(f"missing required file: {path}") from None
    except json.JSONDecodeError as exc:
        raise ComparisonError(f"invalid JSON in {path}: {exc}") from None
    if not isinstance(value, dict):
        raise ComparisonError(f"expected a JSON object in {path}")
    return value


def _manifest_is_complete(manifest: dict) -> bool:
    if manifest.get("status") != "completed":
        return False
    try:
        failure_count = int(
            manifest.get("failure_count", len(manifest.get("failures", [])))
        )
        planned = manifest.get("planned_run_count")
        successful = manifest.get("successful_run_count")
        planned_count = None if planned is None else int(planned)
        successful_count = None if successful is None else int(successful)
    except (TypeError, ValueError):
        return False
    if failure_count != 0:
        return False
    return planned_count is None or successful_count == planned_count


def discover_release_root(results_parent: Path, pattern: str) -> Path:
    """Return the newest complete release matching ``pattern``."""
    results_parent = results_parent.resolve()
    if not results_parent.is_dir():
        raise ComparisonError(f"results parent does not exist: {results_parent}")

    eligible = []
    rejected = []
    for candidate in sorted(results_parent.glob(pattern)):
        if not candidate.is_dir():
            continue
        manifest_path = candidate / "sweep_manifest.json"
        try:
            manifest = _read_json(manifest_path)
        except ComparisonError as exc:
            rejected.append(f"{candidate.name}: {exc}")
            continue
        if not _manifest_is_complete(manifest):
            rejected.append(
                f"{candidate.name}: status={manifest.get('status')!r}, "
                f"failures={manifest.get('failure_count')!r}"
            )
            continue
        timestamp = str(manifest.get("timestamp_utc") or "")
        eligible.append((timestamp, candidate.name, candidate.resolve()))

    if not eligible:
        detail = "; ".join(rejected[-5:]) or "no matching directories"
        raise ComparisonError(
            f"no complete release matching {pattern!r} under {results_parent}: {detail}"
        )
    return max(eligible)[2]


def _load_case_metrics(path: Path) -> dict[CaseKey, CaseRecord]:
    cases: dict[CaseKey, CaseRecord] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ComparisonError(f"missing CSV header: {path}")
        missing_columns = sorted(REQUIRED_CASE_COLUMNS - set(reader.fieldnames))
        if missing_columns:
            raise ComparisonError(
                f"{path} is missing required columns: {', '.join(missing_columns)}"
            )
        for line_number, row in enumerate(reader, start=2):
            experiment = (row.get("experiment") or "").strip()
            algo = (row.get("algo") or "").strip()
            if not experiment or not algo:
                raise ComparisonError(
                    f"{path}:{line_number}: experiment and algo must be nonempty"
                )
            run = RunKey(
                experiment=experiment,
                algo=algo,
                resolution=_parse_decimal(
                    row.get("resolution"), "resolution", path, line_number
                ),
                wiggle=_parse_decimal(row.get("wiggle"), "wiggle", path, line_number),
                seed=_parse_int(row.get("seed"), "seed", path, line_number),
            )
            key = CaseKey(
                run=run,
                case_index=_parse_int(
                    row.get("case_index"), "case_index", path, line_number
                ),
            )
            if key in cases:
                raise ComparisonError(
                    f"duplicate case key in {path}:{line_number}: "
                    f"{key.csv_values()}"
                )
            metrics = {
                metric: _parse_optional_float(
                    row.get(metric), metric, path, line_number
                )
                for metric in SCIENTIFIC_METRICS
            }
            cases[key] = CaseRecord(
                key=key,
                metrics=metrics,
                plic_fallback_cells=_parse_optional_int(
                    row.get("num_plic_fallback_cells"),
                    "num_plic_fallback_cells",
                    path,
                    line_number,
                ),
            )
    if not cases:
        raise ComparisonError(f"no case rows found in {path}")
    return cases


def load_release(root: Path, role: str) -> ReleaseData:
    root = root.resolve()
    if not root.is_dir():
        raise ComparisonError(f"{role} release root does not exist: {root}")
    manifest = _read_json(root / "sweep_manifest.json")
    if not _manifest_is_complete(manifest):
        raise ComparisonError(
            f"{role} release is not complete and failure-free: {root} "
            f"(status={manifest.get('status')!r}, "
            f"successful={manifest.get('successful_run_count')!r}, "
            f"planned={manifest.get('planned_run_count')!r}, "
            f"failures={manifest.get('failure_count')!r})"
        )

    case_path = root / "diagnostics" / "case_metrics.csv"
    try:
        cases = _load_case_metrics(case_path)
    except FileNotFoundError:
        raise ComparisonError(f"missing required file: {case_path}") from None
    cases_by_run: dict[RunKey, dict[int, CaseRecord]] = defaultdict(dict)
    for key, record in cases.items():
        cases_by_run[key.run][key.case_index] = record

    successful = manifest.get("successful_run_count")
    if successful is not None and int(successful) != len(cases_by_run):
        raise ComparisonError(
            f"{role} manifest reports {successful} successful runs but "
            f"case_metrics.csv contains {len(cases_by_run)} run keys"
        )
    planned_cases = manifest.get("planned_case_count")
    if planned_cases is not None and int(planned_cases) != len(cases):
        raise ComparisonError(
            f"{role} manifest reports {planned_cases} planned cases but "
            f"case_metrics.csv contains {len(cases)} case keys"
        )

    source_state_path = root / "diagnostics" / "source_state.json"
    source_state = _read_json(source_state_path) if source_state_path.is_file() else {}
    return ReleaseData(
        root=root,
        manifest=manifest,
        cases=cases,
        cases_by_run=dict(cases_by_run),
        source_state=source_state,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _ratio(baseline: float, candidate: float) -> Optional[float]:
    if baseline == 0.0:
        return 1.0 if candidate == 0.0 else None
    ratio = candidate / baseline
    return ratio if math.isfinite(ratio) else None


def _material_outcome(
    baseline: float, candidate: float, absolute_tolerance: float, relative_tolerance: float
) -> str:
    threshold = max(
        absolute_tolerance,
        relative_tolerance * max(abs(baseline), abs(candidate)),
    )
    delta = candidate - baseline
    if delta < -threshold:
        return "improved"
    if delta > threshold:
        return "worsened"
    return "stable"


def _stats_from_case_rows(
    rows: Sequence[dict], metric: str, tail_threshold: float
) -> dict:
    baseline_values = [
        row["baseline_value"]
        for row in rows
        if row["baseline_value"] is not None
    ]
    candidate_values = [
        row["candidate_value"]
        for row in rows
        if row["candidate_value"] is not None
    ]
    paired = [
        row
        for row in rows
        if row["baseline_value"] is not None
        and row["candidate_value"] is not None
    ]
    baseline_paired = [row["baseline_value"] for row in paired]
    candidate_paired = [row["candidate_value"] for row in paired]
    result = {
        "paired_case_count": len(paired),
        "baseline_value_count": len(baseline_values),
        "candidate_value_count": len(candidate_values),
        "baseline_median": None,
        "candidate_median": None,
        "delta_median": None,
        "candidate_to_baseline_median_ratio": None,
        "baseline_p95": None,
        "candidate_p95": None,
        "delta_p95": None,
        "baseline_max": None,
        "candidate_max": None,
        "delta_max": None,
        "material_improved_cases": 0,
        "material_worsened_cases": 0,
        "material_stable_cases": 0,
        "baseline_above_tail_count": None,
        "candidate_above_tail_count": None,
        "fixed_tail_cases": None,
        "introduced_tail_cases": None,
    }
    if not paired:
        return result

    baseline_median = float(statistics.median(baseline_paired))
    candidate_median = float(statistics.median(candidate_paired))
    baseline_p95 = _percentile(baseline_paired, 0.95)
    candidate_p95 = _percentile(candidate_paired, 0.95)
    outcomes = [row["outcome"] for row in paired]
    result.update(
        {
            "baseline_median": baseline_median,
            "candidate_median": candidate_median,
            "delta_median": candidate_median - baseline_median,
            "candidate_to_baseline_median_ratio": _ratio(
                baseline_median, candidate_median
            ),
            "baseline_p95": baseline_p95,
            "candidate_p95": candidate_p95,
            "delta_p95": candidate_p95 - baseline_p95,
            "baseline_max": max(baseline_paired),
            "candidate_max": max(candidate_paired),
            "delta_max": max(candidate_paired) - max(baseline_paired),
            "material_improved_cases": outcomes.count("improved"),
            "material_worsened_cases": outcomes.count("worsened"),
            "material_stable_cases": outcomes.count("stable"),
        }
    )
    if metric == "hausdorff":
        result.update(
            {
                "baseline_above_tail_count": sum(
                    value > tail_threshold for value in baseline_paired
                ),
                "candidate_above_tail_count": sum(
                    value > tail_threshold for value in candidate_paired
                ),
                "fixed_tail_cases": sum(
                    baseline > tail_threshold and candidate <= tail_threshold
                    for baseline, candidate in zip(
                        baseline_paired, candidate_paired
                    )
                ),
                "introduced_tail_cases": sum(
                    candidate > tail_threshold and baseline <= tail_threshold
                    for baseline, candidate in zip(
                        baseline_paired, candidate_paired
                    )
                ),
            }
        )
    return result


def _build_case_metric_rows(
    baseline: ReleaseData,
    candidate: ReleaseData,
    matched_case_keys: Iterable[CaseKey],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> list[dict]:
    rows = []
    for key in sorted(matched_case_keys):
        baseline_record = baseline.cases[key]
        candidate_record = candidate.cases[key]
        for metric in SCIENTIFIC_METRICS:
            baseline_value = baseline_record.metrics[metric]
            candidate_value = candidate_record.metrics[metric]
            if baseline_value is None and candidate_value is None:
                continue
            if baseline_value is None:
                status = "missing_baseline_value"
                outcome = "incomparable"
            elif candidate_value is None:
                status = "missing_candidate_value"
                outcome = "incomparable"
            else:
                status = "paired"
                outcome = _material_outcome(
                    baseline_value,
                    candidate_value,
                    absolute_tolerance,
                    relative_tolerance,
                )
            rows.append(
                {
                    **key.csv_values(),
                    "cells_per_side": key.run.cells_per_side,
                    "metric": metric,
                    "baseline_value": baseline_value,
                    "candidate_value": candidate_value,
                    "delta": (
                        None
                        if baseline_value is None or candidate_value is None
                        else candidate_value - baseline_value
                    ),
                    "candidate_to_baseline_ratio": (
                        None
                        if baseline_value is None or candidate_value is None
                        else _ratio(baseline_value, candidate_value)
                    ),
                    "outcome": outcome,
                    "value_status": status,
                }
            )
    return rows


def _group_metric_rows(
    case_metric_rows: Sequence[dict], fields: Sequence[str]
) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in case_metric_rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return dict(grouped)


def _build_metric_summaries(
    case_metric_rows: Sequence[dict], tail_threshold: float
) -> tuple[list[dict], list[dict]]:
    method_rows = []
    method_groups = _group_metric_rows(
        case_metric_rows, ("experiment", "algo", "metric")
    )
    for (experiment, algo, metric), rows in sorted(method_groups.items()):
        method_rows.append(
            {
                "experiment": experiment,
                "algo": algo,
                "metric": metric,
                **_stats_from_case_rows(rows, metric, tail_threshold),
            }
        )

    setting_rows = []
    setting_fields = (*RUN_FIELDS, "metric")
    setting_groups = _group_metric_rows(case_metric_rows, setting_fields)
    for values, rows in sorted(setting_groups.items()):
        row = dict(zip(setting_fields, values))
        run = RunKey(
            experiment=row["experiment"],
            algo=row["algo"],
            resolution=Decimal(str(row["resolution"])),
            wiggle=Decimal(str(row["wiggle"])),
            seed=int(row["seed"]),
        )
        setting_rows.append(
            {
                **row,
                "cells_per_side": run.cells_per_side,
                **_stats_from_case_rows(rows, row["metric"], tail_threshold),
            }
        )
    return method_rows, setting_rows


def _build_tail_rows(
    baseline: ReleaseData,
    candidate: ReleaseData,
    case_metric_rows: Sequence[dict],
    tail_threshold: float,
) -> list[dict]:
    tails = []
    for row in case_metric_rows:
        if row["metric"] != "hausdorff" or row["value_status"] != "paired":
            continue
        baseline_value = row["baseline_value"]
        candidate_value = row["candidate_value"]
        reasons = []
        if baseline_value > tail_threshold:
            reasons.append("baseline_tail")
        if candidate_value > tail_threshold:
            reasons.append("candidate_tail")
        if baseline_value <= tail_threshold < candidate_value:
            reasons.append("introduced_tail")
        if candidate_value <= tail_threshold < baseline_value:
            reasons.append("fixed_tail")
        if row["outcome"] == "worsened":
            reasons.append("material_regression")
        if not reasons:
            continue
        run = RunKey(
            experiment=row["experiment"],
            algo=row["algo"],
            resolution=Decimal(str(row["resolution"])),
            wiggle=Decimal(str(row["wiggle"])),
            seed=int(row["seed"]),
        )
        key = CaseKey(run=run, case_index=int(row["case_index"]))
        tails.append(
            {
                **key.csv_values(),
                "cells_per_side": run.cells_per_side,
                "baseline_hausdorff": baseline_value,
                "candidate_hausdorff": candidate_value,
                "delta_hausdorff": row["delta"],
                "candidate_to_baseline_ratio": row[
                    "candidate_to_baseline_ratio"
                ],
                "outcome": row["outcome"],
                "baseline_plic_fallback_cells": baseline.cases[
                    key
                ].plic_fallback_cells,
                "candidate_plic_fallback_cells": candidate.cases[
                    key
                ].plic_fallback_cells,
                "reasons": ";".join(reasons),
            }
        )
    return sorted(
        tails,
        key=lambda row: (
            "introduced_tail" in row["reasons"],
            row["candidate_hausdorff"],
            row["delta_hausdorff"],
        ),
        reverse=True,
    )


def _build_perfect_rows(
    baseline: ReleaseData,
    candidate: ReleaseData,
    matched_runs: Iterable[RunKey],
    threshold: float,
) -> list[dict]:
    rows = []
    for run in sorted(matched_runs):
        if PERFECT_RECONSTRUCTION_METHODS.get(run.experiment) != run.algo:
            continue
        case_indices = sorted(
            set(baseline.cases_by_run[run]) & set(candidate.cases_by_run[run])
        )
        pairs = [
            (baseline.cases_by_run[run][index], candidate.cases_by_run[run][index])
            for index in case_indices
        ]
        if any(
            record.metrics[metric] is None
            for pair in pairs
            for record in pair
            for metric in ("hausdorff", "facet_gap")
        ):
            continue
        baseline_h = [pair[0].metrics["hausdorff"] for pair in pairs]
        candidate_h = [pair[1].metrics["hausdorff"] for pair in pairs]
        baseline_g = [pair[0].metrics["facet_gap"] for pair in pairs]
        candidate_g = [pair[1].metrics["facet_gap"] for pair in pairs]
        baseline_h_median = float(statistics.median(baseline_h))
        candidate_h_median = float(statistics.median(candidate_h))
        baseline_g_median = float(statistics.median(baseline_g))
        candidate_g_median = float(statistics.median(candidate_g))
        baseline_joint_median = (
            baseline_h_median < threshold and baseline_g_median < threshold
        )
        candidate_joint_median = (
            candidate_h_median < threshold and candidate_g_median < threshold
        )
        if baseline_joint_median and candidate_joint_median:
            outcome = "retained"
        elif baseline_joint_median:
            outcome = "lost"
        elif candidate_joint_median:
            outcome = "gained"
        else:
            outcome = "neither"
        baseline_joint_cases = sum(
            h < threshold and g < threshold for h, g in zip(baseline_h, baseline_g)
        )
        candidate_joint_cases = sum(
            h < threshold and g < threshold for h, g in zip(candidate_h, candidate_g)
        )
        rows.append(
            {
                **run.csv_values(),
                "cells_per_side": run.cells_per_side,
                "case_count": len(pairs),
                "threshold": threshold,
                "baseline_hausdorff_median": baseline_h_median,
                "candidate_hausdorff_median": candidate_h_median,
                "baseline_hausdorff_p95": _percentile(baseline_h, 0.95),
                "candidate_hausdorff_p95": _percentile(candidate_h, 0.95),
                "baseline_hausdorff_max": max(baseline_h),
                "candidate_hausdorff_max": max(candidate_h),
                "baseline_facet_gap_median": baseline_g_median,
                "candidate_facet_gap_median": candidate_g_median,
                "baseline_facet_gap_max": max(baseline_g),
                "candidate_facet_gap_max": max(candidate_g),
                "baseline_hausdorff_median_below_threshold": (
                    baseline_h_median < threshold
                ),
                "candidate_hausdorff_median_below_threshold": (
                    candidate_h_median < threshold
                ),
                "baseline_joint_median_below_threshold": baseline_joint_median,
                "candidate_joint_median_below_threshold": candidate_joint_median,
                "baseline_joint_floor_case_count": baseline_joint_cases,
                "candidate_joint_floor_case_count": candidate_joint_cases,
                "baseline_all_cases_joint_floor": (
                    baseline_joint_cases == len(pairs)
                ),
                "candidate_all_cases_joint_floor": (
                    candidate_joint_cases == len(pairs)
                ),
                "threshold_outcome": outcome,
                "high_resolution": run.cells_per_side >= 100,
            }
        )
    return rows


def _perfect_totals(rows: Sequence[dict]) -> list[dict]:
    totals = []
    for experiment, algo in PERFECT_RECONSTRUCTION_METHODS.items():
        selected = [
            row
            for row in rows
            if row["experiment"] == experiment and row["algo"] == algo
        ]
        high = [row for row in selected if row["high_resolution"]]
        totals.append(
            {
                "experiment": experiment,
                "algo": algo,
                "setting_count": len(selected),
                "baseline_hausdorff_median_floor_settings": sum(
                    row["baseline_hausdorff_median_below_threshold"]
                    for row in selected
                ),
                "candidate_hausdorff_median_floor_settings": sum(
                    row["candidate_hausdorff_median_below_threshold"]
                    for row in selected
                ),
                "baseline_joint_median_floor_settings": sum(
                    row["baseline_joint_median_below_threshold"]
                    for row in selected
                ),
                "candidate_joint_median_floor_settings": sum(
                    row["candidate_joint_median_below_threshold"]
                    for row in selected
                ),
                "baseline_all_case_floor_settings": sum(
                    row["baseline_all_cases_joint_floor"] for row in selected
                ),
                "candidate_all_case_floor_settings": sum(
                    row["candidate_all_cases_joint_floor"] for row in selected
                ),
                "lost_joint_median_settings": sum(
                    row["threshold_outcome"] == "lost" for row in selected
                ),
                "gained_joint_median_settings": sum(
                    row["threshold_outcome"] == "gained" for row in selected
                ),
                "high_resolution_setting_count": len(high),
                "baseline_high_resolution_joint_median_floor_settings": sum(
                    row["baseline_joint_median_below_threshold"] for row in high
                ),
                "candidate_high_resolution_joint_median_floor_settings": sum(
                    row["candidate_joint_median_below_threshold"] for row in high
                ),
            }
        )
    return totals


def compare_releases(
    baseline: ReleaseData,
    candidate: ReleaseData,
    *,
    perfect_threshold: float = 1e-6,
    tail_threshold: float = 1.0,
    absolute_tolerance: float = 1e-10,
    relative_tolerance: float = 0.01,
) -> dict:
    if baseline.root == candidate.root:
        raise ComparisonError("baseline and candidate release roots must differ")
    if (
        not math.isfinite(perfect_threshold)
        or not math.isfinite(tail_threshold)
        or perfect_threshold <= 0
        or tail_threshold <= 0
    ):
        raise ComparisonError("perfect and tail thresholds must be positive")
    if (
        not math.isfinite(absolute_tolerance)
        or not math.isfinite(relative_tolerance)
        or absolute_tolerance < 0
        or relative_tolerance < 0
    ):
        raise ComparisonError("material-change tolerances must be nonnegative")

    baseline_runs = baseline.run_keys
    candidate_runs = candidate.run_keys
    matched_runs = baseline_runs & candidate_runs
    run_rows = []
    matched_case_keys = set()
    issues = []
    for run in sorted(baseline_runs | candidate_runs):
        baseline_cases = set(baseline.cases_by_run.get(run, {}))
        candidate_cases = set(candidate.cases_by_run.get(run, {}))
        overlap = baseline_cases & candidate_cases
        if baseline_cases and candidate_cases:
            if baseline_cases == candidate_cases:
                status = "matched"
            else:
                status = "case_key_mismatch"
                issues.append(
                    f"case indices differ for {run.experiment}/{run.algo} "
                    f"r={_decimal_text(run.resolution)} "
                    f"w={_decimal_text(run.wiggle)} s={run.seed}: "
                    f"baseline={len(baseline_cases)}, "
                    f"candidate={len(candidate_cases)}, matched={len(overlap)}"
                )
            matched_case_keys.update(
                CaseKey(run=run, case_index=index) for index in overlap
            )
        elif baseline_cases:
            status = "baseline_only"
        else:
            status = "candidate_only"
        run_rows.append(
            {
                **run.csv_values(),
                "cells_per_side": run.cells_per_side,
                "baseline_present": bool(baseline_cases),
                "candidate_present": bool(candidate_cases),
                "baseline_case_count": len(baseline_cases),
                "candidate_case_count": len(candidate_cases),
                "matched_case_count": len(overlap),
                "status": status,
            }
        )

    baseline_only = baseline_runs - candidate_runs
    if baseline_only:
        issues.append(
            f"candidate is missing {len(baseline_only)} run keys present in baseline"
        )
    if not matched_runs:
        issues.append("the releases have no matched run keys")

    case_metric_rows = _build_case_metric_rows(
        baseline,
        candidate,
        matched_case_keys,
        absolute_tolerance,
        relative_tolerance,
    )
    missing_required_values = [
        row
        for row in case_metric_rows
        if row["metric"] in {"hausdorff", "facet_gap"}
        and row["value_status"] != "paired"
    ]
    if missing_required_values:
        issues.append(
            f"{len(missing_required_values)} matched required metric values are missing"
        )

    method_rows, setting_rows = _build_metric_summaries(
        case_metric_rows, tail_threshold
    )
    tail_rows = _build_tail_rows(
        baseline, candidate, case_metric_rows, tail_threshold
    )
    perfect_rows = _build_perfect_rows(
        baseline, candidate, matched_runs, perfect_threshold
    )
    perfect_totals = _perfect_totals(perfect_rows)

    overall_groups = _group_metric_rows(case_metric_rows, ("metric",))
    overall_metrics = []
    for (metric,), rows in sorted(overall_groups.items()):
        overall_metrics.append(
            {"metric": metric, **_stats_from_case_rows(rows, metric, tail_threshold)}
        )

    return {
        "run_coverage": run_rows,
        "case_metric_comparison": case_metric_rows,
        "method_metric_comparison": method_rows,
        "setting_metric_comparison": setting_rows,
        "tail_cases": tail_rows,
        "perfect_reconstruction": perfect_rows,
        "summary": {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass" if not issues else "attention_required",
            "baseline": _release_summary(baseline),
            "candidate": _release_summary(candidate),
            "thresholds": {
                "perfect_reconstruction": perfect_threshold,
                "hausdorff_tail": tail_threshold,
                "material_absolute": absolute_tolerance,
                "material_relative": relative_tolerance,
            },
            "coverage": {
                "baseline_run_count": len(baseline_runs),
                "candidate_run_count": len(candidate_runs),
                "matched_run_count": len(matched_runs),
                "baseline_only_run_count": len(baseline_runs - candidate_runs),
                "candidate_only_run_count": len(candidate_runs - baseline_runs),
                "exact_case_grid_run_count": sum(
                    row["status"] == "matched" for row in run_rows
                ),
                "case_mismatch_run_count": sum(
                    row["status"] == "case_key_mismatch" for row in run_rows
                ),
                "matched_case_count": len(matched_case_keys),
            },
            "overall_metrics": overall_metrics,
            "method_metric_comparison": method_rows,
            "tail_case_count": len(tail_rows),
            "largest_hausdorff_tails": tail_rows[:20],
            "perfect_reconstruction": perfect_totals,
            "issues": issues,
            "notes": [
                "Candidate-only run keys are expected when comparing the 970-run "
                "all-method release with the 300-run July affected-method subset.",
                "July square and Zalesak `area_error` values predate the "
                "geometry-faithful repair; `area_error` pairs are exported for "
                "audit but are not submission-equivalent evidence.",
                "All error metrics are interpreted as smaller-is-better; tails use "
                "the paired 95th percentile, maximum, and Hausdorff threshold counts.",
            ],
        },
    }


def _release_summary(release: ReleaseData) -> dict:
    manifest = release.manifest
    parameters = manifest.get("parameters") or {}
    return {
        "root": str(release.root),
        "name": release.root.name,
        "status": manifest.get("status"),
        "planned_run_count": manifest.get("planned_run_count"),
        "successful_run_count": manifest.get("successful_run_count"),
        "planned_case_count": manifest.get("planned_case_count"),
        "case_row_count": len(release.cases),
        "source_commit": release.source_state.get("source_commit", ""),
        "source_dirty": release.source_state.get("source_dirty"),
        "profile": {
            "plic_fallback": parameters.get("plic_fallback"),
            "rescue_profile": parameters.get("rescue_profile"),
            "corner_behavior_profile": parameters.get("corner_behavior_profile"),
        },
    }


def _format_number(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3e}"


def _markdown_report(result: dict, tail_limit: int) -> str:
    summary = result["summary"]
    coverage = summary["coverage"]
    baseline = summary["baseline"]
    candidate = summary["candidate"]
    lines = [
        "# Submission release comparison",
        "",
        f"Status: **{summary['status'].replace('_', ' ')}**",
        "",
        f"- Baseline: `{baseline['root']}`",
        f"- Candidate: `{candidate['root']}`",
        f"- Matched coverage: `{coverage['matched_run_count']}` runs / "
        f"`{coverage['matched_case_count']}` cases; "
        f"`{coverage['candidate_only_run_count']}` candidate-only runs and "
        f"`{coverage['baseline_only_run_count']}` baseline-only runs.",
        f"- Exact case grids: `{coverage['exact_case_grid_run_count']}/"
        f"{coverage['matched_run_count']}` matched runs.",
        "",
        "Candidate-only runs are expected here because the final sweep contains all "
        "paper methods while July contains the affected-method subset.",
        "",
        "## Matched Hausdorff summary",
        "",
        "| Benchmark / method | Cases | Median baseline/candidate | "
        "p95 baseline/candidate | Max baseline/candidate | > tail baseline/candidate | I/W |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    hausdorff_rows = [
        row
        for row in result["method_metric_comparison"]
        if row["metric"] == "hausdorff"
    ]
    for row in hausdorff_rows:
        lines.append(
            f"| `{row['experiment']}/{row['algo']}` | {row['paired_case_count']} | "
            f"{_format_number(row['baseline_median'])} / "
            f"{_format_number(row['candidate_median'])} | "
            f"{_format_number(row['baseline_p95'])} / "
            f"{_format_number(row['candidate_p95'])} | "
            f"{_format_number(row['baseline_max'])} / "
            f"{_format_number(row['candidate_max'])} | "
            f"{row['baseline_above_tail_count']} / "
            f"{row['candidate_above_tail_count']} | "
            f"{row['material_improved_cases']} / "
            f"{row['material_worsened_cases']} |"
        )

    lines.extend(
        [
            "",
            "## Perfect-reconstruction checks",
            "",
            f"Threshold: both setting-median Hausdorff and facet gap below "
            f"`{summary['thresholds']['perfect_reconstruction']:.1e}`. The CSV also "
            "records Hausdorff-only and all-case joint checks.",
            "",
            "| Benchmark / method | Settings | H-median floor baseline/candidate | "
            "Joint-median floor baseline/candidate | "
            "All-case joint floor baseline/candidate | Lost/gained |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["perfect_reconstruction"]:
        lines.append(
            f"| `{row['experiment']}/{row['algo']}` | {row['setting_count']} | "
            f"{row['baseline_hausdorff_median_floor_settings']} / "
            f"{row['candidate_hausdorff_median_floor_settings']} | "
            f"{row['baseline_joint_median_floor_settings']} / "
            f"{row['candidate_joint_median_floor_settings']} | "
            f"{row['baseline_all_case_floor_settings']} / "
            f"{row['candidate_all_case_floor_settings']} | "
            f"{row['lost_joint_median_settings']} / "
            f"{row['gained_joint_median_settings']} |"
        )
        if row["experiment"] == "zalesak":
            lines.append(
                f"| `zalesak/circular+corner (N>=100)` | "
                f"{row['high_resolution_setting_count']} | n/a | "
                f"{row['baseline_high_resolution_joint_median_floor_settings']} / "
                f"{row['candidate_high_resolution_joint_median_floor_settings']} | "
                f"n/a | n/a |"
            )

    changed_thresholds = [
        row
        for row in result["perfect_reconstruction"]
        if row["threshold_outcome"] in {"lost", "gained"}
    ]
    if changed_thresholds:
        lines.extend(["", "Threshold changes:"])
        for row in changed_thresholds:
            lines.append(
                f"- `{row['experiment']}/{row['algo']}`, "
                f"N={row['cells_per_side']}, w={row['wiggle']}: "
                f"**{row['threshold_outcome']}** joint-median floor status."
            )

    selected_tails = result["tail_cases"][:tail_limit]
    lines.extend(
        [
            "",
            "## Largest matched tails and regressions",
            "",
            "| Benchmark / method | N | w | Case | H baseline/candidate | Delta | Reasons |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    if selected_tails:
        for row in selected_tails:
            lines.append(
                f"| `{row['experiment']}/{row['algo']}` | "
                f"{row['cells_per_side']} | {row['wiggle']} | "
                f"{row['case_index']} | "
                f"{_format_number(row['baseline_hausdorff'])} / "
                f"{_format_number(row['candidate_hausdorff'])} | "
                f"{_format_number(row['delta_hausdorff'])} | "
                f"`{row['reasons']}` |"
            )
    else:
        lines.append("| None | | | | | | |")

    lines.extend(["", "## Interpretation", ""])
    lines.extend(f"- {note}" for note in summary["notes"])
    if summary["issues"]:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- {issue}" for issue in summary["issues"])
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `comparison.json`: summary, thresholds, coverage, and issues",
            "- `run_coverage.csv`: matched and unmatched run keys",
            "- `method_metric_comparison.csv`: benchmark/method summaries",
            "- `setting_metric_comparison.csv`: resolution/wiggle/seed summaries",
            "- `case_metric_comparison.csv`: paired case-level metric values",
            "- `tail_cases.csv`: Hausdorff tails and material regressions",
            "- `perfect_reconstruction.csv`: square and Zalesak threshold checks",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_value(value: object) -> object:
    return "" if value is None else value


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def write_comparison(result: dict, output_dir: Path, tail_limit: int = 12) -> Path:
    output_dir = output_dir.resolve()
    baseline_root = Path(result["summary"]["baseline"]["root"]).resolve()
    candidate_root = Path(result["summary"]["candidate"]["root"]).resolve()
    if _is_within(output_dir, baseline_root) or _is_within(output_dir, candidate_root):
        raise ComparisonError(
            "output directory must be outside both immutable release roots"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ComparisonError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "run_coverage.csv": (RUN_COVERAGE_FIELDS, result["run_coverage"]),
        "method_metric_comparison.csv": (
            METHOD_METRIC_FIELDS,
            result["method_metric_comparison"],
        ),
        "setting_metric_comparison.csv": (
            SETTING_METRIC_FIELDS,
            result["setting_metric_comparison"],
        ),
        "case_metric_comparison.csv": (
            CASE_METRIC_FIELDS,
            result["case_metric_comparison"],
        ),
        "tail_cases.csv": (TAIL_CASE_FIELDS, result["tail_cases"]),
        "perfect_reconstruction.csv": (
            PERFECT_FIELDS,
            result["perfect_reconstruction"],
        ),
    }
    for name, (fieldnames, rows) in artifacts.items():
        _write_csv(output_dir / name, fieldnames, rows)

    summary = dict(result["summary"])
    summary["artifacts"] = sorted([*artifacts, "REPORT.md", "comparison.json"])
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(
        _markdown_report(result, tail_limit), encoding="utf-8"
    )
    return output_dir / "REPORT.md"


def _default_output_dir(baseline: Path, candidate: Path) -> Path:
    return (
        Path("results")
        / "submission"
        / "release_comparisons"
        / f"{candidate.name}_vs_{baseline.name}"
    )


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative finite number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare matched case metrics from a complete candidate release and a "
            "complete reference release. Incomplete/running releases are rejected."
        )
    )
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument(
        "--results-parent",
        type=Path,
        default=Path("results/static"),
        help="parent used to discover omitted release roots",
    )
    parser.add_argument("--baseline-glob", default=DEFAULT_BASELINE_GLOB)
    parser.add_argument("--candidate-glob", default=DEFAULT_CANDIDATE_GLOB)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--perfect-threshold", type=_positive_float, default=1e-6
    )
    parser.add_argument("--tail-threshold", type=_positive_float, default=1.0)
    parser.add_argument(
        "--material-absolute-tolerance", type=_nonnegative_float, default=1e-10
    )
    parser.add_argument(
        "--material-relative-tolerance", type=_nonnegative_float, default=0.01
    )
    parser.add_argument("--report-tail-limit", type=_positive_int, default=12)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        baseline_root = args.baseline_root or discover_release_root(
            args.results_parent, args.baseline_glob
        )
        candidate_root = args.candidate_root or discover_release_root(
            args.results_parent, args.candidate_glob
        )
        baseline = load_release(baseline_root, "baseline")
        candidate = load_release(candidate_root, "candidate")
        result = compare_releases(
            baseline,
            candidate,
            perfect_threshold=args.perfect_threshold,
            tail_threshold=args.tail_threshold,
            absolute_tolerance=args.material_absolute_tolerance,
            relative_tolerance=args.material_relative_tolerance,
        )
        output_dir = args.output_dir or _default_output_dir(
            baseline.root, candidate.root
        )
        report_path = write_comparison(
            result, output_dir, tail_limit=args.report_tail_limit
        )
    except (ComparisonError, OSError, csv.Error) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(report_path)
    return 0 if result["summary"]["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
