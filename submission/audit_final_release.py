#!/usr/bin/env python3
"""Fail-closed audit for a completed final static-result release."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import statistics
import tarfile
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Mapping, Sequence


FINAL_RUN_COUNT = 970
FINAL_CASE_COUNT = 24_250
DEFAULT_SHA256_MANIFEST = "SHA256SUMS"
MAX_REPORTED_ERRORS = 250

RUN_CONTEXT_FIELDS = (
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
)

REQUIRED_RELEASE_FILES = (
    "submission_config.resolved.json",
    "sweep_manifest.json",
    "environment.json",
    "failures.csv",
    "perturbed_sweep.csv",
    "diagnostics/source_state.json",
    "diagnostics/source_snapshot.tar.gz",
    "diagnostics/run_inventory.csv",
    "diagnostics/run_manifests.jsonl",
    "diagnostics/case_geometry.jsonl",
    "diagnostics/case_metrics.csv",
    "diagnostics/cell_metrics.csv",
    "diagnostics/merge_events.csv",
    "diagnostics/unresolved_plic_fallbacks.csv",
)

REQUIRED_RAW_ARTIFACTS = {
    "case_geometry": "metrics/case_geometry.jsonl",
    "case_metrics": "metrics/case_metrics.csv",
    "cell_metrics": "metrics/cell_metrics.csv",
    "fallback_events": "metrics/unresolved_plic_fallbacks.csv",
    "merge_events": "metrics/merge_events.csv",
    "mesh": "vtk/mesh.vtk",
}

METRICS_BY_EXPERIMENT = {
    "lines": ("hausdorff", "facet_gap"),
    "circles": (
        "curvature_error",
        "facet_gap",
        "hausdorff",
        "tangent_error",
        "curvature_proxy_error",
    ),
    "ellipses": (
        "curvature_error",
        "facet_gap",
        "hausdorff",
        "tangent_error",
        "curvature_proxy_error",
    ),
    "squares": ("area_error", "facet_gap", "hausdorff"),
    "zalesak": ("area_error", "facet_gap", "hausdorff"),
}

AGGREGATE_STATS = ("mean", "median", "p25", "p75")


class ReleaseAuditInputError(ValueError):
    """Raised when a manifest operation receives an unsafe input path."""


@dataclass(frozen=True, order=True)
class RunKey:
    experiment: str
    algo: str
    resolution: str
    wiggle: str
    seed: int

    def display(self) -> str:
        return (
            f"{self.experiment}/{self.algo}/r={self.resolution}/"
            f"w={self.wiggle}/s={self.seed}"
        )


@dataclass
class AuditReport:
    release_root: Path
    errors: list[str] = field(default_factory=list)
    total_errors: int = 0
    summaries: dict[str, int | str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.total_errors == 0

    def add_error(self, message: str) -> None:
        self.total_errors += 1
        if len(self.errors) < MAX_REPORTED_ERRORS:
            self.errors.append(message)

    @property
    def suppressed_errors(self) -> int:
        return self.total_errors - len(self.errors)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value!r}")


def _strict_json_loads(text: str, source: Path | str) -> object:
    try:
        return json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ReleaseAuditInputError(f"invalid JSON in {source}: {exc}") from exc


def _load_json(path: Path, report: AuditReport) -> dict | None:
    if not path.is_file():
        report.add_error(f"missing required JSON file: {path}")
        return None
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"), path)
    except (OSError, UnicodeError, ReleaseAuditInputError) as exc:
        report.add_error(str(exc))
        return None
    if not isinstance(value, dict):
        report.add_error(f"JSON root must be an object: {path}")
        return None
    return value


def _canonical_number(value: object) -> str:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ReleaseAuditInputError(f"not a number: {value!r}") from exc
    if not number.is_finite():
        raise ReleaseAuditInputError(f"nonfinite number: {value!r}")
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def _parse_int(value: object, label: str) -> int:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ReleaseAuditInputError(f"{label} is not an integer: {value!r}") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise ReleaseAuditInputError(f"{label} is not an integer: {value!r}")
    return int(number)


def _finite_metric(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReleaseAuditInputError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ReleaseAuditInputError(f"{label} is nonfinite: {value!r}")
    if number < 0:
        raise ReleaseAuditInputError(f"{label} is negative: {value!r}")
    return number


def _run_key(row: Mapping[str, object], source: str) -> RunKey:
    missing = [
        field
        for field in ("experiment", "algo", "resolution", "wiggle", "seed")
        if row.get(field) in (None, "")
    ]
    if missing:
        raise ReleaseAuditInputError(
            f"{source} is missing run-key fields: {', '.join(missing)}"
        )
    return RunKey(
        experiment=str(row["experiment"]).strip().lower(),
        algo=str(row["algo"]).strip().lower(),
        resolution=_canonical_number(row["resolution"]),
        wiggle=_canonical_number(row["wiggle"]),
        seed=_parse_int(row["seed"], f"{source} seed"),
    )


def _safe_release_path(root: Path, raw_path: object, label: str) -> Path:
    value = str(raw_path or "")
    pure = PurePosixPath(value)
    if not value or pure.is_absolute() or ".." in pure.parts:
        raise ReleaseAuditInputError(f"{label} is not release-relative: {value!r}")
    path = root.joinpath(*pure.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise ReleaseAuditInputError(
            f"{label} escapes the release root: {value!r}"
        ) from exc
    return path


def _require_headers(
    fieldnames: Sequence[str] | None,
    required: Iterable[str],
    path: Path,
    report: AuditReport,
) -> bool:
    if fieldnames is None:
        report.add_error(f"CSV has no header: {path}")
        return False
    missing = sorted(set(required) - set(fieldnames))
    if missing:
        report.add_error(f"CSV {path} is missing columns: {', '.join(missing)}")
        return False
    return True


def _read_csv_rows(
    path: Path,
    required_headers: Iterable[str],
    report: AuditReport,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        report.add_error(f"missing required CSV file: {path}")
        return [], []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
            if not _require_headers(fieldnames, required_headers, path, report):
                return fieldnames, []
            return fieldnames, list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        report.add_error(f"could not read CSV {path}: {exc}")
        return [], []


def _iter_jsonl(path: Path, report: AuditReport) -> Iterator[tuple[int, dict]]:
    if not path.is_file():
        report.add_error(f"missing required JSONL file: {path}")
        return
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                try:
                    value = _strict_json_loads(raw_line, f"{path}:{line_number}")
                except ReleaseAuditInputError as exc:
                    report.add_error(str(exc))
                    continue
                if not isinstance(value, dict):
                    report.add_error(
                        f"JSONL row must be an object: {path}:{line_number}"
                    )
                    continue
                yield line_number, value
    except (OSError, UnicodeError) as exc:
        report.add_error(f"could not read JSONL {path}: {exc}")


def _expected_grid(
    config: Mapping[str, object], report: AuditReport
) -> tuple[set[RunKey], int]:
    try:
        grid = config["benchmark_grid"]
        benchmarks = config["benchmarks"]
        if not isinstance(grid, dict) or not isinstance(benchmarks, dict):
            raise TypeError("benchmark_grid and benchmarks must be objects")
        wiggles = list(grid["wiggles"])
        seed = _parse_int(grid["seed"], "benchmark_grid.seed")
        trials = _parse_int(
            grid["trials_per_setting"], "benchmark_grid.trials_per_setting"
        )
    except (KeyError, TypeError, ReleaseAuditInputError) as exc:
        report.add_error(f"invalid benchmark grid in resolved config: {exc}")
        return set(), 0

    expected: set[RunKey] = set()
    for experiment, raw_benchmark in benchmarks.items():
        if not isinstance(raw_benchmark, dict):
            report.add_error(f"benchmark {experiment!r} is not an object")
            continue
        try:
            resolution_key = str(raw_benchmark["resolutions"])
            resolutions = list(grid[resolution_key])
            methods = list(raw_benchmark["methods"])
        except (KeyError, TypeError) as exc:
            report.add_error(f"invalid benchmark {experiment!r}: {exc}")
            continue
        computed = len(resolutions) * len(wiggles) * len(methods)
        try:
            configured = _parse_int(
                raw_benchmark["planned_runs"], f"{experiment}.planned_runs"
            )
        except (KeyError, ReleaseAuditInputError) as exc:
            report.add_error(f"invalid benchmark {experiment!r}: {exc}")
            continue
        if computed != configured:
            report.add_error(
                f"benchmark {experiment} computes {computed} runs but config records "
                f"{configured}"
            )
        for resolution in resolutions:
            for wiggle in wiggles:
                for method in methods:
                    key = RunKey(
                        str(experiment).lower(),
                        str(method).lower(),
                        _canonical_number(resolution),
                        _canonical_number(wiggle),
                        seed,
                    )
                    if key in expected:
                        report.add_error(f"duplicate configured run key: {key.display()}")
                    expected.add(key)
    return expected, trials


def _check_exact_counts(
    config: Mapping[str, object],
    expected_runs: int,
    trials: int,
    required_runs: int,
    required_cases: int,
    report: AuditReport,
) -> None:
    computed_cases = expected_runs * trials
    if expected_runs != required_runs:
        report.add_error(
            f"resolved config defines {expected_runs} runs; exactly {required_runs} required"
        )
    if computed_cases != required_cases:
        report.add_error(
            f"resolved config defines {computed_cases} cases; exactly {required_cases} required"
        )
    try:
        totals = config["planned_totals"]
        configured_runs = _parse_int(totals["runs"], "planned_totals.runs")
        configured_cases = _parse_int(totals["cases"], "planned_totals.cases")
    except (KeyError, TypeError, ReleaseAuditInputError) as exc:
        report.add_error(f"invalid planned_totals in resolved config: {exc}")
        return
    if configured_runs != required_runs:
        report.add_error(
            f"planned_totals.runs is {configured_runs}; exactly {required_runs} required"
        )
    if configured_cases != required_cases:
        report.add_error(
            f"planned_totals.cases is {configured_cases}; exactly {required_cases} required"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_source_provenance(
    root: Path,
    config: dict,
    report: AuditReport,
) -> str:
    target_commit = str(config.get("source", {}).get("target_commit", ""))
    target_branch = str(config.get("source", {}).get("target_branch", ""))
    if config.get("status") != "frozen":
        report.add_error("resolved config status is not 'frozen'")
    if config.get("launch_approved") is not True:
        report.add_error("resolved config launch_approved is not true")
    if not target_commit:
        report.add_error("resolved config source.target_commit is empty")

    state_path = root / "diagnostics" / "source_state.json"
    state = _load_json(state_path, report)
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    environment = _load_json(root / "environment.json", report)
    if state:
        if state.get("source_commit") != target_commit:
            report.add_error("source_state commit does not match resolved config")
        if state.get("source_branch") != target_branch:
            report.add_error("source_state branch does not match resolved config")
        if state.get("source_dirty") is not False or state.get("source_status"):
            report.add_error("source_state does not record a clean source tree")
        if snapshot_path.is_file():
            try:
                actual_digest = _sha256(snapshot_path)
            except OSError as exc:
                report.add_error(f"could not hash source snapshot: {exc}")
            else:
                if state.get("snapshot_sha256") != actual_digest:
                    report.add_error("source snapshot SHA-256 does not match source_state")

    if environment:
        repository = environment.get("repository")
        if not isinstance(repository, dict):
            report.add_error("environment.json has no repository object")
        else:
            if repository.get("commit") != target_commit:
                report.add_error("environment commit does not match resolved config")
            if repository.get("branch") != target_branch:
                report.add_error("environment branch does not match resolved config")
            if repository.get("source_dirty") is not False:
                report.add_error("environment records a dirty source tree")

    snapshot_members: dict[str, bytes] = {}
    if snapshot_path.is_file():
        try:
            with tarfile.open(snapshot_path, "r:gz") as archive:
                for member in archive.getmembers():
                    pure = PurePosixPath(member.name)
                    if pure.is_absolute() or ".." in pure.parts:
                        report.add_error(
                            f"unsafe path in source snapshot: {member.name!r}"
                        )
                        continue
                    if not member.isfile():
                        continue
                    if member.name in snapshot_members:
                        report.add_error(
                            f"duplicate file in source snapshot: {member.name}"
                        )
                        continue
                    stream = archive.extractfile(member)
                    if stream is None:
                        report.add_error(
                            f"could not read source snapshot member: {member.name}"
                        )
                        continue
                    snapshot_members[member.name] = stream.read()
        except (OSError, tarfile.TarError) as exc:
            report.add_error(f"could not read source snapshot: {exc}")
    if state:
        try:
            recorded_file_count = _parse_int(
                state.get("snapshot_file_count"), "source_state snapshot_file_count"
            )
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
        else:
            if recorded_file_count != len(snapshot_members):
                report.add_error(
                    "source snapshot file count does not match source_state"
                )

    original_config_bytes = snapshot_members.get("submission/submission_config.json")
    if original_config_bytes is None:
        report.add_error("source snapshot lacks submission/submission_config.json")
    else:
        try:
            original_config = _strict_json_loads(
                original_config_bytes.decode("utf-8"),
                "source snapshot submission/submission_config.json",
            )
        except (UnicodeError, ReleaseAuditInputError) as exc:
            report.add_error(str(exc))
        else:
            if not isinstance(original_config, dict):
                report.add_error("source snapshot submission config is not an object")
            else:
                expected_resolved = copy.deepcopy(original_config)
                expected_resolved["status"] = "frozen"
                expected_resolved.setdefault("source", {})["target_commit"] = target_commit
                if expected_resolved != config:
                    report.add_error(
                        "resolved config differs from the snapshotted config beyond "
                        "status and source.target_commit"
                    )

    if environment:
        fingerprints = environment.get("input_fingerprints")
        if not isinstance(fingerprints, list):
            report.add_error("environment input_fingerprints is not a list")
        else:
            seen_fingerprints: set[str] = set()
            for index, fingerprint in enumerate(fingerprints):
                if not isinstance(fingerprint, dict):
                    report.add_error(
                        f"environment fingerprint {index} is not an object"
                    )
                    continue
                relative = str(fingerprint.get("path", ""))
                if relative in seen_fingerprints:
                    report.add_error(f"duplicate environment fingerprint: {relative}")
                    continue
                seen_fingerprints.add(relative)
                data = snapshot_members.get(relative)
                if data is None:
                    report.add_error(
                        f"environment fingerprint path is absent from source snapshot: "
                        f"{relative}"
                    )
                    continue
                digest = hashlib.sha256(data).hexdigest()
                if fingerprint.get("sha256") != digest:
                    report.add_error(
                        f"environment fingerprint digest mismatch: {relative}"
                    )
                try:
                    recorded_size = _parse_int(
                        fingerprint.get("size_bytes"),
                        f"environment fingerprint size for {relative}",
                    )
                except ReleaseAuditInputError as exc:
                    report.add_error(str(exc))
                else:
                    if recorded_size != len(data):
                        report.add_error(
                            f"environment fingerprint size mismatch: {relative}"
                        )
            if "submission/submission_config.json" not in seen_fingerprints:
                report.add_error(
                    "environment lacks a fingerprint for submission/submission_config.json"
                )
    return target_commit


def _check_controller(
    root: Path,
    required_runs: int,
    required_cases: int,
    report: AuditReport,
) -> None:
    manifest = _load_json(root / "sweep_manifest.json", report)
    if manifest:
        expected_values = {
            "planned_run_count": required_runs,
            "planned_case_count": required_cases,
            "successful_run_count": required_runs,
            "failure_count": 0,
        }
        if manifest.get("status") != "completed":
            report.add_error(
                f"controller status is {manifest.get('status')!r}, not 'completed'"
            )
        for field_name, expected in expected_values.items():
            try:
                actual = _parse_int(manifest.get(field_name), field_name)
            except ReleaseAuditInputError as exc:
                report.add_error(str(exc))
                continue
            if actual != expected:
                report.add_error(
                    f"controller {field_name} is {actual}; expected {expected}"
                )
        failures = manifest.get("failures")
        if failures != []:
            report.add_error("controller manifest failures list is not empty")

    _, failure_rows = _read_csv_rows(
        root / "failures.csv",
        ("experiment", "algo", "resolution", "wiggle", "seed", "save_name"),
        report,
    )
    if failure_rows:
        report.add_error(f"failures.csv contains {len(failure_rows)} controller failures")


def _check_context_source(
    row: Mapping[str, object],
    target_commit: str,
    target_branch: str,
    label: str,
    report: AuditReport,
) -> None:
    if row.get("source_commit") != target_commit:
        report.add_error(f"{label} source_commit does not match final source commit")
    if row.get("source_branch") != target_branch:
        report.add_error(f"{label} source_branch does not match final source branch")


def _check_inventory(
    root: Path,
    expected_runs: set[RunKey],
    trials: int,
    target_commit: str,
    target_branch: str,
    production: Mapping[str, object],
    report: AuditReport,
) -> dict[RunKey, dict[str, str]]:
    path = root / "diagnostics" / "run_inventory.csv"
    required = set(RUN_CONTEXT_FIELDS) | {
        "run_bundle",
        "case_geometry_rows",
        "case_metrics_rows",
        "cell_metrics_rows",
        "merge_events_rows",
        "unresolved_plic_fallbacks_rows",
    }
    _, rows = _read_csv_rows(path, required, report)
    inventory: dict[RunKey, dict[str, str]] = {}
    save_names: set[str] = set()
    bundle_paths: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        label = f"{path}:{row_number}"
        try:
            key = _run_key(row, label)
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        if key in inventory:
            report.add_error(f"duplicate run key in inventory: {key.display()}")
            continue
        inventory[key] = row
        if key not in expected_runs:
            report.add_error(f"unexpected run key in inventory: {key.display()}")
        save_name = row.get("save_name", "")
        if not save_name:
            report.add_error(f"inventory run has empty save_name: {key.display()}")
        elif save_name in save_names:
            report.add_error(f"duplicate inventory save_name: {save_name}")
        save_names.add(save_name)
        bundle_value = row.get("run_bundle", "")
        if bundle_value in bundle_paths:
            report.add_error(f"duplicate inventory run_bundle: {bundle_value}")
        bundle_paths.add(bundle_value)
        try:
            bundle = _safe_release_path(root, bundle_value, f"{label} run_bundle")
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
        else:
            expected_bundle = root / "raw_runs" / str(save_name)
            if bundle != expected_bundle:
                report.add_error(
                    f"inventory bundle for {key.display()} must be "
                    f"raw_runs/{save_name}"
                )
            if not bundle.is_dir():
                report.add_error(f"inventory raw bundle is missing: {bundle}")
        _check_context_source(row, target_commit, target_branch, label, report)
        if row.get("plic_fallback") != production.get(
            "unresolved_orientation_fallback"
        ):
            report.add_error(f"inventory fallback differs from production: {key.display()}")
        if row.get("corner_behavior_profile") != production.get(
            "corner_behavior_profile"
        ):
            report.add_error(
                f"inventory corner profile differs from production: {key.display()}"
            )
        for count_field in (
            "case_geometry_rows",
            "case_metrics_rows",
            "cell_metrics_rows",
            "merge_events_rows",
            "unresolved_plic_fallbacks_rows",
        ):
            try:
                count = _parse_int(row.get(count_field), f"{label} {count_field}")
            except ReleaseAuditInputError as exc:
                report.add_error(str(exc))
                continue
            if count < 0:
                report.add_error(f"{label} {count_field} is negative")
            if count_field in {"case_geometry_rows", "case_metrics_rows"} and count != trials:
                report.add_error(
                    f"{label} {count_field} is {count}; expected {trials}"
                )

    missing = expected_runs - set(inventory)
    for key in sorted(missing):
        report.add_error(f"missing run key from inventory: {key.display()}")
    report.summaries["inventory_runs"] = len(rows)
    return inventory


def _check_consolidated_run_manifests(
    root: Path,
    expected_runs: set[RunKey],
    inventory: Mapping[RunKey, Mapping[str, str]],
    target_commit: str,
    target_branch: str,
    report: AuditReport,
) -> None:
    path = root / "diagnostics" / "run_manifests.jsonl"
    seen: set[RunKey] = set()
    save_names: set[str] = set()
    for line_number, row in _iter_jsonl(path, report):
        label = f"{path}:{line_number}"
        try:
            key = _run_key(row, label)
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        if key in seen:
            report.add_error(f"duplicate run key in run_manifests: {key.display()}")
        seen.add(key)
        if key not in expected_runs:
            report.add_error(f"unexpected run manifest key: {key.display()}")
        save_name = str(row.get("save_name", ""))
        if save_name in save_names:
            report.add_error(f"duplicate save_name in run_manifests: {save_name}")
        save_names.add(save_name)
        inventory_row = inventory.get(key)
        if inventory_row and save_name != inventory_row.get("save_name"):
            report.add_error(f"run manifest save_name disagrees with inventory: {key.display()}")
        _check_context_source(row, target_commit, target_branch, label, report)
        manifest = row.get("manifest")
        if not isinstance(manifest, dict):
            report.add_error(f"{label} has no nested manifest object")
            continue
        if manifest.get("source_commit") != target_commit:
            report.add_error(f"nested run manifest commit mismatch: {key.display()}")
        if manifest.get("source_branch") != target_branch:
            report.add_error(f"nested run manifest branch mismatch: {key.display()}")
        if str(manifest.get("experiment", "")).lower() != key.experiment:
            report.add_error(f"nested run manifest experiment mismatch: {key.display()}")
        parameters = manifest.get("parameters")
        if not isinstance(parameters, dict):
            report.add_error(f"nested run manifest parameters missing: {key.display()}")
            continue
        try:
            nested_key = RunKey(
                key.experiment,
                str(parameters.get("facet_algo", "")).lower(),
                _canonical_number(parameters.get("resolution")),
                _canonical_number(parameters.get("perturb_wiggle")),
                _parse_int(parameters.get("perturb_seed"), "perturb_seed"),
            )
        except ReleaseAuditInputError as exc:
            report.add_error(f"invalid nested run manifest for {key.display()}: {exc}")
        else:
            if nested_key != key:
                report.add_error(f"nested run manifest key mismatch: {key.display()}")

    for key in sorted(expected_runs - seen):
        report.add_error(f"missing consolidated run manifest: {key.display()}")


def _case_key(
    row: Mapping[str, object], source: str
) -> tuple[RunKey, int]:
    key = _run_key(row, source)
    if row.get("case_index") in (None, ""):
        raise ReleaseAuditInputError(f"{source} has no case_index")
    return key, _parse_int(row["case_index"], f"{source} case_index")


def _check_case_metrics(
    root: Path,
    expected_runs: set[RunKey],
    trials: int,
    target_commit: str,
    target_branch: str,
    report: AuditReport,
) -> dict[tuple[RunKey, str], list[float]]:
    path = root / "diagnostics" / "case_metrics.csv"
    all_metrics = {metric for metrics in METRICS_BY_EXPERIMENT.values() for metric in metrics}
    required = set(RUN_CONTEXT_FIELDS) | {
        "case_index",
        "num_mixed_cells",
        "num_final_missing_cells",
    } | all_metrics
    _, rows = _read_csv_rows(path, required, report)
    seen: set[tuple[RunKey, int]] = set()
    values: dict[tuple[RunKey, str], list[float]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=2):
        label = f"{path}:{row_number}"
        try:
            key, case_index = _case_key(row, label)
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        case_key = (key, case_index)
        if case_key in seen:
            report.add_error(
                f"duplicate case key in case_metrics: {key.display()}/case={case_index}"
            )
            continue
        seen.add(case_key)
        if key not in expected_runs or not 0 <= case_index < trials:
            report.add_error(
                f"unexpected case key in case_metrics: {key.display()}/case={case_index}"
            )
        _check_context_source(row, target_commit, target_branch, label, report)
        try:
            mixed_cells = _parse_int(row.get("num_mixed_cells"), f"{label} num_mixed_cells")
            missing_cells = _parse_int(
                row.get("num_final_missing_cells"), f"{label} num_final_missing_cells"
            )
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
        else:
            if mixed_cells <= 0:
                report.add_error(f"{label} has no mixed cells")
            if missing_cells != 0:
                report.add_error(f"{label} reports {missing_cells} final missing facets")
        for metric in METRICS_BY_EXPERIMENT.get(key.experiment, ()):
            try:
                value = _finite_metric(row.get(metric), f"{label} {metric}")
            except ReleaseAuditInputError as exc:
                report.add_error(str(exc))
            else:
                values[(key, metric)].append(value)

    expected_cases = {
        (run_key, case_index)
        for run_key in expected_runs
        for case_index in range(trials)
    }
    for key, case_index in sorted(expected_cases - seen):
        report.add_error(
            f"missing case_metrics key: {key.display()}/case={case_index}"
        )
    report.summaries["case_metric_rows"] = len(rows)
    return values


def _check_case_geometry(
    root: Path,
    expected_runs: set[RunKey],
    trials: int,
    target_commit: str,
    target_branch: str,
    report: AuditReport,
) -> None:
    path = root / "diagnostics" / "case_geometry.jsonl"
    seen: set[tuple[RunKey, int]] = set()
    row_count = 0
    for line_number, row in _iter_jsonl(path, report):
        row_count += 1
        label = f"{path}:{line_number}"
        try:
            key, case_index = _case_key(row, label)
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        case_key = (key, case_index)
        if case_key in seen:
            report.add_error(
                f"duplicate case key in case_geometry: {key.display()}/case={case_index}"
            )
        seen.add(case_key)
        if key not in expected_runs or not 0 <= case_index < trials:
            report.add_error(
                f"unexpected case key in case_geometry: {key.display()}/case={case_index}"
            )
        if not row.get("geometry_type"):
            report.add_error(f"{label} has no geometry_type")
        _check_context_source(row, target_commit, target_branch, label, report)
    expected_cases = {
        (run_key, case_index)
        for run_key in expected_runs
        for case_index in range(trials)
    }
    for key, case_index in sorted(expected_cases - seen):
        report.add_error(
            f"missing case_geometry key: {key.display()}/case={case_index}"
        )
    report.summaries["case_geometry_rows"] = row_count


def _count_csv_rows(
    path: Path,
    required_headers: Iterable[str],
    report: AuditReport,
    *,
    validate_cell_rows: bool = False,
    valid_cases: set[int] | None = None,
) -> int:
    if not path.is_file():
        report.add_error(f"missing required CSV file: {path}")
        return 0
    count = 0
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not _require_headers(reader.fieldnames, required_headers, path, report):
                return 0
            for row_number, row in enumerate(reader, start=2):
                count += 1
                if valid_cases is not None:
                    try:
                        case_index = _parse_int(
                            row.get("case_index"), f"{path}:{row_number} case_index"
                        )
                    except ReleaseAuditInputError as exc:
                        report.add_error(str(exc))
                    else:
                        if case_index not in valid_cases:
                            report.add_error(
                                f"{path}:{row_number} references unexpected case {case_index}"
                            )
                if validate_cell_rows:
                    facet_class = row.get("final_facet_class", "")
                    if facet_class in ("", "missing"):
                        report.add_error(
                            f"{path}:{row_number} has no final reconstructed facet"
                        )
                    if not row.get("facet_geometry_json"):
                        report.add_error(
                            f"{path}:{row_number} has no facet geometry metadata"
                        )
    except (OSError, UnicodeError, csv.Error) as exc:
        report.add_error(f"could not read CSV {path}: {exc}")
    return count


def _check_consolidated_table_counts(
    root: Path,
    inventory: Mapping[RunKey, Mapping[str, str]],
    report: AuditReport,
) -> None:
    diagnostics = root / "diagnostics"
    table_specs = {
        "cell_metrics_rows": (
            diagnostics / "cell_metrics.csv",
            ("case_index", "cell_id", "final_facet_class", "facet_geometry_json"),
            True,
        ),
        "merge_events_rows": (
            diagnostics / "merge_events.csv",
            ("case_index", "event_order", "event_kind"),
            False,
        ),
        "unresolved_plic_fallbacks_rows": (
            diagnostics / "unresolved_plic_fallbacks.csv",
            ("case_index", "merge_id", "policy"),
            False,
        ),
    }
    for count_field, (path, headers, validate_cells) in table_specs.items():
        expected = 0
        for key, row in inventory.items():
            try:
                expected += _parse_int(row.get(count_field), f"{key.display()} {count_field}")
            except ReleaseAuditInputError:
                continue
        actual = _count_csv_rows(
            path,
            set(RUN_CONTEXT_FIELDS) | set(headers),
            report,
            validate_cell_rows=validate_cells,
        )
        if actual != expected:
            report.add_error(
                f"consolidated {path.name} has {actual} rows; inventory records {expected}"
            )
        report.summaries[count_field] = actual


def _check_raw_case_rows(
    bundle: Path,
    key: RunKey,
    trials: int,
    report: AuditReport,
) -> tuple[int, int, list[dict]]:
    metrics_path = bundle / "metrics" / "case_metrics.csv"
    required_metrics = METRICS_BY_EXPERIMENT.get(key.experiment, ())
    _, metric_rows = _read_csv_rows(
        metrics_path,
        {"case_index", "num_final_missing_cells", *required_metrics},
        report,
    )
    seen_metrics: set[int] = set()
    for row_number, row in enumerate(metric_rows, start=2):
        label = f"{metrics_path}:{row_number}"
        try:
            case_index = _parse_int(row.get("case_index"), f"{label} case_index")
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        if case_index in seen_metrics:
            report.add_error(f"duplicate raw case metric index in {bundle}: {case_index}")
        seen_metrics.add(case_index)
        if not 0 <= case_index < trials:
            report.add_error(f"unexpected raw case metric index in {bundle}: {case_index}")
        for metric in required_metrics:
            try:
                _finite_metric(row.get(metric), f"{label} {metric}")
            except ReleaseAuditInputError as exc:
                report.add_error(str(exc))
        try:
            missing = _parse_int(
                row.get("num_final_missing_cells"), f"{label} num_final_missing_cells"
            )
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
        else:
            if missing != 0:
                report.add_error(f"{label} reports {missing} final missing facets")
    if seen_metrics != set(range(trials)):
        report.add_error(f"raw case_metrics coverage is incomplete in {bundle}")

    geometry_path = bundle / "metrics" / "case_geometry.jsonl"
    geometry_rows: list[dict] = []
    seen_geometry: set[int] = set()
    for line_number, row in _iter_jsonl(geometry_path, report):
        geometry_rows.append(row)
        try:
            case_index = _parse_int(
                row.get("case_index"), f"{geometry_path}:{line_number} case_index"
            )
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        if case_index in seen_geometry:
            report.add_error(f"duplicate raw case geometry index in {bundle}: {case_index}")
        seen_geometry.add(case_index)
        if not 0 <= case_index < trials:
            report.add_error(f"unexpected raw case geometry index in {bundle}: {case_index}")
    if seen_geometry != set(range(trials)):
        report.add_error(f"raw case_geometry coverage is incomplete in {bundle}")
    return len(metric_rows), len(geometry_rows), geometry_rows


def _require_nonempty_file(path: Path, label: str, report: AuditReport) -> None:
    if not path.is_file():
        report.add_error(f"missing {label}: {path}")
        return
    try:
        if path.stat().st_size == 0:
            report.add_error(f"empty {label}: {path}")
    except OSError as exc:
        report.add_error(f"could not inspect {label} {path}: {exc}")


def _check_raw_bundle(
    root: Path,
    bundle: Path,
    key: RunKey,
    inventory_row: Mapping[str, str],
    trials: int,
    target_commit: str,
    target_branch: str,
    report: AuditReport,
) -> None:
    for path in bundle.rglob("*"):
        if path.is_symlink():
            report.add_error(f"raw bundle contains a symbolic link: {path}")
        if path.is_dir() and path.name == "plt":
            report.add_error(f"raw bundle contains temporary raster previews: {path}")

    manifest_path = bundle / "run_manifest.json"
    manifest = _load_json(manifest_path, report)
    if manifest:
        if manifest.get("source_commit") != target_commit:
            report.add_error(f"raw run manifest commit mismatch: {key.display()}")
        if manifest.get("source_branch") != target_branch:
            report.add_error(f"raw run manifest branch mismatch: {key.display()}")
        if str(manifest.get("experiment", "")).lower() != key.experiment:
            report.add_error(f"raw run manifest experiment mismatch: {key.display()}")
        parameters = manifest.get("parameters")
        if not isinstance(parameters, dict):
            report.add_error(f"raw run manifest parameters missing: {key.display()}")
        else:
            try:
                manifest_key = RunKey(
                    key.experiment,
                    str(parameters.get("facet_algo", "")).lower(),
                    _canonical_number(parameters.get("resolution")),
                    _canonical_number(parameters.get("perturb_wiggle")),
                    _parse_int(parameters.get("perturb_seed"), "perturb_seed"),
                )
            except ReleaseAuditInputError as exc:
                report.add_error(f"invalid raw run manifest for {key.display()}: {exc}")
            else:
                if manifest_key != key:
                    report.add_error(f"raw run manifest key mismatch: {key.display()}")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            report.add_error(f"raw run manifest artifacts missing: {key.display()}")
        else:
            for artifact_name, expected_relative in REQUIRED_RAW_ARTIFACTS.items():
                if artifacts.get(artifact_name) != expected_relative:
                    report.add_error(
                        f"raw artifact mapping {artifact_name!r} is inconsistent for "
                        f"{key.display()}"
                    )

    for artifact_name, relative in REQUIRED_RAW_ARTIFACTS.items():
        _require_nonempty_file(bundle / relative, artifact_name, report)

    raw_case_count, raw_geometry_count, geometry_rows = _check_raw_case_rows(
        bundle, key, trials, report
    )
    actual_counts = {
        "case_metrics_rows": raw_case_count,
        "case_geometry_rows": raw_geometry_count,
        "cell_metrics_rows": _count_csv_rows(
            bundle / "metrics" / "cell_metrics.csv",
            ("case_index", "cell_id", "final_facet_class", "facet_geometry_json"),
            report,
            validate_cell_rows=True,
            valid_cases=set(range(trials)),
        ),
        "merge_events_rows": _count_csv_rows(
            bundle / "metrics" / "merge_events.csv",
            ("case_index", "event_order", "event_kind"),
            report,
            valid_cases=set(range(trials)),
        ),
        "unresolved_plic_fallbacks_rows": _count_csv_rows(
            bundle / "metrics" / "unresolved_plic_fallbacks.csv",
            ("case_index", "merge_id", "policy"),
            report,
            valid_cases=set(range(trials)),
        ),
    }
    for field_name, actual in actual_counts.items():
        try:
            recorded = _parse_int(
                inventory_row.get(field_name), f"inventory {field_name} for {key.display()}"
            )
        except ReleaseAuditInputError:
            continue
        if actual != recorded:
            report.add_error(
                f"raw {field_name} for {key.display()} is {actual}; inventory records "
                f"{recorded}"
            )

    for metric in METRICS_BY_EXPERIMENT.get(key.experiment, ()):
        _require_nonempty_file(bundle / "metrics" / f"{metric}.txt", metric, report)

    for geometry in geometry_rows:
        try:
            case_index = _parse_int(
                geometry.get("case_index"), f"raw geometry case_index in {bundle}"
            )
        except ReleaseAuditInputError:
            continue
        for truth_field in ("truth_vtp", "truth_metadata"):
            if truth_field not in geometry:
                continue
            try:
                truth_path = _safe_release_path(
                    bundle, geometry[truth_field], f"{key.display()} {truth_field}"
                )
            except ReleaseAuditInputError as exc:
                report.add_error(str(exc))
            else:
                _require_nonempty_file(truth_path, truth_field, report)
        reconstructed = (
            bundle / "vtk" / "reconstructed"
        )
        _require_nonempty_file(
            reconstructed / "facets" / f"{case_index}.vtp",
            "reconstructed facets",
            report,
        )
        _require_nonempty_file(
            reconstructed / "facets" / f"{case_index}.facet_metadata.json",
            "reconstructed facet metadata",
            report,
        )
        _require_nonempty_file(
            reconstructed / "mixed_cells" / f"{case_index}.vtp",
            "reconstructed mixed cells",
            report,
        )


def _check_raw_bundles(
    root: Path,
    expected_runs: set[RunKey],
    inventory: Mapping[RunKey, Mapping[str, str]],
    trials: int,
    target_commit: str,
    target_branch: str,
    report: AuditReport,
) -> None:
    raw_root = root / "raw_runs"
    if not raw_root.is_dir():
        report.add_error(f"missing raw bundle directory: {raw_root}")
        return
    children = list(raw_root.iterdir())
    staging = [path for path in children if path.name.startswith(".")]
    for path in staging:
        report.add_error(f"raw bundle root contains temporary/staging path: {path}")
    raw_dirs = {
        path.name: path
        for path in children
        if path.is_dir() and not path.name.startswith(".")
    }
    non_dirs = [path for path in children if not path.is_dir()]
    for path in non_dirs:
        report.add_error(f"unexpected non-directory in raw_runs: {path}")

    inventory_names = {
        str(row.get("save_name", "")) for row in inventory.values() if row.get("save_name")
    }
    if set(raw_dirs) != inventory_names:
        for name in sorted(inventory_names - set(raw_dirs)):
            report.add_error(f"inventory raw bundle directory is missing: {name}")
        for name in sorted(set(raw_dirs) - inventory_names):
            report.add_error(f"unindexed raw bundle directory: {name}")
    if len(raw_dirs) != len(expected_runs):
        report.add_error(
            f"raw_runs contains {len(raw_dirs)} bundles; expected {len(expected_runs)}"
        )

    for key in sorted(expected_runs):
        inventory_row = inventory.get(key)
        if inventory_row is None:
            continue
        bundle = raw_dirs.get(str(inventory_row.get("save_name", "")))
        if bundle is None:
            continue
        _check_raw_bundle(
            root,
            bundle,
            key,
            inventory_row,
            trials,
            target_commit,
            target_branch,
            report,
        )
    report.summaries["raw_bundles"] = len(raw_dirs)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ReleaseAuditInputError("cannot aggregate an empty metric series")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _aggregate_stat(values: Sequence[float], stat: str) -> float:
    if stat == "mean":
        return statistics.fmean(values)
    if stat == "median":
        return statistics.median(values)
    if stat == "p25":
        return _percentile(values, 0.25)
    if stat == "p75":
        return _percentile(values, 0.75)
    raise ReleaseAuditInputError(f"unknown aggregate statistic: {stat}")


def _check_aggregate_metrics(
    root: Path,
    expected_runs: set[RunKey],
    case_values: Mapping[tuple[RunKey, str], Sequence[float]],
    report: AuditReport,
) -> None:
    path = root / "perturbed_sweep.csv"
    _, rows = _read_csv_rows(
        path,
        {
            "experiment",
            "algo",
            "resolution",
            "wiggle",
            "seed",
            "metric_key",
            "metric_value",
            "save_name",
        },
        report,
    )
    expected_keys = {
        (run_key, f"{metric}_{stat}")
        for run_key in expected_runs
        for metric in METRICS_BY_EXPERIMENT.get(run_key.experiment, ())
        for stat in AGGREGATE_STATS
    }
    seen: set[tuple[RunKey, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        label = f"{path}:{row_number}"
        try:
            run_key = _run_key(row, label)
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        metric_key = str(row.get("metric_key", ""))
        key = (run_key, metric_key)
        if key in seen:
            report.add_error(
                f"duplicate aggregate key: {run_key.display()}/{metric_key}"
            )
            continue
        seen.add(key)
        if key not in expected_keys:
            report.add_error(
                f"unexpected aggregate key: {run_key.display()}/{metric_key}"
            )
            continue
        try:
            value = _finite_metric(row.get("metric_value"), f"{label} metric_value")
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        metric = ""
        stat = ""
        for suffix in AGGREGATE_STATS:
            token = f"_{suffix}"
            if metric_key.endswith(token):
                metric = metric_key[: -len(token)]
                stat = suffix
                break
        values = case_values.get((run_key, metric), ())
        if not values:
            report.add_error(
                f"aggregate row has no complete case series: {run_key.display()}/{metric}"
            )
            continue
        expected_value = _aggregate_stat(values, stat)
        if not math.isclose(value, expected_value, rel_tol=1e-12, abs_tol=1e-15):
            report.add_error(
                f"aggregate value mismatch for {run_key.display()}/{metric_key}: "
                f"{value} != {expected_value}"
            )

    for run_key, metric_key in sorted(expected_keys - seen):
        report.add_error(f"missing aggregate key: {run_key.display()}/{metric_key}")
    report.summaries["aggregate_rows"] = len(rows)


def audit_final_release(
    release_root: Path,
    *,
    required_runs: int = FINAL_RUN_COUNT,
    required_cases: int = FINAL_CASE_COUNT,
) -> AuditReport:
    """Audit a release without modifying it and return every detected failure."""
    root = Path(release_root).resolve()
    report = AuditReport(root)
    if not root.is_dir():
        report.add_error(f"release root is not a directory: {root}")
        return report
    for relative in REQUIRED_RELEASE_FILES:
        if not (root / relative).is_file():
            report.add_error(f"missing required release file: {relative}")

    config = _load_json(root / "submission_config.resolved.json", report)
    if config is None:
        return report
    expected_runs, trials = _expected_grid(config, report)
    _check_exact_counts(
        config,
        len(expected_runs),
        trials,
        required_runs,
        required_cases,
        report,
    )
    target_commit = _check_source_provenance(root, config, report)
    target_branch = str(config.get("source", {}).get("target_branch", ""))
    production = config.get("production_method", {})
    if not isinstance(production, dict):
        report.add_error("resolved config production_method is not an object")
        production = {}

    _check_controller(root, required_runs, required_cases, report)
    inventory = _check_inventory(
        root,
        expected_runs,
        trials,
        target_commit,
        target_branch,
        production,
        report,
    )
    _check_consolidated_run_manifests(
        root,
        expected_runs,
        inventory,
        target_commit,
        target_branch,
        report,
    )
    case_values = _check_case_metrics(
        root,
        expected_runs,
        trials,
        target_commit,
        target_branch,
        report,
    )
    _check_case_geometry(
        root,
        expected_runs,
        trials,
        target_commit,
        target_branch,
        report,
    )
    _check_consolidated_table_counts(root, inventory, report)
    _check_raw_bundles(
        root,
        expected_runs,
        inventory,
        trials,
        target_commit,
        target_branch,
        report,
    )
    _check_aggregate_metrics(root, expected_runs, case_values, report)
    report.summaries["expected_runs"] = len(expected_runs)
    report.summaries["expected_cases"] = len(expected_runs) * trials
    return report


def _manifest_path(root: Path, relative_path: Path | str) -> Path:
    pure = PurePosixPath(str(relative_path))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ReleaseAuditInputError(
            f"SHA-256 manifest path must be release-relative: {relative_path!r}"
        )
    path = root.joinpath(*pure.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ReleaseAuditInputError("SHA-256 manifest path escapes release root") from exc
    return path


def _release_files(root: Path, excluded: set[Path]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ReleaseAuditInputError(f"release contains a symbolic link: {path}")
        if not path.is_file() or path in excluded:
            continue
        relative = path.relative_to(root).as_posix()
        if "\n" in relative or "\r" in relative:
            raise ReleaseAuditInputError(
                f"release filename cannot be represented safely: {relative!r}"
            )
        files.append((relative, path))
    files.sort(key=lambda item: item[0])
    return files


def generate_sha256_manifest(
    release_root: Path,
    manifest_relative_path: Path | str = DEFAULT_SHA256_MANIFEST,
) -> Path:
    """Atomically write a sorted SHA-256 manifest for every other release file."""
    root = Path(release_root).resolve()
    if not root.is_dir():
        raise ReleaseAuditInputError(f"release root is not a directory: {root}")
    manifest_path = _manifest_path(root, manifest_relative_path)
    files = _release_files(root, {manifest_path})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for relative, path in files:
                stream.write(f"{_sha256(path)}  {relative}\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(manifest_path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return manifest_path


def verify_sha256_manifest(
    release_root: Path,
    manifest_relative_path: Path | str = DEFAULT_SHA256_MANIFEST,
) -> list[str]:
    """Return manifest verification failures, including incomplete file coverage."""
    root = Path(release_root).resolve()
    errors: list[str] = []
    try:
        manifest_path = _manifest_path(root, manifest_relative_path)
    except ReleaseAuditInputError as exc:
        return [str(exc)]
    if not manifest_path.is_file():
        return [f"SHA-256 manifest is missing: {manifest_path}"]
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [f"could not read SHA-256 manifest: {exc}"]

    records: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if len(line) < 67 or line[64:66] != "  ":
            errors.append(f"invalid SHA-256 manifest line {line_number}")
            continue
        digest, relative = line[:64].lower(), line[66:]
        if any(character not in "0123456789abcdef" for character in digest):
            errors.append(f"invalid SHA-256 digest on line {line_number}")
            continue
        pure = PurePosixPath(relative)
        if not relative or pure.is_absolute() or ".." in pure.parts:
            errors.append(f"unsafe SHA-256 path on line {line_number}: {relative!r}")
            continue
        if relative in seen_paths:
            errors.append(f"duplicate SHA-256 path: {relative}")
            continue
        seen_paths.add(relative)
        records.append((relative, digest))
    record_paths = [relative for relative, _ in records]
    if record_paths != sorted(record_paths):
        errors.append("SHA-256 manifest paths are not sorted")

    try:
        release_files = _release_files(root, {manifest_path})
    except ReleaseAuditInputError as exc:
        errors.append(str(exc))
        return errors
    actual_paths = {relative for relative, _ in release_files}
    manifest_paths = set(record_paths)
    for relative in sorted(actual_paths - manifest_paths):
        errors.append(f"file is absent from SHA-256 manifest: {relative}")
    for relative in sorted(manifest_paths - actual_paths):
        errors.append(f"manifest path is absent from release: {relative}")

    file_lookup = dict(release_files)
    for relative, expected_digest in records:
        path = file_lookup.get(relative)
        if path is None:
            continue
        try:
            actual_digest = _sha256(path)
        except OSError as exc:
            errors.append(f"could not hash {relative}: {exc}")
            continue
        if actual_digest != expected_digest:
            errors.append(f"SHA-256 mismatch: {relative}")
    return errors


def _print_report(report: AuditReport) -> None:
    print(f"Release root: {report.release_root}")
    for key in sorted(report.summaries):
        print(f"{key}: {report.summaries[key]}")
    if report.ok:
        print("FINAL RELEASE AUDIT PASSED")
        return
    print(f"FINAL RELEASE AUDIT FAILED ({report.total_errors} errors)")
    for error in report.errors:
        print(f"- {error}")
    if report.suppressed_errors:
        print(f"- ... {report.suppressed_errors} additional errors suppressed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_root", type=Path)
    parser.add_argument(
        "--write-sha256-manifest",
        action="store_true",
        help="write a sorted manifest after the scientific release audit passes",
    )
    parser.add_argument(
        "--verify-sha256-manifest",
        action="store_true",
        help="verify exact manifest coverage and every recorded digest",
    )
    parser.add_argument(
        "--sha256-manifest",
        default=DEFAULT_SHA256_MANIFEST,
        help="release-relative manifest path (default: SHA256SUMS)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_final_release(args.release_root)
    if report.ok and args.write_sha256_manifest:
        try:
            path = generate_sha256_manifest(
                report.release_root, args.sha256_manifest
            )
        except (OSError, ReleaseAuditInputError) as exc:
            report.add_error(f"could not write SHA-256 manifest: {exc}")
        else:
            print(f"Wrote SHA-256 manifest: {path}")
    if report.ok and args.verify_sha256_manifest:
        for error in verify_sha256_manifest(
            report.release_root, args.sha256_manifest
        ):
            report.add_error(error)
    _print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
