#!/usr/bin/env python3
"""Verify the final-vs-July comparison and emit durable audit tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submission.compare_release_results import (
    CASE_FIELDS,
    RUN_FIELDS,
    SCIENTIFIC_METRICS,
    CaseKey,
    ComparisonError,
    ReleaseData,
    RunKey,
    compare_releases,
    load_release,
)


DEFAULT_RESULTS_ROOT = Path(
    os.environ.get("INTERFACE_RESULTS_ROOT", REPO_ROOT / "results" / "static")
).expanduser()
DEFAULT_BASELINE = (
    DEFAULT_RESULTS_ROOT / "static_paper_simplified_default_20260717_212413"
)
DEFAULT_CANDIDATE = (
    DEFAULT_RESULTS_ROOT / "submission_static_20260731_012430_505aefa45432"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent
PROFILE_FIELDS = (
    "plic_fallback",
    "rescue_profile",
    "corner_behavior_profile",
)
RUN_KEY_FIELDS = (*RUN_FIELDS,)
AGGREGATE_KEY_FIELDS = (*RUN_FIELDS, "metric_key")


def _decimal(value: str) -> Decimal:
    return Decimal(value).normalize()


def _run_key(row: dict[str, str]) -> RunKey:
    return RunKey(
        experiment=row["experiment"].strip(),
        algo=row["algo"].strip(),
        resolution=_decimal(row["resolution"]),
        wiggle=_decimal(row["wiggle"]),
        seed=int(row["seed"]),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ComparisonError(f"missing CSV header: {path}")
        return list(reader)


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key_digest(keys: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        if isinstance(key, CaseKey):
            values = key.csv_values()
            fields = CASE_FIELDS
        elif isinstance(key, RunKey):
            values = key.csv_values()
            fields = RUN_FIELDS
        else:
            values = dict(zip(AGGREGATE_KEY_FIELDS, key))
            fields = AGGREGATE_KEY_FIELDS
        digest.update(
            ("\x1f".join(str(values[field]) for field in fields) + "\n").encode()
        )
    return digest.hexdigest()


def _inventory(release: ReleaseData) -> dict[RunKey, dict[str, str]]:
    path = release.root / "diagnostics" / "run_inventory.csv"
    rows = _read_csv(path)
    inventory: dict[RunKey, dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        key = _run_key(row)
        if key in inventory:
            raise ComparisonError(f"duplicate inventory key at {path}:{line_number}")
        inventory[key] = row
    if set(inventory) != release.run_keys:
        raise ComparisonError(
            f"run inventory and case metrics disagree for {release.root}"
        )
    return inventory


def _aggregate_rows(
    release: ReleaseData,
) -> dict[tuple[str, str, Decimal, Decimal, int, str], dict[str, str]]:
    path = release.root / "perturbed_sweep.csv"
    rows = _read_csv(path)
    aggregate: dict[tuple[str, str, Decimal, Decimal, int, str], dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        run = _run_key(row)
        key = (
            run.experiment,
            run.algo,
            run.resolution,
            run.wiggle,
            run.seed,
            row["metric_key"].strip(),
        )
        if key in aggregate:
            raise ComparisonError(f"duplicate aggregate key at {path}:{line_number}")
        value = float(row["metric_value"])
        if not math.isfinite(value):
            raise ComparisonError(f"nonfinite aggregate value at {path}:{line_number}")
        aggregate[key] = row
    if {RunKey(*key[:5]) for key in aggregate} != release.run_keys:
        raise ComparisonError(
            f"aggregate metrics and case metrics disagree for {release.root}"
        )
    return aggregate


def _verify_aggregate_medians(
    release: ReleaseData,
    aggregate: dict[tuple[str, str, Decimal, Decimal, int, str], dict[str, str]],
) -> int:
    checked = 0
    for run, records in release.cases_by_run.items():
        for metric in SCIENTIFIC_METRICS:
            values = [record.metrics[metric] for record in records.values()]
            present = [value for value in values if value is not None]
            key = (
                run.experiment,
                run.algo,
                run.resolution,
                run.wiggle,
                run.seed,
                f"{metric}_median",
            )
            if not present:
                if key in aggregate:
                    raise ComparisonError(f"unexpected aggregate median: {key}")
                continue
            if len(present) != len(values):
                raise ComparisonError(
                    f"partially missing metric values: {run}/{metric}"
                )
            if key not in aggregate:
                raise ComparisonError(f"missing aggregate median: {key}")
            stored = float(aggregate[key]["metric_value"])
            computed = float(statistics.median(present))
            if stored != computed:
                raise ComparisonError(
                    f"aggregate median mismatch for {key}: {stored} != {computed}"
                )
            checked += 1
    return checked


def _profile_rows(
    label: str,
    release: ReleaseData,
    inventory: dict[RunKey, dict[str, str]],
    baseline_runs: set[RunKey],
) -> list[dict]:
    grouped: dict[tuple[str, str], list[RunKey]] = defaultdict(list)
    for run in release.run_keys:
        grouped[(run.experiment, run.algo)].append(run)
    rows = []
    for (experiment, algo), runs in sorted(grouped.items()):
        profiles = {
            tuple(inventory[run][field] for field in PROFILE_FIELDS) for run in runs
        }
        if len(profiles) != 1:
            raise ComparisonError(
                f"multiple profiles for {label} {experiment}/{algo}: {profiles}"
            )
        profile = next(iter(profiles))
        case_count = sum(len(release.cases_by_run[run]) for run in runs)
        rows.append(
            {
                "release": label,
                "experiment": experiment,
                "algo": algo,
                "coverage_status": (
                    "matched" if set(runs) <= baseline_runs else "candidate_only"
                ),
                "run_count": len(runs),
                "case_count": case_count,
                "cells_per_side": ";".join(
                    str(value) for value in sorted({run.cells_per_side for run in runs})
                ),
                "wiggles": ";".join(
                    format(value, "f") for value in sorted({run.wiggle for run in runs})
                ),
                "seeds": ";".join(
                    str(value) for value in sorted({r.seed for r in runs})
                ),
                **dict(zip(PROFILE_FIELDS, profile)),
            }
        )
    return rows


def _metric_coverage_rows(
    baseline: ReleaseData, candidate: ReleaseData, matched: set[CaseKey]
) -> list[dict]:
    grouped: dict[tuple[str, str], list[CaseKey]] = defaultdict(list)
    for key in matched:
        grouped[(key.run.experiment, key.run.algo)].append(key)
    rows = []
    for (experiment, algo), keys in sorted(grouped.items()):
        for metric in SCIENTIFIC_METRICS:
            baseline_count = sum(
                baseline.cases[key].metrics[metric] is not None for key in keys
            )
            candidate_count = sum(
                candidate.cases[key].metrics[metric] is not None for key in keys
            )
            mismatch_count = sum(
                (baseline.cases[key].metrics[metric] is None)
                != (candidate.cases[key].metrics[metric] is None)
                for key in keys
            )
            rows.append(
                {
                    "experiment": experiment,
                    "algo": algo,
                    "metric": metric,
                    "matched_case_count": len(keys),
                    "baseline_value_count": baseline_count,
                    "candidate_value_count": candidate_count,
                    "missingness_mismatch_count": mismatch_count,
                    "status": "pass" if mismatch_count == 0 else "fail",
                }
            )
    return rows


def _input_rows(label: str, release: ReleaseData) -> list[dict]:
    relatives = (
        "sweep_manifest.json",
        "failures.csv",
        "perturbed_sweep.csv",
        "diagnostics/case_metrics.csv",
        "diagnostics/run_inventory.csv",
        "diagnostics/source_state.json",
        "diagnostics/source_snapshot.tar.gz",
    )
    rows = []
    for relative in relatives:
        path = release.root / relative
        rows.append(
            {
                "release": label,
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _portableize_core_outputs(
    output: Path, baseline: ReleaseData, candidate: ReleaseData
) -> None:
    portable_roots = {
        str(baseline.root): f"$INTERFACE_RESULTS_ROOT/{baseline.root.name}",
        str(candidate.root): f"$INTERFACE_RESULTS_ROOT/{candidate.root.name}",
    }
    report_path = output / "REPORT.md"
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        for absolute, portable in portable_roots.items():
            report = report.replace(absolute, portable)
        report_path.write_text(report, encoding="utf-8")

    comparison_path = output / "comparison.json"
    if comparison_path.is_file():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        comparison["baseline"]["root"] = portable_roots[str(baseline.root)]
        comparison["candidate"]["root"] = portable_roots[str(candidate.root)]
        comparison_path.write_text(
            json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _normalize_csv_line_endings(output: Path) -> None:
    for path in output.glob("*.csv"):
        data = path.read_bytes()
        normalized = data.replace(b"\r\n", b"\n")
        if normalized != data:
            path.write_bytes(normalized)


def run(baseline_root: Path, candidate_root: Path, output: Path) -> None:
    baseline = load_release(baseline_root, "baseline")
    candidate = load_release(candidate_root, "candidate")
    result = compare_releases(baseline, candidate)
    if result["summary"]["status"] != "pass":
        raise ComparisonError(str(result["summary"]["issues"]))

    baseline_inventory = _inventory(baseline)
    candidate_inventory = _inventory(candidate)
    matched_runs = baseline.run_keys & candidate.run_keys
    matched_cases = set(baseline.cases) & set(candidate.cases)
    bad_case_grids = [
        run
        for run in matched_runs
        if set(baseline.cases_by_run[run]) != set(candidate.cases_by_run[run])
        or len(baseline.cases_by_run[run]) != 25
    ]
    if bad_case_grids:
        raise ComparisonError(
            f"nonidentical or non-25-case grids on {len(bad_case_grids)} runs"
        )
    profile_mismatches = [
        run
        for run in matched_runs
        if tuple(baseline_inventory[run][field] for field in PROFILE_FIELDS)
        != tuple(candidate_inventory[run][field] for field in PROFILE_FIELDS)
    ]
    if profile_mismatches:
        raise ComparisonError(f"profile mismatch on {len(profile_mismatches)} runs")

    baseline_aggregate = _aggregate_rows(baseline)
    candidate_aggregate = _aggregate_rows(candidate)
    baseline_aggregate_keys = set(baseline_aggregate)
    candidate_aggregate_keys = set(candidate_aggregate)
    median_checks = _verify_aggregate_medians(
        baseline, baseline_aggregate
    ) + _verify_aggregate_medians(candidate, candidate_aggregate)

    metric_coverage = _metric_coverage_rows(baseline, candidate, matched_cases)
    missingness_mismatches = sum(
        row["missingness_mismatch_count"] for row in metric_coverage
    )
    if missingness_mismatches:
        raise ComparisonError(
            f"metric missingness differs on {missingness_mismatches} matched values"
        )

    snapshot_checks = []
    for label, release in (("baseline", baseline), ("candidate", candidate)):
        snapshot_path = release.root / "diagnostics" / "source_snapshot.tar.gz"
        actual = _sha256(snapshot_path)
        declared = release.source_state.get("snapshot_sha256")
        if actual != declared:
            raise ComparisonError(
                f"{label} source snapshot checksum mismatch: {actual} != {declared}"
            )
        snapshot_checks.append(f"{label}={actual}")

    candidate_matched_aggregate = candidate_aggregate_keys & baseline_aggregate_keys
    integrity_rows = [
        {
            "check": "release_completion",
            "status": "pass",
            "count": 2,
            "details": "both manifests completed with zero failures",
        },
        {
            "check": "baseline_run_keys_subset_of_candidate",
            "status": "pass",
            "count": len(matched_runs),
            "details": f"baseline_only=0; candidate_only={len(candidate.run_keys - baseline.run_keys)}",
        },
        {
            "check": "baseline_case_keys_subset_of_candidate",
            "status": "pass",
            "count": len(matched_cases),
            "details": f"baseline_only=0; candidate_only={len(candidate.cases) - len(matched_cases)}",
        },
        {
            "check": "exact_case_grids",
            "status": "pass",
            "count": len(matched_runs),
            "details": "all matched runs contain the same 25 case indices",
        },
        {
            "check": "matched_run_profiles",
            "status": "pass",
            "count": len(matched_runs),
            "details": "PLIC, rescue, and corner-profile fields agree exactly",
        },
        {
            "check": "metric_missingness",
            "status": "pass",
            "count": sum(row["baseline_value_count"] for row in metric_coverage),
            "details": "zero availability mismatches across six metric columns",
        },
        {
            "check": "aggregate_keys",
            "status": "pass",
            "count": len(baseline_aggregate_keys),
            "details": (
                "baseline_only=0; candidate_only="
                f"{len(candidate_aggregate_keys - baseline_aggregate_keys)}"
            ),
        },
        {
            "check": "aggregate_medians_recomputed",
            "status": "pass",
            "count": median_checks,
            "details": "stored medians equal medians recomputed from case rows",
        },
        {
            "check": "source_snapshot_checksums",
            "status": "pass",
            "count": 2,
            "details": "; ".join(snapshot_checks),
        },
        {
            "check": "baseline_case_key_sha256",
            "status": "pass",
            "count": len(baseline.cases),
            "details": _key_digest(baseline.cases),
        },
        {
            "check": "candidate_matched_case_key_sha256",
            "status": "pass",
            "count": len(matched_cases),
            "details": _key_digest(matched_cases),
        },
        {
            "check": "baseline_aggregate_key_sha256",
            "status": "pass",
            "count": len(baseline_aggregate_keys),
            "details": _key_digest(baseline_aggregate_keys),
        },
        {
            "check": "candidate_matched_aggregate_key_sha256",
            "status": "pass",
            "count": len(candidate_matched_aggregate),
            "details": _key_digest(candidate_matched_aggregate),
        },
    ]
    if integrity_rows[-4]["details"] != integrity_rows[-3]["details"]:
        raise ComparisonError("matched case-key digests differ")
    if integrity_rows[-2]["details"] != integrity_rows[-1]["details"]:
        raise ComparisonError("matched aggregate-key digests differ")

    output = output.resolve()
    for root in (baseline.root, candidate.root):
        try:
            output.relative_to(root)
        except ValueError:
            pass
        else:
            raise ComparisonError("audit output must be outside both release roots")
    output.mkdir(parents=True, exist_ok=True)

    _portableize_core_outputs(output, baseline, candidate)

    _write_csv(
        output / "integrity_audit.csv",
        ("check", "status", "count", "details"),
        integrity_rows,
    )
    _write_csv(
        output / "coverage_profile.csv",
        (
            "release",
            "experiment",
            "algo",
            "coverage_status",
            "run_count",
            "case_count",
            "cells_per_side",
            "wiggles",
            "seeds",
            *PROFILE_FIELDS,
        ),
        [
            *_profile_rows(
                "july_baseline", baseline, baseline_inventory, baseline.run_keys
            ),
            *_profile_rows(
                "authoritative_final",
                candidate,
                candidate_inventory,
                baseline.run_keys,
            ),
        ],
    )
    _write_csv(
        output / "metric_coverage.csv",
        (
            "experiment",
            "algo",
            "metric",
            "matched_case_count",
            "baseline_value_count",
            "candidate_value_count",
            "missingness_mismatch_count",
            "status",
        ),
        metric_coverage,
    )
    material_changes = [
        row
        for row in result["case_metric_comparison"]
        if row["outcome"] in {"improved", "worsened"}
    ]
    _write_csv(
        output / "material_changes.csv",
        (
            *CASE_FIELDS,
            "cells_per_side",
            "metric",
            "baseline_value",
            "candidate_value",
            "delta",
            "candidate_to_baseline_ratio",
            "outcome",
            "value_status",
        ),
        material_changes,
    )
    _write_csv(
        output / "input_checksums.csv",
        ("release", "relative_path", "size_bytes", "sha256"),
        [
            *_input_rows("july_baseline", baseline),
            *_input_rows("authoritative_final", candidate),
        ],
    )
    _normalize_csv_line_endings(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_BASELINE,
        help="July release root (defaults below INTERFACE_RESULTS_ROOT)",
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=DEFAULT_CANDIDATE,
        help="authoritative release root (defaults below INTERFACE_RESULTS_ROOT)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.baseline_root, args.candidate_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
