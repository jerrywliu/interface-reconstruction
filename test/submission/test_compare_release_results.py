import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from submission.compare_release_results import (
    ComparisonError,
    compare_releases,
    discover_release_root,
    load_release,
    write_comparison,
)


CASE_FIELDS = [
    "experiment",
    "algo",
    "resolution",
    "wiggle",
    "seed",
    "case_index",
    "num_plic_fallback_cells",
    "hausdorff",
    "facet_gap",
    "area_error",
    "curvature_error",
    "tangent_error",
    "curvature_proxy_error",
]


def _case(
    experiment,
    algo,
    resolution,
    wiggle,
    case_index,
    hausdorff,
    facet_gap,
    *,
    seed=0,
    plic=0,
):
    return {
        "experiment": experiment,
        "algo": algo,
        "resolution": resolution,
        "wiggle": wiggle,
        "seed": seed,
        "case_index": case_index,
        "num_plic_fallback_cells": plic,
        "hausdorff": hausdorff,
        "facet_gap": facet_gap,
        "area_error": "",
        "curvature_error": "",
        "tangent_error": "",
        "curvature_proxy_error": "",
    }


def _write_release(root: Path, rows, *, status="completed", timestamp="2026-07-01"):
    rows = list(rows)
    run_keys = {
        (
            row["experiment"],
            row["algo"],
            str(row["resolution"]),
            str(row["wiggle"]),
            int(row["seed"]),
        )
        for row in rows
    }
    root.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "status": status,
        "timestamp_utc": timestamp,
        "planned_run_count": len(run_keys),
        "planned_case_count": len(rows),
        "successful_run_count": len(run_keys) if status == "completed" else 0,
        "failure_count": 0,
        "parameters": {
            "plic_fallback": "LVIRA",
            "rescue_profile": "exact_linear_support_only",
            "corner_behavior_profile": "pre_f8_corner",
        },
    }
    (root / "sweep_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    diagnostics = root / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "source_state.json").write_text(
        json.dumps({"source_commit": root.name, "source_dirty": False}),
        encoding="utf-8",
    )
    with (diagnostics / "case_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=CASE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return root


def _synthetic_pair(tmp_path):
    baseline_rows = [
        _case("squares", "linear+corner", "1.0", "0.10", 0, 1e-8, 2e-8),
        _case("squares", "linear+corner", "1.0", "0.10", 1, 2e-8, 3e-8),
        _case("squares", "linear+corner", "1.0", "0.10", 2, 3e-8, 4e-8),
        _case("zalesak", "circular+corner", "1.0", "0.1", 0, 1e-8, 1e-8),
        _case("zalesak", "circular+corner", "1.0", "0.1", 1, 2e-8, 2e-8),
        _case("zalesak", "circular+corner", "1.0", "0.1", 2, 2.0, 3e-8),
    ]
    candidate_rows = [
        _case("squares", "linear+corner", "1.00", "0.1", 0, 1e-8, 2e-8),
        _case("squares", "linear+corner", "1.00", "0.1", 1, 8e-7, 3e-8),
        _case("squares", "linear+corner", "1.00", "0.1", 2, 2e-6, 4e-8),
        _case("zalesak", "circular+corner", "1.0", "0.10", 0, 1e-8, 1e-8),
        _case("zalesak", "circular+corner", "1.0", "0.10", 1, 2e-8, 2e-8),
        _case("zalesak", "circular+corner", "1.0", "0.10", 2, 0.5, 3e-8),
        _case("squares", "Youngs", "1.0", "0.1", 0, 0.2, 0.1),
        _case("squares", "Youngs", "1.0", "0.1", 1, 0.3, 0.2),
        _case("squares", "Youngs", "1.0", "0.1", 2, 0.4, 0.3),
    ]
    baseline_root = _write_release(tmp_path / "july", baseline_rows)
    candidate_root = _write_release(tmp_path / "final", candidate_rows)
    return load_release(baseline_root, "baseline"), load_release(
        candidate_root, "candidate"
    )


def test_compares_shared_subset_and_normalizes_decimal_run_keys(tmp_path):
    baseline, candidate = _synthetic_pair(tmp_path)

    result = compare_releases(baseline, candidate)

    coverage = result["summary"]["coverage"]
    assert coverage == {
        "baseline_run_count": 2,
        "candidate_run_count": 3,
        "matched_run_count": 2,
        "baseline_only_run_count": 0,
        "candidate_only_run_count": 1,
        "exact_case_grid_run_count": 2,
        "case_mismatch_run_count": 0,
        "matched_case_count": 6,
    }
    assert result["summary"]["status"] == "pass"
    squares = next(
        row
        for row in result["method_metric_comparison"]
        if row["experiment"] == "squares"
        and row["algo"] == "linear+corner"
        and row["metric"] == "hausdorff"
    )
    assert squares["paired_case_count"] == 3
    assert squares["baseline_median"] == pytest.approx(2e-8)
    assert squares["candidate_median"] == pytest.approx(8e-7)
    assert squares["candidate_max"] == pytest.approx(2e-6)


def test_highlights_perfect_reconstruction_and_tail_changes(tmp_path):
    baseline, candidate = _synthetic_pair(tmp_path)

    result = compare_releases(baseline, candidate)
    perfect = {row["experiment"]: row for row in result["perfect_reconstruction"]}

    assert perfect["squares"]["candidate_hausdorff_median_below_threshold"] is True
    assert perfect["squares"]["candidate_all_cases_joint_floor"] is False
    assert perfect["zalesak"]["threshold_outcome"] == "retained"
    zalesak_tail = next(
        row
        for row in result["tail_cases"]
        if row["experiment"] == "zalesak" and row["case_index"] == 2
    )
    assert "baseline_tail" in zalesak_tail["reasons"]
    assert "fixed_tail" in zalesak_tail["reasons"]


def test_writes_machine_readable_and_markdown_artifacts(tmp_path):
    baseline, candidate = _synthetic_pair(tmp_path)
    result = compare_releases(baseline, candidate)

    report = write_comparison(result, tmp_path / "comparison")

    expected = {
        "REPORT.md",
        "comparison.json",
        "run_coverage.csv",
        "method_metric_comparison.csv",
        "setting_metric_comparison.csv",
        "case_metric_comparison.csv",
        "tail_cases.csv",
        "perfect_reconstruction.csv",
    }
    assert {path.name for path in report.parent.iterdir()} == expected
    summary = json.loads((report.parent / "comparison.json").read_text())
    assert summary["status"] == "pass"
    assert summary["coverage"]["candidate_only_run_count"] == 1
    report_text = report.read_text(encoding="utf-8")
    assert "Perfect-reconstruction checks" in report_text
    assert "July square and Zalesak `area_error`" in report_text


def test_reports_case_grid_mismatch_without_silently_matching_the_run(tmp_path):
    baseline, candidate = _synthetic_pair(tmp_path)
    missing_key = next(
        key
        for key in candidate.cases
        if key.run.experiment == "squares"
        and key.run.algo == "linear+corner"
        and key.case_index == 2
    )
    del candidate.cases[missing_key]
    del candidate.cases_by_run[missing_key.run][missing_key.case_index]

    result = compare_releases(baseline, candidate)

    assert result["summary"]["status"] == "attention_required"
    assert result["summary"]["coverage"]["case_mismatch_run_count"] == 1
    assert any("case indices differ" in issue for issue in result["summary"]["issues"])


def test_rejects_duplicate_case_keys_and_incomplete_releases(tmp_path):
    row = _case("squares", "linear+corner", "1.0", "0.1", 0, 1e-8, 1e-8)
    duplicate_root = _write_release(tmp_path / "duplicate", [row, row])
    with pytest.raises(ComparisonError, match="duplicate case key"):
        load_release(duplicate_root, "baseline")

    running_root = _write_release(tmp_path / "running", [row], status="running")
    with pytest.raises(ComparisonError, match="not complete"):
        load_release(running_root, "candidate")


def test_discovery_ignores_newer_running_release(tmp_path):
    row = _case("squares", "linear+corner", "1.0", "0.1", 0, 1e-8, 1e-8)
    completed = _write_release(
        tmp_path / "submission_static_20260730_100000_a",
        [row],
        timestamp="2026-07-30T10:00:00+00:00",
    )
    _write_release(
        tmp_path / "submission_static_20260730_110000_b",
        [row],
        status="running",
        timestamp="2026-07-30T11:00:00+00:00",
    )

    discovered = discover_release_root(tmp_path, "submission_static_*")

    assert discovered == completed.resolve()


def test_refuses_to_write_inside_an_input_release(tmp_path):
    baseline, candidate = _synthetic_pair(tmp_path)
    result = compare_releases(baseline, candidate)

    with pytest.raises(ComparisonError, match="outside both immutable"):
        write_comparison(result, candidate.root / "comparison")


def test_cli_writes_report_from_explicit_synthetic_roots(tmp_path):
    baseline, candidate = _synthetic_pair(tmp_path)
    output_dir = tmp_path / "cli-comparison"
    repo = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            "submission/compare_release_results.py",
            "--baseline-root",
            str(baseline.root),
            "--candidate-root",
            str(candidate.root),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str((output_dir / "REPORT.md").resolve())
    assert (output_dir / "comparison.json").is_file()
