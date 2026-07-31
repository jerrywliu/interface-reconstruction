import copy
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from submission.audit_final_release import (
    RUN_CONTEXT_FIELDS,
    audit_final_release,
    generate_sha256_manifest,
    verify_sha256_manifest,
)


COMMIT = "a" * 40
SAVE_NAME = "submission_test_perturb_sweep_lines_linear_r0p5_w0p0_s0"


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _snapshot(path: Path, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for relative, data in sorted(files.items()):
            member = tarfile.TarInfo(relative)
            member.size = len(data)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(data))


def _run_context() -> dict[str, object]:
    return {
        "experiment": "lines",
        "algo": "linear",
        "resolution": 0.5,
        "wiggle": 0.0,
        "seed": 0,
        "save_name": SAVE_NAME,
        "source_commit": COMMIT,
        "source_branch": "main",
        "plic_fallback": "LVIRA",
        "rescue_profile": "exact_linear_support_only",
        "corner_behavior_profile": "pre_f8_corner",
    }


def _run_manifest() -> dict:
    return {
        "schema_version": 1,
        "source_commit": COMMIT,
        "source_branch": "main",
        "experiment": "lines",
        "parameters": {
            "facet_algo": "linear",
            "resolution": 0.5,
            "perturb_wiggle": 0.0,
            "perturb_seed": 0,
            "plic_fallback": "LVIRA",
            "rescue_profile": "exact_linear_support_only",
            "corner_behavior_profile": "pre_f8_corner",
            "num_lines": 2,
        },
        "artifacts": {
            "case_geometry": "metrics/case_geometry.jsonl",
            "case_metrics": "metrics/case_metrics.csv",
            "cell_metrics": "metrics/cell_metrics.csv",
            "fallback_events": "metrics/unresolved_plic_fallbacks.csv",
            "merge_events": "metrics/merge_events.csv",
            "mesh": "vtk/mesh.vtk",
        },
    }


def _config() -> dict:
    return {
        "status": "frozen",
        "launch_approved": True,
        "source": {"target_commit": COMMIT, "target_branch": "main"},
        "production_method": {
            "corner_behavior_profile": "pre_f8_corner",
            "rescue_profile": "exact_linear_support_only",
            "unresolved_orientation_fallback": "LVIRA",
        },
        "benchmark_grid": {
            "seed": 0,
            "trials_per_setting": 2,
            "wiggles": [0.0],
            "full_resolutions": [0.5],
            "short_resolutions": [0.5],
        },
        "benchmarks": {
            "lines": {
                "resolutions": "full_resolutions",
                "methods": ["linear"],
                "planned_runs": 1,
            }
        },
        "planned_totals": {"runs": 1, "cases": 2},
    }


def _make_release(root: Path) -> Path:
    root.mkdir()
    config = _config()
    _write_json(root / "submission_config.resolved.json", config)

    source_config = copy.deepcopy(config)
    source_config["status"] = "candidate_not_frozen"
    source_config["source"]["target_commit"] = None
    source_config_bytes = (json.dumps(source_config, indent=2) + "\n").encode()
    requirements_bytes = b"numpy==1.23.4\n"
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    _snapshot(
        snapshot_path,
        {
            "submission/submission_config.json": source_config_bytes,
            "requirements.txt": requirements_bytes,
        },
    )
    _write_json(
        root / "diagnostics" / "source_state.json",
        {
            "source_commit": COMMIT,
            "source_branch": "main",
            "source_dirty": False,
            "source_status": [],
            "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            "snapshot_file_count": 2,
        },
    )
    _write_json(
        root / "environment.json",
        {
            "repository": {
                "commit": COMMIT,
                "branch": "main",
                "source_dirty": False,
            },
            "input_fingerprints": [
                {
                    "path": "requirements.txt",
                    "size_bytes": len(requirements_bytes),
                    "sha256": hashlib.sha256(requirements_bytes).hexdigest(),
                },
                {
                    "path": "submission/submission_config.json",
                    "size_bytes": len(source_config_bytes),
                    "sha256": hashlib.sha256(source_config_bytes).hexdigest(),
                },
            ],
        },
    )
    _write_json(
        root / "sweep_manifest.json",
        {
            "status": "completed",
            "planned_run_count": 1,
            "planned_case_count": 2,
            "successful_run_count": 1,
            "failure_count": 0,
            "failures": [],
        },
    )
    _write_csv(
        root / "failures.csv",
        [
            "experiment",
            "algo",
            "resolution",
            "wiggle",
            "seed",
            "save_name",
            "reason",
            "code",
            "log_path",
        ],
        [],
    )

    context = _run_context()
    inventory = {
        **context,
        "run_bundle": f"raw_runs/{SAVE_NAME}",
        "case_geometry_rows": 2,
        "case_metrics_rows": 2,
        "cell_metrics_rows": 2,
        "merge_events_rows": 0,
        "unresolved_plic_fallbacks_rows": 0,
    }
    _write_csv(
        root / "diagnostics" / "run_inventory.csv",
        [
            *RUN_CONTEXT_FIELDS,
            "run_bundle",
            "case_geometry_rows",
            "case_metrics_rows",
            "cell_metrics_rows",
            "merge_events_rows",
            "unresolved_plic_fallbacks_rows",
        ],
        [inventory],
    )
    _write_jsonl(
        root / "diagnostics" / "run_manifests.jsonl",
        [{**context, "manifest": _run_manifest()}],
    )

    all_metric_fields = [
        "hausdorff",
        "facet_gap",
        "area_error",
        "curvature_error",
        "tangent_error",
        "curvature_proxy_error",
    ]
    case_rows = []
    geometry_rows = []
    cell_rows = []
    raw_case_rows = []
    raw_geometry_rows = []
    raw_cell_rows = []
    for case_index, (hausdorff, facet_gap) in enumerate(((1.0, 2.0), (3.0, 4.0))):
        raw_case = {
            "case_index": case_index,
            "num_mixed_cells": 1,
            "num_final_missing_cells": 0,
            "hausdorff": hausdorff,
            "facet_gap": facet_gap,
            "area_error": "",
            "curvature_error": "",
            "tangent_error": "",
            "curvature_proxy_error": "",
        }
        raw_case_rows.append(raw_case)
        case_rows.append({**context, **raw_case})
        truth_vtp = f"vtk/true/true_line{case_index}.vtp"
        truth_metadata = f"vtk/true/true_line{case_index}.facet_metadata.json"
        raw_geometry = {
            "case_index": case_index,
            "geometry_type": "line",
            "angle": float(case_index),
            "truth_vtp": truth_vtp,
            "truth_metadata": truth_metadata,
        }
        raw_geometry_rows.append(raw_geometry)
        geometry_rows.append({**context, **raw_geometry})
        raw_cell = {
            "case_index": case_index,
            "cell_id": f"{case_index},0",
            "final_facet_class": "linear",
            "facet_geometry_json": json.dumps(
                {"class": "linear", "p_left": [0, 0], "p_right": [1, 1]}
            ),
        }
        raw_cell_rows.append(raw_cell)
        cell_rows.append({**context, **raw_cell})

    _write_csv(
        root / "diagnostics" / "case_metrics.csv",
        [
            *RUN_CONTEXT_FIELDS,
            "case_index",
            "num_mixed_cells",
            "num_final_missing_cells",
            *all_metric_fields,
        ],
        case_rows,
    )
    _write_jsonl(root / "diagnostics" / "case_geometry.jsonl", geometry_rows)
    _write_csv(
        root / "diagnostics" / "cell_metrics.csv",
        [
            *RUN_CONTEXT_FIELDS,
            "case_index",
            "cell_id",
            "final_facet_class",
            "facet_geometry_json",
        ],
        cell_rows,
    )
    _write_csv(
        root / "diagnostics" / "merge_events.csv",
        [*RUN_CONTEXT_FIELDS, "case_index", "event_order", "event_kind"],
        [],
    )
    _write_csv(
        root / "diagnostics" / "unresolved_plic_fallbacks.csv",
        [*RUN_CONTEXT_FIELDS, "case_index", "merge_id", "policy"],
        [],
    )

    aggregate_rows = []
    for metric, values in (
        ("hausdorff", (1.0, 3.0)),
        ("facet_gap", (2.0, 4.0)),
    ):
        aggregate_values = {
            "mean": sum(values) / 2,
            "median": sum(values) / 2,
            "p25": values[0] * 0.75 + values[1] * 0.25,
            "p75": values[0] * 0.25 + values[1] * 0.75,
        }
        for stat, value in aggregate_values.items():
            aggregate_rows.append(
                {
                    "experiment": context["experiment"],
                    "algo": context["algo"],
                    "resolution": context["resolution"],
                    "wiggle": context["wiggle"],
                    "seed": context["seed"],
                    "corner_behavior_profile": context["corner_behavior_profile"],
                    "metric_key": f"{metric}_{stat}",
                    "metric_value": value,
                    "save_name": SAVE_NAME,
                }
            )
    _write_csv(
        root / "perturbed_sweep.csv",
        [
            "experiment",
            "algo",
            "resolution",
            "wiggle",
            "seed",
            "corner_behavior_profile",
            "metric_key",
            "metric_value",
            "save_name",
        ],
        aggregate_rows,
    )

    bundle = root / "raw_runs" / SAVE_NAME
    _write_json(bundle / "run_manifest.json", _run_manifest())
    _write_jsonl(bundle / "metrics" / "case_geometry.jsonl", raw_geometry_rows)
    _write_csv(
        bundle / "metrics" / "case_metrics.csv",
        [
            "case_index",
            "num_mixed_cells",
            "num_final_missing_cells",
            *all_metric_fields,
        ],
        raw_case_rows,
    )
    _write_csv(
        bundle / "metrics" / "cell_metrics.csv",
        ["case_index", "cell_id", "final_facet_class", "facet_geometry_json"],
        raw_cell_rows,
    )
    _write_csv(
        bundle / "metrics" / "merge_events.csv",
        ["case_index", "event_order", "event_kind"],
        [],
    )
    _write_csv(
        bundle / "metrics" / "unresolved_plic_fallbacks.csv",
        ["case_index", "merge_id", "policy"],
        [],
    )
    (bundle / "metrics" / "hausdorff.txt").write_text("1.0\n3.0\n")
    (bundle / "metrics" / "facet_gap.txt").write_text("2.0\n4.0\n")
    for case_index in range(2):
        files = (
            bundle / "vtk" / "true" / f"true_line{case_index}.vtp",
            bundle
            / "vtk"
            / "true"
            / f"true_line{case_index}.facet_metadata.json",
            bundle / "vtk" / "reconstructed" / "facets" / f"{case_index}.vtp",
            bundle
            / "vtk"
            / "reconstructed"
            / "facets"
            / f"{case_index}.facet_metadata.json",
            bundle
            / "vtk"
            / "reconstructed"
            / "mixed_cells"
            / f"{case_index}.vtp",
        )
        for path in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("scientific artifact\n", encoding="utf-8")
    (bundle / "vtk" / "mesh.vtk").write_text("mesh\n", encoding="utf-8")
    return root


def _messages(report) -> str:
    return "\n".join(report.errors)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def _mutate_csv(path: Path, row_index: int, field_name: str, value: str) -> None:
    fieldnames, rows = _read_csv(path)
    rows[row_index][field_name] = value
    _write_csv(path, fieldnames, rows)


def _add_provenance_rows(root: Path) -> None:
    context = _run_context()
    merge_fields = ["case_index", "event_order", "event_kind"]
    raw_merge = {"case_index": 0, "event_order": 1, "event_kind": "facet_assignment"}
    fallback_fields = ["case_index", "merge_id", "policy"]
    raw_fallback = {"case_index": 0, "merge_id": 7, "policy": "LVIRA"}

    _write_csv(
        root / "diagnostics" / "merge_events.csv",
        [*RUN_CONTEXT_FIELDS, *merge_fields],
        [{**context, **raw_merge}],
    )
    _write_csv(
        root / "diagnostics" / "unresolved_plic_fallbacks.csv",
        [*RUN_CONTEXT_FIELDS, *fallback_fields],
        [{**context, **raw_fallback}],
    )
    bundle = root / "raw_runs" / SAVE_NAME / "metrics"
    _write_csv(bundle / "merge_events.csv", merge_fields, [raw_merge])
    _write_csv(
        bundle / "unresolved_plic_fallbacks.csv", fallback_fields, [raw_fallback]
    )

    inventory_path = root / "diagnostics" / "run_inventory.csv"
    fieldnames, rows = _read_csv(inventory_path)
    rows[0]["merge_events_rows"] = "1"
    rows[0]["unresolved_plic_fallbacks_rows"] = "1"
    _write_csv(inventory_path, fieldnames, rows)


def test_complete_synthetic_release_passes(tmp_path):
    root = _make_release(tmp_path / "release")

    report = audit_final_release(root, required_runs=1, required_cases=2)

    assert report.ok, _messages(report)
    assert report.summaries["raw_bundles"] == 1
    assert report.summaries["case_metric_rows"] == 2
    assert report.summaries["aggregate_rows"] == 8


def test_command_defaults_remain_hard_gated_to_final_scope(tmp_path):
    root = _make_release(tmp_path / "release")

    report = audit_final_release(root)

    assert not report.ok
    assert "exactly 970 required" in _messages(report)
    assert "exactly 24250 required" in _messages(report)


def test_controller_must_complete_every_run_without_failures(tmp_path):
    root = _make_release(tmp_path / "release")
    manifest_path = root / "sweep_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    failure = {
        "experiment": "lines",
        "algo": "linear",
        "resolution": 0.5,
        "wiggle": 0.0,
        "seed": 0,
        "save_name": SAVE_NAME,
        "reason": "failed",
        "code": 1,
        "log_path": "logs/run.log",
    }
    manifest.update(
        {
            "status": "failed",
            "successful_run_count": 0,
            "failure_count": 1,
            "failures": [failure],
        }
    )
    _write_json(manifest_path, manifest)
    _write_csv(root / "failures.csv", list(failure), [failure])

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "controller status is 'failed'" in messages
    assert "successful_run_count is 0" in messages
    assert "failure_count is 1" in messages
    assert "failures.csv contains 1" in messages


def test_duplicate_run_key_and_nonrelative_bundle_fail(tmp_path):
    root = _make_release(tmp_path / "release")
    inventory_path = root / "diagnostics" / "run_inventory.csv"
    with inventory_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        fieldnames = list(rows[0])
    rows[0]["run_bundle"] = "../outside"
    rows.append(dict(rows[0]))
    _write_csv(inventory_path, fieldnames, rows)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "not release-relative" in messages
    assert "duplicate run key in inventory" in messages


def test_duplicate_nonfinite_case_and_aggregate_coverage_fail(tmp_path):
    root = _make_release(tmp_path / "release")
    case_path = root / "diagnostics" / "case_metrics.csv"
    with case_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        fieldnames = list(rows[0])
    rows[0]["hausdorff"] = "nan"
    rows.append(dict(rows[1]))
    _write_csv(case_path, fieldnames, rows)

    aggregate_path = root / "perturbed_sweep.csv"
    with aggregate_path.open(newline="", encoding="utf-8") as stream:
        aggregate = list(csv.DictReader(stream))
        aggregate_fields = list(aggregate[0])
    aggregate[0]["metric_value"] = "inf"
    aggregate.pop()
    _write_csv(aggregate_path, aggregate_fields, aggregate)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "nonfinite" in messages
    assert "duplicate case key" in messages
    assert "missing aggregate key" in messages


def test_missing_scientific_artifact_and_preview_directory_fail(tmp_path):
    root = _make_release(tmp_path / "release")
    bundle = root / "raw_runs" / SAVE_NAME
    (bundle / "vtk" / "reconstructed" / "facets" / "1.vtp").unlink()
    preview = bundle / "plt" / "areas" / "1.png"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"preview")

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "temporary raster previews" in messages
    assert "missing reconstructed facets" in messages


def test_source_config_environment_and_run_commit_must_agree(tmp_path):
    root = _make_release(tmp_path / "release")
    environment_path = root / "environment.json"
    environment = json.loads(environment_path.read_text())
    environment["repository"]["commit"] = "b" * 40
    _write_json(environment_path, environment)

    config_path = root / "submission_config.resolved.json"
    config = json.loads(config_path.read_text())
    config["production_method"]["rescue_profile"] = "different"
    _write_json(config_path, config)

    raw_manifest_path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text())
    raw_manifest["source_commit"] = "c" * 40
    _write_json(raw_manifest_path, raw_manifest)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "environment commit does not match" in messages
    assert "resolved config differs" in messages
    assert "raw run manifest commit mismatch" in messages


@pytest.mark.parametrize(
    ("relative_path", "field_name", "tampered_value", "table_name"),
    [
        (
            "diagnostics/case_metrics.csv",
            "hausdorff",
            "1.25",
            "case_metrics.csv",
        ),
        (
            "diagnostics/cell_metrics.csv",
            "facet_geometry_json",
            '{"class":"linear","tampered":true}',
            "cell_metrics.csv",
        ),
        (
            "diagnostics/merge_events.csv",
            "event_kind",
            "tampered_event",
            "merge_events.csv",
        ),
        (
            "diagnostics/unresolved_plic_fallbacks.csv",
            "policy",
            "ELVIRA",
            "unresolved_plic_fallbacks.csv",
        ),
    ],
)
def test_value_level_reconciliation_rejects_consolidated_tampering(
    tmp_path, relative_path, field_name, tampered_value, table_name
):
    root = _make_release(tmp_path / "release")
    _add_provenance_rows(root)
    _mutate_csv(root / relative_path, 0, field_name, tampered_value)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "raw/consolidated value mismatch" in messages
    assert table_name in messages
    assert field_name in messages


def test_value_level_reconciliation_rejects_raw_tampering(tmp_path):
    root = _make_release(tmp_path / "release")
    raw_path = root / "raw_runs" / SAVE_NAME / "metrics" / "case_metrics.csv"
    _mutate_csv(raw_path, 0, "facet_gap", "2.5")

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "raw/consolidated value mismatch" in messages
    assert "case_metrics.csv/case_index=0 column facet_gap" in messages


def test_reconciliation_reports_missing_and_unexpected_stable_keys(tmp_path):
    root = _make_release(tmp_path / "release")
    _mutate_csv(root / "diagnostics" / "cell_metrics.csv", 0, "cell_id", "99,99")

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "missing consolidated row" in messages
    assert "cell_id=0,0" in messages
    assert "unexpected consolidated row" in messages
    assert "cell_id=99,99" in messages


def test_equivalent_decimal_serialization_reconciles_exactly(tmp_path):
    root = _make_release(tmp_path / "release")
    raw_path = root / "raw_runs" / SAVE_NAME / "metrics" / "case_metrics.csv"
    _mutate_csv(raw_path, 0, "hausdorff", "1.0000000000000000")

    report = audit_final_release(root, required_runs=1, required_cases=2)

    assert report.ok, _messages(report)


def test_expected_case_summary_repair_is_reproduced_before_reconciliation(tmp_path):
    root = _make_release(tmp_path / "release")
    raw_path = root / "raw_runs" / SAVE_NAME / "metrics" / "case_metrics.csv"
    _mutate_csv(raw_path, 0, "num_mixed_cells", "999")

    report = audit_final_release(root, required_runs=1, required_cases=2)

    assert report.ok, _messages(report)


@pytest.mark.parametrize(
    "source",
    [
        "inventory",
        "case_metrics",
        "case_geometry",
        "cell_metrics",
        "merge_events",
        "fallback_events",
        "run_manifest_context",
        "nested_run_manifest",
        "raw_run_manifest",
    ],
)
def test_rescue_profile_is_enforced_across_all_provenance_layers(tmp_path, source):
    root = _make_release(tmp_path / "release")
    _add_provenance_rows(root)
    if source == "inventory":
        _mutate_csv(
            root / "diagnostics" / "run_inventory.csv",
            0,
            "rescue_profile",
            "different",
        )
    elif source == "case_geometry":
        path = root / "diagnostics" / "case_geometry.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["rescue_profile"] = "different"
        _write_jsonl(path, rows)
    elif source in {
        "case_metrics",
        "cell_metrics",
        "merge_events",
        "fallback_events",
    }:
        filename = {
            "case_metrics": "case_metrics.csv",
            "cell_metrics": "cell_metrics.csv",
            "merge_events": "merge_events.csv",
            "fallback_events": "unresolved_plic_fallbacks.csv",
        }[source]
        _mutate_csv(
            root / "diagnostics" / filename,
            0,
            "rescue_profile",
            "different",
        )
    elif source in {"run_manifest_context", "nested_run_manifest"}:
        path = root / "diagnostics" / "run_manifests.jsonl"
        row = json.loads(path.read_text())
        if source == "run_manifest_context":
            row["rescue_profile"] = "different"
        else:
            row["manifest"]["parameters"]["rescue_profile"] = "different"
        _write_jsonl(path, [row])
    else:
        path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["parameters"]["rescue_profile"] = "different"
        _write_json(path, manifest)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "rescue_profile" in messages
    assert "different" in messages
    assert "exact_linear_support_only" in messages


def test_sha256_manifest_is_sorted_complete_and_verifiable(tmp_path):
    root = _make_release(tmp_path / "release")
    assert audit_final_release(root, required_runs=1, required_cases=2).ok

    manifest = generate_sha256_manifest(root)

    lines = manifest.read_text(encoding="utf-8").splitlines()
    paths = [line[66:] for line in lines]
    assert paths == sorted(paths)
    assert "SHA256SUMS" not in paths
    assert verify_sha256_manifest(root) == []

    environment_path = root / "environment.json"
    environment_path.write_text("tampered\n", encoding="utf-8")
    errors = verify_sha256_manifest(root)
    assert "SHA-256 mismatch: environment.json" in errors


def test_sha256_verification_rejects_extra_and_unsorted_files(tmp_path):
    root = _make_release(tmp_path / "release")
    manifest = generate_sha256_manifest(root)
    (root / "final_figures" / "new.pdf").parent.mkdir()
    (root / "final_figures" / "new.pdf").write_bytes(b"vector figure")

    errors = verify_sha256_manifest(root)
    assert "file is absent from SHA-256 manifest: final_figures/new.pdf" in errors

    generate_sha256_manifest(root)
    lines = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    errors = verify_sha256_manifest(root)
    assert "SHA-256 manifest paths are not sorted" in errors
