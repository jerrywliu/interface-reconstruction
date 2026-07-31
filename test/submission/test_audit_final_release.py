import csv
import gzip
import hashlib
import io
import json
import os
import shlex
import subprocess
import tarfile
from pathlib import Path

import pytest
import submission.audit_final_release as audit_module

from submission.audit_final_release import (
    RUN_CONTEXT_FIELDS,
    audit_final_release,
    generate_sha256_manifest,
    verify_sha256_manifest,
)


PRODUCTION_COMMIT = "505aefa454328d4ba34ade5e7247050a0acfc793"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    _snapshot_entries(
        path,
        [(relative, data, 0o644) for relative, data in sorted(files.items())],
    )


def _snapshot_entries(path: Path, entries: list[tuple[str, bytes, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for relative, data, mode in entries:
            member = tarfile.TarInfo(relative)
            member.size = len(data)
            member.mode = mode
            archive.addfile(member, io.BytesIO(data))


def _metadata_bomb_archive(metadata_kind: str, payload_size: int) -> bytes:
    member = tarfile.TarInfo(
        "././@PaxHeader" if metadata_kind == "pax" else "././@LongLink"
    )
    member.type = (
        tarfile.XHDTYPE if metadata_kind == "pax" else tarfile.GNUTYPE_LONGNAME
    )
    member.mode = 0o644
    member.size = payload_size
    archive_format = (
        tarfile.PAX_FORMAT if metadata_kind == "pax" else tarfile.GNU_FORMAT
    )
    header = member.tobuf(format=archive_format)
    payload = b"\0" * payload_size
    padding = b"\0" * ((-payload_size) % tarfile.BLOCKSIZE)
    return gzip.compress(header + payload + padding + b"\0" * tarfile.RECORDSIZE)


def _tar_data_end(archive_bytes: bytes) -> int:
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        members = list(archive)
    return max(
        member.offset_data
        + ((member.size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE)
        * tarfile.BLOCKSIZE
        for member in members
    )


def _snapshot_budget(files: dict[str, bytes]) -> int:
    expected = {
        relative: audit_module.GitBlob("100644", "0" * 40, data)
        for relative, data in files.items()
    }
    return audit_module._source_tar_decompressed_budget(expected)


def _in_memory_tar(files: dict[str, bytes], archive_format: int) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=archive_format) as archive:
        for relative, data in files.items():
            member = tarfile.TarInfo(relative)
            member.size = len(data)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


def _audit_snapshot_bytes(
    tmp_path: Path,
    files: dict[str, bytes],
    raw_archive: bytes,
) -> audit_module.AuditReport:
    snapshot_path = tmp_path / "source_snapshot.tar.gz"
    snapshot_path.write_bytes(gzip.compress(raw_archive))
    expected = {
        relative: audit_module.GitBlob("100644", "0" * 40, data)
        for relative, data in files.items()
    }
    report = audit_module.AuditReport(tmp_path)
    audit_module._read_bounded_source_snapshot(snapshot_path, expected, report)
    return report


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


def _run_manifest(repository=None) -> dict:
    executable = (
        repository / "experiments" / "static" / "lines.py"
        if repository is not None
        else Path("experiments/static/lines.py")
    )
    return {
        "schema_version": 1,
        "source_commit": COMMIT,
        "source_branch": "main",
        "experiment": "lines",
        "command": (
            f"{executable} --facet_algo linear "
            "--resolution 0.5 --perturb_wiggle 0.0 --perturb_seed 0 "
            "--plic_fallback LVIRA "
            "--rescue_profile exact_linear_support_only "
            "--corner_behavior_profile pre_f8_corner"
        ),
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
                "driver": "experiments.static.lines",
                "resolutions": "full_resolutions",
                "methods": ["linear"],
                "planned_runs": 1,
            }
        },
        "planned_totals": {"runs": 1, "cases": 2},
    }


def _git(repository: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repository), *arguments])


@pytest.fixture(autouse=True)
def _trusted_source_repository(tmp_path, monkeypatch):
    global COMMIT
    repository = tmp_path
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Release Audit Test")
    _git(repository, "config", "user.email", "release-audit@example.invalid")

    source_config = _config()
    source_config["status"] = "candidate_not_frozen"
    source_config["source"]["target_commit"] = None
    _write_json(repository / "submission" / "submission_config.json", source_config)
    (repository / "requirements.txt").write_text("numpy==1.23.4\n", encoding="utf-8")
    for relative in audit_module.PRODUCTION_SOURCE_SHA256:
        data = _git(PROJECT_ROOT, "show", f"{PRODUCTION_COMMIT}:{relative}")
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    _git(repository, "add", ".")
    environment = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2026-07-31T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2026-07-31T00:00:00+00:00",
    }
    subprocess.check_call(
        ["git", "-C", str(repository), "commit", "-q", "-m", "fixture source"],
        env=environment,
    )
    COMMIT = _git(repository, "rev-parse", "HEAD").decode().strip()
    monkeypatch.setattr(audit_module, "FINAL_SOURCE_COMMIT", COMMIT)
    monkeypatch.setattr(audit_module, "LEGACY_COMMAND_SOURCE_COMMIT", COMMIT)


def _make_release(root: Path) -> Path:
    root.mkdir()
    config = _config()
    _write_json(root / "submission_config.resolved.json", config)

    repository = root.parent
    source_config_bytes = (
        repository / "submission" / "submission_config.json"
    ).read_bytes()
    requirements_bytes = (repository / "requirements.txt").read_bytes()
    tracked_paths = _git(repository, "ls-files", "-z").rstrip(b"\0").split(b"\0")
    snapshot_files = {
        relative.decode(): (repository / relative.decode()).read_bytes()
        for relative in tracked_paths
        if relative
    }
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    _snapshot(snapshot_path, snapshot_files)
    _write_json(
        root / "diagnostics" / "source_state.json",
        {
            "source_commit": COMMIT,
            "source_branch": "main",
            "source_dirty": False,
            "source_status": [],
            "excluded_roots": sorted(audit_module.SNAPSHOT_EXCLUDED_ROOTS),
            "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            "snapshot_file_count": len(snapshot_files),
        },
    )
    _write_json(
        root / "environment.json",
        {
            "repository": {
                "commit": COMMIT,
                "branch": "main",
                "root": str(repository),
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
            "schema_version": 1,
            "status": "completed",
            "command": (
                f"{repository / 'experiments/static/run_perturbed_sweeps.py'} "
                "--plic_fallback LVIRA "
                "--rescue_profile exact_linear_support_only "
                "--corner_behavior_profile pre_f8_corner"
            ),
            "parameters": {
                "plic_fallback": "LVIRA",
                "rescue_profile": "exact_linear_support_only",
                "corner_behavior_profile": "pre_f8_corner",
            },
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
        [{**context, "manifest": _run_manifest(repository)}],
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
            "cell_x": case_index,
            "cell_y": 0,
            "merge_id": case_index,
            "final_facet_class": "linear",
            "construction_path": "direct_fit",
            "fallback_policy": "",
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
            "cell_x",
            "cell_y",
            "merge_id",
            "final_facet_class",
            "construction_path",
            "fallback_policy",
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
    _write_json(bundle / "run_manifest.json", _run_manifest(repository))
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
        [
            "case_index",
            "cell_id",
            "cell_x",
            "cell_y",
            "merge_id",
            "final_facet_class",
            "construction_path",
            "fallback_policy",
            "facet_geometry_json",
        ],
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
            bundle / "vtk" / "true" / f"true_line{case_index}.facet_metadata.json",
            bundle / "vtk" / "reconstructed" / "facets" / f"{case_index}.vtp",
            bundle
            / "vtk"
            / "reconstructed"
            / "facets"
            / f"{case_index}.facet_metadata.json",
            bundle / "vtk" / "reconstructed" / "mixed_cells" / f"{case_index}.vtp",
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

    for relative in (
        "diagnostics/cell_metrics.csv",
        f"raw_runs/{SAVE_NAME}/metrics/cell_metrics.csv",
    ):
        path = root / relative
        fieldnames, rows = _read_csv(path)
        rows[0]["merge_id"] = "7"
        rows[0]["construction_path"] = "plic_fallback"
        rows[0]["fallback_policy"] = "LVIRA"
        _write_csv(path, fieldnames, rows)

    inventory_path = root / "diagnostics" / "run_inventory.csv"
    fieldnames, rows = _read_csv(inventory_path)
    rows[0]["merge_events_rows"] = "1"
    rows[0]["unresolved_plic_fallbacks_rows"] = "1"
    _write_csv(inventory_path, fieldnames, rows)


def _snapshot_files(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r:gz") as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }


def _rewrite_snapshot(root: Path, files: dict[str, bytes]) -> None:
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    _snapshot(snapshot_path, files)
    state_path = root / "diagnostics" / "source_state.json"
    state = json.loads(state_path.read_text())
    state["snapshot_sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    state["snapshot_file_count"] = len(files)
    _write_json(state_path, state)


def _rewrite_snapshot_entries(
    root: Path,
    entries: list[tuple[str, bytes, int]],
    *,
    recorded_count=None,
) -> None:
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    _snapshot_entries(snapshot_path, entries)
    state_path = root / "diagnostics" / "source_state.json"
    state = json.loads(state_path.read_text())
    state["snapshot_sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    state["snapshot_file_count"] = (
        len(entries) if recorded_count is None else recorded_count
    )
    _write_json(state_path, state)


def _rewrite_snapshot_bytes(
    root: Path, archive_bytes: bytes, recorded_count: int
) -> None:
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    snapshot_path.write_bytes(archive_bytes)
    state_path = root / "diagnostics" / "source_state.json"
    state = json.loads(state_path.read_text())
    state["snapshot_sha256"] = hashlib.sha256(archive_bytes).hexdigest()
    state["snapshot_file_count"] = recorded_count
    _write_json(state_path, state)


def _replace_command_executable(command: str, executable: str) -> str:
    tokens = shlex.split(command)
    tokens[0] = executable
    return shlex.join(tokens)


def _rewrite_historical_command_root(root: Path, historical_root: str) -> None:
    environment_path = root / "environment.json"
    environment = json.loads(environment_path.read_text())
    environment["repository"]["root"] = historical_root
    _write_json(environment_path, environment)

    controller_path = root / "sweep_manifest.json"
    controller = json.loads(controller_path.read_text())
    controller["command"] = _replace_command_executable(
        controller["command"],
        f"{historical_root}/experiments/static/run_perturbed_sweeps.py",
    )
    _write_json(controller_path, controller)

    consolidated_path = root / "diagnostics" / "run_manifests.jsonl"
    consolidated = json.loads(consolidated_path.read_text())
    consolidated["manifest"]["command"] = _replace_command_executable(
        consolidated["manifest"]["command"],
        f"{historical_root}/experiments/static/lines.py",
    )
    _write_jsonl(consolidated_path, [consolidated])

    raw_path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
    raw = json.loads(raw_path.read_text())
    raw["command"] = _replace_command_executable(
        raw["command"], f"{historical_root}/experiments/static/lines.py"
    )
    _write_json(raw_path, raw)


def _add_argv_to_all_manifests(root: Path) -> None:
    controller_path = root / "sweep_manifest.json"
    controller = json.loads(controller_path.read_text())
    controller["argv"] = shlex.split(controller["command"])
    _write_json(controller_path, controller)

    consolidated_path = root / "diagnostics" / "run_manifests.jsonl"
    consolidated = json.loads(consolidated_path.read_text())
    nested = consolidated["manifest"]
    nested["argv"] = shlex.split(nested["command"])
    _write_jsonl(consolidated_path, [consolidated])

    raw_path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
    raw = json.loads(raw_path.read_text())
    raw["argv"] = shlex.split(raw["command"])
    _write_json(raw_path, raw)


def _retarget_release(root: Path, target_commit: str) -> None:
    config_path = root / "submission_config.resolved.json"
    config = json.loads(config_path.read_text())
    config["source"]["target_commit"] = target_commit
    _write_json(config_path, config)

    state_path = root / "diagnostics" / "source_state.json"
    state = json.loads(state_path.read_text())
    state["source_commit"] = target_commit
    _write_json(state_path, state)

    environment_path = root / "environment.json"
    environment = json.loads(environment_path.read_text())
    environment["repository"]["commit"] = target_commit
    _write_json(environment_path, environment)

    for relative in (
        "diagnostics/run_inventory.csv",
        "diagnostics/case_metrics.csv",
        "diagnostics/cell_metrics.csv",
        "diagnostics/merge_events.csv",
        "diagnostics/unresolved_plic_fallbacks.csv",
    ):
        path = root / relative
        fieldnames, rows = _read_csv(path)
        for row in rows:
            row["source_commit"] = target_commit
        _write_csv(path, fieldnames, rows)

    geometry_path = root / "diagnostics" / "case_geometry.jsonl"
    geometry_rows = [
        json.loads(line) for line in geometry_path.read_text().splitlines()
    ]
    for row in geometry_rows:
        row["source_commit"] = target_commit
    _write_jsonl(geometry_path, geometry_rows)

    manifests_path = root / "diagnostics" / "run_manifests.jsonl"
    manifest_rows = [
        json.loads(line) for line in manifests_path.read_text().splitlines()
    ]
    for row in manifest_rows:
        row["source_commit"] = target_commit
        row["manifest"]["source_commit"] = target_commit
    _write_jsonl(manifests_path, manifest_rows)

    raw_path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
    raw = json.loads(raw_path.read_text())
    raw["source_commit"] = target_commit
    _write_json(raw_path, raw)

    repository = root.parent
    tracked = _git(repository, "ls-files", "-z").rstrip(b"\0").split(b"\0")
    _rewrite_snapshot(
        root,
        {
            relative.decode(): (repository / relative.decode()).read_bytes()
            for relative in tracked
            if relative
        },
    )


def _omit_rescue_profile_everywhere(root: Path) -> None:
    for relative in (
        "diagnostics/run_inventory.csv",
        "diagnostics/case_metrics.csv",
        "diagnostics/cell_metrics.csv",
        "diagnostics/merge_events.csv",
        "diagnostics/unresolved_plic_fallbacks.csv",
    ):
        path = root / relative
        fieldnames, rows = _read_csv(path)
        for row in rows:
            row["rescue_profile"] = ""
        _write_csv(path, fieldnames, rows)

    geometry_path = root / "diagnostics" / "case_geometry.jsonl"
    geometry_rows = [
        json.loads(line) for line in geometry_path.read_text().splitlines()
    ]
    for row in geometry_rows:
        row["rescue_profile"] = ""
    _write_jsonl(geometry_path, geometry_rows)

    manifests_path = root / "diagnostics" / "run_manifests.jsonl"
    consolidated = json.loads(manifests_path.read_text())
    consolidated["rescue_profile"] = ""
    consolidated["manifest"]["parameters"].pop("rescue_profile", None)
    consolidated["manifest"]["command"] = consolidated["manifest"]["command"].replace(
        "--rescue_profile exact_linear_support_only ", ""
    )
    _write_jsonl(manifests_path, [consolidated])

    raw_path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
    raw = json.loads(raw_path.read_text())
    raw["parameters"].pop("rescue_profile", None)
    raw["command"] = raw["command"].replace(
        "--rescue_profile exact_linear_support_only ", ""
    )
    _write_json(raw_path, raw)


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


def test_legacy_string_commands_reject_historical_paths_with_spaces(tmp_path):
    root = _make_release(tmp_path / "release")
    _rewrite_historical_command_root(
        root, "/archived experiment roots/Interface Reconstruction 2026"
    )

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "argv is required; legacy command-string parsing is restricted" in messages


def test_argv_commands_support_relocation_and_historical_spaces(tmp_path):
    root = _make_release(tmp_path / "release")
    _rewrite_historical_command_root(
        root, "/archived experiment roots/Interface Reconstruction 2026"
    )
    _add_argv_to_all_manifests(root)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    assert report.ok, _messages(report)


def test_symlink_alias_cannot_impersonate_reviewed_driver(tmp_path):
    root = _make_release(tmp_path / "release")
    alias = tmp_path / "lines-driver-alias.py"
    alias.symlink_to(tmp_path / "experiments" / "static" / "lines.py")

    consolidated_path = root / "diagnostics" / "run_manifests.jsonl"
    consolidated = json.loads(consolidated_path.read_text())
    consolidated["manifest"]["command"] = _replace_command_executable(
        consolidated["manifest"]["command"], str(alias)
    )
    _write_jsonl(consolidated_path, [consolidated])

    raw_path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
    raw = json.loads(raw_path.read_text())
    raw["command"] = _replace_command_executable(raw["command"], str(alias))
    _write_json(raw_path, raw)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "does not invoke reviewed driver" in messages


def test_future_repository_relative_argv_schema_is_supported(tmp_path):
    root = _make_release(tmp_path / "release")

    controller_path = root / "sweep_manifest.json"
    controller = json.loads(controller_path.read_text())
    controller_tokens = shlex.split(controller.pop("command"))
    controller_tokens[0] = "experiments/static/run_perturbed_sweeps.py"
    controller["argv"] = controller_tokens
    _write_json(controller_path, controller)

    consolidated_path = root / "diagnostics" / "run_manifests.jsonl"
    consolidated = json.loads(consolidated_path.read_text())
    child = consolidated["manifest"]
    child_tokens = shlex.split(child.pop("command"))
    child_tokens[0] = "experiments/static/lines.py"
    child["argv"] = child_tokens
    _write_jsonl(consolidated_path, [consolidated])

    raw_path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
    raw = json.loads(raw_path.read_text())
    raw_tokens = shlex.split(raw.pop("command"))
    raw_tokens[0] = "experiments/static/lines.py"
    raw["argv"] = raw_tokens
    _write_json(raw_path, raw)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    assert report.ok, _messages(report)


def test_future_source_release_requires_argv(tmp_path, monkeypatch):
    root = _make_release(tmp_path / "release")
    (tmp_path / "future-source-marker.txt").write_text("future\n", encoding="utf-8")
    _git(tmp_path, "add", "future-source-marker.txt")
    _git(tmp_path, "commit", "-q", "-m", "future source release")
    changed_commit = _git(tmp_path, "rev-parse", "HEAD").decode().strip()
    monkeypatch.setattr(audit_module, "FINAL_SOURCE_COMMIT", changed_commit)
    _retarget_release(root, changed_commit)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "argv is required; legacy command-string parsing is restricted" in messages


def test_conflicting_argv_and_command_are_rejected(tmp_path):
    root = _make_release(tmp_path / "release")
    raw_path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
    raw = json.loads(raw_path.read_text())
    raw["argv"] = shlex.split(raw["command"])
    raw["argv"][0] = "experiments/static/circles.py"
    _write_json(raw_path, raw)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "argv does not exactly match the tokenized command string" in messages


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


def test_csv_trailing_fields_are_rejected_before_reconciliation(tmp_path):
    root = _make_release(tmp_path / "release")
    path = root / "raw_runs" / SAVE_NAME / "metrics" / "case_metrics.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] += ",injected"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "has trailing fields beyond the header" in messages
    assert "injected" in messages


@pytest.mark.parametrize("malicious_number", ["1e999999999", "9" * 1000])
def test_numeric_parsing_rejects_unbounded_exponents_and_lengths(
    tmp_path, malicious_number
):
    root = _make_release(tmp_path / "release")
    _mutate_csv(
        root / "diagnostics" / "run_inventory.csv",
        0,
        "resolution",
        malicious_number,
    )

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "outside the audited range" in messages or "is too long" in messages


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


def test_target_commit_must_exist_as_a_git_object(tmp_path, monkeypatch):
    root = _make_release(tmp_path / "release")
    nonexistent = "b" * 40
    monkeypatch.setattr(audit_module, "FINAL_SOURCE_COMMIT", nonexistent)
    config_path = root / "submission_config.resolved.json"
    config = json.loads(config_path.read_text())
    config["source"]["target_commit"] = nonexistent
    _write_json(config_path, config)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "target_commit does not exist as an exact Git commit object" in messages


def test_self_reported_snapshot_hash_cannot_hide_git_byte_mismatch(tmp_path):
    root = _make_release(tmp_path / "release")
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    files = _snapshot_files(snapshot_path)
    files["requirements.txt"] += b"tampered==1\n"
    _rewrite_snapshot(root, files)

    environment_path = root / "environment.json"
    environment = json.loads(environment_path.read_text())
    for fingerprint in environment["input_fingerprints"]:
        if fingerprint["path"] == "requirements.txt":
            fingerprint["size_bytes"] = len(files["requirements.txt"])
            fingerprint["sha256"] = hashlib.sha256(
                files["requirements.txt"]
            ).hexdigest()
    _write_json(environment_path, environment)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert (
        "source snapshot member size exceeds or differs from target_commit bound "
        "for requirements.txt" in messages
    )


def test_every_audit_git_operation_disables_replace_objects(tmp_path, monkeypatch):
    root = _make_release(tmp_path / "release")
    real_run = subprocess.run
    git_calls = []

    def checked_run(command, *args, **kwargs):
        if command and command[0] == "git":
            git_calls.append((command, kwargs.get("env", {})))
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(audit_module.subprocess, "run", checked_run)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    assert report.ok, _messages(report)
    assert git_calls
    assert all(command[1] == "--no-replace-objects" for command, _ in git_calls)
    assert all(
        environment.get("GIT_NO_REPLACE_OBJECTS") == "1" for _, environment in git_calls
    )


def test_git_replace_ref_cannot_substitute_for_target_commit(tmp_path):
    root = _make_release(tmp_path / "release")
    target_commit = COMMIT
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text(
        "numpy==1.23.4\nreplacement-only==1\n", encoding="utf-8"
    )
    _git(tmp_path, "add", "requirements.txt")
    _git(tmp_path, "commit", "-q", "-m", "replacement tree")
    replacement_commit = _git(tmp_path, "rev-parse", "HEAD").decode().strip()
    _git(tmp_path, "replace", target_commit, replacement_commit)
    assert b"replacement-only==1" in _git(
        tmp_path, "show", f"{target_commit}:requirements.txt"
    )

    snapshot_files = {
        relative.decode(): (tmp_path / relative.decode()).read_bytes()
        for relative in _git(tmp_path, "ls-files", "-z").rstrip(b"\0").split(b"\0")
        if relative
    }
    _rewrite_snapshot(root, snapshot_files)
    environment_path = root / "environment.json"
    environment = json.loads(environment_path.read_text())
    for fingerprint in environment["input_fingerprints"]:
        if fingerprint["path"] == "requirements.txt":
            data = snapshot_files["requirements.txt"]
            fingerprint["size_bytes"] = len(data)
            fingerprint["sha256"] = hashlib.sha256(data).hexdigest()
    _write_json(environment_path, environment)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "target_commit bound for requirements.txt" in messages


def test_git_info_export_ignore_cannot_hide_tracked_source_file(tmp_path):
    root = _make_release(tmp_path / "release")
    (tmp_path / ".git" / "info" / "attributes").write_text(
        "requirements.txt export-ignore\n", encoding="utf-8"
    )

    archive_bytes = _git(tmp_path, "archive", COMMIT)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        assert "requirements.txt" not in archive.getnames()

    files = _snapshot_files(root / "diagnostics" / "source_snapshot.tar.gz")
    files.pop("requirements.txt")
    _rewrite_snapshot(root, files)
    environment_path = root / "environment.json"
    environment = json.loads(environment_path.read_text())
    environment["input_fingerprints"] = [
        row
        for row in environment["input_fingerprints"]
        if row["path"] != "requirements.txt"
    ]
    _write_json(environment_path, environment)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert (
        "source snapshot is missing tracked files from target_commit: "
        "'requirements.txt'" in messages
    )


@pytest.mark.parametrize("bad_mode", [0o755, 0o4644])
def test_source_snapshot_complete_mode_must_match_git_tree(tmp_path, bad_mode):
    root = _make_release(tmp_path / "release")
    files = _snapshot_files(root / "diagnostics" / "source_snapshot.tar.gz")
    entries = [
        (relative, data, bad_mode if relative == "requirements.txt" else 0o644)
        for relative, data in sorted(files.items())
    ]
    _rewrite_snapshot_entries(root, entries)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert (
        "source snapshot complete mode differs from target_commit for requirements.txt"
        in messages
    )


@pytest.mark.parametrize("metadata_kind", ["pax", "gnu"])
def test_source_snapshot_metadata_bomb_hits_decompressed_budget(
    tmp_path, metadata_kind
):
    root = _make_release(tmp_path / "release")
    files = _snapshot_files(root / "diagnostics" / "source_snapshot.tar.gz")
    budget = _snapshot_budget(files)
    archive_bytes = _metadata_bomb_archive(metadata_kind, budget + 1)
    _rewrite_snapshot_bytes(root, archive_bytes, len(files))

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "decompressed source snapshot exceeds Git-derived byte budget" in messages


def test_high_ratio_zero_tail_hits_decompressed_budget(tmp_path):
    root = _make_release(tmp_path / "release")
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    files = _snapshot_files(snapshot_path)
    raw_archive = gzip.decompress(snapshot_path.read_bytes())
    budget = _snapshot_budget(files)
    oversized_tail = b"\0" * (budget - len(raw_archive) + 1)
    _rewrite_snapshot_bytes(
        root, gzip.compress(raw_archive + oversized_tail), len(files)
    )

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "decompressed source snapshot exceeds Git-derived byte budget" in messages


def test_gzip_crc_corruption_is_rejected(tmp_path):
    root = _make_release(tmp_path / "release")
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    files = _snapshot_files(snapshot_path)
    corrupted = bytearray(snapshot_path.read_bytes())
    corrupted[-8] ^= 0x01
    _rewrite_snapshot_bytes(root, bytes(corrupted), len(files))

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "gzip source snapshot CRC/trailer validation failed" in messages


def test_truncated_gzip_trailer_is_rejected(tmp_path):
    root = _make_release(tmp_path / "release")
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    files = _snapshot_files(snapshot_path)
    _rewrite_snapshot_bytes(root, snapshot_path.read_bytes()[:-4], len(files))

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "gzip source snapshot is truncated or missing its CRC/trailer" in messages


def test_missing_second_tar_end_block_is_rejected(tmp_path):
    root = _make_release(tmp_path / "release")
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    files = _snapshot_files(snapshot_path)
    raw_archive = gzip.decompress(snapshot_path.read_bytes())
    data_end = _tar_data_end(raw_archive)
    one_end_block = raw_archive[:data_end] + b"\0" * tarfile.BLOCKSIZE
    _rewrite_snapshot_bytes(root, gzip.compress(one_end_block), len(files))

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "canonical two-zero-block, record-aligned terminator" in messages


def test_extra_zero_tar_record_is_rejected_below_global_budget(tmp_path):
    root = _make_release(tmp_path / "release")
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    files = _snapshot_files(snapshot_path)
    raw_archive = gzip.decompress(snapshot_path.read_bytes())
    _rewrite_snapshot_bytes(
        root,
        gzip.compress(raw_archive + b"\0" * tarfile.RECORDSIZE),
        len(files),
    )

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "canonical two-zero-block, record-aligned terminator" in messages


@pytest.mark.parametrize("tail_kind", ["gzip_member", "compressed_garbage"])
def test_gzip_multi_member_and_compressed_tail_are_rejected(tmp_path, tail_kind):
    root = _make_release(tmp_path / "release")
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    files = _snapshot_files(snapshot_path)
    tail = (
        gzip.compress(b"second member")
        if tail_kind == "gzip_member"
        else b"compressed garbage"
    )
    _rewrite_snapshot_bytes(root, snapshot_path.read_bytes() + tail, len(files))

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "multiple members or trailing compressed data" in messages


def test_nonzero_decompressed_data_after_tar_terminator_is_rejected(tmp_path):
    root = _make_release(tmp_path / "release")
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    files = _snapshot_files(snapshot_path)
    raw_archive = gzip.decompress(snapshot_path.read_bytes())
    _rewrite_snapshot_bytes(root, gzip.compress(raw_archive + b"not zero"), len(files))

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "nonzero trailing decompressed data" in messages


def test_nonzero_padding_hidden_between_two_files_is_rejected(tmp_path):
    files = {"first.txt": b"a", "second.txt": b"second"}
    raw_archive = bytearray(_in_memory_tar(files, tarfile.PAX_FORMAT))
    with tarfile.open(fileobj=io.BytesIO(raw_archive), mode="r:") as archive:
        first, second = archive.getmembers()
    padding_offset = first.offset_data + first.size
    assert padding_offset < second.offset
    raw_archive[padding_offset] = 0x7F

    with tarfile.open(fileobj=io.BytesIO(raw_archive), mode="r:") as archive:
        extracted = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
        }
    assert extracted == files

    report = _audit_snapshot_bytes(tmp_path, files, bytes(raw_archive))

    assert not report.ok
    assert "nonzero padding after member 'first.txt' byte" in _messages(report)


@pytest.mark.parametrize(
    ("archive_format", "extension_type"),
    [
        (tarfile.PAX_FORMAT, tarfile.XHDTYPE),
        (tarfile.GNU_FORMAT, tarfile.GNUTYPE_LONGNAME),
    ],
)
def test_nonzero_pax_and_gnu_extension_padding_is_rejected(
    tmp_path,
    archive_format,
    extension_type,
):
    long_name = f"source/{'nested-name-' * 12}.txt"
    files = {long_name: b"first", "second.txt": b"second"}
    raw_archive = bytearray(_in_memory_tar(files, archive_format))
    extension = tarfile.TarInfo.frombuf(
        raw_archive[: tarfile.BLOCKSIZE], "utf-8", "surrogateescape"
    )
    assert extension.type == extension_type
    padding_offset = tarfile.BLOCKSIZE + extension.size
    padding_size = (-extension.size) % tarfile.BLOCKSIZE
    assert padding_size > 0
    raw_archive[padding_offset] = 0x7F

    report = _audit_snapshot_bytes(tmp_path, files, bytes(raw_archive))

    assert not report.ok
    assert "nonzero extension-member padding byte" in _messages(report)


@pytest.mark.parametrize(
    ("archive_format", "extension_type"),
    [
        (tarfile.PAX_FORMAT, tarfile.XHDTYPE),
        (tarfile.GNU_FORMAT, tarfile.GNUTYPE_LONGNAME),
    ],
)
def test_pax_and_gnu_extension_metadata_size_is_individually_bounded(
    tmp_path,
    archive_format,
    extension_type,
):
    long_name = "source/" + "a" * (audit_module.MAX_TAR_METADATA_BYTES_PER_MEMBER + 128)
    files = {long_name: b"content"}
    raw_archive = _in_memory_tar(files, archive_format)
    extension = tarfile.TarInfo.frombuf(
        raw_archive[: tarfile.BLOCKSIZE], "utf-8", "surrogateescape"
    )
    assert extension.type == extension_type
    assert extension.size > audit_module.MAX_TAR_METADATA_BYTES_PER_MEMBER
    assert len(raw_archive) < _snapshot_budget(files)

    report = _audit_snapshot_bytes(tmp_path, files, raw_archive)

    assert not report.ok
    assert "extension metadata size exceeds per-extension limit" in _messages(report)


def test_source_snapshot_member_count_is_bounded_by_git_tree(tmp_path):
    root = _make_release(tmp_path / "release")
    files = _snapshot_files(root / "diagnostics" / "source_snapshot.tar.gz")
    entries = [(relative, data, 0o644) for relative, data in sorted(files.items())]
    entries.append(("untracked-extra.txt", b"x", 0o644))
    _rewrite_snapshot_entries(root, entries)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "source snapshot member count exceeds target_commit tree bound" in messages


def test_source_snapshot_member_and_total_sizes_are_git_tree_bounded(
    tmp_path, monkeypatch
):
    root = _make_release(tmp_path / "release")
    files = _snapshot_files(root / "diagnostics" / "source_snapshot.tar.gz")
    entries = [
        (
            relative,
            data + b"x" if relative == "requirements.txt" else data,
            0o644,
        )
        for relative, data in sorted(files.items())
    ]
    _rewrite_snapshot_entries(root, entries)

    def fail_if_extracted(*_args, **_kwargs):
        pytest.fail("snapshot payload was extracted before metadata bounds passed")

    monkeypatch.setattr(tarfile.TarFile, "extractfile", fail_if_extracted)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert (
        "source snapshot total uncompressed bytes exceed target_commit tree bound"
        in messages
    )


def test_committed_dead_alternate_call_fails_reviewed_source_fingerprint(
    tmp_path, monkeypatch
):
    root = _make_release(tmp_path / "release")
    source_path = tmp_path / "experiments" / "static" / "lines.py"
    source_path.write_bytes(
        source_path.read_bytes()
        + b"\n\ndef dead_alternate_path():\n    return runReconstruction()\n"
    )
    _git(tmp_path, "add", "experiments/static/lines.py")
    _git(tmp_path, "commit", "-q", "-m", "add dead alternate path")
    changed_commit = _git(tmp_path, "rev-parse", "HEAD").decode().strip()
    monkeypatch.setattr(audit_module, "FINAL_SOURCE_COMMIT", changed_commit)
    _retarget_release(root, changed_commit)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "production source fingerprint mismatch" in messages
    assert "experiments/static/lines.py" in messages
    assert "source snapshot bytes differ from target_commit" not in messages


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


@pytest.mark.parametrize("source", ["consolidated", "raw"])
def test_case_geometry_jsonl_reconciliation_rejects_value_tampering(tmp_path, source):
    root = _make_release(tmp_path / "release")
    path = (
        root / "diagnostics" / "case_geometry.jsonl"
        if source == "consolidated"
        else root / "raw_runs" / SAVE_NAME / "metrics" / "case_geometry.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["angle"] = 99.0
    _write_jsonl(path, rows)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "raw/consolidated JSON value mismatch" in messages
    assert "case_geometry.jsonl" in messages
    assert "$.angle" in messages


@pytest.mark.parametrize("source", ["consolidated", "raw"])
def test_run_manifest_jsonl_reconciliation_rejects_nested_tampering(tmp_path, source):
    root = _make_release(tmp_path / "release")
    if source == "consolidated":
        path = root / "diagnostics" / "run_manifests.jsonl"
        row = json.loads(path.read_text())
        row["manifest"]["timestamp_utc"] = "tampered"
        _write_jsonl(path, [row])
    else:
        path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
        row = json.loads(path.read_text())
        row["timestamp_utc"] = "tampered"
        _write_json(path, row)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "raw/consolidated JSON value mismatch" in messages
    assert "run_manifests.jsonl" in messages
    assert "$.timestamp_utc" in messages


def test_jsonl_reconciliation_normalizes_numbers_and_safe_relative_paths(tmp_path):
    root = _make_release(tmp_path / "release")
    geometry_path = root / "raw_runs" / SAVE_NAME / "metrics" / "case_geometry.jsonl"
    geometry_rows = [
        json.loads(line) for line in geometry_path.read_text().splitlines()
    ]
    geometry_rows[0]["angle"] = 0
    geometry_rows[0]["truth_vtp"] = "./vtk/true/true_line0.vtp"
    _write_jsonl(geometry_path, geometry_rows)

    manifest_path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parameters"]["perturb_wiggle"] = 0
    _write_json(manifest_path, manifest)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    assert report.ok, _messages(report)


def test_matching_raw_and_consolidated_non_lvira_fallbacks_still_fail(tmp_path):
    root = _make_release(tmp_path / "release")
    _add_provenance_rows(root)
    for relative in (
        "diagnostics/unresolved_plic_fallbacks.csv",
        f"raw_runs/{SAVE_NAME}/metrics/unresolved_plic_fallbacks.csv",
    ):
        _mutate_csv(root / relative, 0, "policy", "ELVIRA")

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert messages.count("policy differs from production") >= 2
    assert "'ELVIRA' != 'LVIRA'" in messages


def test_plic_fallback_cell_requires_lvira_policy(tmp_path):
    root = _make_release(tmp_path / "release")
    _add_provenance_rows(root)
    for relative in (
        "diagnostics/cell_metrics.csv",
        f"raw_runs/{SAVE_NAME}/metrics/cell_metrics.csv",
    ):
        _mutate_csv(root / relative, 0, "fallback_policy", "ELVIRA")

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "plic_fallback policy differs from production" in messages
    assert "'ELVIRA' != 'LVIRA'" in messages


def test_plic_fallback_component_requires_matching_event(tmp_path):
    root = _make_release(tmp_path / "release")
    _add_provenance_rows(root)
    for relative in (
        "diagnostics/cell_metrics.csv",
        f"raw_runs/{SAVE_NAME}/metrics/cell_metrics.csv",
    ):
        _mutate_csv(root / relative, 0, "merge_id", "8")

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "plic_fallback component has no unresolved fallback event" in messages
    assert "unresolved fallback event has no plic_fallback cell component" in messages


def test_fallback_event_requires_plic_cell_component(tmp_path):
    root = _make_release(tmp_path / "release")
    _add_provenance_rows(root)
    for relative in (
        "diagnostics/cell_metrics.csv",
        f"raw_runs/{SAVE_NAME}/metrics/cell_metrics.csv",
    ):
        _mutate_csv(root / relative, 0, "construction_path", "direct_fit")
        _mutate_csv(root / relative, 0, "fallback_policy", "")

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "unresolved fallback event has no plic_fallback cell component" in messages


@pytest.mark.parametrize(
    ("field_name", "value", "expected_message"),
    [
        (
            "fallback_policy",
            "LVIRA",
            "nonfallback construction_path 'direct_fit' cannot claim fallback_policy",
        ),
        ("construction_path", "", "has no construction_path provenance"),
    ],
)
def test_nonfallback_cell_provenance_is_fail_closed(
    tmp_path, field_name, value, expected_message
):
    root = _make_release(tmp_path / "release")
    for relative in (
        "diagnostics/cell_metrics.csv",
        f"raw_runs/{SAVE_NAME}/metrics/cell_metrics.csv",
    ):
        _mutate_csv(root / relative, 0, field_name, value)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    assert not report.ok
    assert expected_message in _messages(report)


def test_reconciliation_reports_missing_and_unexpected_stable_keys(tmp_path):
    root = _make_release(tmp_path / "release")
    _mutate_csv(root / "diagnostics" / "cell_metrics.csv", 0, "cell_id", "99,99")

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "missing consolidated row" in messages
    assert "cell_id=0,0" in messages
    assert "cell_id is not canonical for cell_x/cell_y" in messages
    assert "'99,99' != '0,0'" in messages


@pytest.mark.parametrize(
    ("field_name", "value", "expected_message"),
    [
        ("cell_id", "00,0", "cell_id is not canonical"),
        ("cell_x", "0.5", "cell_x is not a canonical integer"),
        ("cell_y", "1e999999999", "cell_y is not a canonical integer"),
    ],
)
def test_cell_ids_require_bounded_integer_coordinates(
    tmp_path, field_name, value, expected_message
):
    root = _make_release(tmp_path / "release")
    for relative in (
        "diagnostics/cell_metrics.csv",
        f"raw_runs/{SAVE_NAME}/metrics/cell_metrics.csv",
    ):
        _mutate_csv(root / relative, 0, field_name, value)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert expected_message in messages


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
    ("field_name", "option", "expected"),
    [
        ("plic_fallback", "--plic_fallback", "LVIRA"),
        ("rescue_profile", "--rescue_profile", "exact_linear_support_only"),
        (
            "corner_behavior_profile",
            "--corner_behavior_profile",
            "pre_f8_corner",
        ),
    ],
)
@pytest.mark.parametrize("mutation", ["parameter", "command"])
def test_controller_binds_every_production_profile(
    tmp_path, field_name, option, expected, mutation
):
    root = _make_release(tmp_path / "release")
    path = root / "sweep_manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "parameter":
        manifest["parameters"][field_name] = "different"
    else:
        manifest["command"] = manifest["command"].replace(
            f"{option} {expected}", f"{option} different"
        )
    _write_json(path, manifest)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "controller" in messages
    assert field_name in messages or option in messages


@pytest.mark.parametrize(
    ("field_name", "option", "expected"),
    [
        ("plic_fallback", "--plic_fallback", "LVIRA"),
        ("rescue_profile", "--rescue_profile", "exact_linear_support_only"),
        (
            "corner_behavior_profile",
            "--corner_behavior_profile",
            "pre_f8_corner",
        ),
    ],
)
def test_explicit_raw_profile_requires_matching_raw_command_evidence(
    tmp_path, field_name, option, expected
):
    root = _make_release(tmp_path / "release")
    path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["command"] = " ".join(
        shlex.split(manifest["command"].replace(f"{option} {expected}", ""))
    )
    _write_json(path, manifest)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert (
        "raw child manifest command does not contain exactly one matching" in messages
    )
    assert option in messages


def test_consolidated_child_command_is_profile_bound(tmp_path):
    root = _make_release(tmp_path / "release")
    path = root / "diagnostics" / "run_manifests.jsonl"
    row = json.loads(path.read_text())
    row["manifest"]["command"] = row["manifest"]["command"].replace(
        "--plic_fallback LVIRA", "--plic_fallback Youngs"
    )
    _write_jsonl(path, [row])

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "consolidated child manifest command" in messages
    assert "--plic_fallback" in messages


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


def test_authoritative_non_zalesak_rescue_inheritance_is_allowed(tmp_path):
    root = _make_release(tmp_path / "release")
    _add_provenance_rows(root)
    _omit_rescue_profile_everywhere(root)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    assert report.ok, _messages(report)
    assert report.summaries["rescue_profile_inherited_runs"] == 1
    assert len(report.warnings) == 1
    assert "Audited rescue_profile='exact_linear_support_only'" in report.warnings[0]
    assert "1 non-Zalesak runs" in report.warnings[0]


def test_rescue_omission_pattern_must_match_across_provenance_layers(tmp_path):
    root = _make_release(tmp_path / "release")
    _omit_rescue_profile_everywhere(root)
    _mutate_csv(
        root / "diagnostics" / "run_inventory.csv",
        0,
        "rescue_profile",
        "exact_linear_support_only",
    )

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "omission pattern is inconsistent across provenance layers" in messages
    assert len(report.warnings) == 1
    assert "Audited rescue_profile" in report.warnings[0]


def test_matching_child_option_allows_redundant_raw_field_omission(tmp_path):
    root = _make_release(tmp_path / "release")
    _omit_rescue_profile_everywhere(root)
    consolidated_path = root / "diagnostics" / "run_manifests.jsonl"
    consolidated = json.loads(consolidated_path.read_text())
    consolidated["manifest"]["command"] += " --rescue_profile exact_linear_support_only"
    _write_jsonl(consolidated_path, [consolidated])
    raw_path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
    raw = json.loads(raw_path.read_text())
    raw["command"] += " --rescue_profile exact_linear_support_only"
    _write_json(raw_path, raw)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    assert report.ok, _messages(report)
    assert report.summaries["rescue_profile_inherited_runs"] == 1
    assert len(report.warnings) == 1


def test_rescue_command_omission_pattern_must_match_between_manifests(tmp_path):
    root = _make_release(tmp_path / "release")
    _omit_rescue_profile_everywhere(root)
    raw_path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
    raw = json.loads(raw_path.read_text())
    raw["command"] += " --rescue_profile exact_linear_support_only"
    _write_json(raw_path, raw)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "command-option omission pattern is inconsistent" in messages
    assert len(report.warnings) == 1


@pytest.mark.parametrize("conflict_source", ["raw_parameter", "raw_command"])
def test_inherited_rescue_profile_rejects_lower_level_conflicts(
    tmp_path, conflict_source
):
    root = _make_release(tmp_path / "release")
    _omit_rescue_profile_everywhere(root)
    raw_path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
    raw = json.loads(raw_path.read_text())
    if conflict_source == "raw_parameter":
        raw["parameters"]["rescue_profile"] = "different"
    else:
        raw["command"] += " --rescue_profile different"
    _write_json(raw_path, raw)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "rescue_profile" in messages
    assert "different" in messages


def test_rescue_inheritance_rejects_conflicting_global_pin(tmp_path):
    root = _make_release(tmp_path / "release")
    _omit_rescue_profile_everywhere(root)
    manifest_path = root / "sweep_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parameters"]["rescue_profile"] = "different"
    _write_json(manifest_path, manifest)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "omission cannot be inherited" in messages
    assert "controller parameter rescue_profile differs from production" in messages


def test_rescue_inheritance_rejects_missing_child_command_proof(tmp_path):
    root = _make_release(tmp_path / "release")
    _omit_rescue_profile_everywhere(root)
    raw_path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
    raw = json.loads(raw_path.read_text())
    raw.pop("command")
    _write_json(raw_path, raw)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "raw child manifest command evidence is invalid" in messages
    assert "command is absent or empty" in messages


def test_rescue_inheritance_rejects_unverified_child_driver(tmp_path):
    root = _make_release(tmp_path / "release")
    _omit_rescue_profile_everywhere(root)
    raw_path = root / "raw_runs" / SAVE_NAME / "run_manifest.json"
    raw = json.loads(raw_path.read_text())
    raw["command"] = raw["command"].replace(
        "experiments/static/lines.py", "experiments/static/circles.py"
    )
    _write_json(raw_path, raw)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "raw child manifest does not invoke reviewed driver" in messages


def test_rescue_inheritance_rejects_unproven_source_default(tmp_path):
    root = _make_release(tmp_path / "release")
    _omit_rescue_profile_everywhere(root)
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    files = _snapshot_files(snapshot_path)
    files[
        "util/reconstruction.py"
    ] = b"""
def _run_with_merge(m, merge_ids, algo_kwargs):
    m.fitFacets(merge_ids, rescue_profile=None)
"""
    _rewrite_snapshot(root, files)

    report = audit_final_release(root, required_runs=1, required_cases=2)

    messages = _messages(report)
    assert not report.ok
    assert "omission cannot be inherited" in messages
    assert (
        "source snapshot member size exceeds or differs from target_commit bound"
        in messages
    )
    assert "source snapshot lacks production fingerprint file" in messages


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
