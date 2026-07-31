#!/usr/bin/env python3
"""Regenerate, prove, accept, and atomically publish final figure candidates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submission.accept_figure_candidates import (
    DEFAULT_ALLOWLIST,
    FigureAcceptanceError,
    accept_figure_candidates,
    load_candidate_allowlist,
)
from submission.audit_final_release import audit_final_release, verify_sha256_manifest
from submission.final_figure_provenance import (
    atomic_write_json,
    file_sha256,
    load_json_object,
    release_figure_anchor,
    snapshot_record,
)
from submission.generator_checkout import verify_generator_checkout


class FinalFigureOrchestrationError(RuntimeError):
    """Raised when final figures cannot be proven and published."""


PROFILE = {
    "plic_fallback": "LVIRA",
    "corner_behavior_profile": "pre_f8_corner",
    "rescue_profile": "exact_linear_support_only",
}
EXPERIMENTS = ("lines", "squares", "circles", "ellipses", "zalesak")
RELEASE_METHODS = {
    "lines": ("Youngs", "ELVIRA", "LVIRA", "safe_linear", "linear"),
    "circles": (
        "Youngs",
        "ELVIRA",
        "LVIRA",
        "safe_linear",
        "linear",
        "safe_circle",
        "circular",
    ),
    "ellipses": (
        "Youngs",
        "ELVIRA",
        "LVIRA",
        "safe_linear",
        "linear",
        "safe_circle",
        "circular",
    ),
    "squares": (
        "Youngs",
        "ELVIRA",
        "LVIRA",
        "safe_linear",
        "linear",
        "linear+corner",
        "safe_circle",
        "circular",
    ),
    "zalesak": (
        "Youngs",
        "ELVIRA",
        "LVIRA",
        "safe_linear",
        "linear",
        "safe_circle",
        "circular",
        "circular+corner",
    ),
}
RELEASE_RUN_COUNTS = {
    "lines": 150,
    "circles": 210,
    "ellipses": 210,
    "squares": 200,
    "zalesak": 200,
}
MAINTEXT_METHODS = {
    "lines": ("Youngs", "ELVIRA", "LVIRA", "linear"),
    "squares": ("ELVIRA", "LVIRA", "linear", "linear+corner"),
    "circles": ("ELVIRA", "LVIRA", "linear", "circular"),
    "ellipses": ("ELVIRA", "LVIRA", "linear", "circular"),
    "zalesak": ("ELVIRA", "LVIRA", "circular", "circular+corner"),
}
MAINTEXT_CASES = {
    "lines": {"case_index": 6, "resolution": 0.32, "wiggle": 0.3, "seed": 0},
    "squares": {"case_index": 24, "resolution": 0.5, "wiggle": 0.1, "seed": 0},
    "circles": {"case_index": 12, "resolution": 0.32, "wiggle": 0.1, "seed": 0},
    "ellipses": {"case_index": 12, "resolution": 0.32, "wiggle": 0.1, "seed": 0},
    "zalesak": {"case_index": 12, "resolution": 1.0, "wiggle": 0.1, "seed": 0},
}
RESOLUTION_CASES = {
    "lines": (0, "linear"),
    "squares": (22, "linear+corner"),
    "circles": (12, "circular"),
    "ellipses": (12, "circular"),
    "zalesak": (20, "circular+corner"),
}
RESOLUTION_VALUES = (0.16, 0.32, 0.64)
RESOLUTION_WIGGLES = (0.0, 0.1)
C0_RESOLUTIONS = {
    "ellipses": (0.32, 0.5, 0.64, 1.0, 1.28, 1.5),
    "zalesak": (0.5, 0.64, 1.0, 1.28, 1.5),
}
C0_VARIANTS = {
    "ellipses": {
        "linear": ("linear", False),
        "linear+C0": ("linear", True),
        "circular": ("circular", False),
    },
    "zalesak": {
        "circular": ("circular", False),
        "circular+C0": ("circular", True),
        "circular+corner": ("circular+corner", False),
    },
}
C0_WIGGLES = (0.0, 0.05, 0.1, 0.2, 0.3)
ALL_METHOD_FILES = {
    "lines": "lines_all_methods_2x2.pdf",
    "squares": "squares_all_methods_2x2.pdf",
    "circles": "circles_all_methods_5x2_axes.pdf",
    "ellipses": "ellipses_all_methods_5x2_axes.pdf",
    "zalesak": "zalesak_all_methods_2x2.pdf",
}
ORCHESTRATION_MANIFEST = "provenance/final_figure_orchestration.json"


def _numbers(values: Sequence[float]) -> str:
    return ",".join(str(value) for value in values)


def _same_number(actual: object, expected: float) -> bool:
    try:
        return abs(float(actual) - expected) <= 1e-12
    except (TypeError, ValueError):
        return False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalFigureOrchestrationError(message)


def _profile_from_generation(payload: Mapping[str, object], label: str) -> dict:
    generation = payload.get("generation_provenance")
    _require(isinstance(generation, dict), f"{label} lacks generation provenance")
    return generation


def _validate_generation(
    payload: Mapping[str, object], commit: str, label: str
) -> None:
    generation = _profile_from_generation(payload, label)
    _require(
        generation.get("source_commit") == commit, f"{label} has wrong source commit"
    )
    _require(generation.get("source_dirty") is False, f"{label} reports dirty source")
    _require(not generation.get("source_status"), f"{label} reports source changes")
    _require(
        generation.get("reconstruction_profile") == PROFILE,
        f"{label} has wrong profile",
    )


def validate_final_release_contract(release_root: Path) -> dict:
    report = audit_final_release(release_root)
    _require(report.ok, "Final release audit failed: " + "; ".join(report.errors))
    checksum_errors = verify_sha256_manifest(release_root)
    _require(
        not checksum_errors,
        "Final release checksum failed: " + "; ".join(checksum_errors),
    )
    config = load_json_object(Path(release_root) / "submission_config.resolved.json")
    sweep = load_json_object(Path(release_root) / "sweep_manifest.json")
    grid = config.get("benchmark_grid")
    benchmarks = config.get("benchmarks")
    totals = config.get("planned_totals")
    production = config.get("production_method")
    _require(
        isinstance(grid, dict) and isinstance(benchmarks, dict),
        "Release grid is missing",
    )
    _require(
        isinstance(totals, dict) and isinstance(production, dict),
        "Release totals are missing",
    )
    _require(grid.get("seed") == 0, "Release seed must be 0")
    _require(
        grid.get("trials_per_setting") == 25, "Release must use 25 cases per setting"
    )
    _require(
        grid.get("wiggles") == [0.0, 0.05, 0.1, 0.2, 0.3], "Release wiggle grid differs"
    )
    _require(
        grid.get("full_resolutions") == [0.32, 0.5, 0.64, 1.0, 1.28, 1.5],
        "Full resolution grid differs",
    )
    _require(
        grid.get("short_resolutions") == [0.5, 0.64, 1.0, 1.28, 1.5],
        "Short resolution grid differs",
    )
    _require(
        {
            "plic_fallback": production.get("unresolved_orientation_fallback"),
            "corner_behavior_profile": production.get("corner_behavior_profile"),
            "rescue_profile": production.get("rescue_profile"),
        }
        == PROFILE,
        "Release reconstruction profile differs",
    )
    for experiment, methods in RELEASE_METHODS.items():
        item = benchmarks.get(experiment)
        _require(isinstance(item, dict), f"Release lacks {experiment}")
        _require(
            item.get("methods") == list(methods),
            f"Release methods differ for {experiment}",
        )
        _require(
            item.get("planned_runs") == RELEASE_RUN_COUNTS[experiment],
            f"Release run count differs for {experiment}",
        )
    _require(
        totals.get("runs") == 970 and totals.get("cases") == 24250,
        "Release totals must be 970 runs and 24,250 cases",
    )
    _require(sweep.get("status") == "completed", "Release sweep is not completed")
    _require(sweep.get("planned_run_count") == 970, "Sweep planned run count differs")
    _require(sweep.get("successful_run_count") == 970, "Sweep is not 970/970 complete")
    _require(
        sweep.get("failure_count") == 0 and not sweep.get("failures"),
        "Sweep contains failures",
    )
    return {
        "status": "validated",
        "methods": {key: list(value) for key, value in RELEASE_METHODS.items()},
        "run_counts": RELEASE_RUN_COUNTS,
        "seed": 0,
        "cases_per_setting": 25,
        "total_runs": 970,
        "total_cases": 24250,
        "profile": PROFILE,
    }


def validate_maintext_manifest(path: Path) -> dict:
    payload = load_json_object(path)
    specs = payload.get("specs")
    _require(isinstance(specs, dict), "Main-text manifest lacks scientific specs")
    representative = specs.get("representative")
    _require(
        isinstance(representative, dict),
        "Main-text manifest lacks representative specs",
    )
    _require(
        set(representative) == set(EXPERIMENTS), "Main-text experiment set differs"
    )
    for experiment, expected in MAINTEXT_CASES.items():
        actual = representative.get(experiment)
        _require(isinstance(actual, dict), f"Main-text {experiment} spec is missing")
        for key in ("case_index", "seed"):
            _require(
                actual.get(key) == expected[key],
                f"Main-text {experiment} {key} differs",
            )
        for key in ("resolution", "wiggle"):
            _require(
                _same_number(actual.get(key), expected[key]),
                f"Main-text {experiment} {key} differs",
            )
        methods = tuple(
            item[0] for item in actual.get("methods", []) if isinstance(item, list)
        )
        _require(
            methods == MAINTEXT_METHODS[experiment],
            f"Main-text methods differ for {experiment}",
        )
    _require(
        set(payload.get("quantitative", {})) == set(EXPERIMENTS),
        "Main-text quantitative outputs incomplete",
    )
    reps = payload.get("representative")
    _require(
        isinstance(reps, dict) and set(reps) == set(EXPERIMENTS),
        "Main-text representatives incomplete",
    )
    for experiment, variants in reps.items():
        _require(
            isinstance(variants, dict) and set(variants) == {"with_endpoints", "clean"},
            f"Main-text endpoint variants incomplete for {experiment}",
        )
    return {
        "status": "validated",
        "case_settings": MAINTEXT_CASES,
        "methods": {key: list(value) for key, value in MAINTEXT_METHODS.items()},
        "endpoint_variants": ["with_endpoints", "clean"],
    }


def _validate_run_manifest(
    path: Path,
    *,
    commit: str,
    experiment: str,
    method: str,
    resolution: float,
    wiggle: float,
    seed: int,
    case_index: Optional[int],
    do_c0: Optional[bool] = None,
) -> None:
    payload = load_json_object(path)
    _require(
        payload.get("source_commit") == commit, f"Run manifest has wrong commit: {path}"
    )
    _require(
        payload.get("experiment") == experiment,
        f"Run manifest has wrong experiment: {path}",
    )
    params = payload.get("parameters")
    _require(isinstance(params, dict), f"Run manifest is parameterless: {path}")
    _require(
        params.get("facet_algo") == method, f"Run manifest has wrong method: {path}"
    )
    _require(
        _same_number(params.get("resolution"), resolution),
        f"Run manifest has wrong resolution: {path}",
    )
    _require(
        _same_number(params.get("perturb_wiggle"), wiggle),
        f"Run manifest has wrong wiggle: {path}",
    )
    _require(params.get("perturb_seed") == seed, f"Run manifest has wrong seed: {path}")
    _require(
        params.get("plic_fallback") == PROFILE["plic_fallback"],
        f"Run manifest has wrong PLIC fallback: {path}",
    )
    _require(
        params.get("corner_behavior_profile") == PROFILE["corner_behavior_profile"],
        f"Run manifest has wrong corner profile: {path}",
    )
    if experiment == "zalesak":
        _require(
            params.get("rescue_profile") == PROFILE["rescue_profile"],
            f"Run manifest has wrong rescue profile: {path}",
        )
    count_key = {
        "lines": "num_lines",
        "squares": "num_squares",
        "circles": "num_circles",
        "ellipses": "num_ellipses",
        "zalesak": "num_cases",
    }[experiment]
    _require(params.get(count_key) == 25, f"Run manifest must request 25 cases: {path}")
    if case_index is None:
        _require(
            params.get("case_indices") is None,
            f"Full run unexpectedly filters cases: {path}",
        )
    else:
        _require(
            params.get("case_indices") in (str(case_index), [case_index]),
            f"Run manifest has wrong case selection: {path}",
        )
    if do_c0 is not None:
        _require(
            params.get("do_c0") is do_c0, f"Run manifest has wrong C0 setting: {path}"
        )


def validate_resolution_manifest(
    path: Path, plots_root: Path, experiment: str, commit: str
) -> list[Path]:
    payload = load_json_object(path)
    _validate_generation(payload, commit, f"Resolution {experiment} manifest")
    _require(
        payload.get("status") == "completed",
        f"Resolution {experiment} is not completed",
    )
    _require(
        payload.get("endpoint_variants") == "paired",
        f"Resolution {experiment} is not paired",
    )
    runs = payload.get("runs")
    _require(
        isinstance(runs, list) and len(runs) == 6,
        f"Resolution {experiment} must have six runs",
    )
    case_index, method = RESOLUTION_CASES[experiment]
    expected = {
        (r, w, 0, case_index, method)
        for r in RESOLUTION_VALUES
        for w in RESOLUTION_WIGGLES
    }
    actual = set()
    manifests = []
    for run in runs:
        _require(isinstance(run, dict), f"Resolution {experiment} has malformed run")
        _require(
            run.get("status") == "completed",
            f"Resolution {experiment} contains planned/plot-only/existing run",
        )
        key = (
            float(run.get("resolution")),
            float(run.get("wiggle")),
            run.get("seed"),
            run.get("case_index"),
            run.get("algo"),
        )
        actual.add(key)
        manifest = Path(plots_root) / str(run.get("save_name")) / "run_manifest.json"
        _validate_run_manifest(
            manifest,
            commit=commit,
            experiment=experiment,
            method=method,
            resolution=key[0],
            wiggle=key[1],
            seed=0,
            case_index=case_index,
        )
        manifests.append(manifest)
    _require(actual == expected, f"Resolution {experiment} scientific grid differs")
    summary = payload.get("summary_plots", {}).get(experiment)
    _require(
        isinstance(summary, dict) and set(summary) == {"with_endpoints", "clean"},
        f"Resolution {experiment} outputs incomplete",
    )
    return manifests


def validate_c0_manifests(
    paths: Mapping[str, Path], plots_root: Path, commit: str
) -> list[Path]:
    run_manifests = []
    setting_count = 0
    for experiment in ("ellipses", "zalesak"):
        payload = load_json_object(paths[experiment])
        _validate_generation(payload, commit, f"Guarded-C0 {experiment} manifest")
        _require(
            payload.get("status") == "completed",
            f"Guarded-C0 {experiment} is incomplete",
        )
        params = payload.get("parameters")
        _require(
            isinstance(params, dict),
            f"Guarded-C0 {experiment} manifest is parameterless",
        )
        _require(
            params.get("only") == experiment,
            f"Guarded-C0 {experiment} selector differs",
        )
        _require(
            params.get("seeds") == "0" and params.get("case_indices") is None,
            f"Guarded-C0 {experiment} seed/cases differ",
        )
        _require(
            params.get("endpoint_variants") == "paired",
            f"Guarded-C0 {experiment} endpoint variants differ",
        )
        _require(
            params.get("resolutions") == _numbers(C0_RESOLUTIONS[experiment]),
            f"Guarded-C0 {experiment} resolutions differ",
        )
        _require(
            params.get("wiggles") == _numbers(C0_WIGGLES),
            f"Guarded-C0 {experiment} wiggles differ",
        )
        variants = C0_VARIANTS[experiment]
        _require(
            params.get("algos") == ",".join(variants),
            f"Guarded-C0 {experiment} variants differ",
        )
        runs = payload.get("runs")
        expected_count = (
            len(C0_RESOLUTIONS[experiment]) * len(C0_WIGGLES) * len(variants)
        )
        _require(
            isinstance(runs, list) and len(runs) == expected_count,
            f"Guarded-C0 {experiment} setting count differs",
        )
        expected = {
            (r, w, label)
            for r in C0_RESOLUTIONS[experiment]
            for w in C0_WIGGLES
            for label in variants
        }
        actual = set()
        for run in runs:
            _require(
                isinstance(run, dict) and run.get("status") == "completed",
                f"Guarded-C0 {experiment} has non-completed run",
            )
            resolution = float(run.get("resolution"))
            wiggle = float(run.get("wiggle"))
            label = run.get("variant")
            actual.add((resolution, wiggle, label))
            method, do_c0 = variants.get(label, (None, None))
            manifest = (
                Path(plots_root) / str(run.get("save_name")) / "run_manifest.json"
            )
            _validate_run_manifest(
                manifest,
                commit=commit,
                experiment=experiment,
                method=method,
                resolution=resolution,
                wiggle=wiggle,
                seed=0,
                case_index=None,
                do_c0=do_c0,
            )
            run_manifests.append(manifest)
        _require(actual == expected, f"Guarded-C0 {experiment} scientific grid differs")
        outputs = payload.get("outputs")
        _require(isinstance(outputs, dict), f"Guarded-C0 {experiment} outputs missing")
        _require(
            set(outputs.get("summary", {})) == {experiment},
            f"Guarded-C0 {experiment} summary missing",
        )
        reps = outputs.get("representative", {}).get(experiment)
        _require(
            isinstance(reps, dict) and set(reps) == {"with_endpoints", "clean"},
            f"Guarded-C0 {experiment} paired outputs missing",
        )
        setting_count += expected_count
    _require(
        setting_count == 165 and len(run_manifests) == 165,
        "Guarded-C0 contract must contain exactly 165 settings",
    )
    return run_manifests


def validate_plic_metadata(path: Path, commit: str) -> dict:
    payload = load_json_object(path)
    _validate_generation(payload, commit, "PLIC stencil metadata")
    expected = {
        "case_index": 4,
        "center_cell": [14, 13],
        "resolution": 0.32,
        "perturbation_magnitude": 0.3,
        "mesh_seed": 0,
    }
    for key, value in expected.items():
        if isinstance(value, float):
            _require(
                _same_number(payload.get(key), value), f"PLIC parameter {key} differs"
            )
        else:
            _require(payload.get(key) == value, f"PLIC parameter {key} differs")
    return {"status": "validated", **expected}


def validate_staged_metadata(path: Path, commit: str) -> dict:
    payload = load_json_object(path)
    metadata = payload.get("metadata")
    _require(isinstance(metadata, dict), "Staged reconstruction metadata is missing")
    _validate_generation(metadata, commit, "Staged reconstruction metadata")
    expected = {
        "case_index": 22,
        "resolution": 1.0,
        "wiggle": 0.1,
        "seed": 0,
        "radius": 15.0,
        "slot_width": 5.0,
        "slot_top_rel": 10.0,
    }
    for key, value in expected.items():
        if isinstance(value, float):
            _require(
                _same_number(metadata.get(key), value),
                f"Staged parameter {key} differs",
            )
        else:
            _require(metadata.get(key) == value, f"Staged parameter {key} differs")
    return {"status": "validated", **expected}


def run_command(
    command: Sequence[str], cwd: Path, env: Mapping[str, str], log_path: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=cwd, env=dict(env), stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode != 0:
        raise FinalFigureOrchestrationError(
            f"Generator failed ({result.returncode}); see {log_path}"
        )


def stage_all_method_candidates(source: Path, destination: Path) -> list[Path]:
    """Copy only the five allowlisted all-method PDFs, ignoring auxiliaries."""
    copied = []
    for filename in ALL_METHOD_FILES.values():
        source_pdf = Path(source) / filename
        _require(source_pdf.is_file(), f"All-method candidate is missing: {source_pdf}")
        target = Path(destination) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_pdf, target)
        copied.append(target)
    return copied


def _copy(source: Path, destination: Path) -> Path:
    _require(Path(source).is_file(), f"Generated artifact is missing: {source}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    _require(
        file_sha256(source) == file_sha256(destination),
        f"Snapshot copy changed bytes: {source}",
    )
    return destination


def _write_command_record(path: Path, command: Sequence[str], commit: str) -> Path:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "status": "completed",
            "approved_generator_commit": commit,
            "command": list(command),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return path


def _prepare_release_plot_view(release_root: Path, view: Path) -> dict[str, Path]:
    view.mkdir(parents=True)
    aliases: dict[str, Path] = {}
    with (Path(release_root) / "perturbed_sweep.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            required = (
                "experiment",
                "algo",
                "resolution",
                "wiggle",
                "seed",
                "save_name",
            )
            _require(
                all(row.get(key) for key in required),
                "Release CSV lacks run identity columns",
            )
            alias = "perturb_sweep_{}_{}_r{}_w{}_s{}".format(
                row["experiment"],
                row["algo"].lower().replace("+", "plus"),
                row["resolution"].replace(".", "p"),
                row["wiggle"].replace(".", "p"),
                row["seed"],
            )
            source = Path(release_root) / "raw_runs" / row["save_name"]
            _require(source.is_dir(), f"Release run bundle is missing: {source}")
            previous = aliases.setdefault(alias, source)
            _require(
                previous == source, f"Release CSV maps {alias} to multiple bundles"
            )
    for alias, source in aliases.items():
        (view / alias).symlink_to(source, target_is_directory=True)
    return aliases


def _generator_environment(repository: Path) -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repository) if not existing else f"{repository}{os.pathsep}{existing}"
    )
    env["SLACK_NOTIFY"] = "0"
    git_dir = subprocess.run(
        ["git", "rev-parse", "--absolute-git-dir"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    env["GIT_DIR"] = git_dir
    env["GIT_WORK_TREE"] = str(repository)
    return env


def _stage_candidate(source: Path, staging: Path, spec, generator: str) -> dict:
    target = staging / "candidates" / spec.root / spec.pdf
    _copy(source, target)
    return {
        "candidate_id": spec.candidate_id,
        "root": spec.root,
        "pdf": spec.pdf,
        "sha256": file_sha256(target),
        "generator": generator,
    }


def _copy_manifest(
    source: Path, staging: Path, relative: str, role: str, records: list[dict]
) -> Path:
    target = _copy(source, staging / relative)
    records.append(snapshot_record(target, staging, role))
    return target


def _rehash_before_publish(
    staging: Path, manifest_path: Path, manifest_digest: str
) -> None:
    _require(
        file_sha256(manifest_path) == manifest_digest,
        "Orchestration manifest mutated before publish",
    )
    payload = load_json_object(manifest_path)
    for record in payload.get("snapshot_artifacts", []):
        path = staging / record["path"]
        _require(
            path.is_file() and file_sha256(path) == record["sha256"],
            f"Snapshot mutated before publish: {record['path']}",
        )
    specs = {spec.candidate_id: spec for spec in load_candidate_allowlist()}
    for record in payload.get("candidates", []):
        spec = specs[record["candidate_id"]]
        path = staging / "candidates" / spec.root / spec.pdf
        _require(
            path.is_file() and file_sha256(path) == record["sha256"],
            f"Candidate mutated before publish: {spec.candidate_id}",
        )


def finalize_publication(
    *,
    staging: Path,
    output_root: Path,
    manifest_path: Path,
    acceptance_runner: Callable[..., object],
    acceptance_kwargs: Mapping[str, object],
    after_acceptance_hook: Optional[Callable[[Path], None]] = None,
) -> None:
    """Accept a private snapshot and atomically publish it after one last rehash."""

    manifest_digest = file_sha256(manifest_path)
    try:
        acceptance_runner(**dict(acceptance_kwargs))
        if after_acceptance_hook is not None:
            after_acceptance_hook(staging)
        _rehash_before_publish(staging, manifest_path, manifest_digest)
        tree_records = [
            snapshot_record(path, staging, "published_artifact")
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        atomic_write_json(
            staging / "provenance" / "published_tree_sha256.json",
            {"schema_version": 1, "files": tree_records},
        )
        os.replace(staging, output_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def orchestrate_final_figures(
    *,
    repository: Path,
    release_root: Path,
    approved_generator_commit: str,
    output_root: Path,
    allowlist_path: Path = DEFAULT_ALLOWLIST,
    command_runner: Callable[
        [Sequence[str], Path, Mapping[str, str], Path], None
    ] = run_command,
    acceptance_runner: Callable[..., object] = accept_figure_candidates,
    after_acceptance_hook: Optional[Callable[[Path], None]] = None,
) -> Path:
    repository = Path(repository).resolve()
    release_root = Path(release_root).resolve()
    output_root = Path(output_root).resolve()
    _require(
        repository == REPO_ROOT.resolve(),
        "--repository must be the checkout containing this reviewed wrapper",
    )
    _require(not output_root.exists(), f"Output root must not exist: {output_root}")
    release_contract = validate_final_release_contract(release_root)
    anchor = release_figure_anchor(release_root)
    attestation = verify_generator_checkout(
        repository, approved_generator_commit, anchor["source_commit"]
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    execution = Path(tempfile.mkdtemp(prefix="final-figure-execution-"))
    try:
        figure_root = staging / "candidates" / "figure_root"
        c0_root = staging / "candidates" / "c0_root"
        _require(
            not figure_root.exists() and not c0_root.exists(),
            "Candidate roots must start nonexistent",
        )
        (execution / "config").symlink_to(
            repository / "config", target_is_directory=True
        )
        plots_root = execution / "plots"
        plots_root.mkdir()
        release_view = execution / "release_plots"
        release_aliases = _prepare_release_plot_view(release_root, release_view)
        env = _generator_environment(repository)
        python = sys.executable
        generated = execution / "generated"
        logs = staging / "provenance" / "logs"
        snapshot_artifacts: list[dict] = []
        candidates: list[dict] = []
        contracts: dict[str, dict] = {"final_release": release_contract}

        for filename in (
            "submission_config.resolved.json",
            "sweep_manifest.json",
            "perturbed_sweep.csv",
            "SHA256SUMS",
        ):
            _copy_manifest(
                release_root / filename,
                staging,
                f"provenance/release/{filename}",
                f"release_{filename}",
                snapshot_artifacts,
            )

        main_out = generated / "maintext"
        main_cmd = [
            python,
            "-m",
            "experiments.static.generate_section6_maintext_figures",
            "--csv",
            str(release_root / "perturbed_sweep.csv"),
            "--plots_root",
            str(release_view),
            "--out_dir",
            str(main_out),
            "--experiments",
            "all",
            "--figure_groups",
            "quantitative,representative",
            "--case_overrides",
            "lines=6,squares=24,circles=12,ellipses=12,zalesak=12",
            "--endpoint_variants",
            "paired",
        ]
        _require(not main_out.exists(), "Main-text generator root already exists")
        command_runner(main_cmd, execution, env, logs / "maintext.log")
        contracts["maintext"] = validate_maintext_manifest(
            main_out / "maintext_manifest.json"
        )
        _copy_manifest(
            main_out / "maintext_manifest.json",
            staging,
            "provenance/maintext/maintext_manifest.json",
            "maintext_producer_manifest",
            snapshot_artifacts,
        )
        _copy_manifest(
            _write_command_record(
                execution / "maintext_command.json", main_cmd, approved_generator_commit
            ),
            staging,
            "provenance/maintext/command.json",
            "generator_command",
            snapshot_artifacts,
        )

        specs = load_candidate_allowlist(allowlist_path)
        by_id = {spec.candidate_id: spec for spec in specs}
        for experiment in EXPERIMENTS:
            for candidate_id, source in (
                (
                    f"{experiment}_maintext_metrics",
                    main_out / "summary_plots" / f"{experiment}_maintext_metrics.pdf",
                ),
                (
                    f"{experiment}_maintext_representative_with_endpoints",
                    main_out
                    / "representative_cases"
                    / f"{experiment}_maintext_representative_with_endpoints.pdf",
                ),
                (
                    f"{experiment}_maintext_representative_clean",
                    main_out
                    / "representative_cases"
                    / f"{experiment}_maintext_representative_clean.pdf",
                ),
            ):
                candidates.append(
                    _stage_candidate(
                        source, staging, by_id[candidate_id], "section6_maintext"
                    )
                )
            expected = MAINTEXT_CASES[experiment]
            for method in MAINTEXT_METHODS[experiment]:
                alias = "perturb_sweep_{}_{}_r{}_w{}_s{}".format(
                    experiment,
                    method.lower().replace("+", "plus"),
                    str(expected["resolution"]).replace(".", "p"),
                    str(expected["wiggle"]).replace(".", "p"),
                    expected["seed"],
                )
                source_manifest = release_aliases[alias] / "run_manifest.json"
                safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", alias)
                _copy_manifest(
                    source_manifest,
                    staging,
                    f"provenance/maintext/release_run_manifests/{safe}.json",
                    "maintext_release_run_manifest",
                    snapshot_artifacts,
                )

        all_out = generated / "all_methods"
        all_cmd = [
            python,
            "-m",
            "experiments.static.run_perturbed_sweeps",
            "--plot_from_csv",
            str(release_root / "perturbed_sweep.csv"),
            "--summary_dir",
            str(all_out),
            "--no-notify",
        ]
        _require(not all_out.exists(), "All-method generator root already exists")
        command_runner(all_cmd, execution, env, logs / "all_methods.log")
        staged_all = stage_all_method_candidates(
            all_out, figure_root / "all_method_summary_plots"
        )
        _require(
            len(staged_all) == 5, "Exactly five all-method candidates must be staged"
        )
        contracts["all_methods"] = {
            "status": "validated",
            "source": "audited final release perturbed_sweep.csv",
            "methods": {key: list(value) for key, value in RELEASE_METHODS.items()},
            "staged_pdf_count": 5,
        }
        _copy_manifest(
            _write_command_record(
                execution / "all_methods_command.json",
                all_cmd,
                approved_generator_commit,
            ),
            staging,
            "provenance/all_methods/command.json",
            "generator_command",
            snapshot_artifacts,
        )
        for experiment in EXPERIMENTS:
            spec = by_id[f"{experiment}_all_methods"]
            target = figure_root / spec.pdf
            candidates.append(
                {
                    "candidate_id": spec.candidate_id,
                    "root": spec.root,
                    "pdf": spec.pdf,
                    "sha256": file_sha256(target),
                    "generator": "all_method_summary_plots",
                }
            )

        resolution_contract = {
            "status": "validated",
            "resolutions": list(RESOLUTION_VALUES),
            "wiggles": list(RESOLUTION_WIGGLES),
            "seed": 0,
            "experiments": {},
        }
        for experiment in EXPERIMENTS:
            case_index, method = RESOLUTION_CASES[experiment]
            out = generated / "resolution" / experiment
            prefix = f"final_resolution_{experiment}"
            cmd = [
                python,
                "-m",
                "experiments.static.run_appendix_resolution_visuals",
                "--only",
                experiment,
                "--out_dir",
                str(out),
                "--log_dir",
                str(out / "logs"),
                "--resolutions",
                _numbers(RESOLUTION_VALUES),
                "--wiggles",
                _numbers(RESOLUTION_WIGGLES),
                "--case_index",
                str(case_index),
                "--save_prefix",
                prefix,
                "--endpoint_variants",
                "paired",
                "--plots_root",
                str(plots_root),
            ]
            _require(
                not out.exists(),
                f"Resolution generator root already exists: {experiment}",
            )
            command_runner(cmd, execution, env, logs / f"resolution_{experiment}.log")
            run_manifests = validate_resolution_manifest(
                out / "manifest.json", plots_root, experiment, approved_generator_commit
            )
            resolution_contract["experiments"][experiment] = {
                "case_index": case_index,
                "method": method,
                "completed_runs": 6,
            }
            _copy_manifest(
                out / "manifest.json",
                staging,
                f"provenance/resolution/{experiment}/manifest.json",
                "resolution_producer_manifest",
                snapshot_artifacts,
            )
            _copy_manifest(
                _write_command_record(
                    execution / f"resolution_{experiment}_command.json",
                    cmd,
                    approved_generator_commit,
                ),
                staging,
                f"provenance/resolution/{experiment}/command.json",
                "generator_command",
                snapshot_artifacts,
            )
            for index, run_manifest in enumerate(sorted(run_manifests)):
                _copy_manifest(
                    run_manifest,
                    staging,
                    f"provenance/resolution/{experiment}/run_manifests/{index:02d}.json",
                    "resolution_companion_run_manifest",
                    snapshot_artifacts,
                )
            for variant in ("with_endpoints", "clean"):
                candidate_id = f"{experiment}_resolution_{variant}"
                source = (
                    out
                    / "summary_plots"
                    / f"{experiment}_resolution_cartesian_vs_perturbed_{variant}.pdf"
                )
                candidates.append(
                    _stage_candidate(
                        source, staging, by_id[candidate_id], "appendix_resolution"
                    )
                )
        contracts["resolution"] = resolution_contract

        c0_paths = {}
        c0_commands = {}
        for experiment in ("ellipses", "zalesak"):
            out = generated / "guarded_c0" / experiment
            prefix = f"final_guarded_c0_{experiment}"
            cmd = [
                python,
                "-m",
                "experiments.static.run_appendix_c0_study",
                "--only",
                experiment,
                "--algos",
                ",".join(C0_VARIANTS[experiment]),
                "--resolutions",
                _numbers(C0_RESOLUTIONS[experiment]),
                "--wiggles",
                _numbers(C0_WIGGLES),
                "--seeds",
                "0",
                f"--{experiment}",
                "25",
                "--out_csv",
                str(out / "metrics.csv"),
                "--out_dir",
                str(out),
                "--log_dir",
                str(out / "logs"),
                "--save_prefix",
                prefix,
                "--endpoint_variants",
                "paired",
                "--plots_root",
                str(plots_root),
            ]
            _require(
                not out.exists(),
                f"Guarded-C0 generator root already exists: {experiment}",
            )
            command_runner(cmd, execution, env, logs / f"guarded_c0_{experiment}.log")
            c0_paths[experiment] = out / "manifest.json"
            c0_commands[experiment] = cmd
        c0_run_manifests = validate_c0_manifests(
            c0_paths, plots_root, approved_generator_commit
        )
        contracts["guarded_c0"] = {
            "status": "validated",
            "setting_count": 165,
            "seed": 0,
            "cases_per_setting": 25,
            "experiments": {
                key: {
                    "resolutions": list(C0_RESOLUTIONS[key]),
                    "wiggles": list(C0_WIGGLES),
                    "variants": list(C0_VARIANTS[key]),
                }
                for key in C0_RESOLUTIONS
            },
        }
        for experiment in ("ellipses", "zalesak"):
            out = generated / "guarded_c0" / experiment
            _copy_manifest(
                c0_paths[experiment],
                staging,
                f"provenance/guarded_c0/{experiment}/manifest.json",
                "guarded_c0_producer_manifest",
                snapshot_artifacts,
            )
            _copy_manifest(
                _write_command_record(
                    execution / f"guarded_c0_{experiment}_command.json",
                    c0_commands[experiment],
                    approved_generator_commit,
                ),
                staging,
                f"provenance/guarded_c0/{experiment}/command.json",
                "generator_command",
                snapshot_artifacts,
            )
            for candidate_id, source in (
                (
                    f"{experiment}_appendix_c0_metrics",
                    out / "summary_plots" / f"{experiment}_appendix_c0_2x2.pdf",
                ),
                (
                    f"{experiment}_appendix_c0_representative_with_endpoints",
                    out
                    / "representative_cases"
                    / f"{experiment}_appendix_c0_representative_with_endpoints.pdf",
                ),
                (
                    f"{experiment}_appendix_c0_representative_clean",
                    out
                    / "representative_cases"
                    / f"{experiment}_appendix_c0_representative_clean.pdf",
                ),
            ):
                candidates.append(
                    _stage_candidate(
                        source, staging, by_id[candidate_id], "appendix_guarded_c0"
                    )
                )
        for index, run_manifest in enumerate(sorted(c0_run_manifests)):
            _copy_manifest(
                run_manifest,
                staging,
                f"provenance/guarded_c0/run_manifests/{index:03d}.json",
                "guarded_c0_companion_run_manifest",
                snapshot_artifacts,
            )

        deterministic = generated / "deterministic"
        plic_base = deterministic / "perfect_reconstruction_plic_stencil"
        plic_cmd = [
            python,
            "-m",
            "experiments.static.generate_plic_baseline_stencil_figure",
            "--out",
            str(plic_base),
            "--case-index",
            "4",
            "--cell-x",
            "14",
            "--cell-y",
            "13",
            "--resolution",
            "0.32",
            "--wiggle",
            "0.3",
            "--seed",
            "0",
        ]
        _require(
            not deterministic.exists(), "Deterministic generator root already exists"
        )
        command_runner(plic_cmd, execution, env, logs / "deterministic_plic.log")
        contracts["deterministic_plic"] = validate_plic_metadata(
            plic_base.with_name(f"{plic_base.name}_data.json"),
            approved_generator_commit,
        )
        candidates.append(
            _stage_candidate(
                plic_base.with_suffix(".pdf"),
                staging,
                by_id["perfect_reconstruction_plic_stencil"],
                "deterministic_plic_stencil",
            )
        )
        _copy_manifest(
            plic_base.with_name(f"{plic_base.name}_data.json"),
            staging,
            "provenance/deterministic/perfect_reconstruction_plic_stencil_data.json",
            "deterministic_producer_manifest",
            snapshot_artifacts,
        )
        _copy_manifest(
            _write_command_record(
                execution / "plic_command.json", plic_cmd, approved_generator_commit
            ),
            staging,
            "provenance/deterministic/plic_command.json",
            "generator_command",
            snapshot_artifacts,
        )

        staged_out = deterministic / "staged"
        staged_cmd = [
            python,
            "-m",
            "experiments.static.generate_staged_reconstruction_figure",
            "--case-index",
            "22",
            "--resolution",
            "1.0",
            "--wiggle",
            "0.1",
            "--seed",
            "0",
            "--radius",
            "15.0",
            "--slot-width",
            "5.0",
            "--slot-top-rel",
            "10.0",
            "--output-dir",
            str(staged_out),
            "--prefix",
            "staged_reconstruction_zalesak",
        ]
        _require(not staged_out.exists(), "Staged generator root already exists")
        command_runner(staged_cmd, execution, env, logs / "deterministic_staged.log")
        staged_data = staged_out / "staged_reconstruction_zalesak_data.json"
        contracts["deterministic_staged"] = validate_staged_metadata(
            staged_data, approved_generator_commit
        )
        candidates.append(
            _stage_candidate(
                staged_out / "staged_reconstruction_zalesak.pdf",
                staging,
                by_id["staged_reconstruction_zalesak"],
                "deterministic_staged_reconstruction",
            )
        )
        _copy_manifest(
            staged_data,
            staging,
            "provenance/deterministic/staged_reconstruction_zalesak_data.json",
            "deterministic_producer_manifest",
            snapshot_artifacts,
        )
        _copy_manifest(
            _write_command_record(
                execution / "staged_command.json", staged_cmd, approved_generator_commit
            ),
            staging,
            "provenance/deterministic/staged_command.json",
            "generator_command",
            snapshot_artifacts,
        )

        _require(
            len(candidates) == 38
            and {row["candidate_id"] for row in candidates} == set(by_id),
            "Staged candidate inventory is not the explicit 38-PDF allowlist",
        )
        _require(
            sum(
                row["role"] == "resolution_companion_run_manifest"
                for row in snapshot_artifacts
            )
            == 30,
            "Exactly 30 resolution run manifests must be snapshotted",
        )
        _require(
            sum(
                row["role"] == "guarded_c0_companion_run_manifest"
                for row in snapshot_artifacts
            )
            == 165,
            "Exactly 165 C0 run manifests must be snapshotted",
        )

        orchestration = {
            "schema_version": 1,
            "manifest_type": "final_figure_orchestration",
            "status": "completed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator_checkout": attestation.to_dict(),
            "scientific_release": anchor,
            "scientific_contracts": contracts,
            "allowlist": {
                "path": str(Path(allowlist_path).resolve()),
                "sha256": file_sha256(allowlist_path),
            },
            "candidates": sorted(
                candidates,
                key=lambda row: next(
                    i
                    for i, spec in enumerate(specs)
                    if spec.candidate_id == row["candidate_id"]
                ),
            ),
            "snapshot_artifacts": sorted(
                snapshot_artifacts, key=lambda row: row["path"]
            ),
        }
        manifest_path = staging / ORCHESTRATION_MANIFEST
        atomic_write_json(manifest_path, orchestration)
        finalize_publication(
            staging=staging,
            output_root=output_root,
            manifest_path=manifest_path,
            acceptance_runner=acceptance_runner,
            acceptance_kwargs={
                "figure_root": figure_root,
                "c0_root": c0_root,
                "release_root": release_root,
                "orchestration_manifest": manifest_path,
                "output_dir": staging / "review",
                "allowlist_path": allowlist_path,
            },
            after_acceptance_hook=after_acceptance_hook,
        )
    except Exception as exc:
        if staging.exists():
            shutil.rmtree(staging)
        if isinstance(exc, (FinalFigureOrchestrationError, FigureAcceptanceError)):
            raise
        raise FinalFigureOrchestrationError(str(exc)) from exc
    finally:
        if execution.exists():
            shutil.rmtree(execution)
    return output_root


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument(
        "--approved-generator-commit",
        required=True,
        help="full 40-hex reviewed generator commit",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="new, nonexistent atomic publication root",
    )
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        output = orchestrate_final_figures(
            repository=args.repository,
            release_root=args.release_root,
            approved_generator_commit=args.approved_generator_commit,
            output_root=args.output_root,
            allowlist_path=args.allowlist,
        )
    except (FinalFigureOrchestrationError, FigureAcceptanceError) as exc:
        print(f"FINAL FIGURE ORCHESTRATION ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Final figure publication: {output}")
    print(f"Review PDF: {output / 'review' / 'figure_candidate_review.pdf'}")
    print(f"Provenance: {output / ORCHESTRATION_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
