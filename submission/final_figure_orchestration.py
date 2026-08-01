"""Non-publishing validation and freeze helpers for final figure orchestration.

This module deliberately has no reservation, generator-execution, acceptance,
or atomic-publication capability. The script-only boundary lives in
``submission/final_figure_orchestrator.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submission.audit_final_release import audit_final_release, verify_sha256_manifest
from submission.final_figure_provenance import (
    RELEASE_ANCHOR_FILES,
    atomic_write_json,
    copy_verified_file,
    file_sha256,
    load_json_object,
    make_tree_read_only,
    parse_sha256_manifest,
    release_figure_anchor,
    snapshot_record,
    stable_file_bytes,
)
from submission.generator_checkout import (
    GeneratorCheckoutError,
    materialize_approved_source,
    sanitized_git_environment,
    verify_external_approval_record,
    verify_generator_checkout,
    verify_materialized_source,
)
from submission.trusted_figure_runtime import (
    TrustedFigureRuntime,
)


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
    "zalesak": (6, "circular+corner"),
}
RESOLUTION_VALUES = (0.16, 0.32, 0.64)
RESOLUTION_WIGGLES = (0.0, 0.1)
HYBRID_RESOLUTION_VARIANT = "hybrid_endpoints_n16_n32"
HYBRID_RESOLUTION_MODE = "paired_with_hybrid_endpoints_n16_n32"
HYBRID_RESOLUTION_EXPERIMENTS = frozenset({"lines", "ellipses", "zalesak"})
HYBRID_RESOLUTION_VISIBILITY = {"16": True, "32": True, "64": False}
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
C0_REPRESENTATIVES = {
    "ellipses": {"resolution": 0.32, "wiggle": 0.1, "seed": 0, "case_index": 12},
    "zalesak": {"resolution": 1.0, "wiggle": 0.1, "seed": 0, "case_index": 12},
}
ALL_METHOD_FILES = {
    "lines": "lines_all_methods_2x2.pdf",
    "squares": "squares_all_methods_2x2.pdf",
    "circles": "circles_all_methods_5x2_axes.pdf",
    "ellipses": "ellipses_all_methods_5x2_axes.pdf",
    "zalesak": "zalesak_all_methods_2x2.pdf",
}
ORCHESTRATION_MANIFEST = "provenance/final_figure_orchestration.json"
PRIVATE_ALLOWLIST = "provenance/approved_candidate_allowlist.json"
PUBLISHED_TREE_LEDGER = "provenance/published_tree_sha256.json"
ORCHESTRATION_SCHEMA_VERSION = 4
C0_METRIC_BASES = {
    "ellipses": (
        "curvature_error",
        "facet_gap",
        "hausdorff",
        "tangent_error",
        "curvature_proxy_error",
    ),
    "zalesak": ("area_error", "facet_gap", "hausdorff"),
}
METRIC_STATS = ("mean", "median", "p25", "p75")


def resolution_endpoint_mode(experiment: str) -> str:
    if experiment in HYBRID_RESOLUTION_EXPERIMENTS:
        return HYBRID_RESOLUTION_MODE
    return "paired"


def resolution_endpoint_variants(experiment: str) -> tuple[str, ...]:
    variants = ("with_endpoints", "clean")
    if experiment in HYBRID_RESOLUTION_EXPERIMENTS:
        return variants + (HYBRID_RESOLUTION_VARIANT,)
    return variants


def resolution_endpoint_visibility_contract(experiment: str) -> dict:
    resolutions = tuple(str(int(round(value * 100))) for value in RESOLUTION_VALUES)
    contract = {
        "with_endpoints": {
            "main_endpoint_visibility_by_resolution": {
                resolution: True for resolution in resolutions
            },
            "show_inset_endpoints": True,
        },
        "clean": {
            "main_endpoint_visibility_by_resolution": {
                resolution: False for resolution in resolutions
            },
            "show_inset_endpoints": True,
        },
    }
    if experiment in HYBRID_RESOLUTION_EXPERIMENTS:
        contract[HYBRID_RESOLUTION_VARIANT] = {
            "main_endpoint_visibility_by_resolution": dict(
                HYBRID_RESOLUTION_VISIBILITY
            ),
            "show_inset_endpoints": True,
        }
    return contract


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


def _remove_tree(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    for candidate in path.rglob("*"):
        if candidate.is_dir() and not candidate.is_symlink():
            try:
                candidate.chmod(0o700)
            except OSError:
                pass
    try:
        path.chmod(0o700)
    except OSError:
        pass
    shutil.rmtree(path)


@dataclass(frozen=True)
class ReleaseInputSnapshot:
    root: Path
    csv_path: Path
    plots_root: Path
    anchor: dict
    artifact_records: tuple[dict, ...]
    alias_sources: Mapping[str, str]


@dataclass(frozen=True)
class CompleteReleaseSnapshot:
    root: Path
    file_count: int
    total_size_bytes: int
    live_sha256sums_sha256: str


@dataclass(frozen=True)
class ExecutionConfigAuthority:
    config_root: Path
    manifest_path: Path
    manifest_sha256: str
    file_count: int


@dataclass(frozen=True)
class PrivateGeneratorGitView:
    git_dir: Path
    work_tree: Path
    index_file: Path
    approved_commit: str
    approved_tree: str
    object_format: str
    alternate_objects: Path
    alternate_objects_device: int
    alternate_objects_inode: int
    git_dir_device: int
    git_dir_inode: int
    work_tree_device: int
    work_tree_inode: int
    metadata_sha256: str


@dataclass(frozen=True)
class ReleaseAuditPin:
    root: Path
    device: int
    inode: int
    source_commit: str
    sha256sums_bytes: bytes
    sha256sums_sha256: str
    resolved_config_sha256: str
    contract: Mapping[str, object]


def _capture_release_audit_pin(
    release_root: Path, *, contract: Optional[Mapping[str, object]] = None
) -> ReleaseAuditPin:
    """Capture the exact release identity and authority bytes used by an audit."""

    supplied = Path(release_root).expanduser().absolute()
    _require(
        not supplied.is_symlink(), f"Release root must not be a symlink: {supplied}"
    )
    try:
        root_info = supplied.lstat()
    except FileNotFoundError as exc:
        raise FinalFigureOrchestrationError(
            f"Release root does not exist: {supplied}"
        ) from exc
    _require(
        stat.S_ISDIR(root_info.st_mode), f"Release root is not a directory: {supplied}"
    )
    manifest_bytes = stable_file_bytes(supplied / "SHA256SUMS")
    config_bytes = stable_file_bytes(supplied / "submission_config.resolved.json")
    try:
        config = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FinalFigureOrchestrationError(
            f"Resolved release config is invalid JSON: {exc}"
        ) from exc
    _require(isinstance(config, dict), "Resolved release config must be an object")
    source = config.get("source")
    source_commit = source.get("target_commit") if isinstance(source, dict) else None
    _require(
        isinstance(source_commit, str)
        and re.fullmatch(r"[0-9a-fA-F]{40}", source_commit) is not None,
        "Resolved release config lacks a full source commit",
    )
    after = supplied.lstat()
    _require(
        (after.st_dev, after.st_ino) == (root_info.st_dev, root_info.st_ino),
        "Release root changed while audit authority was captured",
    )
    return ReleaseAuditPin(
        root=supplied,
        device=root_info.st_dev,
        inode=root_info.st_ino,
        source_commit=source_commit,
        sha256sums_bytes=manifest_bytes,
        sha256sums_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        resolved_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        contract=dict(contract or {}),
    )


def _assert_release_matches_audit_pin(
    release_root: Path, audit_pin: ReleaseAuditPin
) -> None:
    supplied = Path(release_root).expanduser().absolute()
    _require(supplied == audit_pin.root, "Release root differs from audited root")
    _require(not supplied.is_symlink(), "Audited release root became a symlink")
    try:
        info = supplied.lstat()
    except FileNotFoundError as exc:
        raise FinalFigureOrchestrationError("Audited release root disappeared") from exc
    _require(
        stat.S_ISDIR(info.st_mode)
        and (info.st_dev, info.st_ino) == (audit_pin.device, audit_pin.inode),
        "Audited release root was replaced",
    )
    manifest_bytes = stable_file_bytes(supplied / "SHA256SUMS")
    _require(
        manifest_bytes == audit_pin.sha256sums_bytes
        and hashlib.sha256(manifest_bytes).hexdigest() == audit_pin.sha256sums_sha256,
        "Release SHA256SUMS differs from the audited bytes",
    )
    config_bytes = stable_file_bytes(supplied / "submission_config.resolved.json")
    _require(
        hashlib.sha256(config_bytes).hexdigest() == audit_pin.resolved_config_sha256,
        "Resolved release config differs from the audited bytes",
    )
    config = json.loads(config_bytes.decode("utf-8"))
    _require(
        config.get("source", {}).get("target_commit") == audit_pin.source_commit,
        "Release source commit differs from the audited source commit",
    )


def _snapshot_complete_release(
    release_root: Path,
    destination: Path,
    *,
    live_pin: ReleaseAuditPin,
    after_open_hook: Optional[Callable[[Path], None]] = None,
) -> CompleteReleaseSnapshot:
    """Materialize the complete live ledger before any scientific audit."""

    release_root = Path(release_root).expanduser().absolute()
    destination = Path(destination).resolve()
    _assert_release_matches_audit_pin(release_root, live_pin)
    _require(
        not destination.exists(), f"Complete release snapshot exists: {destination}"
    )
    destination.mkdir(parents=True, mode=0o700)
    try:
        ledger_path = release_root / "SHA256SUMS"
        ledger_bytes = stable_file_bytes(ledger_path)
        _require(
            ledger_bytes == live_pin.sha256sums_bytes,
            "Live release ledger changed before complete snapshot",
        )
        snapshot_ledger = destination / "SHA256SUMS"
        snapshot_ledger.write_bytes(ledger_bytes)
        ledger = parse_sha256_manifest(snapshot_ledger)

        actual_paths = set()
        for path in sorted(release_root.rglob("*")):
            _require(not path.is_symlink(), f"Live release contains symlink: {path}")
            _require(
                path.is_file() or path.is_dir(),
                f"Live release contains non-regular entry: {path}",
            )
            if path.is_file() and path != ledger_path:
                actual_paths.add(path.relative_to(release_root).as_posix())
        _require(
            actual_paths == set(ledger),
            "Live release inventory differs from its pinned SHA256SUMS",
        )

        total_size = 0
        for relative, expected_sha256 in sorted(ledger.items()):
            source = release_root / relative
            target = destination / relative
            copy_verified_file(
                source,
                target,
                expected_sha256=expected_sha256,
                after_open_hook=after_open_hook,
            )
            total_size += target.stat().st_size

        _assert_release_matches_audit_pin(release_root, live_pin)
        _require(
            snapshot_ledger.read_bytes() == live_pin.sha256sums_bytes,
            "Complete release snapshot has different ledger bytes",
        )
        make_tree_read_only(destination)
    except Exception:
        if destination.exists():
            _remove_tree(destination)
        raise
    return CompleteReleaseSnapshot(
        root=destination,
        file_count=len(ledger),
        total_size_bytes=total_size,
        live_sha256sums_sha256=live_pin.sha256sums_sha256,
    )


def _release_alias(row: Mapping[str, str]) -> str:
    return "perturb_sweep_{}_{}_r{}_w{}_s{}".format(
        row["experiment"],
        row["algo"].lower().replace("+", "plus"),
        row["resolution"].replace(".", "p"),
        row["wiggle"].replace(".", "p"),
        row["seed"],
    )


def _required_release_aliases(csv_path: Path) -> dict[str, str]:
    rows = []
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_identity: dict[tuple, str] = {}
    for row in rows:
        required = ("experiment", "algo", "resolution", "wiggle", "seed", "save_name")
        _require(
            all(row.get(key) for key in required), "Release CSV lacks run identity"
        )
        identity = (
            row["experiment"],
            row["algo"],
            float(row["resolution"]),
            float(row["wiggle"]),
            int(row["seed"]),
        )
        previous = by_identity.setdefault(identity, row["save_name"])
        _require(previous == row["save_name"], "Release CSV identity is ambiguous")

    aliases: dict[str, str] = {}
    for experiment, expected in MAINTEXT_CASES.items():
        for method in MAINTEXT_METHODS[experiment]:
            identity = (
                experiment,
                method,
                float(expected["resolution"]),
                float(expected["wiggle"]),
                int(expected["seed"]),
            )
            _require(
                identity in by_identity,
                f"Release CSV lacks representative run {identity}",
            )
            row = {
                "experiment": experiment,
                "algo": method,
                "resolution": str(expected["resolution"]),
                "wiggle": str(expected["wiggle"]),
                "seed": str(expected["seed"]),
            }
            alias = _release_alias(row)
            previous = aliases.setdefault(alias, by_identity[identity])
            _require(
                previous == by_identity[identity], f"Release alias {alias} is ambiguous"
            )
    return aliases


def _snapshot_release_inputs(
    release_root: Path,
    destination: Path,
    *,
    audit_pin: ReleaseAuditPin,
    staging_root: Path,
    after_open_hook: Optional[Callable[[Path], None]] = None,
) -> ReleaseInputSnapshot:
    """Copy every release byte consumed by generators into immutable staging."""

    release_root = Path(release_root).expanduser().absolute()
    destination = Path(destination).resolve()
    _assert_release_matches_audit_pin(release_root, audit_pin)
    _require(
        not destination.exists(), f"Release snapshot already exists: {destination}"
    )
    destination.mkdir(parents=True)
    try:
        live_manifest = release_root / "SHA256SUMS"
        manifest_bytes = stable_file_bytes(live_manifest)
        _require(
            manifest_bytes == audit_pin.sha256sums_bytes,
            "Release snapshot ledger differs byte-for-byte from the live audit",
        )
        manifest_target = destination / "SHA256SUMS"
        manifest_target.write_bytes(manifest_bytes)
        manifest_digest = file_sha256(manifest_target)
        ledger = parse_sha256_manifest(manifest_target)

        for relative in RELEASE_ANCHOR_FILES[:-1]:
            expected = ledger.get(relative)
            _require(expected is not None, f"Release ledger lacks {relative}")
            copy_verified_file(
                release_root / relative,
                destination / relative,
                expected_sha256=expected,
                after_open_hook=after_open_hook,
            )

        aliases = _required_release_aliases(destination / "perturbed_sweep.csv")
        plots_root = destination / "plots"
        plots_root.mkdir()
        for alias, save_name in sorted(aliases.items()):
            source_bundle = release_root / "raw_runs" / save_name
            _require(
                source_bundle.is_dir(), f"Release raw bundle is missing: {save_name}"
            )
            target_bundle = plots_root / alias
            target_bundle.mkdir(parents=True)
            source_files = []
            for path in sorted(source_bundle.rglob("*")):
                _require(
                    not path.is_symlink(), f"Release bundle contains symlink: {path}"
                )
                if path.is_file():
                    source_files.append(path)
            _require(source_files, f"Release raw bundle is empty: {save_name}")
            prefix = f"raw_runs/{save_name}/"
            expected_bundle_paths = {
                relative for relative in ledger if relative.startswith(prefix)
            }
            actual_bundle_paths = {
                (
                    Path("raw_runs") / save_name / path.relative_to(source_bundle)
                ).as_posix()
                for path in source_files
            }
            _require(
                actual_bundle_paths == expected_bundle_paths,
                f"Release raw bundle inventory changed: {save_name}",
            )
            for source in source_files:
                bundle_relative = source.relative_to(source_bundle)
                release_relative = (
                    Path("raw_runs") / save_name / bundle_relative
                ).as_posix()
                expected = ledger.get(release_relative)
                _require(
                    expected is not None,
                    f"Release ledger lacks required bundle file: {release_relative}",
                )
                copy_verified_file(
                    source,
                    target_bundle / bundle_relative,
                    expected_sha256=expected,
                    after_open_hook=after_open_hook,
                )

        _require(
            stable_file_bytes(live_manifest) == manifest_bytes
            and file_sha256(manifest_target) == manifest_digest,
            "Release checksum ledger changed during input snapshot",
        )
        _assert_release_matches_audit_pin(release_root, audit_pin)
        _require(
            manifest_target.read_bytes() == audit_pin.sha256sums_bytes
            and file_sha256(destination / "submission_config.resolved.json")
            == audit_pin.resolved_config_sha256,
            "Private release snapshot differs from the audited authority bytes",
        )
        snapshot_anchor = release_figure_anchor(destination)
        _require(
            snapshot_anchor["source_commit"] == audit_pin.source_commit,
            "Private release snapshot has a different audited source commit",
        )
        snapshot_anchor["root"] = "provenance/release_input_snapshot"
        for relative, record in snapshot_anchor["artifacts"].items():
            record["path"] = f"provenance/release_input_snapshot/{relative}"
        make_tree_read_only(destination)
        artifact_records = tuple(
            snapshot_record(path, staging_root, "release_input_snapshot")
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        )
    except Exception:
        if destination.exists():
            for path in destination.rglob("*"):
                if path.is_dir() and not path.is_symlink():
                    path.chmod(0o700)
            destination.chmod(0o700)
            shutil.rmtree(destination)
        raise
    return ReleaseInputSnapshot(
        root=destination,
        csv_path=destination / "perturbed_sweep.csv",
        plots_root=destination / "plots",
        anchor=snapshot_anchor,
        artifact_records=artifact_records,
        alias_sources=aliases,
    )


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


def validate_final_release_contract(release_root: Path) -> ReleaseAuditPin:
    audit_pin = _capture_release_audit_pin(release_root)
    report = audit_final_release(audit_pin.root)
    _require(report.ok, "Final release audit failed: " + "; ".join(report.errors))
    checksum_errors = verify_sha256_manifest(audit_pin.root)
    _require(
        not checksum_errors,
        "Final release checksum failed: " + "; ".join(checksum_errors),
    )
    config = load_json_object(audit_pin.root / "submission_config.resolved.json")
    sweep = load_json_object(audit_pin.root / "sweep_manifest.json")
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
    contract = {
        "status": "validated",
        "audited_source_commit": audit_pin.source_commit,
        "audited_sha256sums_sha256": audit_pin.sha256sums_sha256,
        "methods": {key: list(value) for key, value in RELEASE_METHODS.items()},
        "run_counts": RELEASE_RUN_COUNTS,
        "seed": 0,
        "cases_per_setting": 25,
        "total_runs": 970,
        "total_cases": 24250,
        "profile": PROFILE,
    }
    _assert_release_matches_audit_pin(audit_pin.root, audit_pin)
    return ReleaseAuditPin(
        **{
            **audit_pin.__dict__,
            "contract": contract,
        }
    )


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
    expected_endpoint_mode = resolution_endpoint_mode(experiment)
    _require(
        payload.get("endpoint_variants") == expected_endpoint_mode,
        f"Resolution {experiment} has the wrong endpoint variant mode",
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
    expected_variants = set(resolution_endpoint_variants(experiment))
    _require(
        isinstance(summary, dict) and set(summary) == expected_variants,
        f"Resolution {experiment} outputs incomplete",
    )
    endpoint_visibility = payload.get("endpoint_visibility", {}).get(experiment)
    _require(
        endpoint_visibility == resolution_endpoint_visibility_contract(experiment),
        f"Resolution {experiment} endpoint visibility contract differs",
    )
    return manifests


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    _require(Path(path).is_file(), f"Required CSV is missing: {path}")
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    _require(rows, f"Required CSV is empty: {path}")
    return rows


def _read_jsonl(path: Path) -> list[dict]:
    _require(Path(path).is_file(), f"Required JSONL is missing: {path}")
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FinalFigureOrchestrationError(
                f"Malformed JSONL {path}:{line_number}: {exc}"
            ) from exc
        _require(isinstance(value, dict), f"JSONL row is not an object: {path}")
        rows.append(value)
    _require(rows, f"Required JSONL is empty: {path}")
    return rows


def resolution_input_paths(
    plots_root: Path,
    *,
    experiment: str,
    save_name: str,
    case_index: int,
    include_consumed_truth: bool = False,
) -> list[tuple[Path, str]]:
    """Validate and return exact quantitative/geometry inputs for one panel run."""

    run_root = Path(plots_root) / save_name
    case_metrics = run_root / "metrics" / "case_metrics.csv"
    metric_rows = _read_csv_rows(case_metrics)
    selected = [row for row in metric_rows if row.get("case_index") == str(case_index)]
    _require(
        len(selected) == 1 and len(metric_rows) == 1,
        f"Resolution case metrics must contain exactly case {case_index}: {case_metrics}",
    )
    required_metrics = {
        "lines": ("hausdorff", "facet_gap"),
        "squares": ("hausdorff", "facet_gap", "area_error"),
        "circles": (
            "hausdorff",
            "facet_gap",
            "curvature_error",
            "tangent_error",
            "curvature_proxy_error",
        ),
        "ellipses": (
            "hausdorff",
            "facet_gap",
            "curvature_error",
            "tangent_error",
            "curvature_proxy_error",
        ),
        "zalesak": ("hausdorff", "facet_gap", "area_error"),
    }[experiment]
    for metric in required_metrics:
        try:
            value = float(selected[0][metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise FinalFigureOrchestrationError(
                f"Resolution metric {metric} is missing/non-numeric: {case_metrics}"
            ) from exc
        _require(value == value and abs(value) != float("inf"), f"Non-finite {metric}")

    geometry_path = run_root / "metrics" / "case_geometry.jsonl"
    geometry_rows = _read_jsonl(geometry_path)
    geometry_selected = [
        row for row in geometry_rows if row.get("case_index") == case_index
    ]
    _require(
        len(geometry_selected) == 1 and len(geometry_rows) == 1,
        f"Resolution geometry must contain exactly case {case_index}: {geometry_path}",
    )

    facet = run_root / "vtk" / "reconstructed" / "facets" / f"{case_index}.vtp"
    facet_metadata = facet.with_suffix(".facet_metadata.json")
    metadata = load_json_object(facet_metadata)
    _require(
        metadata.get("schema_version", 0) >= 2
        and isinstance(metadata.get("primitives"), list),
        f"Resolution facet metadata is incomplete: {facet_metadata}",
    )
    paths = [
        (case_metrics, "resolution_case_metrics"),
        (geometry_path, "resolution_case_geometry"),
        (run_root / "vtk" / "mesh.vtk", "resolution_mesh_geometry"),
        (facet, "resolution_reconstructed_geometry"),
        (facet_metadata, "resolution_facet_metadata"),
    ]
    if include_consumed_truth and experiment in {"squares", "circles"}:
        stem = {"squares": "true_square", "circles": "true_circle"}[experiment]
        paths.append(
            (
                run_root / "vtk" / "true" / f"{stem}{case_index}.vtp",
                "resolution_truth_geometry",
            )
        )
    for path, _role in paths:
        _require(
            path.is_file() and not path.is_symlink(),
            f"Resolution input missing: {path}",
        )
    return paths


def validate_c0_metrics(
    path: Path, experiment: str, producer_manifest: Optional[Path] = None
) -> dict:
    """Require exact setting-by-metric coverage for one guarded-C0 CSV."""

    rows = _read_csv_rows(path)
    variants = C0_VARIANTS[experiment]
    expected_settings = {
        (experiment, label, method, int(do_c0), resolution, wiggle, 0)
        for resolution in C0_RESOLUTIONS[experiment]
        for wiggle in C0_WIGGLES
        for label, (method, do_c0) in variants.items()
    }
    expected_metric_keys = {
        f"{base}_{stat}"
        for base in C0_METRIC_BASES[experiment]
        for stat in METRIC_STATS
    }
    expected_save_names = None
    if producer_manifest is not None:
        producer = load_json_object(producer_manifest)
        producer_runs = producer.get("runs")
        _require(
            isinstance(producer_runs, list),
            f"Guarded-C0 producer manifest lacks runs: {producer_manifest}",
        )
        expected_save_names = {}
        for run in producer_runs:
            _require(isinstance(run, dict), "Malformed guarded-C0 producer run")
            key = (
                run.get("experiment"),
                run.get("variant"),
                C0_VARIANTS[experiment].get(run.get("variant"), (None, None))[0],
                int(C0_VARIANTS[experiment].get(run.get("variant"), (None, False))[1]),
                float(run.get("resolution")),
                float(run.get("wiggle")),
                int(run.get("seed")),
            )
            save_name = run.get("save_name")
            _require(
                isinstance(save_name, str) and save_name,
                "Guarded-C0 producer run lacks save_name",
            )
            previous = expected_save_names.setdefault(key, save_name)
            _require(previous == save_name, "Guarded-C0 producer run is ambiguous")
        _require(
            set(expected_save_names) == expected_settings,
            "Guarded-C0 producer settings differ from metric contract",
        )
    actual: dict[tuple, set[str]] = {}
    seen_rows = set()
    for row in rows:
        try:
            setting = (
                row["experiment"],
                row["algo"],
                row["facet_algo"],
                int(row["do_c0"]),
                float(row["resolution"]),
                float(row["wiggle"]),
                int(row["seed"]),
            )
            metric_key = row["metric_key"]
            metric_value = float(row["metric_value"])
            save_name = row["save_name"]
        except (KeyError, TypeError, ValueError) as exc:
            raise FinalFigureOrchestrationError(
                f"Guarded-C0 metrics row is malformed: {path}"
            ) from exc
        _require(
            metric_value == metric_value and abs(metric_value) != float("inf"),
            f"Guarded-C0 metric is non-finite: {path}",
        )
        if expected_save_names is not None:
            _require(
                save_name == expected_save_names.get(setting),
                f"Guarded-C0 metric save_name differs for {setting}: {path}",
            )
        row_key = setting + (metric_key, save_name)
        _require(
            row_key not in seen_rows, f"Duplicate guarded-C0 metric row: {row_key}"
        )
        seen_rows.add(row_key)
        actual.setdefault(setting, set()).add(metric_key)
    _require(
        set(actual) == expected_settings, f"Guarded-C0 setting coverage differs: {path}"
    )
    for setting, keys in actual.items():
        _require(
            keys == expected_metric_keys,
            f"Guarded-C0 metric coverage differs for {setting}: {path}",
        )
    expected_rows = len(expected_settings) * len(expected_metric_keys)
    _require(len(rows) == expected_rows, f"Guarded-C0 metric row count differs: {path}")
    return {
        "status": "validated",
        "experiment": experiment,
        "setting_count": len(expected_settings),
        "metric_keys": sorted(expected_metric_keys),
        "row_count": expected_rows,
        "sha256": file_sha256(path),
    }


def representative_geometry_input_paths(
    run_root: Path, *, case_index: int
) -> list[tuple[Path, str]]:
    """Return exact mesh/facet/metadata inputs consumed by a representative panel."""

    run_root = Path(run_root)
    geometry_path = run_root / "metrics" / "case_geometry.jsonl"
    rows = _read_jsonl(geometry_path)
    _require(
        sum(row.get("case_index") == case_index for row in rows) == 1,
        f"Representative geometry lacks exact case {case_index}: {geometry_path}",
    )
    facet = run_root / "vtk" / "reconstructed" / "facets" / f"{case_index}.vtp"
    metadata_path = facet.with_suffix(".facet_metadata.json")
    metadata = load_json_object(metadata_path)
    _require(
        metadata.get("schema_version", 0) >= 2
        and isinstance(metadata.get("primitives"), list),
        f"Representative facet metadata is incomplete: {metadata_path}",
    )
    paths = [
        (geometry_path, "representative_case_geometry"),
        (run_root / "vtk" / "mesh.vtk", "representative_mesh_geometry"),
        (facet, "representative_reconstructed_geometry"),
        (metadata_path, "representative_facet_metadata"),
    ]
    for path, _role in paths:
        _require(
            path.is_file() and not path.is_symlink(),
            f"Representative geometry input is missing: {path}",
        )
    return paths


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


def stage_all_method_candidates(source: Path, destination: Path) -> list[Path]:
    """Copy only the five allowlisted all-method PDFs, ignoring auxiliaries."""
    copied = []
    for filename in ALL_METHOD_FILES.values():
        source_pdf = Path(source) / filename
        _require(source_pdf.is_file(), f"All-method candidate is missing: {source_pdf}")
        target = Path(destination) / filename
        _copy(source_pdf, target)
        copied.append(target)
    return copied


def _copy(source: Path, destination: Path) -> Path:
    source = Path(source)
    _require(
        source.is_file() and not source.is_symlink(),
        f"Generated artifact is missing or is a symlink: {source}",
    )
    destination = Path(destination)
    data = stable_file_bytes(source)
    digest = hashlib.sha256(data).hexdigest()
    copy_verified_file(
        source,
        destination,
        expected_sha256=digest,
    )
    return destination


def _portable_command(
    command: Sequence[str], *, staging_root: Path, execution_root: Path
) -> list[str]:
    replacements = (
        (str(Path(staging_root).resolve()), "<publication-root>"),
        (str(Path(execution_root).resolve()), "<private-execution>"),
    )
    portable = []
    for raw in command:
        value = str(raw)
        for prefix, replacement in replacements:
            value = value.replace(prefix, replacement)
        portable.append(value)
    return portable


def _write_command_record(
    path: Path,
    command: Sequence[str],
    commit: str,
    *,
    staging_root: Path,
    execution_root: Path,
) -> Path:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "status": "completed",
            "approved_generator_commit": commit,
            "command": _portable_command(
                command, staging_root=staging_root, execution_root=execution_root
            ),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return path


def _config_authority_payload(config_root: Path) -> dict:
    config_root = Path(config_root).resolve()
    _require(config_root.is_dir(), f"Approved config root is missing: {config_root}")
    records = []
    for path in sorted(config_root.rglob("*")):
        _require(not path.is_symlink(), f"Approved config contains symlink: {path}")
        _require(
            path.is_file() or path.is_dir(),
            f"Approved config contains non-regular entry: {path}",
        )
        if path.is_file():
            relative = path.relative_to(config_root).as_posix()
            data = stable_file_bytes(path)
            records.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            )
    _require(records, "Approved config tree is empty")
    return {
        "schema_version": 1,
        "authority": "approved_generator_config",
        "files": records,
    }


def _seal_execution_config(
    config_root: Path, manifest_path: Path
) -> ExecutionConfigAuthority:
    payload = _config_authority_payload(config_root)
    atomic_write_json(manifest_path, payload)
    manifest_path.chmod(0o400)
    return ExecutionConfigAuthority(
        config_root=Path(config_root).resolve(),
        manifest_path=Path(manifest_path).resolve(),
        manifest_sha256=file_sha256(manifest_path),
        file_count=len(payload["files"]),
    )


def _verify_execution_config(authority: ExecutionConfigAuthority) -> None:
    manifest_bytes = stable_file_bytes(authority.manifest_path)
    _require(
        hashlib.sha256(manifest_bytes).hexdigest() == authority.manifest_sha256,
        "Execution config authority manifest mutated",
    )
    try:
        expected = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FinalFigureOrchestrationError(
            f"Execution config authority manifest is invalid: {exc}"
        ) from exc
    _require(
        expected == _config_authority_payload(authority.config_root)
        and len(expected.get("files", [])) == authority.file_count,
        "Attested execution config bytes mutated",
    )


def _run_git_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "--no-pager", *command],
            cwd=cwd,
            env=dict(environment),
            check=check,
            capture_output=True,
            text=text,
        )
    except FileNotFoundError as exc:
        raise FinalFigureOrchestrationError("git is unavailable") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout
        detail = (stderr or stdout or "git command failed").strip()
        raise FinalFigureOrchestrationError(detail) from exc


def _private_git_metadata_sha256(git_dir: Path) -> str:
    git_dir = Path(git_dir).resolve()
    _require(git_dir.is_dir(), f"Private Git directory is missing: {git_dir}")
    digest = hashlib.sha256()
    paths = [git_dir, *git_dir.rglob("*")]
    for path in sorted(
        paths, key=lambda candidate: candidate.relative_to(git_dir).as_posix()
    ):
        relative = path.relative_to(git_dir).as_posix() or "."
        info = path.lstat()
        _require(
            info.st_uid == os.getuid(),
            f"Private Git metadata has the wrong owner: {relative}",
        )
        if stat.S_ISDIR(info.st_mode):
            _require(
                stat.S_IMODE(info.st_mode) == 0o500,
                f"Private Git directory mode changed: {relative}",
            )
            digest.update(f"D\0{relative}\0{stat.S_IMODE(info.st_mode):04o}\n".encode())
        else:
            _require(
                stat.S_ISREG(info.st_mode),
                f"Private Git metadata contains a non-regular entry: {relative}",
            )
            _require(
                stat.S_IMODE(info.st_mode) == 0o400,
                f"Private Git file mode changed: {relative}",
            )
            data = stable_file_bytes(path)
            digest.update(
                (
                    f"F\0{relative}\0{stat.S_IMODE(info.st_mode):04o}\0"
                    f"{len(data)}\0{hashlib.sha256(data).hexdigest()}\n"
                ).encode()
            )
    return digest.hexdigest()


def _private_git_environment(
    view: PrivateGeneratorGitView,
    base: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    env = {} if base is None else dict(base)
    for key in tuple(env):
        if key.startswith("GIT_"):
            env.pop(key)
    env.update(sanitized_git_environment(env))
    env.update(
        {
            "GIT_DIR": str(view.git_dir),
            "GIT_WORK_TREE": str(view.work_tree),
            "GIT_INDEX_FILE": str(view.index_file),
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return env


def _run_private_git(
    view: PrivateGeneratorGitView,
    *command: str,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    return _run_git_command(
        command,
        cwd=view.work_tree,
        environment=_private_git_environment(view),
        check=check,
        text=text,
    )


def _git_object_oid(data: bytes, object_type: str, object_format: str) -> str:
    try:
        digest = hashlib.new(object_format)
    except ValueError as exc:
        raise FinalFigureOrchestrationError(
            f"Unsupported Git object format: {object_format}"
        ) from exc
    digest.update(f"{object_type} {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def _tree_index_records(data: bytes, *, index: bool) -> list[tuple[str, str, str]]:
    records = []
    for raw in data.split(b"\0"):
        if not raw:
            continue
        metadata, path_bytes = raw.split(b"\t", 1)
        fields = metadata.decode("ascii").split(" ")
        if index:
            _require(len(fields) == 3, "Private Git index record is malformed")
            mode, oid, stage_number = fields
            _require(stage_number == "0", "Private Git index has an unresolved stage")
        else:
            _require(len(fields) == 3, "Private Git tree record is malformed")
            mode, object_type, oid = fields
            _require(
                object_type == "blob",
                "Private Git tree contains a non-blob tracked object",
            )
        records.append((mode, oid, path_bytes.decode("utf-8", errors="strict")))
    return records


def _verify_private_generator_git_view(view: PrivateGeneratorGitView) -> None:
    _require(not view.git_dir.is_symlink(), "Private Git directory became a symlink")
    git_info = view.git_dir.lstat()
    _require(
        stat.S_ISDIR(git_info.st_mode)
        and (git_info.st_dev, git_info.st_ino)
        == (view.git_dir_device, view.git_dir_inode),
        "Private Git directory identity changed",
    )
    _require(not view.work_tree.is_symlink(), "Immutable source became a symlink")
    source_info = view.work_tree.lstat()
    _require(
        stat.S_ISDIR(source_info.st_mode)
        and (source_info.st_dev, source_info.st_ino)
        == (view.work_tree_device, view.work_tree_inode),
        "Immutable source root identity changed",
    )
    _require(
        not view.alternate_objects.is_symlink(),
        "Approved Git object directory became a symlink",
    )
    object_info = view.alternate_objects.lstat()
    _require(
        stat.S_ISDIR(object_info.st_mode)
        and (object_info.st_dev, object_info.st_ino)
        == (view.alternate_objects_device, view.alternate_objects_inode),
        "Approved Git object directory identity changed",
    )
    _require(
        _private_git_metadata_sha256(view.git_dir) == view.metadata_sha256,
        "Private Git metadata changed",
    )

    absolute_git_dir = _run_private_git(
        view, "rev-parse", "--absolute-git-dir"
    ).stdout.strip()
    top_level = _run_private_git(view, "rev-parse", "--show-toplevel").stdout.strip()
    commit = _run_private_git(view, "rev-parse", "HEAD").stdout.strip().lower()
    tree = _run_private_git(view, "rev-parse", "HEAD^{tree}").stdout.strip().lower()
    object_format = _run_private_git(
        view, "rev-parse", "--show-object-format"
    ).stdout.strip()
    _require(
        Path(absolute_git_dir).resolve() == view.git_dir,
        "Child Git directory differs from the private authority",
    )
    _require(
        Path(top_level).resolve() == view.work_tree,
        "Child Git work tree differs from the immutable source",
    )
    _require(commit == view.approved_commit, "Private Git HEAD differs from approval")
    _require(tree == view.approved_tree, "Private Git tree differs from approval")
    _require(
        object_format == view.object_format,
        "Private Git object format differs from the approved repository",
    )

    symbolic = _run_private_git(view, "symbolic-ref", "-q", "HEAD", check=False)
    _require(
        symbolic.returncode == 1 and not symbolic.stdout,
        "Private Git HEAD is not detached",
    )
    commit_data = _run_private_git(
        view, "cat-file", "commit", "HEAD", text=False
    ).stdout
    tree_data = _run_private_git(
        view, "cat-file", "tree", view.approved_tree, text=False
    ).stdout
    _require(
        _git_object_oid(commit_data, "commit", view.object_format)
        == view.approved_commit,
        "Private Git commit object failed hash verification",
    )
    _require(
        _git_object_oid(tree_data, "tree", view.object_format) == view.approved_tree,
        "Private Git tree object failed hash verification",
    )

    tree_records = _tree_index_records(
        _run_private_git(
            view, "ls-tree", "-rz", "--full-tree", "HEAD", text=False
        ).stdout,
        index=False,
    )
    index_records = _tree_index_records(
        _run_private_git(view, "ls-files", "--stage", "-z", text=False).stdout,
        index=True,
    )
    _require(index_records == tree_records, "Private Git index differs from approval")
    status = _run_private_git(
        view,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        text=False,
    )
    _require(not status.stdout, "Immutable source reports Git status changes")
    for command in (
        ("diff", "--quiet", "HEAD", "--"),
        ("diff", "--cached", "--quiet", "HEAD", "--"),
    ):
        result = _run_private_git(view, *command, check=False)
        _require(
            result.returncode == 0, "Immutable source differs from private Git HEAD"
        )
    _require(
        _private_git_metadata_sha256(view.git_dir) == view.metadata_sha256,
        "Private Git metadata changed during verification",
    )


def _create_private_generator_git_view(
    repository: Path,
    immutable_source: Path,
    approved_generator_commit: str,
    approved_generator_tree: str,
    destination: Path,
) -> PrivateGeneratorGitView:
    repository = Path(repository).resolve()
    immutable_source = Path(immutable_source).resolve()
    supplied_destination = Path(destination).expanduser().absolute()
    _require(
        not os.path.lexists(supplied_destination),
        f"Private Git destination already exists: {supplied_destination}",
    )
    _require(
        immutable_source.is_dir() and not immutable_source.is_symlink(),
        "Immutable generator source is missing or is a symlink",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{40}", approved_generator_commit or "") is not None,
        "Approved generator commit must be full lowercase 40-hex",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{40}", approved_generator_tree or "") is not None,
        "Approved generator tree must be full lowercase 40-hex",
    )
    base_environment = sanitized_git_environment()
    object_format = _run_git_command(
        ("rev-parse", "--show-object-format"),
        cwd=repository,
        environment=base_environment,
    ).stdout.strip()
    object_path = _run_git_command(
        ("rev-parse", "--path-format=absolute", "--git-path", "objects"),
        cwd=repository,
        environment=base_environment,
    ).stdout.strip()
    alternate_objects = Path(object_path).expanduser().absolute()
    _require(
        not alternate_objects.is_symlink(),
        "Approved Git object directory must not be a symlink",
    )
    alternate_objects = alternate_objects.resolve()
    object_info = alternate_objects.lstat()
    _require(
        stat.S_ISDIR(object_info.st_mode),
        "Approved Git object directory is not a directory",
    )
    _require(
        "\n" not in str(alternate_objects) and "\r" not in str(alternate_objects),
        "Approved Git object directory path contains a newline",
    )

    destination_parent = supplied_destination.parent.resolve()
    _require(destination_parent.is_dir(), "Private Git parent directory is missing")
    git_dir = destination_parent / supplied_destination.name
    try:
        _run_git_command(
            ("init", "-q", "--bare", f"--object-format={object_format}", str(git_dir)),
            cwd=destination_parent,
            environment=base_environment,
        )
        alternates = git_dir / "objects" / "info" / "alternates"
        alternates.write_text(f"{alternate_objects}\n", encoding="utf-8")
        provisional = PrivateGeneratorGitView(
            git_dir=git_dir,
            work_tree=immutable_source,
            index_file=git_dir / "index",
            approved_commit=approved_generator_commit,
            approved_tree=approved_generator_tree,
            object_format=object_format,
            alternate_objects=alternate_objects,
            alternate_objects_device=object_info.st_dev,
            alternate_objects_inode=object_info.st_ino,
            git_dir_device=0,
            git_dir_inode=0,
            work_tree_device=0,
            work_tree_inode=0,
            metadata_sha256="",
        )
        setup_environment = _private_git_environment(provisional)
        _run_git_command(
            ("update-ref", "--no-deref", "HEAD", approved_generator_commit),
            cwd=immutable_source,
            environment=setup_environment,
        )
        _run_git_command(
            ("read-tree", approved_generator_commit),
            cwd=immutable_source,
            environment=setup_environment,
        )
        for path in git_dir.rglob("*"):
            _require(
                not path.is_symlink(),
                f"Private Git initialization created a symlink: {path}",
            )
            _require(
                path.is_file() or path.is_dir(),
                f"Private Git initialization created a special entry: {path}",
            )
            if path.is_file():
                path.chmod(0o400)
        for directory in sorted(
            (path for path in git_dir.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o500)
        git_dir.chmod(0o500)
        git_info = git_dir.lstat()
        source_info = immutable_source.lstat()
        view = PrivateGeneratorGitView(
            **{
                **provisional.__dict__,
                "git_dir_device": git_info.st_dev,
                "git_dir_inode": git_info.st_ino,
                "work_tree_device": source_info.st_dev,
                "work_tree_inode": source_info.st_ino,
                "metadata_sha256": _private_git_metadata_sha256(git_dir),
            }
        )
        _verify_private_generator_git_view(view)
        return view
    except Exception:
        if git_dir.exists():
            _remove_tree(git_dir)
        raise


def _generator_environment(
    immutable_source: Path,
    runtime: TrustedFigureRuntime,
    git_view: PrivateGeneratorGitView,
) -> dict[str, str]:
    immutable_source = Path(immutable_source).resolve()
    _require(
        immutable_source == git_view.work_tree,
        "Generator environment source differs from private Git work tree",
    )
    _verify_private_generator_git_view(git_view)
    env = dict(runtime.environment)
    env = _private_git_environment(git_view, env)
    env["PYTHONPATH"] = str(immutable_source)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["SLACK_NOTIFY"] = "0"
    return env


def _stage_candidate(source: Path, staging: Path, spec, generator: str) -> dict:
    target = staging / "candidates" / spec.root / spec.pdf
    _copy(source, target)
    return {
        "candidate_id": spec.candidate_id,
        "path": target.relative_to(staging).as_posix(),
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
    staging: Path,
    manifest_path: Path,
    manifest_digest: str,
    candidate_specs: Sequence[object],
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
    specs = {spec.candidate_id: spec for spec in candidate_specs}
    for record in payload.get("candidates", []):
        _require(
            record.get("candidate_id") in specs,
            "Orchestration manifest names a candidate outside the private allowlist",
        )
        spec = specs[record["candidate_id"]]
        expected_relative = (Path("candidates") / spec.root / spec.pdf).as_posix()
        _require(
            record.get("path") == expected_relative,
            f"Candidate path changed before publish: {spec.candidate_id}",
        )
        path = staging / expected_relative
        _require(
            path.is_file() and file_sha256(path) == record["sha256"],
            f"Candidate mutated before publish: {spec.candidate_id}",
        )


def _verify_tree_records(root: Path, records: Sequence[Mapping[str, object]]) -> None:
    indexed = {str(record["path"]): record for record in records}
    _require(
        len(indexed) == len(records), "Accepted staging inventory contains duplicates"
    )
    actual = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"Accepted staging contains symlink: {path}")
        _require(
            path.is_file() or path.is_dir(),
            f"Accepted staging contains non-regular entry: {path}",
        )
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    _require(
        actual == set(indexed), "Accepted staging inventory changed after acceptance"
    )
    for relative, record in indexed.items():
        path = root / relative
        _require(
            not path.is_symlink()
            and path.stat().st_size == record["size_bytes"]
            and file_sha256(path) == record["sha256"],
            f"Accepted staging artifact mutated: {relative}",
        )


def _copy_publication_tree(
    source: Path,
    destination: Path,
    accepted_records: Sequence[Mapping[str, object]],
) -> None:
    """Checksum-copy the accepted build into a distinct private tree."""

    _require(destination.is_dir(), f"Publication tree is unavailable: {destination}")
    _verify_tree_records(source, accepted_records)
    for record in sorted(accepted_records, key=lambda item: str(item["path"])):
        relative = Path(str(record["path"]))
        path = source / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        copy_verified_file(path, target, expected_sha256=str(record["sha256"]))
    _verify_tree_records(source, accepted_records)


def _published_tree_records(root: Path) -> list[dict]:
    ledger = root / PUBLISHED_TREE_LEDGER
    return [
        snapshot_record(path, root, "published_artifact")
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != ledger
    ]


def _verify_frozen_publication_tree(root: Path, *, ledger_sha256: str) -> None:
    """Rehash the exact sealed tree immediately before atomic publication."""

    ledger = root / PUBLISHED_TREE_LEDGER
    _require(
        ledger.is_file() and file_sha256(ledger) == ledger_sha256,
        "Published-tree ledger mutated before publication",
    )
    payload = load_json_object(ledger)
    _require(
        set(payload) == {"schema_version", "files"}
        and payload.get("schema_version") == 1
        and isinstance(payload.get("files"), list),
        "Published-tree ledger schema is invalid",
    )
    records = payload["files"]
    indexed: dict[str, dict] = {}
    for record in records:
        _require(
            isinstance(record, dict)
            and set(record) == {"role", "path", "sha256", "size_bytes"}
            and record.get("role") == "published_artifact",
            "Published-tree ledger contains a malformed record",
        )
        relative = record["path"]
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        _require(
            pure is not None
            and not pure.is_absolute()
            and "." not in pure.parts
            and ".." not in pure.parts
            and relative not in indexed
            and relative != PUBLISHED_TREE_LEDGER,
            "Published-tree ledger contains an unsafe or duplicate path",
        )
        indexed[relative] = record

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != ledger
    }
    _require(
        set(indexed) == actual_files,
        "Published-tree inventory differs from its ledger",
    )
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"Frozen publication contains symlink: {path}")
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_file():
            _require(mode == 0o400, f"Frozen publication file mode changed: {path}")
        elif path.is_dir():
            _require(
                mode == 0o500, f"Frozen publication directory mode changed: {path}"
            )
    _require(
        stat.S_IMODE(root.lstat().st_mode) == 0o500,
        "Frozen publication root mode changed",
    )
    for relative, record in indexed.items():
        path = root / relative
        _require(
            path.stat().st_size == record["size_bytes"]
            and file_sha256(path) == record["sha256"],
            f"Frozen publication artifact mutated: {relative}",
        )


def _published_logical_path(
    root: Path, raw: object, *, label: str, directory: bool = False
) -> str:
    _require(
        isinstance(raw, str) and raw and "\\" not in raw,
        f"{label} must be a nonempty publication-root-relative POSIX path",
    )
    pure = PurePosixPath(raw)
    _require(
        not pure.is_absolute() and "." not in pure.parts and ".." not in pure.parts,
        f"{label} is not a safe publication-root-relative path: {raw}",
    )
    target = root.joinpath(*pure.parts)
    _require(not target.is_symlink(), f"{label} resolves through a symlink: {raw}")
    _require(
        target.is_dir() if directory else target.is_file(),
        f"{label} does not resolve to an existing {'directory' if directory else 'file'}: {raw}",
    )
    return pure.as_posix()


def validate_published_logical_paths(root: Path) -> tuple[str, ...]:
    """Verify that every wrapper-owned artifact path resolves in the final tree."""

    root = Path(root).resolve()
    _require(root.is_dir(), f"Publication root does not exist: {root}")
    checked: list[str] = []

    def check(raw: object, label: str, *, directory: bool = False) -> None:
        checked.append(
            _published_logical_path(root, raw, label=label, directory=directory)
        )

    orchestration_path = root / ORCHESTRATION_MANIFEST
    if orchestration_path.is_file():
        orchestration = load_json_object(orchestration_path)
        external_approval = orchestration.get("external_approval", {})
        if isinstance(external_approval, dict) and "snapshot_path" in external_approval:
            check(
                external_approval["snapshot_path"],
                "orchestration external approval snapshot_path",
            )
        config_authority = orchestration.get("execution_config_authority", {})
        if isinstance(config_authority, dict) and "path" in config_authority:
            check(
                config_authority["path"],
                "orchestration execution config authority path",
            )
        scientific_release = orchestration.get("scientific_release", {})
        if isinstance(scientific_release, dict):
            if "root" in scientific_release:
                check(
                    scientific_release["root"],
                    "orchestration scientific release root",
                    directory=True,
                )
            artifacts = scientific_release.get("artifacts", {})
            if isinstance(artifacts, dict):
                for name, record in artifacts.items():
                    if isinstance(record, dict) and "path" in record:
                        check(
                            record["path"],
                            f"orchestration scientific release artifact {name}",
                        )
        release_snapshot = orchestration.get("release_input_snapshot", {})
        if isinstance(release_snapshot, dict) and "root" in release_snapshot:
            check(
                release_snapshot["root"],
                "orchestration release input snapshot root",
                directory=True,
            )
        allowlist = orchestration.get("allowlist", {})
        if isinstance(allowlist, dict) and "path" in allowlist:
            check(allowlist["path"], "orchestration allowlist path")
        for index, record in enumerate(orchestration.get("candidates", [])):
            if isinstance(record, dict) and "path" in record:
                check(record["path"], f"orchestration candidate {index} path")
        for index, record in enumerate(orchestration.get("snapshot_artifacts", [])):
            if isinstance(record, dict) and "path" in record:
                check(record["path"], f"orchestration snapshot artifact {index} path")

    source_map_path = root / "review" / "figure_candidate_source_map.json"
    if source_map_path.is_file():
        source_map = load_json_object(source_map_path)
        release = source_map.get("release", {})
        if isinstance(release, dict):
            if "root" in release:
                check(release["root"], "source map release root", directory=True)
            artifacts = release.get("artifacts", {})
            if isinstance(artifacts, dict):
                for name, record in artifacts.items():
                    if isinstance(record, dict) and "path" in record:
                        check(record["path"], f"source map release artifact {name}")
        check(source_map["allowlist"]["path"], "source map allowlist path")
        for name, path in source_map.get("roots", {}).items():
            check(path, f"source map root {name}", directory=True)
        check(source_map["review"]["path"], "source map review PDF path")
        check(source_map["vector_qa"]["path"], "source map vector QA path")
        for index, record in enumerate(source_map.get("candidates", [])):
            check(record["pdf_path"], f"source map candidate {index} PDF path")
            check(record["png_path"], f"source map candidate {index} preview path")
            check(
                record["provenance_manifest"],
                f"source map candidate {index} provenance path",
            )

    qa_path = root / "review" / "figure_candidate_vector_qa.json"
    if qa_path.is_file():
        qa = load_json_object(qa_path)
        for index, report in enumerate(qa.get("candidate_reports", [])):
            check(report["path"], f"vector QA candidate {index} path")
        check(qa["review_report"]["path"], "vector QA review PDF path")

    source_csv = root / "review" / "figure_candidate_source_map.csv"
    if source_csv.is_file():
        with source_csv.open(newline="", encoding="utf-8") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                check(row["pdf_relative_path"], f"source CSV candidate {index} PDF")
                check(row["png_relative_path"], f"source CSV candidate {index} preview")
                check(
                    row["provenance_manifest"],
                    f"source CSV candidate {index} provenance",
                )

    ledger_path = root / PUBLISHED_TREE_LEDGER
    if ledger_path.is_file():
        ledger = load_json_object(ledger_path)
        for index, record in enumerate(ledger.get("files", [])):
            check(record["path"], f"published ledger artifact {index}")
    return tuple(checked)
