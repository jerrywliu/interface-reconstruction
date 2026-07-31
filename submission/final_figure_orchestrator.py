#!/usr/bin/env python3
"""Regenerate, prove, accept, and atomically publish final figure candidates."""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submission.accept_figure_candidates import (
    DEFAULT_ALLOWLIST,
    EXPECTED_COUNTS,
    FigureAcceptanceError,
    _accept_orchestrated_candidates,
    _create_orchestrated_acceptance_state,
    build_vector_review_pdf,
    load_candidate_allowlist,
    pdf_page_info,
    render_pdf_preview,
)
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
from submission.pdf_vector_qa import inspect_pdf
from submission.trusted_figure_runtime import (
    TrustedFigureRuntime,
    TrustedFigureRuntimeError,
    prepare_trusted_figure_runtime,
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


@dataclass
class PublicationReservation:
    path: Path
    device: int
    inode: int
    descriptor: int
    released: bool = False


@dataclass(frozen=True)
class ReleaseInputSnapshot:
    root: Path
    csv_path: Path
    plots_root: Path
    anchor: dict
    artifact_records: tuple[dict, ...]
    alias_sources: Mapping[str, str]


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


def _reserve_publication(output_root: Path) -> PublicationReservation:
    supplied = Path(output_root).expanduser().absolute()
    _require(
        not os.path.lexists(supplied),
        f"Output root must not exist, including as a symlink: {supplied}",
    )
    output_root = supplied.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    reservation = output_root.parent / f".{output_root.name}.final-figure-reservation"
    try:
        descriptor = os.open(
            reservation,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise FinalFigureOrchestrationError(
            f"Publication destination is already reserved: {output_root}"
        ) from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        payload = f"pid={os.getpid()}\noutput={output_root}\n".encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
    except Exception:
        os.close(descriptor)
        reservation.unlink(missing_ok=True)
        raise
    if os.path.lexists(output_root):
        reservation.unlink(missing_ok=True)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise FinalFigureOrchestrationError(
            f"Output root appeared during reservation: {output_root}"
        )
    return PublicationReservation(reservation, info.st_dev, info.st_ino, descriptor)


def _verify_reservation(reservation: PublicationReservation) -> None:
    _require(not reservation.released, "Publication reservation is already released")
    try:
        descriptor_info = os.fstat(reservation.descriptor)
    except OSError as exc:
        raise FinalFigureOrchestrationError(
            "Publication reservation lock descriptor is unavailable"
        ) from exc
    try:
        info = reservation.path.lstat()
    except FileNotFoundError as exc:
        raise FinalFigureOrchestrationError(
            "Publication reservation disappeared"
        ) from exc
    _require(
        stat.S_ISREG(info.st_mode)
        and (info.st_dev, info.st_ino) == (reservation.device, reservation.inode),
        "Publication reservation was replaced",
    )
    _require(
        (descriptor_info.st_dev, descriptor_info.st_ino)
        == (reservation.device, reservation.inode),
        "Publication reservation descriptor was replaced",
    )


def _release_reservation(reservation: PublicationReservation) -> None:
    if reservation.released:
        return
    try:
        info = reservation.path.lstat()
    except FileNotFoundError:
        info = None
    try:
        if info is not None and (info.st_dev, info.st_ino) == (
            reservation.device,
            reservation.inode,
        ):
            reservation.path.unlink()
    finally:
        try:
            fcntl.flock(reservation.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(reservation.descriptor)
            reservation.released = True


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory and fail if destination exists."""

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        at_fdcwd = -2
        rename_excl = 0x00000004
        rename = libc.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            at_fdcwd,
            ctypes.c_char_p(source_bytes),
            at_fdcwd,
            ctypes.c_char_p(destination_bytes),
            rename_excl,
        )
    elif hasattr(libc, "renameat2"):
        at_fdcwd = -100
        rename_noreplace = 1
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            at_fdcwd,
            ctypes.c_char_p(source_bytes),
            at_fdcwd,
            ctypes.c_char_p(destination_bytes),
            rename_noreplace,
        )
    else:
        raise FinalFigureOrchestrationError(
            "Atomic no-replace directory publication is unavailable"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FinalFigureOrchestrationError(
                f"Publication destination appeared before publish: {destination}"
            )
        raise FinalFigureOrchestrationError(
            f"Atomic no-replace publication failed: {os.strerror(error_number)}"
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


def _copy_config_tree(source: Path, destination: Path) -> None:
    _require(not destination.exists(), f"Config snapshot already exists: {destination}")
    destination.mkdir(parents=True)
    for path in sorted(Path(source).rglob("*")):
        _require(not path.is_symlink(), f"Approved config contains symlink: {path}")
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


def _generator_environment(
    repository: Path,
    immutable_source: Path,
    runtime: TrustedFigureRuntime,
) -> dict[str, str]:
    env = dict(runtime.environment)
    env.update(sanitized_git_environment(env))
    env["PYTHONPATH"] = str(Path(immutable_source).resolve())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["SLACK_NOTIFY"] = "0"
    git_dir = subprocess.run(
        ["git", "--no-pager", "rev-parse", "--absolute-git-dir"],
        cwd=repository,
        env=sanitized_git_environment(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    env["GIT_DIR"] = git_dir
    env["GIT_WORK_TREE"] = str(Path(immutable_source).resolve())
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
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


def finalize_publication(
    *,
    staging: Path,
    output_root: Path,
    reservation: PublicationReservation,
    manifest_path: Path,
    acceptance_runner: Callable[..., object],
    acceptance_kwargs: Mapping[str, object],
    candidate_specs: Sequence[object] = (),
    after_acceptance_hook: Optional[Callable[[Path], None]] = None,
    after_publish_freeze_hook: Optional[Callable[[Path], None]] = None,
) -> None:
    """Accept, clone, seal, rehash, and atomically publish one private tree."""

    manifest_digest = file_sha256(manifest_path)
    publish_tree = None
    try:
        acceptance_runner(**dict(acceptance_kwargs))
        accepted_records = [
            snapshot_record(path, staging, "accepted_artifact")
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        if after_acceptance_hook is not None:
            after_acceptance_hook(staging)
        _rehash_before_publish(staging, manifest_path, manifest_digest, candidate_specs)
        publish_tree = Path(
            tempfile.mkdtemp(
                prefix=f".{output_root.name}.publish-", dir=output_root.parent
            )
        )
        publish_tree.chmod(0o700)
        _verify_tree_records(staging, accepted_records)
        _copy_publication_tree(staging, publish_tree, accepted_records)
        tree_records = _published_tree_records(publish_tree)
        atomic_write_json(
            publish_tree / PUBLISHED_TREE_LEDGER,
            {"schema_version": 1, "files": tree_records},
        )
        ledger_sha256 = file_sha256(publish_tree / PUBLISHED_TREE_LEDGER)
        validate_published_logical_paths(publish_tree)
        _remove_tree(staging)
        make_tree_read_only(publish_tree)
        if after_publish_freeze_hook is not None:
            after_publish_freeze_hook(publish_tree)
        _verify_reservation(reservation)
        _require(
            not os.path.lexists(output_root),
            f"Publication destination appeared before publish: {output_root}",
        )
        _verify_frozen_publication_tree(publish_tree, ledger_sha256=ledger_sha256)
        _rename_directory_noreplace(publish_tree, output_root)
        publish_tree = None
    except Exception:
        if staging.exists():
            _remove_tree(staging)
        if publish_tree is not None and publish_tree.exists():
            _remove_tree(publish_tree)
        raise
    finally:
        _release_reservation(reservation)


def orchestrate_final_figures(
    *,
    repository: Path,
    release_root: Path,
    approved_generator_commit: str,
    approval_record: Path,
    approval_record_sha256: str,
    output_root: Path,
    allowlist_path: Path = DEFAULT_ALLOWLIST,
    command_runner: Callable[
        [Sequence[str], Path, Mapping[str, str], Path], None
    ] = run_command,
    after_acceptance_hook: Optional[Callable[[Path], None]] = None,
    after_publish_freeze_hook: Optional[Callable[[Path], None]] = None,
    source_materialized_hook: Optional[Callable[[Path, Path], None]] = None,
    release_after_open_hook: Optional[Callable[[Path], None]] = None,
) -> Path:
    repository = Path(repository).resolve()
    release_root = Path(release_root).expanduser().absolute()
    supplied_output = Path(output_root).expanduser().absolute()
    _require(
        not os.path.lexists(supplied_output),
        f"Output root must not exist, including as a symlink: {supplied_output}",
    )
    output_root = supplied_output.resolve()
    _require(
        repository == REPO_ROOT.resolve(),
        "--repository must be the checkout containing this reviewed wrapper",
    )
    expected_allowlist = repository / "submission" / "final_figure_candidates.json"
    _require(
        Path(allowlist_path).resolve() == expected_allowlist.resolve(),
        "Final acceptance must use the approved repository allowlist",
    )
    _require(not output_root.exists(), f"Output root must not exist: {output_root}")
    release_audit_pin = validate_final_release_contract(release_root)
    attestation = verify_generator_checkout(
        repository, approved_generator_commit, release_audit_pin.source_commit
    )
    reservation = _reserve_publication(output_root)
    staging = None
    execution = None
    try:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_root.name}.staging-", dir=output_root.parent
            )
        )
        execution = Path(tempfile.mkdtemp(prefix="final-figure-execution-"))
        immutable_source = execution / "approved_source"
        attestation = materialize_approved_source(
            repository,
            approved_generator_commit,
            immutable_source,
            attestation,
        )
        materialized_allowlist = (
            immutable_source / "submission" / "final_figure_candidates.json"
        )
        private_allowlist = _copy(materialized_allowlist, staging / PRIVATE_ALLOWLIST)
        private_allowlist.chmod(0o400)
        specs = load_candidate_allowlist(private_allowlist)
        _require(
            len(specs) == EXPECTED_COUNTS["candidate_pdfs"],
            "Private allowlist does not encode the exact candidate contract",
        )
        if source_materialized_hook is not None:
            source_materialized_hook(repository, immutable_source)
        verify_materialized_source(
            repository, approved_generator_commit, immutable_source, attestation
        )

        def approved_command_runner(
            command: Sequence[str],
            cwd: Path,
            command_environment: Mapping[str, str],
            log_path: Path,
        ) -> None:
            verify_materialized_source(
                repository, approved_generator_commit, immutable_source, attestation
            )
            command_runner(command, cwd, command_environment, log_path)
            verify_materialized_source(
                repository, approved_generator_commit, immutable_source, attestation
            )

        figure_root = staging / "candidates" / "figure_root"
        c0_root = staging / "candidates" / "c0_root"
        _require(
            not figure_root.exists() and not c0_root.exists(),
            "Candidate roots must start nonexistent",
        )
        _copy_config_tree(
            immutable_source / "config",
            execution / "config",
        )
        plots_root = execution / "plots"
        plots_root.mkdir()
        release_snapshot = _snapshot_release_inputs(
            release_root,
            staging / "provenance" / "release_input_snapshot",
            audit_pin=release_audit_pin,
            staging_root=staging,
            after_open_hook=release_after_open_hook,
        )
        anchor = release_snapshot.anchor
        release_view = release_snapshot.plots_root
        release_aliases = release_snapshot.alias_sources
        release_sha256sums_sha256 = file_sha256(release_snapshot.root / "SHA256SUMS")
        _require(
            release_sha256sums_sha256 == release_audit_pin.sha256sums_sha256
            and anchor["source_commit"] == release_audit_pin.source_commit,
            "Private release snapshot differs from the completed live audit",
        )
        approval = verify_external_approval_record(
            approval_record,
            approval_record_sha256,
            repository=repository,
            approved_commit=approved_generator_commit,
            approved_tree=attestation.commit_tree,
            scientific_release_commit=anchor["source_commit"],
            release_sha256sums_sha256=release_sha256sums_sha256,
            allowlist_sha256=file_sha256(private_allowlist),
            candidate_contract=EXPECTED_COUNTS,
            orchestrator_schema_version=ORCHESTRATION_SCHEMA_VERSION,
        )
        runtime = prepare_trusted_figure_runtime(execution / "trusted_runtime")
        env = _generator_environment(repository, immutable_source, runtime)
        python = str(runtime.attestation["python"]["executable"])
        generated = execution / "generated"
        logs = staging / "provenance" / "logs"
        snapshot_artifacts: list[dict] = list(release_snapshot.artifact_records)
        snapshot_artifacts.append(
            snapshot_record(private_allowlist, staging, "approved_candidate_allowlist")
        )
        runtime_record = staging / "provenance" / "trusted_runtime.json"
        atomic_write_json(runtime_record, runtime.attestation)
        snapshot_artifacts.append(
            snapshot_record(runtime_record, staging, "trusted_figure_runtime")
        )
        candidates: list[dict] = []
        contracts: dict[str, dict] = {"final_release": dict(release_audit_pin.contract)}
        approval_snapshot = copy_verified_file(
            Path(approval.path),
            staging / "provenance" / "external_approval_record.json",
            expected_sha256=approval.sha256,
        )
        snapshot_artifacts.append(
            snapshot_record(approval_snapshot, staging, "external_approval_record")
        )

        main_out = generated / "maintext"
        main_cmd = [
            python,
            "-B",
            "-m",
            "experiments.static.generate_section6_maintext_figures",
            "--csv",
            str(release_snapshot.csv_path),
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
        approved_command_runner(main_cmd, execution, env, logs / "maintext.log")
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
                execution / "maintext_command.json",
                main_cmd,
                approved_generator_commit,
                staging_root=staging,
                execution_root=execution,
            ),
            staging,
            "provenance/maintext/command.json",
            "generator_command",
            snapshot_artifacts,
        )

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
                _require(
                    alias in release_aliases, f"Release snapshot lacks alias {alias}"
                )
                source_manifest = release_view / alias / "run_manifest.json"
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
            "-B",
            "-m",
            "experiments.static.run_perturbed_sweeps",
            "--plot_from_csv",
            str(release_snapshot.csv_path),
            "--summary_dir",
            str(all_out),
            "--no-notify",
        ]
        _require(not all_out.exists(), "All-method generator root already exists")
        approved_command_runner(all_cmd, execution, env, logs / "all_methods.log")
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
                staging_root=staging,
                execution_root=execution,
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
                    "path": target.relative_to(staging).as_posix(),
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
                "-B",
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
            approved_command_runner(
                cmd, execution, env, logs / f"resolution_{experiment}.log"
            )
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
                    staging_root=staging,
                    execution_root=execution,
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
            resolution_payload = load_json_object(out / "manifest.json")
            input_count = 0
            for run_index, run in enumerate(
                sorted(resolution_payload["runs"], key=lambda item: item["save_name"])
            ):
                run_root = plots_root / str(run["save_name"])
                for input_path, role in resolution_input_paths(
                    plots_root,
                    experiment=experiment,
                    save_name=str(run["save_name"]),
                    case_index=case_index,
                    include_consumed_truth=(
                        experiment in {"squares", "circles"}
                        and _same_number(run.get("resolution"), RESOLUTION_VALUES[0])
                        and _same_number(run.get("wiggle"), RESOLUTION_WIGGLES[0])
                    ),
                ):
                    relative = input_path.relative_to(run_root)
                    _copy_manifest(
                        input_path,
                        staging,
                        (
                            f"provenance/resolution/{experiment}/inputs/"
                            f"{run_index:02d}/{relative.as_posix()}"
                        ),
                        role,
                        snapshot_artifacts,
                    )
                    input_count += 1
            resolution_contract["experiments"][experiment][
                "snapshotted_input_files"
            ] = input_count
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
                "-B",
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
            approved_command_runner(
                cmd, execution, env, logs / f"guarded_c0_{experiment}.log"
            )
            c0_paths[experiment] = out / "manifest.json"
            c0_commands[experiment] = cmd
        c0_run_manifests = validate_c0_manifests(
            c0_paths, plots_root, approved_generator_commit
        )
        c0_metric_contracts = {
            experiment: validate_c0_metrics(
                generated / "guarded_c0" / experiment / "metrics.csv",
                experiment,
                c0_paths[experiment],
            )
            for experiment in ("ellipses", "zalesak")
        }
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
            "metrics": c0_metric_contracts,
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
                    staging_root=staging,
                    execution_root=execution,
                ),
                staging,
                f"provenance/guarded_c0/{experiment}/command.json",
                "generator_command",
                snapshot_artifacts,
            )
            _copy_manifest(
                out / "metrics.csv",
                staging,
                f"provenance/guarded_c0/{experiment}/metrics.csv",
                "guarded_c0_aggregate_metrics",
                snapshot_artifacts,
            )
            representative = C0_REPRESENTATIVES[experiment]
            c0_payload = load_json_object(c0_paths[experiment])
            representative_runs = [
                run
                for run in c0_payload["runs"]
                if _same_number(run.get("resolution"), representative["resolution"])
                and _same_number(run.get("wiggle"), representative["wiggle"])
                and run.get("seed") == representative["seed"]
            ]
            _require(
                {run.get("variant") for run in representative_runs}
                == set(C0_VARIANTS[experiment]),
                f"Guarded-C0 representative run set differs for {experiment}",
            )
            for run_index, run in enumerate(
                sorted(representative_runs, key=lambda item: item["variant"])
            ):
                run_root = plots_root / str(run["save_name"])
                for input_path, role in representative_geometry_input_paths(
                    run_root, case_index=representative["case_index"]
                ):
                    relative = input_path.relative_to(run_root)
                    _copy_manifest(
                        input_path,
                        staging,
                        (
                            f"provenance/guarded_c0/{experiment}/representative_inputs/"
                            f"{run_index:02d}/{relative.as_posix()}"
                        ),
                        f"guarded_c0_{role}",
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
            "-B",
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
        approved_command_runner(
            plic_cmd, execution, env, logs / "deterministic_plic.log"
        )
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
                execution / "plic_command.json",
                plic_cmd,
                approved_generator_commit,
                staging_root=staging,
                execution_root=execution,
            ),
            staging,
            "provenance/deterministic/plic_command.json",
            "generator_command",
            snapshot_artifacts,
        )

        staged_out = deterministic / "staged"
        staged_cmd = [
            python,
            "-B",
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
        approved_command_runner(
            staged_cmd, execution, env, logs / "deterministic_staged.log"
        )
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
                execution / "staged_command.json",
                staged_cmd,
                approved_generator_commit,
                staging_root=staging,
                execution_root=execution,
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
        for role, expected_count in {
            "resolution_case_metrics": 30,
            "resolution_case_geometry": 30,
            "resolution_mesh_geometry": 30,
            "resolution_reconstructed_geometry": 30,
            "resolution_facet_metadata": 30,
            "resolution_truth_geometry": 2,
            "guarded_c0_aggregate_metrics": 2,
            "guarded_c0_representative_case_geometry": 6,
            "guarded_c0_representative_mesh_geometry": 6,
            "guarded_c0_representative_reconstructed_geometry": 6,
            "guarded_c0_representative_facet_metadata": 6,
        }.items():
            actual_count = sum(row["role"] == role for row in snapshot_artifacts)
            _require(
                actual_count == expected_count,
                f"Snapshot role {role} has {actual_count}, expected {expected_count}",
            )

        orchestration = {
            "schema_version": ORCHESTRATION_SCHEMA_VERSION,
            "manifest_type": "final_figure_orchestration",
            "status": "ready_for_internal_acceptance",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator_checkout": attestation.to_dict(),
            "trusted_figure_runtime": runtime.attestation,
            "external_approval": {
                key: value for key, value in approval.to_dict().items() if key != "path"
            }
            | {"snapshot_path": "provenance/external_approval_record.json"},
            "scientific_release": anchor,
            "audited_release_authority": {
                "source_commit": release_audit_pin.source_commit,
                "sha256sums_sha256": release_audit_pin.sha256sums_sha256,
                "resolved_config_sha256": release_audit_pin.resolved_config_sha256,
            },
            "release_input_snapshot": {
                "root": "provenance/release_input_snapshot",
                "representative_alias_sources": dict(
                    sorted(release_snapshot.alias_sources.items())
                ),
                "artifact_count": len(release_snapshot.artifact_records),
            },
            "scientific_contracts": contracts,
            "allowlist": {
                "path": PRIVATE_ALLOWLIST,
                "sha256": file_sha256(private_allowlist),
                "expected_counts": EXPECTED_COUNTS,
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
        acceptance_state = _create_orchestrated_acceptance_state(
            figure_root=figure_root,
            c0_root=c0_root,
            snapshot_root=staging,
            release_anchor=anchor,
            generator_source_commit=approved_generator_commit,
            orchestration_record=manifest_path,
            allowlist_path=private_allowlist,
            candidate_records=orchestration["candidates"],
        )
        finalize_publication(
            staging=staging,
            output_root=output_root,
            reservation=reservation,
            manifest_path=manifest_path,
            acceptance_runner=_accept_orchestrated_candidates,
            acceptance_kwargs={
                "orchestration_state": acceptance_state,
                "output_dir": staging / "review",
                "pdf_inspector": partial(inspect_pdf, runtime=runtime),
                "page_inspector": partial(pdf_page_info, runtime=runtime),
                "preview_renderer": partial(render_pdf_preview, runtime=runtime),
                "review_builder": partial(build_vector_review_pdf, runtime=runtime),
            },
            candidate_specs=specs,
            after_acceptance_hook=after_acceptance_hook,
            after_publish_freeze_hook=after_publish_freeze_hook,
        )
    except Exception as exc:
        if staging is not None and staging.exists():
            _remove_tree(staging)
        _release_reservation(reservation)
        if isinstance(
            exc,
            (
                FinalFigureOrchestrationError,
                FigureAcceptanceError,
                GeneratorCheckoutError,
                TrustedFigureRuntimeError,
            ),
        ):
            raise
        raise FinalFigureOrchestrationError(str(exc)) from exc
    finally:
        if execution is not None and execution.exists():
            for path in execution.rglob("*"):
                if path.is_dir() and not path.is_symlink():
                    try:
                        path.chmod(0o700)
                    except OSError:
                        pass
            try:
                (execution / "approved_source").chmod(0o700)
            except OSError:
                pass
            _remove_tree(execution)
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
        "--approval-record",
        type=Path,
        required=True,
        help="external reviewer approval JSON for the exact final generator commit",
    )
    parser.add_argument(
        "--approval-record-sha256",
        required=True,
        help="externally recorded SHA-256 of --approval-record",
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
            approval_record=args.approval_record,
            approval_record_sha256=args.approval_record_sha256,
            output_root=args.output_root,
            allowlist_path=args.allowlist,
        )
    except (
        FinalFigureOrchestrationError,
        FigureAcceptanceError,
        GeneratorCheckoutError,
        TrustedFigureRuntimeError,
    ) as exc:
        print(f"FINAL FIGURE ORCHESTRATION ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Final figure publication: {output}")
    print(f"Review PDF: {output / 'review' / 'figure_candidate_review.pdf'}")
    print(f"Provenance: {output / ORCHESTRATION_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
