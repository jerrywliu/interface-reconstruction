#!/usr/bin/env python3
"""Regenerate, prove, accept, and atomically publish final figure candidates."""

from __future__ import annotations

import os
import stat
import sys


def _require_isolated_cli_startup() -> None:
    flags = sys.flags
    forbidden_modules = sorted(
        name
        for name in sys.modules
        if name in {"sitecustomize", "usercustomize", "submission", "experiments"}
        or name.startswith(("submission.", "experiments."))
    )
    if not (
        flags.isolated
        and flags.ignore_environment
        and flags.no_user_site
        and not forbidden_modules
    ):
        detail = (
            f"; preloaded={','.join(forbidden_modules)}" if forbidden_modules else ""
        )
        sys.stderr.write(
            "FINAL FIGURE STARTUP ERROR: use the trusted "
            "submission/run_final_figure_orchestrator launcher; the CLI requires "
            f"a fresh isolated Python process{detail}\n"
        )
        raise SystemExit(2)


def _require_trusted_launcher_descriptor() -> None:
    launcher = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "run_final_figure_orchestrator"
    )
    try:
        expected = os.lstat(launcher)
        inherited = os.fstat(9)
    except OSError:
        expected = inherited = None
    if not (
        expected is not None
        and inherited is not None
        and stat.S_ISREG(expected.st_mode)
        and (expected.st_dev, expected.st_ino) == (inherited.st_dev, inherited.st_ino)
    ):
        sys.stderr.write(
            "FINAL FIGURE STARTUP ERROR: use the trusted "
            "submission/run_final_figure_orchestrator launcher; its inherited "
            "descriptor is missing or invalid\n"
        )
        raise SystemExit(2)


if __name__ != "__main__":
    raise ImportError(
        "submission.final_figure_orchestrator is a script-only publication "
        "boundary; use submission/run_final_figure_orchestrator"
    )
_require_isolated_cli_startup()
_require_trusted_launcher_descriptor()

import argparse
import csv
import ctypes
import errno
import fcntl
import json
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submission.accept_figure_candidates import (
    INDEX_ROWS_PER_PAGE,
    REVIEW_PDF,
    SOURCE_MAP_CSV,
    SOURCE_MAP_JSON,
    VECTOR_QA_JSON,
    AcceptanceOutputs,
    AcceptedCandidate,
    CandidateSpec,
    DEFAULT_ALLOWLIST,
    EXPECTED_COUNTS,
    FigureAcceptanceError,
    ProvenanceEvidence,
    _candidate_pdf_logical_path,
    _candidate_preview_logical_path,
    _qa_dict,
    _root_contains,
    _sha256,
    _validate_invocation,
    _write_csv,
    build_vector_review_pdf,
    inspect_generated_preview,
    load_candidate_allowlist,
    pdf_page_info,
    render_pdf_preview,
    verify_candidate_inventory,
    verify_review_page_map,
)
from submission.audit_final_release import audit_final_release, verify_sha256_manifest
from submission.final_figure_orchestration import (
    ALL_METHOD_FILES,
    C0_REPRESENTATIVES,
    C0_RESOLUTIONS,
    C0_VARIANTS,
    C0_WIGGLES,
    EXPERIMENTS,
    MAINTEXT_CASES,
    MAINTEXT_METHODS,
    ORCHESTRATION_MANIFEST,
    ORCHESTRATION_SCHEMA_VERSION,
    PRIVATE_ALLOWLIST,
    PROFILE,
    PUBLISHED_TREE_LEDGER,
    RELEASE_METHODS,
    RESOLUTION_CASES,
    RESOLUTION_VALUES,
    RESOLUTION_WIGGLES,
    FinalFigureOrchestrationError,
    _capture_release_audit_pin,
    _copy,
    _copy_manifest,
    _copy_publication_tree,
    _generator_environment,
    _numbers,
    _published_tree_records,
    _rehash_before_publish,
    _remove_tree,
    _require,
    _same_number,
    _seal_execution_config,
    _snapshot_complete_release,
    _snapshot_release_inputs,
    _stage_candidate,
    _verify_execution_config,
    _verify_frozen_publication_tree,
    _verify_tree_records,
    _write_command_record,
    representative_geometry_input_paths,
    resolution_input_paths,
    stage_all_method_candidates,
    validate_c0_manifests,
    validate_c0_metrics,
    validate_final_release_contract,
    validate_maintext_manifest,
    validate_plic_metadata,
    validate_published_logical_paths,
    validate_resolution_manifest,
    validate_staged_metadata,
)
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
from submission.pdf_vector_qa import PdfQaError, inspect_pdf
from submission.trusted_figure_runtime import (
    TrustedFigureRuntime,
    TrustedFigureRuntimeError,
    prepare_trusted_figure_runtime,
)


@dataclass(frozen=True)
class _AcceptanceState:
    figure_root: Path
    c0_root: Path
    snapshot_root: Path
    release_anchor: Mapping[str, object]
    generator_source_commit: str
    orchestration_record: Path
    orchestration_record_sha256: str
    allowlist_path: Path
    allowlist_sha256: str
    candidate_records: Tuple[Mapping[str, object], ...]


def _create_acceptance_state(
    *,
    figure_root: Path,
    c0_root: Path,
    snapshot_root: Path,
    release_anchor: Mapping[str, object],
    generator_source_commit: str,
    orchestration_record: Path,
    allowlist_path: Path,
    candidate_records: Sequence[Mapping[str, object]],
) -> _AcceptanceState:
    """Bind acceptance to this script's private staging tree."""

    snapshot_root = Path(snapshot_root).resolve()
    figure_root = Path(figure_root).resolve()
    c0_root = Path(c0_root).resolve()
    orchestration_record = Path(orchestration_record).resolve()
    allowlist_path = Path(allowlist_path).resolve()
    expected_roots = {
        "figure_root": (snapshot_root / "candidates" / "figure_root").resolve(),
        "c0_root": (snapshot_root / "candidates" / "c0_root").resolve(),
    }
    if (
        figure_root != expected_roots["figure_root"]
        or c0_root != expected_roots["c0_root"]
    ):
        raise FigureAcceptanceError(
            "Acceptance roots are not the orchestrator's private candidate roots"
        )
    if not orchestration_record.is_file() or not _root_contains(
        snapshot_root, orchestration_record
    ):
        raise FigureAcceptanceError("Orchestration record is outside private staging")
    if not allowlist_path.is_file() or not _root_contains(
        snapshot_root, allowlist_path
    ):
        raise FigureAcceptanceError("Approved allowlist is outside private staging")
    if not isinstance(release_anchor.get("source_commit"), str):
        raise FigureAcceptanceError("Orchestration state lacks a release source commit")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", generator_source_commit or ""):
        raise FigureAcceptanceError(
            "Orchestration state lacks an approved generator commit"
        )
    return _AcceptanceState(
        figure_root=figure_root,
        c0_root=c0_root,
        snapshot_root=snapshot_root,
        release_anchor=dict(release_anchor),
        generator_source_commit=generator_source_commit,
        orchestration_record=orchestration_record,
        orchestration_record_sha256=file_sha256(orchestration_record),
        allowlist_path=allowlist_path,
        allowlist_sha256=file_sha256(allowlist_path),
        candidate_records=tuple(dict(record) for record in candidate_records),
    )


def _candidate_provenance_from_state(
    specs: Sequence[CandidateSpec],
    roots: Mapping[str, Path],
    state: _AcceptanceState,
) -> Dict[str, ProvenanceEvidence]:
    expected = {spec.candidate_id: spec for spec in specs}
    if len(state.candidate_records) != len(expected):
        raise FigureAcceptanceError("Orchestrated candidate set is incomplete")
    evidence: Dict[str, ProvenanceEvidence] = {}
    for raw in state.candidate_records:
        candidate_id = raw.get("candidate_id")
        if candidate_id not in expected or candidate_id in evidence:
            raise FigureAcceptanceError(
                "Orchestrated candidate is unknown or duplicated"
            )
        spec = expected[candidate_id]
        expected_path = _candidate_pdf_logical_path(spec).as_posix()
        if raw.get("path") != expected_path:
            raise FigureAcceptanceError(
                f"Orchestrated candidate path mismatch for {candidate_id}"
            )
        candidate_path = roots[spec.root] / spec.pdf
        if raw.get("sha256") != _sha256(candidate_path):
            raise FigureAcceptanceError(
                f"Orchestrated candidate checksum mismatch for {candidate_id}"
            )
        generator = raw.get("generator")
        if not isinstance(generator, str) or not generator:
            raise FigureAcceptanceError(
                f"Orchestrated candidate lacks generator for {candidate_id}"
            )
        evidence[candidate_id] = ProvenanceEvidence(
            manifest_path=state.orchestration_record,
            manifest_sha256=state.orchestration_record_sha256,
            generator=generator,
            generator_source_commit=state.generator_source_commit,
        )
    if set(evidence) != set(expected):
        raise FigureAcceptanceError(
            "Orchestration does not cover exactly 38 candidates"
        )
    return evidence


def _accept_candidates(
    *,
    state: _AcceptanceState,
    output_dir: Path,
    runtime: TrustedFigureRuntime,
) -> AcceptanceOutputs:
    roots = {
        "figure_root": Path(state.figure_root).resolve(),
        "c0_root": Path(state.c0_root).resolve(),
    }
    output_dir = Path(output_dir).resolve()
    _validate_invocation(roots, output_dir)
    allowlist_path = state.allowlist_path
    if file_sha256(allowlist_path) != state.allowlist_sha256:
        raise FigureAcceptanceError(
            "Private approved allowlist mutated before acceptance"
        )
    specs = load_candidate_allowlist(allowlist_path)
    verify_candidate_inventory(specs, roots)
    provenance = _candidate_provenance_from_state(specs, roots, state)
    anchor = dict(state.release_anchor)
    source_commit = anchor["source_commit"]
    snapshot_root = Path(state.snapshot_root).resolve()

    candidate_audits = []
    errors: List[str] = []
    for order, spec in enumerate(specs, start=1):
        pdf_path = roots[spec.root] / spec.pdf
        try:
            report = inspect_pdf(pdf_path, require_fonts=True, runtime=runtime)
        except PdfQaError as exc:
            errors.append(f"PDF QA could not inspect {pdf_path}: {exc}")
            continue
        if not report.passed:
            errors.append(f"PDF QA failed for {pdf_path}: " + "; ".join(report.issues))
            continue
        try:
            page_info = pdf_page_info(pdf_path, runtime=runtime)
        except FigureAcceptanceError as exc:
            errors.append(str(exc))
            continue
        if page_info.page_count != 1:
            errors.append(
                f"Candidate PDF must contain exactly one page: {pdf_path} has "
                f"{page_info.page_count}"
            )
            continue
        candidate_audits.append(
            (order, spec, pdf_path, _sha256(pdf_path), page_info, report)
        )
    if errors:
        raise FigureAcceptanceError("\n".join(errors))
    if len(candidate_audits) != EXPECTED_COUNTS["candidate_pdfs"]:
        raise FigureAcceptanceError(
            f"Internal error: audited {len(candidate_audits)} of 38 candidates"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=str(output_dir.parent)
        )
    )
    try:
        preview_dir = staging_dir / "previews"
        preview_dir.mkdir()
        records: List[AcceptedCandidate] = []
        actual_preview_paths = {}
        for order, spec, pdf_path, digest, page_info, report in candidate_audits:
            staged_preview = preview_dir / f"{spec.candidate_id}.png"
            render_pdf_preview(
                pdf_path, staged_preview, dpi=300, page=1, runtime=runtime
            )
            logical_preview = _candidate_preview_logical_path(spec)
            preview = inspect_generated_preview(
                staged_preview,
                page_info,
                required_dpi=300.0,
                logical_path=logical_preview,
            )
            actual_preview_paths[spec.candidate_id] = staged_preview
            records.append(
                AcceptedCandidate(
                    order=order,
                    spec=spec,
                    pdf_path=pdf_path,
                    pdf_sha256=digest,
                    pdf_page_count=1,
                    pdf_width_points=page_info.width_points,
                    pdf_height_points=page_info.height_points,
                    preview=preview,
                    pdf_qa=report,
                    provenance=provenance[spec.candidate_id],
                )
            )

        index_pages = max(1, math.ceil(len(records) / INDEX_ROWS_PER_PAGE))
        numbered_records = [
            replace(
                record,
                review_page_start=index_pages + record.order,
                review_page_end=index_pages + record.order,
            )
            for record in records
        ]
        staged_review_pdf = staging_dir / REVIEW_PDF
        build_vector_review_pdf(
            numbered_records,
            staged_review_pdf,
            source_commit=source_commit,
            runtime=runtime,
        )
        review_report = inspect_pdf(
            staged_review_pdf, require_fonts=True, runtime=runtime
        )
        if not review_report.passed:
            raise FigureAcceptanceError(
                "Review PDF QA failed: " + "; ".join(review_report.issues)
            )
        review_info = pdf_page_info(staged_review_pdf, runtime=runtime)
        expected_review_pages = index_pages + len(numbered_records)
        if review_info.page_count != expected_review_pages:
            raise FigureAcceptanceError(
                f"Merged review page count is {review_info.page_count}; expected "
                f"{expected_review_pages}"
            )
        verify_review_page_map(
            numbered_records,
            staged_review_pdf,
            actual_preview_paths,
            runtime=runtime,
        )

        staged_qa = staging_dir / VECTOR_QA_JSON
        logical_qa = Path("review") / VECTOR_QA_JSON
        qa_payload = {
            "schema_version": 2,
            "passed": True,
            "candidate_pdf_count": len(numbered_records),
            "candidate_reports": [
                _qa_dict(
                    record.pdf_qa,
                    logical_path=_candidate_pdf_logical_path(record.spec),
                )
                for record in numbered_records
            ],
            "review_report": _qa_dict(
                review_report, logical_path=Path("review") / REVIEW_PDF
            ),
            "measured_review_page_count": review_info.page_count,
            "review_page_map_verified": True,
        }
        staged_qa.write_text(json.dumps(qa_payload, indent=2) + "\n", encoding="utf-8")

        staged_source_map_json = staging_dir / SOURCE_MAP_JSON
        staged_source_map_csv = staging_dir / SOURCE_MAP_CSV
        source_payload = {
            "schema_version": 2,
            "passed": True,
            "source_commit": source_commit,
            "release_source_commit": source_commit,
            "release": anchor,
            "allowlist": {
                "path": allowlist_path.relative_to(snapshot_root).as_posix(),
                "sha256": _sha256(Path(allowlist_path).resolve()),
                "expected_counts": EXPECTED_COUNTS,
            },
            "roots": {
                key: path.relative_to(snapshot_root).as_posix()
                for key, path in roots.items()
            },
            "review": {
                "path": f"review/{REVIEW_PDF}",
                "sha256": _sha256(staged_review_pdf),
                "index_pages": index_pages,
                "page_count": review_info.page_count,
                "page_map_verified": True,
            },
            "vector_qa": {
                "path": str(logical_qa),
                "sha256": _sha256(staged_qa),
            },
            "candidates": [
                {
                    "order": record.order,
                    "candidate_id": record.spec.candidate_id,
                    "slot_id": record.spec.slot_id,
                    "section": record.spec.section,
                    "title": record.spec.title,
                    "variant": record.spec.variant,
                    "root": record.spec.root,
                    "pdf_path": _candidate_pdf_logical_path(record.spec).as_posix(),
                    "pdf_sha256": record.pdf_sha256,
                    "pdf_page_count": record.pdf_page_count,
                    "pdf_width_points": record.pdf_width_points,
                    "pdf_height_points": record.pdf_height_points,
                    "png_path": record.preview.path.as_posix(),
                    "png_sha256": record.preview.sha256,
                    "png_width_px": record.preview.width_px,
                    "png_height_px": record.preview.height_px,
                    "png_dpi_x": record.preview.dpi_x,
                    "png_dpi_y": record.preview.dpi_y,
                    "provenance_manifest": record.provenance.manifest_path.relative_to(
                        snapshot_root
                    ).as_posix(),
                    "provenance_manifest_sha256": (record.provenance.manifest_sha256),
                    "provenance_generator": record.provenance.generator,
                    "generator_source_commit": (
                        record.provenance.generator_source_commit
                    ),
                    "review_page_start": record.review_page_start,
                    "review_page_end": record.review_page_end,
                }
                for record in numbered_records
            ],
        }
        staged_source_map_json.write_text(
            json.dumps(source_payload, indent=2) + "\n", encoding="utf-8"
        )
        _write_csv(
            staged_source_map_csv,
            numbered_records,
            source_commit,
            snapshot_root,
        )

        expected_files = {
            REVIEW_PDF,
            SOURCE_MAP_JSON,
            SOURCE_MAP_CSV,
            VECTOR_QA_JSON,
            *{
                f"previews/{record.spec.candidate_id}.png"
                for record in numbered_records
            },
        }
        actual_files = {
            path.relative_to(staging_dir).as_posix()
            for path in staging_dir.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise FigureAcceptanceError(
                "Staged acceptance artifact inventory is incomplete or contaminated"
            )
        os.replace(staging_dir, output_dir)
    except Exception as exc:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if isinstance(exc, FigureAcceptanceError):
            raise
        if isinstance(exc, PdfQaError):
            raise FigureAcceptanceError(str(exc)) from exc
        raise FigureAcceptanceError(f"Acceptance staging failed: {exc}") from exc

    return AcceptanceOutputs(
        review_pdf=output_dir / REVIEW_PDF,
        source_map_json=output_dir / SOURCE_MAP_JSON,
        source_map_csv=output_dir / SOURCE_MAP_CSV,
        vector_qa_json=output_dir / VECTOR_QA_JSON,
    )


@dataclass
class PublicationReservation:
    path: Path
    device: int
    inode: int
    descriptor: int
    released: bool = False


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


def run_command(
    command: Sequence[str], cwd: Path, env: Mapping[str, str], log_path: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=cwd, env=dict(env), stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode != 0:
        try:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            log_tail = "<generator log unavailable>"
        raise FinalFigureOrchestrationError(
            f"Generator failed ({result.returncode}); {log_path}\n{log_tail}"
        )


def _complete_publication_transaction(
    *,
    staging: Path,
    output_root: Path,
    reservation: PublicationReservation,
    manifest_path: Path,
    acceptance_state: _AcceptanceState,
    candidate_specs: Sequence[CandidateSpec],
    runtime: TrustedFigureRuntime,
) -> None:
    """Accept, freeze, rehash, and publish while holding one reservation."""

    publish_tree = None
    try:
        _accept_candidates(
            state=acceptance_state,
            output_dir=staging / "review",
            runtime=runtime,
        )
        manifest_digest = file_sha256(manifest_path)
        accepted_records = [
            snapshot_record(path, staging, "accepted_artifact")
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
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
        _verify_reservation(reservation)
        _require(
            not os.path.lexists(output_root),
            f"Publication destination appeared before publish: {output_root}",
        )
        _verify_frozen_publication_tree(publish_tree, ledger_sha256=ledger_sha256)
        _rename_directory_noreplace(publish_tree, output_root)
        publish_tree = None
    finally:
        if staging.exists():
            _remove_tree(staging)
        if publish_tree is not None and publish_tree.exists():
            _remove_tree(publish_tree)
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
) -> Path:
    """Run the sole attested generation, acceptance, and publication path."""

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
    live_release_pin = _capture_release_audit_pin(release_root)
    attestation = verify_generator_checkout(
        repository, approved_generator_commit, live_release_pin.source_commit
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
        verify_materialized_source(
            repository, approved_generator_commit, immutable_source, attestation
        )
        config_authority = _seal_execution_config(
            immutable_source / "config", execution / "config_authority.json"
        )
        _verify_execution_config(config_authority)

        def run_approved_command(
            command: Sequence[str],
            cwd: Path,
            command_environment: Mapping[str, str],
            log_path: Path,
        ) -> None:
            verify_materialized_source(
                repository, approved_generator_commit, immutable_source, attestation
            )
            _verify_execution_config(config_authority)
            try:
                run_command(command, cwd, command_environment, log_path)
            finally:
                _verify_execution_config(config_authority)
                verify_materialized_source(
                    repository, approved_generator_commit, immutable_source, attestation
                )

        figure_root = staging / "candidates" / "figure_root"
        c0_root = staging / "candidates" / "c0_root"
        _require(
            not figure_root.exists() and not c0_root.exists(),
            "Candidate roots must start nonexistent",
        )
        plots_root = execution / "plots"
        plots_root.mkdir()
        complete_release = _snapshot_complete_release(
            release_root,
            execution / "audited_release",
            live_pin=live_release_pin,
        )
        release_audit_pin = validate_final_release_contract(complete_release.root)
        _require(
            release_audit_pin.sha256sums_bytes == live_release_pin.sha256sums_bytes
            and release_audit_pin.source_commit == live_release_pin.source_commit
            and release_audit_pin.resolved_config_sha256
            == live_release_pin.resolved_config_sha256,
            "Immutable release audit authority differs from the pinned live release",
        )
        release_snapshot = _snapshot_release_inputs(
            complete_release.root,
            staging / "provenance" / "release_input_snapshot",
            audit_pin=release_audit_pin,
            staging_root=staging,
        )
        anchor = release_snapshot.anchor
        release_view = release_snapshot.plots_root
        release_aliases = release_snapshot.alias_sources
        release_sha256sums_sha256 = file_sha256(release_snapshot.root / "SHA256SUMS")
        _require(
            release_sha256sums_sha256 == release_audit_pin.sha256sums_sha256
            and anchor["source_commit"] == release_audit_pin.source_commit,
            "Compact release input view differs from the immutable release audit",
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
        env.update(
            {
                "INTERFACE_CONFIG_ROOT": str(config_authority.config_root),
                "INTERFACE_CONFIG_AUTHORITY": str(config_authority.manifest_path),
                "INTERFACE_CONFIG_AUTHORITY_SHA256": (config_authority.manifest_sha256),
            }
        )
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
        config_authority_snapshot = _copy_manifest(
            config_authority.manifest_path,
            staging,
            "provenance/execution_config_authority.json",
            "execution_config_authority",
            snapshot_artifacts,
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
        run_approved_command(main_cmd, execution, env, logs / "maintext.log")
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
        run_approved_command(all_cmd, execution, env, logs / "all_methods.log")
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

        # Run the cheap deterministic generators before any companion sweep so
        # runtime/font incompatibilities fail before the expensive studies.
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
        run_approved_command(plic_cmd, execution, env, logs / "deterministic_plic.log")
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
        run_approved_command(
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
            run_approved_command(
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
            run_approved_command(
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
            "execution_config_authority": 1,
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
            "execution_config_authority": {
                "path": config_authority_snapshot.relative_to(staging).as_posix(),
                "sha256": config_authority.manifest_sha256,
                "file_count": config_authority.file_count,
                "source": "approved_materialized_generator_commit",
                "verification": "per_yaml_read_and_before_after_generator",
            },
            "external_approval": {
                key: value for key, value in approval.to_dict().items() if key != "path"
            }
            | {"snapshot_path": "provenance/external_approval_record.json"},
            "scientific_release": anchor,
            "audited_release_authority": {
                "source_commit": release_audit_pin.source_commit,
                "sha256sums_sha256": release_audit_pin.sha256sums_sha256,
                "resolved_config_sha256": release_audit_pin.resolved_config_sha256,
                "audit_root": "private_complete_release_snapshot",
                "checksum_verification": "complete_inventory_passed",
                "snapshotted_file_count": complete_release.file_count,
                "snapshotted_size_bytes": complete_release.total_size_bytes,
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
        acceptance_state = _create_acceptance_state(
            figure_root=figure_root,
            c0_root=c0_root,
            snapshot_root=staging,
            release_anchor=anchor,
            generator_source_commit=approved_generator_commit,
            orchestration_record=manifest_path,
            allowlist_path=private_allowlist,
            candidate_records=orchestration["candidates"],
        )
        _complete_publication_transaction(
            staging=staging,
            output_root=output_root,
            reservation=reservation,
            manifest_path=manifest_path,
            acceptance_state=acceptance_state,
            candidate_specs=specs,
            runtime=runtime,
        )
        staging = None
    except Exception as exc:
        if staging is not None and staging.exists():
            _remove_tree(staging)
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
        _release_reservation(reservation)
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
