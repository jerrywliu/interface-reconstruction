#!/usr/bin/env python3
"""Build a deterministic, fail-closed submission package from audited inputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, Sequence

from submission.audit_final_release import (
    AuditReport,
    audit_final_release,
    verify_sha256_manifest,
)
from submission.pdf_vector_qa import PdfQaReport, inspect_pdf


PACKAGE_SCHEMA_VERSION = 1
RELEASE_SHA256_MANIFEST = "SHA256SUMS"

RELEASE_PAYLOADS = (
    (
        "submission_config.resolved.json",
        "provenance/release/submission_config.resolved.json",
        "release_configuration",
    ),
    ("sweep_manifest.json", "provenance/release/sweep_manifest.json", "release_manifest"),
    ("environment.json", "provenance/release/environment.json", "environment_manifest"),
    ("failures.csv", "provenance/release/failures.csv", "failure_ledger"),
    ("perturbed_sweep.csv", "results/perturbed_sweep.csv", "aggregate_results"),
    ("diagnostics/source_state.json", "provenance/release/source_state.json", "source_manifest"),
    ("diagnostics/run_inventory.csv", "provenance/release/run_inventory.csv", "run_inventory"),
    ("diagnostics/run_manifests.jsonl", "provenance/release/run_manifests.jsonl", "run_manifests"),
    (RELEASE_SHA256_MANIFEST, "provenance/release/SHA256SUMS", "full_release_checksums"),
    ("diagnostics/source_snapshot.tar.gz", "code/source_snapshot.tar.gz", "code_archive"),
)

PAPER_SOURCE_SUFFIXES = {
    ".bbx",
    ".bib",
    ".bst",
    ".cbx",
    ".cfg",
    ".clo",
    ".cls",
    ".def",
    ".lbx",
    ".lua",
    ".md",
    ".sty",
    ".tex",
    ".txt",
}
PAPER_SOURCE_NAMES = {
    ".latexmkrc",
    "latexmkrc",
    "Makefile",
}
PAPER_SOURCE_PREFIXES = ("README", "LICENSE", "COPYING")
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "build",
    "dist",
    "output",
}
REVIEW_BUNDLE_SUFFIXES = {".csv", ".json", ".md", ".pdf", ".txt"}
DEPOSITION_PATTERN = re.compile(r"^(?:https?://\S+|doi:\s*10\.\S+)$", re.IGNORECASE)


class SubmissionPackagingError(RuntimeError):
    """Raised when a submission package cannot be created safely."""


@dataclass(frozen=True)
class ApprovedFigure:
    paper_path: str
    source_path: Path
    sha256: str
    approval_reference: str


@dataclass(frozen=True)
class PlannedFile:
    destination: str
    source: Path
    role: str


@dataclass(frozen=True)
class InventoryEntry:
    path: str
    role: str
    source: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PackagePlan:
    release_root: Path
    paper_source_root: Path
    output_dir: Path
    files: tuple[PlannedFile, ...]
    approved_figures: tuple[ApprovedFigure, ...]
    figure_qa: tuple[PdfQaReport, ...]
    excluded_paper_files: tuple[str, ...]
    audit_summary: Mapping[str, int | str]
    raw_data_deposition: str
    review_bundle: Path | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str, label: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or ".." in pure.parts
        or "\n" in value
        or "\r" in value
    ):
        raise SubmissionPackagingError(f"{label} is not a safe relative path: {value!r}")
    normalized = pure.as_posix()
    if normalized in {".", ""}:
        raise SubmissionPackagingError(f"{label} is empty")
    return normalized


def _require_regular_file(path: Path, label: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_symlink():
        raise SubmissionPackagingError(f"{label} cannot be a symbolic link: {raw_path}")
    path = raw_path.resolve()
    if not path.is_file():
        raise SubmissionPackagingError(f"{label} is not a file: {path}")
    return path


def _validate_deposition(value: str) -> str:
    value = value.strip()
    if not DEPOSITION_PATTERN.match(value):
        raise SubmissionPackagingError(
            "raw-data deposition must be an http(s) URL or a 'doi:10....' identifier"
        )
    if any(token in value.lower() for token in ("pending", "placeholder", "example.com")):
        raise SubmissionPackagingError("raw-data deposition contains a placeholder")
    return value


def _paper_source_is_allowed(path: Path) -> bool:
    return (
        path.suffix.lower() in PAPER_SOURCE_SUFFIXES
        or path.name in PAPER_SOURCE_NAMES
        or path.name.startswith(PAPER_SOURCE_PREFIXES)
    )


def discover_paper_source_files(root: Path) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Return allowlisted manuscript source and a record of everything excluded."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise SubmissionPackagingError(f"paper source root is not a directory: {root}")

    included: list[Path] = []
    excluded: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts[:-1]):
            continue
        if path.is_symlink():
            raise SubmissionPackagingError(
                f"paper source contains a symbolic link: {relative.as_posix()}"
            )
        if not path.is_file():
            continue
        relative_text = relative.as_posix()
        if _paper_source_is_allowed(path):
            included.append(path)
        else:
            excluded.append(relative_text)
    if not included:
        raise SubmissionPackagingError("paper source root contains no allowlisted source files")
    if not any(path.suffix.lower() == ".tex" for path in included):
        raise SubmissionPackagingError("paper source root contains no TeX source")
    return tuple(included), tuple(excluded)


def discover_imported_graphics(
    paper_source_root: Path, paper_source_files: Sequence[Path]
) -> tuple[str, ...]:
    """Return uncommented ``includegraphics`` targets from manuscript TeX files."""
    pattern = re.compile(
        r"\\includegraphics(?:\s*\[[^\]]*\])?\s*\{([^}]+)\}",
        re.MULTILINE,
    )
    targets: set[str] = set()
    for path in paper_source_files:
        if path.suffix.lower() != ".tex":
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise SubmissionPackagingError(f"could not read TeX source {path}: {exc}") from exc
        uncommented = []
        for line in lines:
            pieces = re.split(r"(?<!\\)%", line, maxsplit=1)
            uncommented.append(pieces[0])
        for match in pattern.finditer("\n".join(uncommented)):
            raw_target = match.group(1).strip()
            target = _safe_relative_path(raw_target, "includegraphics target")
            suffix = PurePosixPath(target).suffix.lower()
            if not suffix:
                target += ".pdf"
                suffix = ".pdf"
            if suffix != ".pdf":
                raise SubmissionPackagingError(
                    f"manuscript imports a non-PDF graphic: {raw_target}"
                )
            targets.add(target)
    return tuple(sorted(targets))


def load_approved_figures(
    manifest_path: Path, paper_source_root: Path
) -> tuple[ApprovedFigure, ...]:
    """Load checksum-pinned, explicitly approved vector figures."""
    manifest_path = _require_regular_file(manifest_path, "approved-figures manifest")
    paper_source_root = Path(paper_source_root).resolve()
    try:
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or ())
            required = {"paper_path", "sha256", "approval_status", "approval_reference"}
            missing = sorted(required - fieldnames)
            if missing:
                raise SubmissionPackagingError(
                    "approved-figures manifest is missing columns: " + ", ".join(missing)
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise SubmissionPackagingError(
            f"could not read approved-figures manifest: {exc}"
        ) from exc

    if not rows:
        raise SubmissionPackagingError("approved-figures manifest is empty")
    figures: list[ApprovedFigure] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        if str(row.get("approval_status", "")).strip().lower() != "approved":
            raise SubmissionPackagingError(
                f"figure row {row_number} is not explicitly approved"
            )
        paper_path = _safe_relative_path(
            str(row.get("paper_path", "")).strip(), f"figure row {row_number} paper_path"
        )
        if not paper_path.lower().endswith(".pdf"):
            raise SubmissionPackagingError(
                f"approved figure must be a PDF: {paper_path}"
            )
        if paper_path in seen:
            raise SubmissionPackagingError(f"duplicate approved figure: {paper_path}")
        seen.add(paper_path)

        source_value = str(row.get("source_path", "")).strip() or paper_path
        source_relative = _safe_relative_path(
            source_value, f"figure row {row_number} source_path"
        )
        source_path = paper_source_root.joinpath(*PurePosixPath(source_relative).parts)
        try:
            source_path.resolve().relative_to(paper_source_root)
        except ValueError as exc:
            raise SubmissionPackagingError(
                f"approved figure escapes the paper source root: {source_value}"
            ) from exc
        source_path = _require_regular_file(source_path, f"approved figure {paper_path}")

        expected_digest = str(row.get("sha256", "")).strip().lower()
        if len(expected_digest) != 64 or any(
            character not in "0123456789abcdef" for character in expected_digest
        ):
            raise SubmissionPackagingError(
                f"figure row {row_number} has an invalid SHA-256 digest"
            )
        actual_digest = _sha256(source_path)
        if actual_digest != expected_digest:
            raise SubmissionPackagingError(
                f"approved figure checksum mismatch: {paper_path}"
            )
        approval_reference = str(row.get("approval_reference", "")).strip()
        if not approval_reference:
            raise SubmissionPackagingError(
                f"figure row {row_number} has no approval reference"
            )
        figures.append(
            ApprovedFigure(
                paper_path=paper_path,
                source_path=source_path,
                sha256=expected_digest,
                approval_reference=approval_reference,
            )
        )
    return tuple(sorted(figures, key=lambda item: item.paper_path))


def _review_bundle_files(path: Path | None) -> tuple[tuple[Path, str], ...]:
    if path is None:
        return ()
    raw_path = Path(path)
    if raw_path.is_symlink():
        raise SubmissionPackagingError(
            f"review bundle cannot be a symbolic link: {raw_path}"
        )
    path = raw_path.resolve()
    if path.is_file():
        if path.suffix.lower() not in REVIEW_BUNDLE_SUFFIXES:
            raise SubmissionPackagingError(f"unsupported review-bundle file: {path}")
        return ((path, path.name),)
    if not path.is_dir():
        raise SubmissionPackagingError(f"review bundle does not exist: {path}")

    files: list[tuple[Path, str]] = []
    for candidate in sorted(path.rglob("*")):
        relative = candidate.relative_to(path)
        if candidate.is_symlink():
            raise SubmissionPackagingError(
                f"review bundle contains a symbolic link: {relative.as_posix()}"
            )
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in REVIEW_BUNDLE_SUFFIXES:
            raise SubmissionPackagingError(
                f"unsupported review-bundle file: {relative.as_posix()}"
            )
        files.append((candidate, relative.as_posix()))
    if not files:
        raise SubmissionPackagingError("review bundle contains no supported files")
    return tuple(files)


def _extract_experiment_map(source_snapshot: Path, destination: Path) -> None:
    member_name = "docs/PAPER_EXPERIMENT_MAP.md"
    try:
        with tarfile.open(source_snapshot, "r:gz") as archive:
            members = [member for member in archive.getmembers() if member.name == member_name]
            if len(members) != 1 or not members[0].isfile():
                raise SubmissionPackagingError(
                    f"audited source snapshot must contain exactly one {member_name}"
                )
            stream = archive.extractfile(members[0])
            if stream is None:
                raise SubmissionPackagingError(
                    f"could not read {member_name} from audited source snapshot"
                )
            data = stream.read()
    except (OSError, tarfile.TarError) as exc:
        raise SubmissionPackagingError(f"could not inspect source snapshot: {exc}") from exc
    _write_bytes(destination, data)


def _audit_release_or_fail(
    release_root: Path,
    audit_runner: Callable[[Path], AuditReport],
    checksum_verifier: Callable[[Path, str], list[str]],
) -> AuditReport:
    report = audit_runner(release_root)
    if not report.ok:
        details = "; ".join(report.errors[:5])
        suffix = f": {details}" if details else ""
        raise SubmissionPackagingError(
            f"final release audit failed with {report.total_errors} error(s){suffix}"
        )
    manifest = release_root / RELEASE_SHA256_MANIFEST
    if not manifest.is_file():
        raise SubmissionPackagingError(
            f"audited release lacks {RELEASE_SHA256_MANIFEST}; generate and verify it first"
        )
    errors = checksum_verifier(release_root, RELEASE_SHA256_MANIFEST)
    if errors:
        raise SubmissionPackagingError(
            "full-release checksum verification failed: " + "; ".join(errors[:5])
        )
    return report


def plan_submission_package(
    *,
    release_root: Path,
    paper_source_root: Path,
    approved_figures_manifest: Path,
    raw_data_deposition: str,
    output_dir: Path,
    review_bundle: Path | None = None,
    audit_runner: Callable[[Path], AuditReport] = audit_final_release,
    checksum_verifier: Callable[[Path, str], list[str]] = verify_sha256_manifest,
    pdf_inspector: Callable[..., PdfQaReport] = inspect_pdf,
) -> PackagePlan:
    """Validate every input and return the exact package plan without writing it."""
    release_root = Path(release_root).resolve()
    paper_source_root = Path(paper_source_root).resolve()
    output_dir = Path(output_dir).resolve()
    if not release_root.is_dir():
        raise SubmissionPackagingError(f"release root is not a directory: {release_root}")
    deposition = _validate_deposition(raw_data_deposition)
    archive_path = output_dir.with_suffix(output_dir.suffix + ".tar.gz")
    if output_dir.exists():
        raise SubmissionPackagingError(f"output directory already exists: {output_dir}")
    if archive_path.exists():
        raise SubmissionPackagingError(f"output archive already exists: {archive_path}")
    for protected_root, label in (
        (release_root, "release root"),
        (paper_source_root, "paper source root"),
    ):
        try:
            output_dir.relative_to(protected_root)
        except ValueError:
            pass
        else:
            raise SubmissionPackagingError(
                f"output directory cannot be inside the {label}: {output_dir}"
            )
    audit_report = _audit_release_or_fail(
        release_root, audit_runner, checksum_verifier
    )

    paper_files, excluded = discover_paper_source_files(paper_source_root)
    figures = load_approved_figures(approved_figures_manifest, paper_source_root)
    approved_source_paths = {
        figure.source_path.relative_to(paper_source_root).as_posix()
        for figure in figures
    }
    excluded = tuple(
        relative for relative in excluded if relative not in approved_source_paths
    )
    approved_paths = {figure.paper_path for figure in figures}
    imported_graphics = set(
        discover_imported_graphics(paper_source_root, paper_files)
    )
    missing_approvals = sorted(imported_graphics - approved_paths)
    if missing_approvals:
        raise SubmissionPackagingError(
            "manuscript imports graphics absent from the approved-figures manifest: "
            + ", ".join(missing_approvals)
        )
    figure_qa: list[PdfQaReport] = []
    for figure in figures:
        try:
            report = pdf_inspector(figure.source_path, require_fonts=False)
        except Exception as exc:
            raise SubmissionPackagingError(
                f"could not inspect approved figure {figure.paper_path}: {exc}"
            ) from exc
        if not report.passed:
            raise SubmissionPackagingError(
                f"approved figure is not submission-grade vector PDF: "
                f"{figure.paper_path}: {', '.join(report.issues)}"
            )
        figure_qa.append(report)

    planned: list[PlannedFile] = []
    destinations: set[str] = set()

    def add(destination: str, source: Path, role: str) -> None:
        destination = _safe_relative_path(destination, "package destination")
        if destination in destinations:
            raise SubmissionPackagingError(f"duplicate package destination: {destination}")
        destinations.add(destination)
        planned.append(PlannedFile(destination, _require_regular_file(source, role), role))

    for source_relative, destination, role in RELEASE_PAYLOADS:
        add(destination, release_root / source_relative, role)
    for source in paper_files:
        relative = source.relative_to(paper_source_root).as_posix()
        add(f"manuscript/source/{relative}", source, "manuscript_source")
    for figure in figures:
        add(
            f"manuscript/source/{figure.paper_path}",
            figure.source_path,
            "approved_vector_figure",
        )
    for source, relative in _review_bundle_files(review_bundle):
        add(f"manuscript/review/{relative}", source, "review_bundle")

    return PackagePlan(
        release_root=release_root,
        paper_source_root=paper_source_root,
        output_dir=output_dir,
        files=tuple(sorted(planned, key=lambda item: item.destination)),
        approved_figures=figures,
        figure_qa=tuple(figure_qa),
        excluded_paper_files=excluded,
        audit_summary=dict(audit_report.summaries),
        raw_data_deposition=deposition,
        review_bundle=Path(review_bundle).resolve() if review_bundle else None,
    )


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o644)
    os.utime(path, (0, 0), follow_symlinks=False)


def _write_text(path: Path, text: str) -> None:
    _write_bytes(path, text.encode("utf-8"))


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream:
        with destination.open("wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
    destination.chmod(0o644)
    os.utime(destination, (0, 0), follow_symlinks=False)


def _source_label(plan: PackagePlan, source: Path) -> str:
    for root, prefix in (
        (plan.release_root, "release"),
        (plan.paper_source_root, "paper"),
    ):
        try:
            relative = source.relative_to(root)
        except ValueError:
            continue
        return f"{prefix}:{relative.as_posix()}"
    if plan.review_bundle is not None:
        review_root = (
            plan.review_bundle
            if plan.review_bundle.is_dir()
            else plan.review_bundle.parent
        )
        try:
            relative = source.relative_to(review_root)
        except ValueError:
            pass
        else:
            return f"review:{relative.as_posix()}"
    return source.name


def _inventory_entry(
    root: Path, planned: PlannedFile, plan: PackagePlan
) -> InventoryEntry:
    path = root.joinpath(*PurePosixPath(planned.destination).parts)
    return InventoryEntry(
        path=planned.destination,
        role=planned.role,
        source=_source_label(plan, planned.source),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _generated_inventory_entry(root: Path, relative: str, role: str) -> InventoryEntry:
    path = root.joinpath(*PurePosixPath(relative).parts)
    return InventoryEntry(
        path=relative,
        role=role,
        source="generated:package_submission.py",
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _write_inventory(
    root: Path,
    plan: PackagePlan,
    entries: Sequence[InventoryEntry],
) -> None:
    payload = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "release": {
            "name": plan.release_root.name,
            "audit_passed": True,
            "audit_summary": dict(sorted(plan.audit_summary.items())),
            "full_release_sha256_manifest": "provenance/release/SHA256SUMS",
        },
        "raw_data": {
            "included": False,
            "deposition": plan.raw_data_deposition,
            "excluded_paths": [
                "raw_runs/",
                "diagnostics/case_*",
                "diagnostics/cell_metrics.csv",
                "diagnostics/merge_events.csv",
            ],
        },
        "approved_figures": [
            {
                "paper_path": figure.paper_path,
                "sha256": figure.sha256,
                "approval_reference": figure.approval_reference,
            }
            for figure in plan.approved_figures
        ],
        "excluded_paper_files": list(plan.excluded_paper_files),
        "files": [asdict(entry) for entry in entries],
    }
    _write_text(
        root / "INVENTORY.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )

    csv_path = root / "INVENTORY.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("path", "role", "source", "size_bytes", "sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))
    csv_path.chmod(0o644)
    os.utime(csv_path, (0, 0), follow_symlinks=False)


def _write_package_readme(root: Path, plan: PackagePlan) -> None:
    text = f"""# Submission Package

This package was assembled from the completed, programmatically audited release
`{plan.release_root.name}`. The package checksum manifest is `SHA256SUMS`.

## Contents

- `code/source_snapshot.tar.gz`: exact clean source archive recorded by the final sweep.
- `manuscript/source/`: allowlisted manuscript source plus checksum-pinned approved vector figures.
- `manuscript/review/`: optional review material supplied to the packager.
- `results/perturbed_sweep.csv`: audited aggregate result table.
- `docs/PAPER_EXPERIMENT_MAP.md`: paper-to-code and paper-to-result map from
  the audited source snapshot.
- `provenance/release/`: resolved configuration, environment, source state,
  run inventory, and full-release checksums.
- `provenance/vector_pdf_qa.json`: zero-raster and font-embedding audit for every approved figure.
- `INVENTORY.json` and `INVENTORY.csv`: machine- and human-readable payload inventories.

## Raw Scientific Data

The 970 raw run bundles and large case/cell/merge diagnostic tables are deliberately
excluded from this compact submission package. They are deposited at:

{plan.raw_data_deposition}

`provenance/release/SHA256SUMS` identifies every file in the complete deposited
release. `provenance/RAW_DATA_DEPOSITION.md` records the exclusion boundary.

## Verification

From the package root, run:

```sh
sha256sum -c SHA256SUMS
```

On macOS, `shasum -a 256 -c SHA256SUMS` provides the equivalent check.
"""
    _write_text(root / "README.md", text)


def _write_deposition_record(root: Path, plan: PackagePlan) -> None:
    text = f"""# Raw Data Deposition

- Complete release identifier: `{plan.release_root.name}`
- Deposition: {plan.raw_data_deposition}
- Raw data included in this compact package: no
- Complete release checksum manifest: `provenance/release/SHA256SUMS`

The external deposit should contain the complete audited release, including
`raw_runs/` and the case-, cell-, merge-, and fallback-indexed diagnostic tables.
The compact package retains the aggregate results and run inventory needed to map
paper results to those deposited bundles.
"""
    _write_text(root / "provenance" / "RAW_DATA_DEPOSITION.md", text)


def _write_release_audit_record(root: Path, plan: PackagePlan) -> None:
    payload = {
        "schema_version": 1,
        "passed": True,
        "release_name": plan.release_root.name,
        "summaries": dict(sorted(plan.audit_summary.items())),
        "full_release_checksums_verified": True,
    }
    _write_text(
        root / "provenance" / "release" / "audit_report.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _write_figure_approval_record(root: Path, plan: PackagePlan) -> None:
    path = root / "provenance" / "approved_figures.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("paper_path", "sha256", "approval_reference"),
            lineterminator="\n",
        )
        writer.writeheader()
        for figure in plan.approved_figures:
            writer.writerow(
                {
                    "paper_path": figure.paper_path,
                    "sha256": figure.sha256,
                    "approval_reference": figure.approval_reference,
                }
            )
    path.chmod(0o644)
    os.utime(path, (0, 0), follow_symlinks=False)


def _write_vector_qa_record(root: Path, plan: PackagePlan) -> None:
    reports = []
    for figure, report in zip(plan.approved_figures, plan.figure_qa):
        reports.append(
            {
                "paper_path": figure.paper_path,
                "passed": report.passed,
                "image_objects": report.image_objects,
                "fonts": [asdict(font) for font in report.fonts],
                "issues": list(report.issues),
            }
        )
    payload = {
        "schema_version": 1,
        "passed": all(report["passed"] for report in reports),
        "pdf_count": len(reports),
        "reports": reports,
    }
    _write_text(
        root / "provenance" / "vector_pdf_qa.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _all_files(root: Path, excluded: Iterable[Path] = ()) -> tuple[Path, ...]:
    excluded_set = {path.resolve() for path in excluded}
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SubmissionPackagingError(f"package contains a symbolic link: {path}")
        if path.is_file() and path.resolve() not in excluded_set:
            files.append(path)
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


def _write_package_checksums(root: Path) -> None:
    manifest = root / "SHA256SUMS"
    lines = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in _all_files(root, (manifest,))
    ]
    _write_text(manifest, "\n".join(lines) + "\n")


def verify_package_checksums(root: Path) -> list[str]:
    """Return checksum or coverage failures for a staged package."""
    root = Path(root).resolve()
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        return ["package SHA256SUMS is missing"]
    errors: list[str] = []
    records: dict[str, str] = {}
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [f"could not read package SHA256SUMS: {exc}"]
    for line_number, line in enumerate(lines, start=1):
        if len(line) < 67 or line[64:66] != "  ":
            errors.append(f"invalid checksum line {line_number}")
            continue
        digest, relative = line[:64], line[66:]
        try:
            relative = _safe_relative_path(relative, "checksum path")
        except SubmissionPackagingError as exc:
            errors.append(str(exc))
            continue
        if relative in records:
            errors.append(f"duplicate checksum path: {relative}")
            continue
        records[relative] = digest
    actual = {
        path.relative_to(root).as_posix(): path
        for path in _all_files(root, (manifest,))
    }
    for relative in sorted(set(actual) - set(records)):
        errors.append(f"file missing from checksum manifest: {relative}")
    for relative in sorted(set(records) - set(actual)):
        errors.append(f"checksum path missing from package: {relative}")
    for relative in sorted(set(records) & set(actual)):
        if _sha256(actual[relative]) != records[relative]:
            errors.append(f"checksum mismatch: {relative}")
    return errors


def _normalize_directories(root: Path) -> None:
    directories = (candidate for candidate in root.rglob("*") if candidate.is_dir())
    for path in sorted(directories, reverse=True):
        path.chmod(0o755)
        os.utime(path, (0, 0), follow_symlinks=False)
    root.chmod(0o755)
    os.utime(root, (0, 0), follow_symlinks=False)


def _write_deterministic_tar_gz(root: Path, archive_path: Path) -> None:
    temporary = archive_path.with_name(f".{archive_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("wb") as raw_stream:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as gzip_stream:
                with tarfile.open(
                    fileobj=gzip_stream,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    paths = (root,) + tuple(
                        sorted(
                            root.rglob("*"),
                            key=lambda path: path.relative_to(root.parent).as_posix(),
                        )
                    )
                    for path in paths:
                        if path.is_symlink():
                            raise SubmissionPackagingError(
                                f"cannot archive symbolic link: {path}"
                            )
                        relative = path.relative_to(root.parent).as_posix()
                        info = tarfile.TarInfo(relative)
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        if path.is_dir():
                            info.type = tarfile.DIRTYPE
                            info.mode = 0o755
                            archive.addfile(info)
                        elif path.is_file():
                            info.size = path.stat().st_size
                            info.mode = 0o644
                            with path.open("rb") as stream:
                                archive.addfile(info, stream)
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        temporary.replace(archive_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_submission_package(
    plan: PackagePlan, *, create_archive: bool = True
) -> tuple[Path, Path | None]:
    """Atomically materialize a validated package and optional deterministic archive."""
    output_dir = plan.output_dir
    archive_path = output_dir.with_suffix(output_dir.suffix + ".tar.gz")
    if output_dir.exists():
        raise SubmissionPackagingError(f"output directory already exists: {output_dir}")
    if create_archive and archive_path.exists():
        raise SubmissionPackagingError(f"output archive already exists: {archive_path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        for item in plan.files:
            destination = staging.joinpath(*PurePosixPath(item.destination).parts)
            _copy_file(item.source, destination)

        _extract_experiment_map(
            staging / "code" / "source_snapshot.tar.gz",
            staging / "docs" / "PAPER_EXPERIMENT_MAP.md",
        )
        _write_deposition_record(staging, plan)
        _write_release_audit_record(staging, plan)
        _write_figure_approval_record(staging, plan)
        _write_vector_qa_record(staging, plan)
        _write_package_readme(staging, plan)

        entries = tuple(
            [_inventory_entry(staging, item, plan) for item in plan.files]
            + [
                _generated_inventory_entry(
                    staging,
                    "docs/PAPER_EXPERIMENT_MAP.md",
                    "paper_experiment_map",
                ),
                _generated_inventory_entry(
                    staging,
                    "provenance/RAW_DATA_DEPOSITION.md",
                    "raw_data_deposition",
                ),
                _generated_inventory_entry(
                    staging,
                    "provenance/release/audit_report.json",
                    "release_audit_record",
                ),
                _generated_inventory_entry(
                    staging,
                    "provenance/approved_figures.csv",
                    "figure_approval_record",
                ),
                _generated_inventory_entry(
                    staging,
                    "provenance/vector_pdf_qa.json",
                    "vector_pdf_qa",
                ),
                _generated_inventory_entry(staging, "README.md", "package_readme"),
            ]
        )
        _write_inventory(staging, plan, entries)
        _write_package_checksums(staging)
        checksum_errors = verify_package_checksums(staging)
        if checksum_errors:
            raise SubmissionPackagingError(
                "staged package checksum verification failed: "
                + "; ".join(checksum_errors[:5])
            )
        _normalize_directories(staging)
        staging.replace(output_dir)
        if create_archive:
            _write_deterministic_tar_gz(output_dir, archive_path)
        return output_dir, archive_path if create_archive else None
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        archive_path.unlink(missing_ok=True)
        raise


def _plan_payload(plan: PackagePlan) -> dict:
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "dry_run": True,
        "release_name": plan.release_root.name,
        "audit_summary": dict(sorted(plan.audit_summary.items())),
        "output_dir": str(plan.output_dir),
        "raw_data_included": False,
        "raw_data_deposition": plan.raw_data_deposition,
        "approved_figure_count": len(plan.approved_figures),
        "paper_source_file_count": sum(
            item.role == "manuscript_source" for item in plan.files
        ),
        "excluded_paper_file_count": len(plan.excluded_paper_files),
        "planned_files": [item.destination for item in plan.files],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--paper-source-root", type=Path, required=True)
    parser.add_argument("--approved-figures-manifest", type=Path, required=True)
    parser.add_argument("--review-bundle", type=Path)
    parser.add_argument("--raw-data-deposition", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run all audits and print the exact plan without writing package files",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="stage the package directory without the deterministic outer tar.gz",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = plan_submission_package(
            release_root=args.release_root,
            paper_source_root=args.paper_source_root,
            approved_figures_manifest=args.approved_figures_manifest,
            review_bundle=args.review_bundle,
            raw_data_deposition=args.raw_data_deposition,
            output_dir=args.output_dir,
        )
        if args.dry_run:
            print(json.dumps(_plan_payload(plan), indent=2, sort_keys=True))
            return 0
        output_dir, archive = build_submission_package(
            plan, create_archive=not args.no_archive
        )
    except (OSError, SubmissionPackagingError) as exc:
        print(f"SUBMISSION PACKAGING FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Submission package: {output_dir}")
    if archive is not None:
        print(f"Deterministic archive: {archive}")
    print("SUBMISSION PACKAGING PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
