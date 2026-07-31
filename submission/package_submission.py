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
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submission.audit_final_release import (
    AuditReport,
    audit_final_release,
    verify_sha256_manifest,
)
from submission.pdf_vector_qa import PdfQaReport, inspect_pdf


PACKAGE_SCHEMA_VERSION = 3
RELEASE_SHA256_MANIFEST = "SHA256SUMS"
PACKAGE_ROOT_MARKERS = ("INVENTORY.json", "SHA256SUMS")
DEFAULT_PAPER_SOURCE_SUBDIR = "interface-reconstruction-paper"
DEFAULT_PAPER_ENTRYPOINT = "interface-reconstruction-paper/interface-reconstruction.tex"
DEFAULT_MANUSCRIPT_COMPILE_TIMEOUT_SECONDS = 300

RELEASE_PAYLOADS = (
    (
        "submission_config.resolved.json",
        "provenance/release/submission_config.resolved.json",
        "release_configuration",
    ),
    (
        "sweep_manifest.json",
        "provenance/release/sweep_manifest.json",
        "release_manifest",
    ),
    ("environment.json", "provenance/release/environment.json", "environment_manifest"),
    ("failures.csv", "provenance/release/failures.csv", "failure_ledger"),
    ("perturbed_sweep.csv", "results/perturbed_sweep.csv", "aggregate_results"),
    (
        "diagnostics/source_state.json",
        "provenance/release/source_state.json",
        "source_manifest",
    ),
    (
        "diagnostics/run_inventory.csv",
        "provenance/release/run_inventory.csv",
        "run_inventory",
    ),
    (
        "diagnostics/run_manifests.jsonl",
        "provenance/release/run_manifests.jsonl",
        "run_manifests",
    ),
    (
        RELEASE_SHA256_MANIFEST,
        "provenance/release/SHA256SUMS",
        "full_release_checksums",
    ),
    (
        "diagnostics/source_snapshot.tar.gz",
        "code/source_snapshot.tar.gz",
        "code_archive",
    ),
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
FULL_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_IDENTIFIER_PATTERN = re.compile(r"^sha256:([0-9a-f]{64})$")


class SubmissionPackagingError(RuntimeError):
    """Raised when a submission package cannot be created safely."""


@dataclass(frozen=True)
class ApprovedFigure:
    paper_path: str
    source_path: str
    source: "ContentSource"
    sha256: str
    approval_reference: str


@dataclass(frozen=True)
class GitTreeEntry:
    path: str
    object_id: str
    mode: str


@dataclass(frozen=True)
class ContentSource:
    kind: str
    label: str
    expected_sha256: str
    path: Path | None = None
    git_repository: Path | None = None
    git_object_id: str | None = None


@dataclass(frozen=True)
class PlannedFile:
    destination: str
    source: ContentSource
    role: str


@dataclass(frozen=True)
class InventoryEntry:
    path: str
    role: str
    source: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PaperGitState:
    worktree_root: Path
    commit: str
    source_subdir: str
    entrypoint: str
    tracked_paths: frozenset[str]
    tree_entries: tuple[GitTreeEntry, ...]


@dataclass(frozen=True)
class RawDataDeposit:
    location: str
    release_name: str
    manifest_name: str
    manifest_identifier: str
    manifest_sha256: str
    verification_status: str
    supplied_manifest_bytes_verified: bool
    network_assertion_made: bool


@dataclass(frozen=True)
class PackagePlan:
    release_root: Path
    paper_worktree_root: Path
    paper_commit: str
    paper_source_subdir: str
    paper_entrypoint: str
    paper_tracked_file_count: int
    latexmk_executable: str
    output_dir: Path
    output_parent_device: int
    output_parent_inode: int
    files: tuple[PlannedFile, ...]
    approved_figures: tuple[ApprovedFigure, ...]
    figure_qa: tuple[PdfQaReport, ...]
    excluded_paper_files: tuple[str, ...]
    audit_summary: Mapping[str, int | str]
    raw_data_deposit: RawDataDeposit
    review_bundle: Path | None


@dataclass(frozen=True)
class _OwnedPath:
    path: Path
    device: int
    inode: int

    @classmethod
    def capture(cls, path: Path) -> "_OwnedPath":
        path = Path(path)
        metadata = os.lstat(path)
        return cls(path=path, device=metadata.st_dev, inode=metadata.st_ino)

    def moved_to(self, path: Path) -> "_OwnedPath":
        return _OwnedPath(path=Path(path), device=self.device, inode=self.inode)

    def matches(self) -> bool:
        try:
            metadata = os.lstat(self.path)
        except FileNotFoundError:
            return False
        return metadata.st_dev == self.device and metadata.st_ino == self.inode

    def remove(self) -> bool:
        """Remove the path only while it still names this invocation's inode."""
        try:
            metadata = os.lstat(self.path)
        except FileNotFoundError:
            return False
        if metadata.st_dev != self.device or metadata.st_ino != self.inode:
            return False
        if stat.S_ISDIR(metadata.st_mode):
            shutil.rmtree(self.path)
        else:
            self.path.unlink()
        return True


@dataclass
class _DestinationLock:
    target: Path
    path: Path
    file_descriptor: int
    device: int
    inode: int

    def is_owned(self) -> bool:
        try:
            path_metadata = os.lstat(self.path)
            descriptor_metadata = os.fstat(self.file_descriptor)
        except (FileNotFoundError, OSError):
            return False
        return (
            path_metadata.st_dev == self.device
            and path_metadata.st_ino == self.inode
            and descriptor_metadata.st_dev == self.device
            and descriptor_metadata.st_ino == self.inode
        )

    def release(self) -> None:
        try:
            if self.is_owned():
                self.path.unlink()
        finally:
            os.close(self.file_descriptor)


class _DestinationReservations:
    """Exclusive sidecar locks for every final publication destination."""

    def __init__(self, locks: Sequence[_DestinationLock]) -> None:
        self._locks = tuple(locks)

    @classmethod
    def acquire(cls, targets: Sequence[Path]) -> "_DestinationReservations":
        owner = secrets.token_hex(16)
        locks: list[_DestinationLock] = []
        try:
            for target in sorted({Path(path) for path in targets}, key=str):
                lock_path = target.with_name(f".{target.name}.package_submission.lock")
                flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
                flags |= getattr(os, "O_CLOEXEC", 0)
                try:
                    descriptor = os.open(lock_path, flags, 0o600)
                except FileExistsError as exc:
                    raise SubmissionPackagingError(
                        "submission destination is reserved by another packaging "
                        f"invocation: {target}; inspect stale lock {lock_path}"
                    ) from exc
                metadata = os.fstat(descriptor)
                lock = _DestinationLock(
                    target=target,
                    path=lock_path,
                    file_descriptor=descriptor,
                    device=metadata.st_dev,
                    inode=metadata.st_ino,
                )
                locks.append(lock)
                payload = (
                    json.dumps(
                        {
                            "owner": owner,
                            "pid": os.getpid(),
                            "target": str(target),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(descriptor, remaining)
                    remaining = remaining[written:]
                os.fsync(descriptor)
            return cls(locks)
        except Exception:
            for lock in reversed(locks):
                lock.release()
            raise

    def assert_owned(self) -> None:
        for lock in self._locks:
            if not lock.is_owned():
                raise SubmissionPackagingError(
                    f"submission destination reservation was lost: {lock.target}"
                )

    def release(self) -> None:
        for lock in reversed(self._locks):
            lock.release()

    def __enter__(self) -> "_DestinationReservations":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative_path(value: str, label: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or ".." in pure.parts
        or "\n" in value
        or "\r" in value
    ):
        raise SubmissionPackagingError(
            f"{label} is not a safe relative path: {value!r}"
        )
    normalized = pure.as_posix()
    if normalized in {".", ""}:
        raise SubmissionPackagingError(f"{label} is empty")
    return normalized


def _resolve_output_destination(value: Path) -> Path:
    raw = Path(value).expanduser()
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    ancestors = tuple(reversed(absolute.parent.parents)) + (absolute.parent,)
    for ancestor in ancestors:
        if ancestor.is_symlink():
            raise SubmissionPackagingError(
                f"output path cannot traverse a symbolic-link directory: {ancestor}"
            )
    return absolute.resolve()


def _validate_private_output_parent(
    output_dir: Path,
    *,
    expected_device: int | None = None,
    expected_inode: int | None = None,
) -> _OwnedPath:
    parent = output_dir.parent
    try:
        metadata = os.lstat(parent)
    except FileNotFoundError as exc:
        raise SubmissionPackagingError(
            f"output parent must already exist: {parent}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise SubmissionPackagingError(
            f"output parent must be a real directory: {parent}"
        )
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise SubmissionPackagingError(
            f"output parent must be owned by the current user: {parent}"
        )
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise SubmissionPackagingError(
            f"output parent cannot be group- or other-writable: {parent}"
        )
    if expected_device is not None and expected_inode is not None:
        if metadata.st_dev != expected_device or metadata.st_ino != expected_inode:
            raise SubmissionPackagingError(
                f"output parent changed after package planning: {parent}"
            )
    return _OwnedPath(parent, metadata.st_dev, metadata.st_ino)


def _is_package_root(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        return False
    for marker in PACKAGE_ROOT_MARKERS:
        try:
            marker_metadata = os.lstat(path / marker)
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(marker_metadata.st_mode):
            return False
    return True


def _find_contained_package_root(destination: Path) -> Path | None:
    try:
        metadata = os.lstat(destination)
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        return None
    for root, directory_names, file_names in os.walk(
        destination, topdown=True, followlinks=False
    ):
        root_path = Path(root)
        directory_names[:] = [
            name for name in directory_names if not (root_path / name).is_symlink()
        ]
        if set(PACKAGE_ROOT_MARKERS).issubset(file_names) and _is_package_root(
            root_path
        ):
            return root_path
    return None


def _reject_package_destination_conflicts(destinations: Sequence[Path]) -> None:
    for destination in destinations:
        for ancestor in (destination, *destination.parents):
            if _is_package_root(ancestor):
                raise SubmissionPackagingError(
                    f"submission destination {destination} is inside existing "
                    f"package {ancestor}"
                )
        contained = _find_contained_package_root(destination)
        if contained is not None:
            raise SubmissionPackagingError(
                f"submission destination {destination} contains existing package "
                f"{contained}"
            )


def _require_regular_file(path: Path, label: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_symlink():
        raise SubmissionPackagingError(f"{label} cannot be a symbolic link: {raw_path}")
    path = raw_path.resolve()
    if not path.is_file():
        raise SubmissionPackagingError(f"{label} is not a file: {path}")
    return path


def _filesystem_source(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> ContentSource:
    path = _require_regular_file(path, label)
    digest = expected_sha256 or _sha256(path)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise SubmissionPackagingError(f"invalid expected SHA-256 for {label}")
    return ContentSource(
        kind="filesystem",
        label=label,
        expected_sha256=digest,
        path=path,
    )


def _load_release_sha256_manifest(path: Path) -> dict[str, str]:
    path = _require_regular_file(path, "complete-release checksum manifest")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SubmissionPackagingError(
            f"could not read complete-release checksum manifest: {exc}"
        ) from exc
    records: dict[str, str] = {}
    ordered_paths: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if len(line) < 67 or line[64:66] != "  ":
            raise SubmissionPackagingError(
                f"invalid complete-release checksum line {line_number}"
            )
        digest = line[:64].lower()
        relative = _safe_relative_path(
            line[66:], f"complete-release checksum path on line {line_number}"
        )
        if any(character not in "0123456789abcdef" for character in digest):
            raise SubmissionPackagingError(
                f"invalid complete-release SHA-256 on line {line_number}"
            )
        if relative in records:
            raise SubmissionPackagingError(
                f"duplicate complete-release checksum path: {relative}"
            )
        records[relative] = digest
        ordered_paths.append(relative)
    if not records:
        raise SubmissionPackagingError("complete-release checksum manifest is empty")
    if ordered_paths != sorted(ordered_paths):
        raise SubmissionPackagingError(
            "complete-release checksum manifest paths are not sorted"
        )
    return records


def _validate_deposition_location(value: str) -> str:
    value = value.strip()
    if not DEPOSITION_PATTERN.match(value):
        raise SubmissionPackagingError(
            "raw-data deposition must be an http(s) URL or a 'doi:10....' identifier"
        )
    if any(
        token in value.lower() for token in ("pending", "placeholder", "example.com")
    ):
        raise SubmissionPackagingError("raw-data deposition contains a placeholder")
    return value


def _validate_deposition(
    location: str,
    manifest_identifier: str,
    release_root: Path,
    *,
    deposited_manifest_file: Path | None,
    acknowledge_unverified_remote_deposit: bool,
) -> tuple[RawDataDeposit, ContentSource | None]:
    location = _validate_deposition_location(location)
    identifier = manifest_identifier.strip().lower()
    match = SHA256_IDENTIFIER_PATTERN.fullmatch(identifier)
    if match is None:
        raise SubmissionPackagingError(
            "raw-data manifest identifier must have the form 'sha256:<64-hex-digest>'"
        )
    manifest = _require_regular_file(
        release_root / RELEASE_SHA256_MANIFEST,
        "complete-release checksum manifest",
    )
    manifest_bytes = manifest.read_bytes()
    actual_digest = _sha256_bytes(manifest_bytes)
    expected_digest = match.group(1)
    if actual_digest != expected_digest:
        raise SubmissionPackagingError(
            "raw-data manifest identifier does not match the audited release "
            f"{RELEASE_SHA256_MANIFEST}: expected sha256:{actual_digest}"
        )
    if deposited_manifest_file is not None and acknowledge_unverified_remote_deposit:
        raise SubmissionPackagingError(
            "provide either a deposited manifest file or the explicit manual "
            "acknowledgment, not both"
        )
    evidence: ContentSource | None = None
    if deposited_manifest_file is not None:
        supplied = _require_regular_file(
            deposited_manifest_file,
            "supplied deposited release manifest",
        )
        supplied_bytes = supplied.read_bytes()
        if supplied_bytes != manifest_bytes:
            raise SubmissionPackagingError(
                "supplied deposited release manifest bytes do not match the "
                "audited release SHA256SUMS"
            )
        evidence = _filesystem_source(
            supplied,
            label="supplied deposited release manifest",
            expected_sha256=actual_digest,
        )
        verification_status = "supplied_manifest_bytes_verified"
        supplied_verified = True
    else:
        if not acknowledge_unverified_remote_deposit:
            raise SubmissionPackagingError(
                "remote deposit contents are unverified; supply "
                "--deposited-release-manifest or explicitly acknowledge the "
                "manual gate with --acknowledge-unverified-remote-deposit"
            )
        verification_status = "manual_acknowledgment_remote_contents_unverified"
        supplied_verified = False
    return (
        RawDataDeposit(
            location=location,
            release_name=release_root.name,
            manifest_name=RELEASE_SHA256_MANIFEST,
            manifest_identifier=identifier,
            manifest_sha256=actual_digest,
            verification_status=verification_status,
            supplied_manifest_bytes_verified=supplied_verified,
            network_assertion_made=False,
        ),
        evidence,
    )


def _run_git(worktree_root: Path, arguments: Sequence[str]) -> bytes:
    environment = os.environ.copy()
    for variable in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        environment.pop(variable, None)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        completed = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(worktree_root), *arguments],
            env=environment,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise SubmissionPackagingError(f"could not run git: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        if not detail:
            detail = completed.stdout.decode("utf-8", errors="replace").strip()
        raise SubmissionPackagingError(
            f"paper Git inspection failed ({' '.join(arguments)}): {detail}"
        )
    return completed.stdout


def _read_git_blob(repository: Path, object_id: str) -> bytes:
    return _run_git(repository, ("cat-file", "blob", object_id))


def _paper_entry_map(state: PaperGitState) -> dict[str, GitTreeEntry]:
    return {entry.path: entry for entry in state.tree_entries}


def _git_content_source(
    state: PaperGitState,
    path: str,
    *,
    label: str,
) -> ContentSource:
    entry = _paper_entry_map(state).get(path)
    if entry is None:
        raise SubmissionPackagingError(
            f"{label} is not present in paper commit {state.commit}: {path}"
        )
    data = _read_git_blob(state.worktree_root, entry.object_id)
    return ContentSource(
        kind="git_blob",
        label=f"paper-git:{state.commit}:{path}",
        expected_sha256=_sha256_bytes(data),
        git_repository=state.worktree_root,
        git_object_id=entry.object_id,
    )


def inspect_paper_worktree(
    worktree_root: Path,
    expected_commit: str,
    *,
    source_subdir: str = DEFAULT_PAPER_SOURCE_SUBDIR,
    entrypoint: str = DEFAULT_PAPER_ENTRYPOINT,
) -> PaperGitState:
    """Require an exact, clean Git worktree with the expected paper layout."""
    raw_root = Path(worktree_root)
    if raw_root.is_symlink():
        raise SubmissionPackagingError(
            f"paper worktree root cannot be a symbolic link: {raw_root}"
        )
    worktree_root = raw_root.resolve()
    if not worktree_root.is_dir():
        raise SubmissionPackagingError(
            f"paper worktree root is not a directory: {worktree_root}"
        )
    expected_commit = expected_commit.strip().lower()
    if FULL_GIT_COMMIT_PATTERN.fullmatch(expected_commit) is None:
        raise SubmissionPackagingError(
            "paper commit must be a full 40-character hexadecimal Git SHA"
        )
    source_subdir = _safe_relative_path(source_subdir, "paper source subdirectory")
    entrypoint = _safe_relative_path(entrypoint, "paper entrypoint")
    source_parts = PurePosixPath(source_subdir).parts
    entrypoint_parts = PurePosixPath(entrypoint).parts
    if entrypoint_parts[: len(source_parts)] != source_parts:
        raise SubmissionPackagingError(
            "paper entrypoint must be inside the paper source subdirectory"
        )

    top_level = Path(
        _run_git(worktree_root, ("rev-parse", "--show-toplevel"))
        .decode("utf-8")
        .strip()
    ).resolve()
    if top_level != worktree_root:
        raise SubmissionPackagingError(
            "paper worktree root must be the Git top level containing "
            f"{source_subdir}/: {top_level}"
        )
    actual_commit = (
        _run_git(worktree_root, ("rev-parse", "HEAD")).decode("ascii").strip().lower()
    )
    if actual_commit != expected_commit:
        raise SubmissionPackagingError(
            f"paper commit mismatch: expected {expected_commit}, found {actual_commit}"
        )
    status = _run_git(
        worktree_root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
    )
    if status:
        entries = [
            item.decode("utf-8", errors="replace")
            for item in status.split(b"\0")
            if item
        ]
        raise SubmissionPackagingError(
            "paper worktree is not clean: " + "; ".join(entries[:5])
        )

    tree = _run_git(
        worktree_root,
        ("ls-tree", "-r", "-z", "--full-tree", actual_commit, "--", source_subdir),
    )
    entries: list[GitTreeEntry] = []
    for raw_record in tree.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ", 2)
            path = _safe_relative_path(raw_path.decode("utf-8"), "paper Git tree path")
        except (UnicodeError, ValueError) as exc:
            raise SubmissionPackagingError(
                "could not parse pinned paper Git tree"
            ) from exc
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise SubmissionPackagingError(
                f"paper commit contains unsupported entry {path}: {mode} {object_type}"
            )
        entries.append(GitTreeEntry(path=path, object_id=object_id, mode=mode))
    tracked_paths = frozenset(entry.path for entry in entries)
    if not tracked_paths:
        raise SubmissionPackagingError(
            f"paper commit tracks no files under {source_subdir}/"
        )
    if entrypoint not in tracked_paths:
        raise SubmissionPackagingError(
            f"paper entrypoint is not tracked at {expected_commit}: {entrypoint}"
        )
    return PaperGitState(
        worktree_root=worktree_root,
        commit=actual_commit,
        source_subdir=source_subdir,
        entrypoint=entrypoint,
        tracked_paths=tracked_paths,
        tree_entries=tuple(sorted(entries, key=lambda item: item.path)),
    )


def _paper_source_is_allowed(path: Path | PurePosixPath) -> bool:
    return (
        path.suffix.lower() in PAPER_SOURCE_SUFFIXES
        or path.name in PAPER_SOURCE_NAMES
        or path.name.startswith(PAPER_SOURCE_PREFIXES)
    )


def discover_paper_source_files(
    root: Path,
    *,
    tracked_paths: frozenset[str] | None = None,
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
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
        if tracked_paths is not None and relative_text not in tracked_paths:
            continue
        if _paper_source_is_allowed(path):
            included.append(path)
        else:
            excluded.append(relative_text)
    if not included:
        raise SubmissionPackagingError(
            "paper source root contains no allowlisted source files"
        )
    if not any(path.suffix.lower() == ".tex" for path in included):
        raise SubmissionPackagingError("paper source root contains no TeX source")
    return tuple(included), tuple(excluded)


def discover_paper_source_paths(
    state: PaperGitState,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select manuscript source paths from the pinned commit tree."""
    included: list[str] = []
    excluded: list[str] = []
    for relative in sorted(state.tracked_paths):
        pure = PurePosixPath(relative)
        if any(part in EXCLUDED_DIRECTORY_NAMES for part in pure.parts[:-1]):
            continue
        if _paper_source_is_allowed(pure):
            included.append(relative)
        else:
            excluded.append(relative)
    if not included:
        raise SubmissionPackagingError(
            "pinned paper commit contains no allowlisted source files"
        )
    if not any(PurePosixPath(path).suffix.lower() == ".tex" for path in included):
        raise SubmissionPackagingError("pinned paper commit contains no TeX source")
    return tuple(included), tuple(excluded)


def discover_imported_graphics(
    paper_source_root: Path, paper_source_files: Sequence[Path]
) -> tuple[str, ...]:
    """Return uncommented ``includegraphics`` targets from manuscript TeX files."""
    texts: list[tuple[str, str]] = []
    for path in paper_source_files:
        if path.suffix.lower() != ".tex":
            continue
        try:
            texts.append((str(path), path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError) as exc:
            raise SubmissionPackagingError(
                f"could not read TeX source {path}: {exc}"
            ) from exc
    return _discover_imported_graphics_from_texts(texts)


def _discover_imported_graphics_from_texts(
    texts: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    pattern = re.compile(
        r"\\includegraphics(?:\s*\[[^\]]*\])?\s*\{([^}]+)\}",
        re.MULTILINE,
    )
    targets: set[str] = set()
    for label, text_value in texts:
        lines = text_value.splitlines()
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


def discover_imported_graphics_from_git(
    state: PaperGitState,
    paper_source_paths: Sequence[str],
) -> tuple[str, ...]:
    entries = _paper_entry_map(state)
    texts: list[tuple[str, str]] = []
    for path in paper_source_paths:
        if PurePosixPath(path).suffix.lower() != ".tex":
            continue
        entry = entries[path]
        try:
            text_value = _read_git_blob(state.worktree_root, entry.object_id).decode(
                "utf-8"
            )
        except UnicodeError as exc:
            raise SubmissionPackagingError(
                f"could not decode TeX source from paper commit: {path}"
            ) from exc
        texts.append((path, text_value))
    return _discover_imported_graphics_from_texts(texts)


def load_approved_figures(
    manifest_path: Path, paper_source: Path | PaperGitState
) -> tuple[ApprovedFigure, ...]:
    """Load checksum-pinned, explicitly approved vector figures."""
    manifest_path = _require_regular_file(manifest_path, "approved-figures manifest")
    paper_state = paper_source if isinstance(paper_source, PaperGitState) else None
    paper_source_root = (
        None if paper_state is not None else Path(paper_source).resolve()
    )
    try:
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or ())
            required = {"paper_path", "sha256", "approval_status", "approval_reference"}
            missing = sorted(required - fieldnames)
            if missing:
                raise SubmissionPackagingError(
                    "approved-figures manifest is missing columns: "
                    + ", ".join(missing)
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
            str(row.get("paper_path", "")).strip(),
            f"figure row {row_number} paper_path",
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
        if paper_state is not None:
            source = _git_content_source(
                paper_state,
                source_relative,
                label=f"approved figure {paper_path}",
            )
        else:
            assert paper_source_root is not None
            filesystem_path = paper_source_root.joinpath(
                *PurePosixPath(source_relative).parts
            )
            try:
                filesystem_path.resolve().relative_to(paper_source_root)
            except ValueError as exc:
                raise SubmissionPackagingError(
                    f"approved figure escapes the paper source root: {source_value}"
                ) from exc
            source = _filesystem_source(
                filesystem_path,
                label=f"approved figure {paper_path}",
            )

        expected_digest = str(row.get("sha256", "")).strip().lower()
        if len(expected_digest) != 64 or any(
            character not in "0123456789abcdef" for character in expected_digest
        ):
            raise SubmissionPackagingError(
                f"figure row {row_number} has an invalid SHA-256 digest"
            )
        actual_digest = source.expected_sha256
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
                source_path=source_relative,
                source=source,
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
            members = [
                member for member in archive.getmembers() if member.name == member_name
            ]
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
        raise SubmissionPackagingError(
            f"could not inspect source snapshot: {exc}"
        ) from exc
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


def _compile_manuscript_tree(
    source_root: Path,
    *,
    source_subdir: str,
    entrypoint: str,
    latexmk_executable: str,
    timeout_seconds: int = DEFAULT_MANUSCRIPT_COMPILE_TIMEOUT_SECONDS,
) -> None:
    """Compile a disposable copy of one exact manuscript source tree."""
    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        raise SubmissionPackagingError(
            f"manuscript compile source is not a directory: {source_root}"
        )
    executable = shutil.which(latexmk_executable)
    if executable is None:
        explicit = Path(latexmk_executable)
        if explicit.is_file() and os.access(explicit, os.X_OK):
            executable = str(explicit.resolve())
    if executable is None:
        raise SubmissionPackagingError(
            f"manuscript compiler is unavailable: {latexmk_executable}"
        )

    with tempfile.TemporaryDirectory(prefix="submission-manuscript-compile-") as temp:
        scratch = Path(temp)
        compile_root = scratch / "paper-worktree"
        shutil.copytree(source_root, compile_root)
        compile_entrypoint = compile_root.joinpath(*PurePosixPath(entrypoint).parts)
        if not compile_entrypoint.is_file():
            raise SubmissionPackagingError(
                f"staged manuscript entrypoint is missing: {entrypoint}"
            )
        build_dir = scratch / "build"
        build_dir.mkdir()
        environment = os.environ.copy()
        for variable in tuple(environment):
            if variable.startswith("TEXMF") or variable in {
                "TEXINPUTS",
                "BIBINPUTS",
                "BSTINPUTS",
                "LATEXMKRC",
                "LATEXMKRCSYS",
                "PERL5LIB",
                "PERL5OPT",
            }:
                environment.pop(variable, None)
        search_prefix = f"{source_subdir}//:"
        for variable in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS"):
            environment[variable] = search_prefix
        isolated_home = scratch / "home"
        isolated_texmf_home = scratch / "texmf-home"
        isolated_texmf_config = scratch / "texmf-config"
        isolated_texmf_var = scratch / "texmf-var"
        for directory in (
            isolated_home,
            isolated_home / ".config",
            isolated_texmf_home,
            isolated_texmf_config,
            isolated_texmf_var,
        ):
            directory.mkdir()
        environment["HOME"] = str(isolated_home)
        environment["XDG_CONFIG_HOME"] = str(isolated_home / ".config")
        environment["TEXMFHOME"] = str(isolated_texmf_home)
        environment["TEXMFCONFIG"] = str(isolated_texmf_config)
        environment["TEXMFVAR"] = str(isolated_texmf_var)
        environment["TEXMFCACHE"] = str(isolated_texmf_var)
        environment["TEXMFOUTPUT"] = str(build_dir)
        environment["SOURCE_DATE_EPOCH"] = "0"
        environment["TZ"] = "UTC"
        command = [
            executable,
            "-norc",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={build_dir}",
            entrypoint,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=compile_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise SubmissionPackagingError(
                f"manuscript compile exceeded {timeout_seconds} seconds"
            ) from exc
        except OSError as exc:
            raise SubmissionPackagingError(
                f"could not run manuscript compiler: {exc}"
            ) from exc
        if completed.returncode != 0:
            transcript = "\n".join(
                (completed.stdout + "\n" + completed.stderr).splitlines()[-30:]
            )
            raise SubmissionPackagingError(
                "manuscript compile failed" + (f":\n{transcript}" if transcript else "")
            )
        expected_name = f"{PurePosixPath(entrypoint).stem}.pdf"
        outputs = [
            path
            for path in build_dir.rglob(expected_name)
            if path.is_file() and path.stat().st_size > 0
        ]
        if not outputs:
            raise SubmissionPackagingError(
                "manuscript compiler returned success without producing "
                f"{expected_name}"
            )


def _materialize_manuscript_files(
    planned_files: Sequence[PlannedFile],
    destination_root: Path,
) -> None:
    prefix = "manuscript/source/"
    copied = 0
    for item in planned_files:
        if not item.destination.startswith(prefix):
            continue
        relative = item.destination[len(prefix) :]
        if not relative:
            raise SubmissionPackagingError("empty manuscript package destination")
        _copy_content_source(
            item.source,
            destination_root.joinpath(*PurePosixPath(relative).parts),
        )
        copied += 1
    if copied == 0:
        raise SubmissionPackagingError("package plan contains no manuscript files")


def _preflight_manuscript_compile(
    planned_files: Sequence[PlannedFile],
    *,
    source_subdir: str,
    entrypoint: str,
    latexmk_executable: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="submission-manuscript-plan-") as temp:
        source_root = Path(temp) / "source"
        source_root.mkdir()
        _materialize_manuscript_files(planned_files, source_root)
        _compile_manuscript_tree(
            source_root,
            source_subdir=source_subdir,
            entrypoint=entrypoint,
            latexmk_executable=latexmk_executable,
        )


def plan_submission_package(
    *,
    release_root: Path,
    paper_worktree_root: Path,
    paper_commit: str,
    approved_figures_manifest: Path,
    raw_data_deposition: str,
    raw_data_manifest_identifier: str,
    output_dir: Path,
    deposited_release_manifest: Path | None = None,
    acknowledge_unverified_remote_deposit: bool = False,
    paper_source_subdir: str = DEFAULT_PAPER_SOURCE_SUBDIR,
    paper_entrypoint: str = DEFAULT_PAPER_ENTRYPOINT,
    latexmk_executable: str = "latexmk",
    review_bundle: Path | None = None,
    audit_runner: Callable[[Path], AuditReport] = audit_final_release,
    checksum_verifier: Callable[[Path, str], list[str]] = verify_sha256_manifest,
    pdf_inspector: Callable[..., PdfQaReport] = inspect_pdf,
) -> PackagePlan:
    """Validate every input and return the exact package plan without writing it."""
    release_root = Path(release_root).resolve()
    output_dir = _resolve_output_destination(output_dir)
    output_parent = _validate_private_output_parent(output_dir)
    if not release_root.is_dir():
        raise SubmissionPackagingError(
            f"release root is not a directory: {release_root}"
        )
    paper_state = inspect_paper_worktree(
        paper_worktree_root,
        paper_commit,
        source_subdir=paper_source_subdir,
        entrypoint=paper_entrypoint,
    )
    deposition, deposit_evidence = _validate_deposition(
        raw_data_deposition,
        raw_data_manifest_identifier,
        release_root,
        deposited_manifest_file=deposited_release_manifest,
        acknowledge_unverified_remote_deposit=(acknowledge_unverified_remote_deposit),
    )
    archive_path = output_dir.with_suffix(output_dir.suffix + ".tar.gz")
    _reject_package_destination_conflicts((output_dir, archive_path))
    if os.path.lexists(output_dir):
        raise SubmissionPackagingError(f"output directory already exists: {output_dir}")
    if os.path.lexists(archive_path):
        raise SubmissionPackagingError(f"output archive already exists: {archive_path}")
    for protected_root, label in (
        (release_root, "release root"),
        (paper_state.worktree_root, "paper worktree root"),
    ):
        try:
            output_dir.relative_to(protected_root)
        except ValueError:
            pass
        else:
            raise SubmissionPackagingError(
                f"output directory cannot be inside the {label}: {output_dir}"
            )
    audit_report = _audit_release_or_fail(release_root, audit_runner, checksum_verifier)
    release_manifest_records = _load_release_sha256_manifest(
        release_root / RELEASE_SHA256_MANIFEST
    )

    paper_files, excluded = discover_paper_source_paths(paper_state)
    figures = load_approved_figures(
        approved_figures_manifest,
        paper_state,
    )
    approved_source_paths = {figure.source_path for figure in figures}
    excluded = tuple(
        relative for relative in excluded if relative not in approved_source_paths
    )
    approved_paths = {figure.paper_path for figure in figures}
    imported_graphics = set(
        discover_imported_graphics_from_git(paper_state, paper_files)
    )
    missing_approvals = sorted(imported_graphics - approved_paths)
    if missing_approvals:
        raise SubmissionPackagingError(
            "manuscript imports graphics absent from the approved-figures manifest: "
            + ", ".join(missing_approvals)
        )
    figure_qa: list[PdfQaReport] = []
    with tempfile.TemporaryDirectory(prefix="submission-figure-qa-") as temp:
        for index, figure in enumerate(figures):
            figure_path = Path(temp) / f"figure-{index}.pdf"
            _copy_content_source(figure.source, figure_path)
            try:
                report = pdf_inspector(figure_path, require_fonts=False)
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

    def add(destination: str, source: ContentSource, role: str) -> None:
        destination = _safe_relative_path(destination, "package destination")
        if destination in destinations:
            raise SubmissionPackagingError(
                f"duplicate package destination: {destination}"
            )
        destinations.add(destination)
        planned.append(PlannedFile(destination, source, role))

    for source_relative, destination, role in RELEASE_PAYLOADS:
        if source_relative == RELEASE_SHA256_MANIFEST:
            expected_digest = deposition.manifest_sha256
        else:
            expected_digest = release_manifest_records.get(source_relative)
            if expected_digest is None:
                raise SubmissionPackagingError(
                    "release payload is absent from the complete-release checksum "
                    f"manifest: {source_relative}"
                )
        add(
            destination,
            _filesystem_source(
                release_root / source_relative,
                label=f"release:{source_relative}",
                expected_sha256=expected_digest,
            ),
            role,
        )
    for relative in paper_files:
        add(
            f"manuscript/source/{relative}",
            _git_content_source(
                paper_state,
                relative,
                label=f"manuscript source {relative}",
            ),
            "manuscript_source",
        )
    for figure in figures:
        add(
            f"manuscript/source/{figure.paper_path}",
            figure.source,
            "approved_vector_figure",
        )
    for source, relative in _review_bundle_files(review_bundle):
        add(
            f"manuscript/review/{relative}",
            _filesystem_source(source, label=f"review:{relative}"),
            "review_bundle",
        )
    if deposit_evidence is not None:
        add(
            "provenance/deposit/SHA256SUMS.downloaded",
            deposit_evidence,
            "deposited_release_manifest_evidence",
        )

    planned_files = tuple(sorted(planned, key=lambda item: item.destination))
    _preflight_manuscript_compile(
        planned_files,
        source_subdir=paper_state.source_subdir,
        entrypoint=paper_state.entrypoint,
        latexmk_executable=latexmk_executable,
    )

    return PackagePlan(
        release_root=release_root,
        paper_worktree_root=paper_state.worktree_root,
        paper_commit=paper_state.commit,
        paper_source_subdir=paper_state.source_subdir,
        paper_entrypoint=paper_state.entrypoint,
        paper_tracked_file_count=len(paper_state.tracked_paths),
        latexmk_executable=latexmk_executable,
        output_dir=output_dir,
        output_parent_device=output_parent.device,
        output_parent_inode=output_parent.inode,
        files=planned_files,
        approved_figures=figures,
        figure_qa=tuple(figure_qa),
        excluded_paper_files=excluded,
        audit_summary=dict(audit_report.summaries),
        raw_data_deposit=deposition,
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


def _copy_content_source(source: ContentSource, destination: Path) -> None:
    if source.kind == "filesystem":
        if source.path is None:
            raise SubmissionPackagingError(
                f"filesystem source lacks a path: {source.label}"
            )
        _copy_file(_require_regular_file(source.path, source.label), destination)
    elif source.kind == "git_blob":
        if source.git_repository is None or source.git_object_id is None:
            raise SubmissionPackagingError(f"Git source is incomplete: {source.label}")
        _write_bytes(
            destination,
            _read_git_blob(source.git_repository, source.git_object_id),
        )
    else:
        raise SubmissionPackagingError(
            f"unsupported content-source kind {source.kind!r}: {source.label}"
        )
    actual_digest = _sha256(destination)
    if actual_digest != source.expected_sha256:
        destination.unlink(missing_ok=True)
        raise SubmissionPackagingError(
            f"staged source checksum mismatch for {source.label}: "
            f"expected {source.expected_sha256}, found {actual_digest}"
        )


def _inventory_entry(
    root: Path, planned: PlannedFile, plan: PackagePlan
) -> InventoryEntry:
    path = root.joinpath(*PurePosixPath(planned.destination).parts)
    return InventoryEntry(
        path=planned.destination,
        role=planned.role,
        source=planned.source.label,
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
            "staged_payloads_verified_against_release_manifest": True,
        },
        "raw_data": {
            "included": False,
            "deposition": asdict(plan.raw_data_deposit),
            "excluded_paths": [
                "raw_runs/",
                "diagnostics/case_*",
                "diagnostics/cell_metrics.csv",
                "diagnostics/merge_events.csv",
            ],
        },
        "paper": {
            "git_commit": plan.paper_commit,
            "source_subdirectory": plan.paper_source_subdir,
            "entrypoint": plan.paper_entrypoint,
            "clean_pinned_worktree_verified": True,
            "bytes_materialized_from_pinned_git_objects": True,
            "tracked_file_count": plan.paper_tracked_file_count,
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
excluded from this compact submission package. Their declared deposit location is:

{plan.raw_data_deposit.location}

`provenance/release/SHA256SUMS` identifies every file in the complete deposited
release. Its deposit binding is `{plan.raw_data_deposit.manifest_identifier}`.
Deposit verification status is `{plan.raw_data_deposit.verification_status}`.
The packager makes no network-access assertion.
`provenance/RAW_DATA_DEPOSITION.md` records the exclusion boundary.

## Manuscript Provenance

The manuscript source came from paper commit `{plan.paper_commit}`. Source and
figure bytes were read from pinned Git objects, not from the worktree. The
packager compiled the staged source from a disposable copy with external TeX
search paths, user TEXMF trees, and latexmk configuration disabled. When this
package was created with its outer archive, it also verified package checksums
and compiled a disposable copy of the extracted manuscript before reporting
success.

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

- Complete release identifier: `{plan.raw_data_deposit.release_name}`
- Declared deposition: {plan.raw_data_deposit.location}
- Raw data included in this compact package: no
- Complete release checksum manifest: `provenance/release/SHA256SUMS`
- Deposited release-manifest identifier: `{plan.raw_data_deposit.manifest_identifier}`
- Release-manifest filename: `{plan.raw_data_deposit.manifest_name}`
- Release-manifest SHA-256: `{plan.raw_data_deposit.manifest_sha256}`
- Verification status: `{plan.raw_data_deposit.verification_status}`
- Supplied manifest bytes verified: `{str(plan.raw_data_deposit.supplied_manifest_bytes_verified).lower()}`
- Network assertion made by packager: `{str(plan.raw_data_deposit.network_assertion_made).lower()}`

The external deposit should contain the complete audited release, including
`raw_runs/` and the case-, cell-, merge-, and fallback-indexed diagnostic tables.
The compact package retains the aggregate results and run inventory needed to map
paper results to those deposited bundles. When
`provenance/deposit/SHA256SUMS.downloaded` is present, its bytes were supplied to
the packager and matched exactly. Otherwise remote contents remain an explicit
manual submission gate; this record does not claim that the DOI/URL was fetched.
"""
    _write_text(root / "provenance" / "RAW_DATA_DEPOSITION.md", text)


def _write_manuscript_build_record(
    root: Path,
    plan: PackagePlan,
    *,
    archive_verification_required: bool,
) -> None:
    payload = {
        "schema_version": 1,
        "paper_git_commit": plan.paper_commit,
        "paper_source_subdirectory": plan.paper_source_subdir,
        "entrypoint": plan.paper_entrypoint,
        "compiler": Path(plan.latexmk_executable).name,
        "compile_arguments": [
            "-norc",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-outdir=<temporary-build-directory>",
            plan.paper_entrypoint,
        ],
        "clean_pinned_worktree_verified": True,
        "paper_bytes_materialized_from_git_objects": True,
        "external_tex_search_environment_discarded": True,
        "user_texmf_and_latexmkrc_disabled": True,
        "preflight_compile_passed": True,
        "staged_compile_passed": True,
        "compile_outputs_in_package": False,
        "extracted_archive_compile_required_before_packager_success": (
            archive_verification_required
        ),
    }
    _write_text(
        root / "provenance" / "manuscript_build.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _write_release_audit_record(root: Path, plan: PackagePlan) -> None:
    payload = {
        "schema_version": 1,
        "passed": True,
        "release_name": plan.release_root.name,
        "summaries": dict(sorted(plan.audit_summary.items())),
        "full_release_checksums_verified": True,
        "staged_release_payloads_verified_against_manifest": True,
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


def _write_deterministic_tar_gz_temporary(
    root: Path,
    archive_path: Path,
    *,
    archive_root_name: str,
) -> _OwnedPath:
    archive_root = PurePosixPath(
        _safe_relative_path(archive_root_name, "archive root name")
    )
    if len(archive_root.parts) != 1:
        raise SubmissionPackagingError(
            f"archive root name must have one path component: {archive_root_name}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive_path.name}.tmp-",
        dir=archive_path.parent,
    )
    temporary = Path(temporary_name)
    temporary_owner = _OwnedPath.capture(temporary)
    try:
        os.fchmod(descriptor, 0o644)
        raw_stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with raw_stream:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_stream, mtime=0
            ) as gzip_stream:
                with tarfile.open(
                    fileobj=gzip_stream,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    paths = (root,) + tuple(
                        sorted(
                            root.rglob("*"),
                            key=lambda path: path.relative_to(root).as_posix(),
                        )
                    )
                    for path in paths:
                        if path.is_symlink():
                            raise SubmissionPackagingError(
                                f"cannot archive symbolic link: {path}"
                            )
                        if path == root:
                            relative = archive_root.as_posix()
                        else:
                            relative = (
                                archive_root / path.relative_to(root).as_posix()
                            ).as_posix()
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
        return temporary_owner
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_owner.remove()
        raise


def _publish_archive(temporary_owner: _OwnedPath, archive_path: Path) -> _OwnedPath:
    try:
        metadata = os.lstat(temporary_owner.path)
    except FileNotFoundError:
        metadata = None
    if (
        metadata is None
        or not stat.S_ISREG(metadata.st_mode)
        or not temporary_owner.matches()
    ):
        raise SubmissionPackagingError(
            f"archive temporary is unavailable: {temporary_owner.path}"
        )
    try:
        os.link(temporary_owner.path, archive_path)
    except FileExistsError as exc:
        raise SubmissionPackagingError(
            f"output archive already exists: {archive_path}"
        ) from exc
    return temporary_owner.moved_to(archive_path)


def _extract_archive_safely(archive_path: Path, destination: Path) -> Path:
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if not members:
                raise SubmissionPackagingError("submission archive is empty")
            top_levels: set[str] = set()
            for member in members:
                pure = PurePosixPath(member.name)
                if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                    raise SubmissionPackagingError(
                        f"unsafe path in submission archive: {member.name}"
                    )
                if member.issym() or member.islnk():
                    raise SubmissionPackagingError(
                        f"submission archive contains a link: {member.name}"
                    )
                if not member.isdir() and not member.isfile():
                    raise SubmissionPackagingError(
                        f"submission archive contains a special file: {member.name}"
                    )
                top_levels.add(pure.parts[0])
            if len(top_levels) != 1:
                raise SubmissionPackagingError(
                    "submission archive must contain exactly one top-level directory"
                )
            archive.extractall(destination)
    except (OSError, tarfile.TarError) as exc:
        raise SubmissionPackagingError(
            f"could not extract submission archive: {exc}"
        ) from exc
    extracted_root = destination / next(iter(top_levels))
    if not extracted_root.is_dir():
        raise SubmissionPackagingError(
            "submission archive did not extract to a package directory"
        )
    return extracted_root


def _verify_extracted_archive(archive_path: Path, plan: PackagePlan) -> None:
    """Verify and compile an extracted archive without modifying its contents."""
    with tempfile.TemporaryDirectory(prefix="submission-archive-check-") as temp:
        extracted_root = _extract_archive_safely(archive_path, Path(temp))
        checksum_errors = verify_package_checksums(extracted_root)
        if checksum_errors:
            raise SubmissionPackagingError(
                "extracted package checksum verification failed: "
                + "; ".join(checksum_errors[:5])
            )
        _compile_manuscript_tree(
            extracted_root / "manuscript" / "source",
            source_subdir=plan.paper_source_subdir,
            entrypoint=plan.paper_entrypoint,
            latexmk_executable=plan.latexmk_executable,
        )
        checksum_errors = verify_package_checksums(extracted_root)
        if checksum_errors:
            raise SubmissionPackagingError(
                "manuscript compile contaminated the extracted package: "
                + "; ".join(checksum_errors[:5])
            )


def build_submission_package(
    plan: PackagePlan, *, create_archive: bool = True
) -> tuple[Path, Path | None]:
    """Atomically materialize a validated package and optional deterministic archive."""
    paper_state = inspect_paper_worktree(
        plan.paper_worktree_root,
        plan.paper_commit,
        source_subdir=plan.paper_source_subdir,
        entrypoint=plan.paper_entrypoint,
    )
    if len(paper_state.tracked_paths) != plan.paper_tracked_file_count:
        raise SubmissionPackagingError(
            "paper tracked-file inventory changed after package planning"
        )
    output_dir = plan.output_dir
    archive_path = output_dir.with_suffix(output_dir.suffix + ".tar.gz")
    reservation_targets = [output_dir]
    if create_archive:
        reservation_targets.append(archive_path)
    _validate_private_output_parent(
        output_dir,
        expected_device=plan.output_parent_device,
        expected_inode=plan.output_parent_inode,
    )
    _reject_package_destination_conflicts(reservation_targets)
    with _DestinationReservations.acquire(reservation_targets) as reservations:
        _validate_private_output_parent(
            output_dir,
            expected_device=plan.output_parent_device,
            expected_inode=plan.output_parent_inode,
        )
        _reject_package_destination_conflicts(reservation_targets)
        if os.path.lexists(output_dir):
            raise SubmissionPackagingError(
                f"output directory already exists: {output_dir}"
            )
        if create_archive and os.path.lexists(archive_path):
            raise SubmissionPackagingError(
                f"output archive already exists: {archive_path}"
            )

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
            )
        )
        staging_owner: _OwnedPath | None = _OwnedPath.capture(staging)
        archive_temporary: _OwnedPath | None = None
        published_output: _OwnedPath | None = None
        published_archive: _OwnedPath | None = None
        try:
            for item in plan.files:
                destination = staging.joinpath(*PurePosixPath(item.destination).parts)
                _copy_content_source(item.source, destination)

            _compile_manuscript_tree(
                staging / "manuscript" / "source",
                source_subdir=plan.paper_source_subdir,
                entrypoint=plan.paper_entrypoint,
                latexmk_executable=plan.latexmk_executable,
            )
            _extract_experiment_map(
                staging / "code" / "source_snapshot.tar.gz",
                staging / "docs" / "PAPER_EXPERIMENT_MAP.md",
            )
            _write_deposition_record(staging, plan)
            _write_manuscript_build_record(
                staging,
                plan,
                archive_verification_required=create_archive,
            )
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
                        "provenance/manuscript_build.json",
                        "manuscript_build_record",
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
            if create_archive:
                archive_temporary = _write_deterministic_tar_gz_temporary(
                    staging,
                    archive_path,
                    archive_root_name=output_dir.name,
                )
                _verify_extracted_archive(archive_temporary.path, plan)

            reservations.assert_owned()
            _validate_private_output_parent(
                output_dir,
                expected_device=plan.output_parent_device,
                expected_inode=plan.output_parent_inode,
            )
            _reject_package_destination_conflicts(reservation_targets)
            if os.path.lexists(output_dir):
                raise SubmissionPackagingError(
                    f"output directory appeared during packaging: {output_dir}"
                )
            if create_archive and os.path.lexists(archive_path):
                raise SubmissionPackagingError(
                    f"output archive appeared during packaging: {archive_path}"
                )
            staging.rename(output_dir)
            published_output = staging_owner.moved_to(output_dir)
            staging_owner = None
            if create_archive:
                reservations.assert_owned()
                assert archive_temporary is not None
                published_archive = _publish_archive(archive_temporary, archive_path)
                if not archive_temporary.remove():
                    raise SubmissionPackagingError(
                        "archive temporary changed during publication"
                    )
                archive_temporary = None
            return output_dir, archive_path if create_archive else None
        except Exception as exc:
            if staging_owner is not None:
                staging_owner.remove()
            if archive_temporary is not None:
                archive_temporary.remove()
            if published_output is not None or published_archive is not None:
                raise SubmissionPackagingError(
                    "submission publication failed after a final path was created; "
                    "published paths were left untouched for manual inspection"
                ) from exc
            raise


def _plan_payload(plan: PackagePlan) -> dict:
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "dry_run": True,
        "release_name": plan.release_root.name,
        "audit_summary": dict(sorted(plan.audit_summary.items())),
        "output_dir": str(plan.output_dir),
        "output_parent": str(plan.output_dir.parent),
        "output_parent_device": plan.output_parent_device,
        "output_parent_inode": plan.output_parent_inode,
        "private_output_parent_verified": True,
        "package_namespace_conflict_check_passed": True,
        "raw_data_included": False,
        "raw_data_deposition": plan.raw_data_deposit.location,
        "raw_data_manifest_identifier": (plan.raw_data_deposit.manifest_identifier),
        "raw_data_deposit_verification_status": (
            plan.raw_data_deposit.verification_status
        ),
        "network_assertion_made": False,
        "paper_commit": plan.paper_commit,
        "paper_entrypoint": plan.paper_entrypoint,
        "manuscript_compile_preflight_passed": True,
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
    parser.add_argument(
        "--paper-worktree-root",
        "--paper-source-root",
        dest="paper_worktree_root",
        type=Path,
        required=True,
        help=(
            "clean paper Git top level containing interface-reconstruction-paper/; "
            "--paper-source-root is retained as a compatibility alias"
        ),
    )
    parser.add_argument(
        "--paper-commit",
        required=True,
        help="full 40-character paper Git commit required at the worktree HEAD",
    )
    parser.add_argument(
        "--paper-source-subdir",
        default=DEFAULT_PAPER_SOURCE_SUBDIR,
    )
    parser.add_argument(
        "--paper-entrypoint",
        default=DEFAULT_PAPER_ENTRYPOINT,
    )
    parser.add_argument(
        "--latexmk-executable",
        default="latexmk",
        help="latexmk executable used for disposable manuscript compile gates",
    )
    parser.add_argument("--approved-figures-manifest", type=Path, required=True)
    parser.add_argument("--review-bundle", type=Path)
    parser.add_argument("--raw-data-deposition", required=True)
    parser.add_argument(
        "--raw-data-manifest-id",
        dest="raw_data_manifest_identifier",
        required=True,
        help="sha256:<digest> of the complete release SHA256SUMS file",
    )
    parser.add_argument(
        "--deposited-release-manifest",
        type=Path,
        help=(
            "optional locally downloaded/fetched SHA256SUMS from the deposit; "
            "its bytes must exactly match the audited release manifest"
        ),
    )
    parser.add_argument(
        "--acknowledge-unverified-remote-deposit",
        action="store_true",
        help=(
            "explicitly acknowledge that the packager did not verify remote "
            "deposit contents; required when no deposited manifest is supplied"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "new package path beneath a pre-existing, current-user-owned, "
            "non-group/other-writable directory"
        ),
    )
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
            paper_worktree_root=args.paper_worktree_root,
            paper_commit=args.paper_commit,
            paper_source_subdir=args.paper_source_subdir,
            paper_entrypoint=args.paper_entrypoint,
            latexmk_executable=args.latexmk_executable,
            approved_figures_manifest=args.approved_figures_manifest,
            review_bundle=args.review_bundle,
            raw_data_deposition=args.raw_data_deposition,
            raw_data_manifest_identifier=args.raw_data_manifest_identifier,
            deposited_release_manifest=args.deposited_release_manifest,
            acknowledge_unverified_remote_deposit=(
                args.acknowledge_unverified_remote_deposit
            ),
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
