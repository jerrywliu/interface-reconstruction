"""Attest and materialize one approved generator commit without live-source trust."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Optional


FULL_COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class GeneratorCheckoutError(RuntimeError):
    """Raised when approved generator source bytes cannot be proven."""


@dataclass(frozen=True)
class CheckoutAttestation:
    repository: str
    approved_commit: str
    scientific_release_commit: str
    commit_tree: str
    tracked_file_count: int
    checkout_manifest_sha256: str
    materialized_manifest_sha256: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ApprovalRecord:
    path: str
    sha256: str
    approved_generator_commit: str
    approved_generator_tree: str
    scientific_release_commit: str
    allowlist_sha256: str
    approved_by: str
    approved_at_utc: str

    def to_dict(self) -> dict:
        return asdict(self)


def sanitized_git_environment(
    base: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Return a minimal Git environment with caller Git configuration removed."""

    source = os.environ if base is None else base
    keep = (
        "TMPDIR",
        "TMP",
        "TEMP",
        "SYSTEMROOT",
    )
    env = {key: source[key] for key in keep if key in source}
    env.update(
        {
            "PATH": os.defpath,
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return env


def _git(
    repository: Path,
    *args: str,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "--no-pager", *args],
            cwd=repository,
            env=sanitized_git_environment(),
            check=check,
            capture_output=True,
            text=text,
        )
    except FileNotFoundError as exc:
        raise GeneratorCheckoutError("git is unavailable") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout
        detail = (stderr or stdout or "git command failed").strip()
        raise GeneratorCheckoutError(detail) from exc


def _object_format(repository: Path) -> str:
    return _git(repository, "rev-parse", "--show-object-format").stdout.strip()


def _object_oid(data: bytes, object_type: str, object_format: str) -> str:
    try:
        digest = hashlib.new(object_format)
    except ValueError as exc:
        raise GeneratorCheckoutError(
            f"Unsupported Git object format: {object_format}"
        ) from exc
    digest.update(f"{object_type} {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def _cat_file(repository: Path, object_type: str, oid: str) -> bytes:
    type_result = _git(repository, "cat-file", "-t", oid, text=False)
    actual_type = type_result.stdout.decode("ascii").strip()
    if actual_type != object_type:
        raise GeneratorCheckoutError(
            f"Git object {oid} is {actual_type}, expected {object_type}"
        )
    return _git(repository, "cat-file", object_type, oid, text=False).stdout


def _commit_tree(repository: Path, commit: str, object_format: str) -> str:
    data = _cat_file(repository, "commit", commit)
    if _object_oid(data, "commit", object_format) != commit.lower():
        raise GeneratorCheckoutError(f"Commit bytes do not hash to {commit}")
    first_line = data.splitlines()[0].decode("ascii", errors="strict")
    if not first_line.startswith("tree "):
        raise GeneratorCheckoutError(f"Commit {commit} lacks a tree header")
    tree = first_line[5:]
    _cat_file(repository, "tree", tree)
    return tree


def _commit_exists(repository: Path, commit: str) -> bool:
    result = _git(repository, "cat-file", "-e", commit, check=False)
    if result.returncode != 0:
        return False
    result = _git(repository, "cat-file", "-t", commit, check=False)
    return result.returncode == 0 and result.stdout.strip() == "commit"


def _tree_records(repository: Path, treeish: str) -> list[tuple[str, str, str]]:
    result = _git(
        repository,
        "ls-tree",
        "-rz",
        "--full-tree",
        treeish,
        text=False,
    )
    records = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        metadata, path_bytes = raw.split(b"\t", 1)
        mode, object_type, oid = metadata.decode("ascii").split(" ")
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise GeneratorCheckoutError(
                f"Unsupported tracked object or symlink {object_type}/{mode}: {path_bytes!r}"
            )
        path = path_bytes.decode("utf-8", errors="strict")
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
            raise GeneratorCheckoutError(f"Unsafe tracked path: {path!r}")
        records.append((mode, oid, path))
    return records


def _index_records(repository: Path) -> list[tuple[str, str, str]]:
    result = _git(repository, "ls-files", "--stage", "-z", text=False)
    records = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        metadata, path_bytes = raw.split(b"\t", 1)
        mode, oid, stage_number = metadata.decode("ascii").split(" ")
        if stage_number != "0":
            raise GeneratorCheckoutError("Git index contains an unresolved stage")
        records.append((mode, oid, path_bytes.decode("utf-8", errors="strict")))
    return records


def _working_tree_blob(repository: Path, mode: str, relative: str) -> bytes:
    path = repository / relative
    if mode == "120000":
        if not path.is_symlink():
            raise GeneratorCheckoutError(f"Tracked symlink is missing: {relative}")
        return os.readlink(path).encode("utf-8")
    if not path.is_file() or path.is_symlink():
        raise GeneratorCheckoutError(f"Tracked file is missing: {relative}")
    actual_mode = path.stat().st_mode
    executable = bool(actual_mode & stat.S_IXUSR)
    if executable != (mode == "100755"):
        raise GeneratorCheckoutError(f"Tracked file mode differs: {relative}")
    return path.read_bytes()


def _reject_hidden_index_flags(repository: Path) -> None:
    result = _git(repository, "ls-files", "-v", "-z", text=False)
    flagged = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        marker = chr(raw[0])
        path = raw[2:].decode("utf-8", errors="replace")
        if marker != "H":
            flagged.append(f"{marker} {path}")
    if flagged:
        raise GeneratorCheckoutError(
            "assume-unchanged/skip-worktree or nonstandard index flags are set: "
            + ", ".join(flagged[:10])
        )


def _inventory_digest(records: list[tuple[str, str, str]]) -> str:
    digest = hashlib.sha256()
    for mode, oid, relative in records:
        digest.update(mode.encode("ascii"))
        digest.update(b"\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(oid.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_generator_checkout(
    repository: Path,
    approved_commit: str,
    scientific_release_commit: str,
) -> CheckoutAttestation:
    """Prove that the live checkout is one clean approved descendant."""

    repository = Path(repository).resolve()
    if not repository.is_dir():
        raise GeneratorCheckoutError(f"Generator repository is missing: {repository}")
    for label, commit in (
        ("approved generator", approved_commit),
        ("scientific release", scientific_release_commit),
    ):
        if not FULL_COMMIT_RE.fullmatch(commit or ""):
            raise GeneratorCheckoutError(f"{label} commit must be full 40-hex")
        if not _commit_exists(repository, commit):
            raise GeneratorCheckoutError(f"{label} commit does not exist: {commit}")

    ancestry = _git(
        repository,
        "merge-base",
        "--is-ancestor",
        scientific_release_commit,
        approved_commit,
        check=False,
    )
    if ancestry.returncode != 0:
        raise GeneratorCheckoutError(
            "Approved generator commit does not descend from scientific release commit"
        )

    head = _git(repository, "rev-parse", "HEAD").stdout.strip()
    if head != approved_commit:
        raise GeneratorCheckoutError(
            f"Checkout HEAD {head} is not approved generator commit {approved_commit}"
        )

    object_format = _object_format(repository)
    commit_tree = _commit_tree(repository, approved_commit, object_format)
    records = _tree_records(repository, commit_tree)
    if _index_records(repository) != records:
        raise GeneratorCheckoutError("Git index differs from approved commit tree")
    status_result = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status_result.stdout:
        raise GeneratorCheckoutError("Generator checkout is not clean")
    _reject_hidden_index_flags(repository)

    for mode, expected_oid, relative in records:
        data = _working_tree_blob(repository, mode, relative)
        if _object_oid(data, "blob", object_format) != expected_oid:
            raise GeneratorCheckoutError(
                f"Checkout bytes differ from approved commit: {relative}"
            )

    return CheckoutAttestation(
        repository=str(repository),
        approved_commit=approved_commit,
        scientific_release_commit=scientific_release_commit,
        commit_tree=commit_tree,
        tracked_file_count=len(records),
        checkout_manifest_sha256=_inventory_digest(records),
    )


def materialize_approved_source(
    repository: Path,
    approved_commit: str,
    destination: Path,
    attestation: CheckoutAttestation,
) -> CheckoutAttestation:
    """Build a read-only source tree only from the approved commit's blobs."""

    repository = Path(repository).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise GeneratorCheckoutError(
            f"Materialized source destination already exists: {destination}"
        )
    if attestation.approved_commit != approved_commit:
        raise GeneratorCheckoutError(
            "Checkout attestation does not match approved commit"
        )

    object_format = _object_format(repository)
    commit_tree = _commit_tree(repository, approved_commit, object_format)
    if commit_tree != attestation.commit_tree:
        raise GeneratorCheckoutError(
            "Approved commit tree changed during materialization"
        )
    records = _tree_records(repository, commit_tree)
    if _inventory_digest(records) != attestation.checkout_manifest_sha256:
        raise GeneratorCheckoutError(
            "Approved tree inventory changed during materialization"
        )

    destination.mkdir(mode=0o700)
    try:
        for mode, expected_oid, relative in records:
            data = _cat_file(repository, "blob", expected_oid)
            if _object_oid(data, "blob", object_format) != expected_oid:
                raise GeneratorCheckoutError(
                    f"Materialized blob bytes do not hash correctly: {relative}"
                )
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "120000":
                os.symlink(data.decode("utf-8", errors="strict"), target)
            else:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                descriptor = os.open(
                    target, flags, 0o700 if mode == "100755" else 0o600
                )
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                target.chmod(0o555 if mode == "100755" else 0o444)

        materialized_records = []
        for mode, expected_oid, relative in records:
            data = _working_tree_blob(destination, mode, relative)
            actual_oid = _object_oid(data, "blob", object_format)
            if actual_oid != expected_oid:
                raise GeneratorCheckoutError(
                    f"Materialized source verification failed: {relative}"
                )
            materialized_records.append((mode, actual_oid, relative))
        for directory in sorted(
            (path for path in destination.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        destination.chmod(0o555)
    except Exception:
        for path in destination.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)
        destination.chmod(0o700)
        import shutil

        shutil.rmtree(destination)
        raise

    digest = _inventory_digest(materialized_records)
    if digest != attestation.checkout_manifest_sha256:
        raise GeneratorCheckoutError("Materialized source inventory digest differs")
    return CheckoutAttestation(
        **{
            **attestation.to_dict(),
            "materialized_manifest_sha256": digest,
        }
    )


def verify_external_approval_record(
    path: Path,
    expected_sha256: str,
    *,
    repository: Path,
    approved_commit: str,
    approved_tree: str,
    scientific_release_commit: str,
    allowlist_sha256: str,
) -> ApprovalRecord:
    """Verify the separately reviewed approval record for the exact final commit."""

    supplied_path = Path(path).expanduser().absolute()
    if supplied_path.is_symlink():
        raise GeneratorCheckoutError("Approval record must not be a symbolic link")
    path = supplied_path.resolve()
    repository = Path(repository).resolve()
    if not SHA256_RE.fullmatch((expected_sha256 or "").lower()):
        raise GeneratorCheckoutError("Approval record SHA-256 must be full 64-hex")
    try:
        path.relative_to(repository)
    except ValueError:
        pass
    else:
        raise GeneratorCheckoutError("Approval record must be outside the repository")
    if path.is_symlink() or not path.is_file():
        raise GeneratorCheckoutError(f"Approval record is not a regular file: {path}")
    mode = path.stat().st_mode
    if path.stat().st_uid != os.getuid():
        raise GeneratorCheckoutError(
            "Approval record must be owned by the current user"
        )
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise GeneratorCheckoutError("Approval record must not be group/world writable")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256.lower():
        raise GeneratorCheckoutError("Approval record SHA-256 does not match")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GeneratorCheckoutError(f"Approval record is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GeneratorCheckoutError("Approval record JSON root must be an object")
    expected = {
        "schema_version": 1,
        "record_type": "final_figure_generator_approval",
        "approved_generator_commit": approved_commit,
        "approved_generator_tree": approved_tree,
        "scientific_release_commit": scientific_release_commit,
        "allowlist_sha256": allowlist_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise GeneratorCheckoutError(f"Approval record field {key} does not match")
    approved_by = payload.get("approved_by")
    approved_at = payload.get("approved_at_utc")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise GeneratorCheckoutError("Approval record lacks approved_by")
    if not isinstance(approved_at, str) or not approved_at.strip():
        raise GeneratorCheckoutError("Approval record lacks approved_at_utc")
    return ApprovalRecord(
        path=str(path),
        sha256=digest,
        approved_generator_commit=approved_commit,
        approved_generator_tree=approved_tree,
        scientific_release_commit=scientific_release_commit,
        allowlist_sha256=allowlist_sha256,
        approved_by=approved_by.strip(),
        approved_at_utc=approved_at.strip(),
    )
