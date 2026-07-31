"""Attest a clean generator checkout against one approved Git commit."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


FULL_COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")


class GeneratorCheckoutError(RuntimeError):
    """Raised when generator source bytes cannot be proven."""


@dataclass(frozen=True)
class CheckoutAttestation:
    repository: str
    approved_commit: str
    scientific_release_commit: str
    commit_tree: str
    tracked_file_count: int
    checkout_manifest_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def _git(
    repository: Path,
    *args: str,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=check,
            capture_output=True,
            text=text,
        )
    except FileNotFoundError as exc:
        raise GeneratorCheckoutError("git is unavailable") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "git command failed").strip()
        raise GeneratorCheckoutError(detail) from exc


def _commit_exists(repository: Path, commit: str) -> bool:
    result = _git(
        repository,
        "cat-file",
        "-e",
        f"{commit}^{{commit}}",
        check=False,
    )
    return result.returncode == 0


def _blob_oid(data: bytes, object_format: str) -> str:
    try:
        digest = hashlib.new(object_format)
    except ValueError as exc:
        raise GeneratorCheckoutError(
            f"Unsupported Git object format: {object_format}"
        ) from exc
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def _tree_records(repository: Path, commit: str) -> list[tuple[str, str, str]]:
    result = _git(
        repository,
        "ls-tree",
        "-rz",
        "--full-tree",
        commit,
        text=False,
    )
    records = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        metadata, path_bytes = raw.split(b"\t", 1)
        mode, object_type, oid = metadata.decode("ascii").split(" ")
        if object_type != "blob":
            raise GeneratorCheckoutError(
                f"Unsupported tracked object {object_type}: {path_bytes!r}"
            )
        path = path_bytes.decode("utf-8", errors="strict")
        records.append((mode, oid, path))
    return records


def _working_tree_blob(repository: Path, mode: str, relative: str) -> tuple[str, bytes]:
    path = repository / relative
    if mode == "120000":
        if not path.is_symlink():
            raise GeneratorCheckoutError(f"Tracked symlink is missing: {relative}")
        return mode, os.readlink(path).encode("utf-8")
    if not path.is_file() or path.is_symlink():
        raise GeneratorCheckoutError(f"Tracked file is missing: {relative}")
    actual_mode = path.stat().st_mode
    executable = bool(actual_mode & stat.S_IXUSR)
    expected_executable = mode == "100755"
    if executable != expected_executable:
        raise GeneratorCheckoutError(f"Tracked file mode differs: {relative}")
    return mode, path.read_bytes()


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


def verify_generator_checkout(
    repository: Path,
    approved_commit: str,
    scientific_release_commit: str,
) -> CheckoutAttestation:
    """Prove that the current checkout is exactly one clean approved descendant."""

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
    commit_tree = _git(
        repository, "rev-parse", f"{approved_commit}^{{tree}}"
    ).stdout.strip()
    index_tree = _git(repository, "write-tree").stdout.strip()
    if index_tree != commit_tree:
        raise GeneratorCheckoutError("Git index tree differs from approved commit")

    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout
    if status:
        raise GeneratorCheckoutError("Generator checkout is not clean")
    _reject_hidden_index_flags(repository)

    object_format = _git(repository, "rev-parse", "--show-object-format").stdout.strip()
    records = _tree_records(repository, approved_commit)
    checkout_digest = hashlib.sha256()
    for mode, expected_oid, relative in records:
        _actual_mode, data = _working_tree_blob(repository, mode, relative)
        actual_oid = _blob_oid(data, object_format)
        if actual_oid != expected_oid:
            raise GeneratorCheckoutError(
                f"Checkout bytes differ from approved commit: {relative}"
            )
        checkout_digest.update(mode.encode("ascii"))
        checkout_digest.update(b"\0")
        checkout_digest.update(relative.encode("utf-8"))
        checkout_digest.update(b"\0")
        checkout_digest.update(expected_oid.encode("ascii"))
        checkout_digest.update(b"\n")

    return CheckoutAttestation(
        repository=str(repository),
        approved_commit=approved_commit,
        scientific_release_commit=scientific_release_commit,
        commit_tree=commit_tree,
        tracked_file_count=len(records),
        checkout_manifest_sha256=checkout_digest.hexdigest(),
    )
