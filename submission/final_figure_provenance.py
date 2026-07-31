"""Checksum and release helpers for the final-figure orchestration gate."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Callable, Mapping, Optional


RELEASE_ANCHOR_FILES = (
    "submission_config.resolved.json",
    "sweep_manifest.json",
    "perturbed_sweep.csv",
    "SHA256SUMS",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_sha256_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or not SHA256_RE.fullmatch(parts[0].lower()):
            raise ValueError(f"Malformed SHA256SUMS line {line_number}")
        digest, relative = parts[0].lower(), parts[1]
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or "." in pure.parts
        ):
            raise ValueError(f"Unsafe SHA256SUMS path: {relative!r}")
        if relative in entries:
            raise ValueError(f"Duplicate SHA256SUMS path: {relative}")
        entries[relative] = digest
    return entries


def stable_file_bytes(
    path: Path,
    *,
    expected_sha256: Optional[str] = None,
    after_open_hook: Optional[Callable[[Path], None]] = None,
) -> bytes:
    """Read one regular file once and reject concurrent byte/metadata changes."""

    path = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Could not open stable input {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Stable input is not a regular file: {path}")
        if after_open_hook is not None:
            after_open_hook(path)
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ValueError(f"Stable input changed while being read: {path}")
    data = b"".join(chunks)
    if len(data) != before.st_size:
        raise ValueError(f"Stable input size changed while being read: {path}")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError(f"Stable input checksum mismatch: {path}")
    return data


def copy_verified_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    after_open_hook: Optional[Callable[[Path], None]] = None,
) -> Path:
    """Copy one checksum-ledger file without following links or replacing output."""

    source = Path(source)
    destination = Path(destination)
    data = stable_file_bytes(
        source,
        expected_sha256=expected_sha256,
        after_open_hook=after_open_hook,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    if file_sha256(destination) != expected_sha256.lower():
        destination.unlink(missing_ok=True)
        raise ValueError(f"Snapshot checksum mismatch after copy: {destination}")
    return destination


def make_tree_read_only(root: Path) -> None:
    """Make a private snapshot readable/traversable only by its owner."""

    root = Path(root)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Immutable snapshot contains a symlink: {path}")
        if path.is_file():
            path.chmod(0o400)
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o500)
    root.chmod(0o500)


def release_figure_anchor(release_root: Path) -> dict:
    root = Path(release_root).resolve()
    if not root.is_dir():
        raise ValueError(f"Final release root is not a directory: {root}")
    required = {name: root / name for name in RELEASE_ANCHOR_FILES}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Final release lacks required files: {', '.join(missing)}")

    config = load_json_object(required["submission_config.resolved.json"])
    sweep = load_json_object(required["sweep_manifest.json"])
    source = config.get("source")
    production = config.get("production_method")
    if not isinstance(source, dict) or not isinstance(production, dict):
        raise ValueError("Resolved release config lacks source or production_method")
    source_commit = source.get("target_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("Resolved release config lacks a full target commit")
    if sweep.get("status") != "completed":
        raise ValueError("Final release sweep manifest is not completed")

    checksums = parse_sha256_manifest(required["SHA256SUMS"])
    for relative in RELEASE_ANCHOR_FILES[:-1]:
        actual = file_sha256(required[relative])
        if checksums.get(relative) != actual:
            raise ValueError(f"Final release checksum does not prove {relative}")

    profile = {
        "plic_fallback": production.get("unresolved_orientation_fallback"),
        "corner_behavior_profile": production.get("corner_behavior_profile"),
        "rescue_profile": production.get("rescue_profile"),
    }
    if any(not isinstance(value, str) or not value for value in profile.values()):
        raise ValueError(
            "Resolved release config has an incomplete reconstruction profile"
        )

    return {
        "root": str(root),
        "name": root.name,
        "source_commit": source_commit,
        "reconstruction_profile": profile,
        "artifacts": {
            relative: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for relative, path in required.items()
        },
    }


def snapshot_record(path: Path, root: Path, role: str) -> dict[str, object]:
    path = Path(path).resolve()
    root = Path(root).resolve()
    if not path.is_file():
        raise ValueError(f"Snapshot artifact is missing: {path}")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Snapshot artifact is outside staging: {path}") from exc
    return {
        "role": role,
        "path": relative,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }
