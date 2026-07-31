"""Checksum and release helpers for the final-figure orchestration gate."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping


RELEASE_ANCHOR_FILES = (
    "submission_config.resolved.json",
    "sweep_manifest.json",
    "perturbed_sweep.csv",
    "SHA256SUMS",
)


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
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"Malformed SHA256SUMS line {line_number}")
        digest, relative = parts
        if relative in entries:
            raise ValueError(f"Duplicate SHA256SUMS path: {relative}")
        entries[relative] = digest
    return entries


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
