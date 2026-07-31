from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from copy import deepcopy
from pathlib import Path, PurePosixPath

import yaml


CONFIG_ROOT_ENV = "INTERFACE_CONFIG_ROOT"
CONFIG_AUTHORITY_ENV = "INTERFACE_CONFIG_AUTHORITY"
CONFIG_AUTHORITY_SHA256_ENV = "INTERFACE_CONFIG_AUTHORITY_SHA256"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ConfigAuthorityError(RuntimeError):
    """Raised when an orchestrated config read cannot prove its source bytes."""


def deep_update(base_dict, update_dict):
    """Recursively update nested dictionaries."""
    for key, value in update_dict.items():
        if (
            isinstance(value, dict)
            and key in base_dict
            and isinstance(base_dict[key], dict)
        ):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def _stable_regular_file_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ConfigAuthorityError(
            f"Could not open attested config file: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ConfigAuthorityError(f"Attested config is not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ConfigAuthorityError(f"Attested config changed while read: {path}")
    return b"".join(chunks)


def _authority() -> tuple[Path, dict[str, dict]] | None:
    values = (
        os.environ.get(CONFIG_ROOT_ENV),
        os.environ.get(CONFIG_AUTHORITY_ENV),
        os.environ.get(CONFIG_AUTHORITY_SHA256_ENV),
    )
    if not any(values):
        return None
    if not all(values):
        raise ConfigAuthorityError("Orchestrated config authority is incomplete")
    root_raw, manifest_raw, expected_digest = values
    if not SHA256_RE.fullmatch(expected_digest or ""):
        raise ConfigAuthorityError(
            "Config authority digest must be full lowercase SHA-256"
        )
    root = Path(root_raw)
    manifest = Path(manifest_raw)
    if not root.is_absolute() or not manifest.is_absolute():
        raise ConfigAuthorityError("Config authority paths must be absolute")
    if root.is_symlink() or manifest.is_symlink():
        raise ConfigAuthorityError("Config authority paths must not be symlinks")
    manifest_bytes = _stable_regular_file_bytes(manifest)
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_digest:
        raise ConfigAuthorityError("Config authority manifest digest differs")
    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigAuthorityError(
            f"Config authority manifest is invalid: {exc}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "authority", "files"}
        or payload.get("schema_version") != 1
        or payload.get("authority") != "approved_generator_config"
        or not isinstance(payload.get("files"), list)
    ):
        raise ConfigAuthorityError("Config authority manifest contract differs")
    records = {}
    for raw in payload["files"]:
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "size_bytes"}:
            raise ConfigAuthorityError("Config authority contains malformed record")
        relative = raw.get("path")
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or "." in pure.parts
            or ".." in pure.parts
            or relative in records
            or not SHA256_RE.fullmatch(str(raw.get("sha256", "")))
            or not isinstance(raw.get("size_bytes"), int)
            or raw["size_bytes"] < 0
        ):
            raise ConfigAuthorityError("Config authority contains unsafe record")
        records[relative] = raw
    if not records:
        raise ConfigAuthorityError("Config authority is empty")
    return root, records


def _config_relative(file_path) -> str:
    raw = str(file_path).replace(os.sep, "/")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or "." in pure.parts or ".." in pure.parts:
        raise ConfigAuthorityError(f"Unsafe orchestrated config path: {file_path}")
    parts = pure.parts[1:] if pure.parts and pure.parts[0] == "config" else pure.parts
    if not parts:
        raise ConfigAuthorityError(f"Empty orchestrated config path: {file_path}")
    return PurePosixPath(*parts).as_posix()


def _config_bytes(file_path) -> bytes:
    authority = _authority()
    if authority is None:
        return Path(file_path).read_bytes()
    root, records = authority
    relative = _config_relative(file_path)
    record = records.get(relative)
    if record is None:
        raise ConfigAuthorityError(f"Config is outside sealed authority: {relative}")
    target = root.joinpath(*PurePosixPath(relative).parts)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ConfigAuthorityError(f"Config escapes sealed root: {relative}") from exc
    for parent in (root, *target.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ConfigAuthorityError(f"Config path contains symlink: {relative}")
    data = _stable_regular_file_bytes(target)
    if len(data) != record["size_bytes"]:
        raise ConfigAuthorityError(f"Config size differs from authority: {relative}")
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ConfigAuthorityError(f"Config digest differs from authority: {relative}")
    return data


def _yaml(file_path):
    try:
        return yaml.safe_load(_config_bytes(file_path))
    except yaml.YAMLError as exc:
        raise ConfigAuthorityError(
            f"Invalid YAML in config {file_path}: {exc}"
        ) from exc


def read_yaml(file_path):
    base_config = _yaml("config/base.yaml")
    override_config = _yaml(file_path)
    config = deepcopy(base_config)
    deep_update(config, override_config)
    return config


def override_yaml(file_path, override):
    base_config = _yaml("config/base.yaml")
    override_config = _yaml(file_path)
    config = deepcopy(base_config)
    deep_update(config, override_config)
    deep_update(config, override)
    return config
