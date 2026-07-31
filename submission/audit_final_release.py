#!/usr/bin/env python3
"""Fail-closed audit for a completed final static-result release."""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import errno
import gzip
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import statistics
import subprocess
import sys
import tarfile
import tempfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, DecimalException
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Mapping, Sequence

import yaml


FINAL_RUN_COUNT = 970
FINAL_CASE_COUNT = 24_250
DEFAULT_SHA256_MANIFEST = "SHA256SUMS"
MAX_REPORTED_ERRORS = 250
FINAL_SOURCE_COMMIT = "505aefa454328d4ba34ade5e7247050a0acfc793"
LEGACY_COMMAND_SOURCE_COMMIT = "505aefa454328d4ba34ade5e7247050a0acfc793"
LEGACY_COMMAND_SCHEMA_VERSION = 1
MAX_NUMERIC_TEXT_LENGTH = 128
MAX_INTEGER_TEXT_LENGTH = 32
MAX_DECIMAL_DIGITS = 64
MIN_DECIMAL_ADJUSTED = -100
MAX_DECIMAL_ADJUSTED = 100
TAR_BLOCK_SIZE = 512
TAR_RECORD_SIZE = 10_240
MAX_TAR_METADATA_BYTES_PER_MEMBER = 4_096
MAX_TAR_GLOBAL_METADATA_BYTES = 65_536
GZIP_VALIDATION_CHUNK_SIZE = 65_536
TRUSTED_GIT_CANDIDATES = (
    Path("/usr/bin/git"),
    Path("/usr/local/bin/git"),
    Path("/opt/homebrew/bin/git"),
)
TAR_EXTENSION_TYPES = frozenset(
    {
        tarfile.GNUTYPE_LONGNAME,
        tarfile.GNUTYPE_LONGLINK,
        tarfile.XHDTYPE,
        tarfile.XGLTYPE,
        tarfile.SOLARIS_XHDTYPE,
    }
)

SNAPSHOT_EXCLUDED_ROOTS = frozenset(
    {".git", "logs", "output", "plots", "results", "tmp"}
)

PRODUCTION_CONTROLLER_MODULE = "experiments.static.run_perturbed_sweeps"
PRODUCTION_DRIVER_MODULES = {
    "lines": "experiments.static.lines",
    "circles": "experiments.static.circles",
    "ellipses": "experiments.static.ellipses",
    "squares": "experiments.static.squares",
    "zalesak": "experiments.static.zalesak",
}

BENCHMARK_COUNT_PARAMETERS = {
    "lines": "num_lines",
    "circles": "num_circles",
    "ellipses": "num_ellipses",
    "squares": "num_squares",
    "zalesak": "num_cases",
}
BENCHMARK_RANDOM_SEEDS = {
    "lines": 42,
    "circles": 41,
    "ellipses": 42,
    "squares": 42,
    "zalesak": 43,
}
BENCHMARK_GEOMETRY_TYPES = {
    "lines": "line",
    "circles": "circle",
    "ellipses": "ellipse",
    "squares": "square",
    "zalesak": "zalesak",
}

# Full-file fingerprints from the reviewed production commit. Binding the entire
# file rejects alternate/dead call sites as well as changes to the live path.
PRODUCTION_SOURCE_SHA256 = {
    "submission/run_final_static_sweep.sh": (
        "be7ce2e3553a8c2443368639e79698a78fa5ff86c4e3df17133996a56f8032ec"
    ),
    "experiments/static/run_perturbed_sweeps.py": (
        "1b2253d1b0c16c1a47d044855ac088af4a9dc18529ac23db70803ae2fa779682"
    ),
    "util/reconstruction.py": (
        "28c412e8400e08a31741193a9d5279a2b276201b542431d0a919b7c938911295"
    ),
    "main/structs/meshes/merge_mesh.py": (
        "67b22195f0689339d91c64a60f2bffcb707d203a07efee648bb8d0c0ee8851b7"
    ),
    "experiments/static/lines.py": (
        "463639def93cdc3a3e64100df8b79ea19efcb97a04326b00c6f01f65d572df4b"
    ),
    "experiments/static/circles.py": (
        "013450ef16afade444a44655be2e0129f98b84ff8df5fa9eade51557f7c533cd"
    ),
    "experiments/static/ellipses.py": (
        "c1ae97acaa8d2def738a531dd7d9d346184ef88131f4851cda5c2385bc039769"
    ),
    "experiments/static/squares.py": (
        "f8ff5c32f50c89ad518dcff24136f6d86f78e1f3ff67f8de39c697b8eedcbf53"
    ),
    "experiments/static/zalesak.py": (
        "c74a4f3547e4b51b7e1bcc5f8eef3ec04de9c2e1930f35a7b33fdab0e1f8d202"
    ),
}

RUN_CONTEXT_FIELDS = (
    "experiment",
    "algo",
    "resolution",
    "wiggle",
    "seed",
    "save_name",
    "source_commit",
    "source_branch",
    "plic_fallback",
    "rescue_profile",
    "corner_behavior_profile",
)

REQUIRED_RELEASE_FILES = (
    "submission_config.resolved.json",
    "sweep_manifest.json",
    "environment.json",
    "failures.csv",
    "perturbed_sweep.csv",
    "diagnostics/source_state.json",
    "diagnostics/source_snapshot.tar.gz",
    "diagnostics/run_inventory.csv",
    "diagnostics/run_manifests.jsonl",
    "diagnostics/case_geometry.jsonl",
    "diagnostics/case_metrics.csv",
    "diagnostics/cell_metrics.csv",
    "diagnostics/merge_events.csv",
    "diagnostics/unresolved_plic_fallbacks.csv",
)

REQUIRED_RAW_ARTIFACTS = {
    "case_geometry": "metrics/case_geometry.jsonl",
    "case_metrics": "metrics/case_metrics.csv",
    "cell_metrics": "metrics/cell_metrics.csv",
    "fallback_events": "metrics/unresolved_plic_fallbacks.csv",
    "merge_events": "metrics/merge_events.csv",
    "mesh": "vtk/mesh.vtk",
}

METRICS_BY_EXPERIMENT = {
    "lines": ("hausdorff", "facet_gap"),
    "circles": (
        "curvature_error",
        "facet_gap",
        "hausdorff",
        "tangent_error",
        "curvature_proxy_error",
    ),
    "ellipses": (
        "curvature_error",
        "facet_gap",
        "hausdorff",
        "tangent_error",
        "curvature_proxy_error",
    ),
    "squares": ("area_error", "facet_gap", "hausdorff"),
    "zalesak": ("area_error", "facet_gap", "hausdorff"),
}

AGGREGATE_STATS = ("mean", "median", "p25", "p75")

PRODUCTION_CONTEXT_CONFIG_FIELDS = {
    "plic_fallback": "unresolved_orientation_fallback",
    "rescue_profile": "rescue_profile",
    "corner_behavior_profile": "corner_behavior_profile",
}

PRODUCTION_COMMAND_OPTIONS = {
    "plic_fallback": "--plic_fallback",
    "rescue_profile": "--rescue_profile",
    "corner_behavior_profile": "--corner_behavior_profile",
}

RECONCILIATION_TABLES = (
    ("cell_metrics.csv", ("case_index", "cell_id")),
    ("case_metrics.csv", ("case_index",)),
    ("merge_events.csv", ("case_index", "event_order")),
    ("unresolved_plic_fallbacks.csv", ("case_index", "merge_id")),
)

INTEGER_KEY_FIELDS = frozenset({"case_index", "event_order", "merge_id"})
JSON_RELATIVE_PATH_FIELDS = frozenset({"truth_vtp", "truth_metadata"})
CELL_INTEGER_FIELDS = frozenset(
    {
        "merge_id",
        "merge_component_size",
        "is_merged",
        "has_3x3_stencil",
        "used_circular",
        "used_linear_corner",
        "used_curved_corner",
        "used_curved_corner_rescue",
        "event_count",
    }
)

RESCUE_INHERITANCE_EXPERIMENTS = frozenset({"lines", "circles", "ellipses", "squares"})


class ReleaseAuditInputError(ValueError):
    """Raised when a manifest operation receives an unsafe input path."""


class DecompressedSnapshotBudgetExceeded(OSError):
    """Raised before tarfile can consume data beyond the verified source budget."""


class SourceSnapshotFormatError(OSError):
    """Raised when gzip or tar termination is not canonical and complete."""


class _DecompressedByteBudgetStream:
    def __init__(self, stream: gzip.GzipFile, byte_budget: int):
        self._stream = stream
        self._byte_budget = byte_budget

    def _raise_budget_error(self) -> None:
        raise DecompressedSnapshotBudgetExceeded(
            "decompressed source snapshot exceeds Git-derived byte budget "
            f"of {self._byte_budget} bytes"
        )

    def tell(self) -> int:
        position = self._stream.tell()
        if position > self._byte_budget:
            self._raise_budget_error()
        return position

    def read(self, size: int = -1) -> bytes:
        position = self.tell()
        remaining = self._byte_budget - position
        if size is None or size < 0 or size > remaining + 1:
            size = remaining + 1
        data = self._stream.read(size)
        if len(data) > remaining:
            self._raise_budget_error()
        return data

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = self.tell() + offset
        else:
            raise DecompressedSnapshotBudgetExceeded(
                "end-relative seeks are forbidden for bounded source snapshots"
            )
        if target < 0 or target > self._byte_budget:
            self._raise_budget_error()
        position = self._stream.seek(offset, whence)
        if position != target:
            raise OSError(
                f"decompressed source snapshot seek returned {position}, "
                f"expected {target}"
            )
        return position


@dataclass(frozen=True, order=True)
class RunKey:
    experiment: str
    algo: str
    resolution: str
    wiggle: str
    seed: int

    def display(self) -> str:
        return (
            f"{self.experiment}/{self.algo}/r={self.resolution}/"
            f"w={self.wiggle}/s={self.seed}"
        )


@dataclass(frozen=True)
class GitBlob:
    mode: str
    object_id: str
    data: bytes


@dataclass
class RescueProfileInheritance:
    value: str
    eligible_experiments: frozenset[str]
    driver_modules: dict[str, str] = field(default_factory=dict)
    historical_root: PurePosixPath | None = None
    legacy_command_strings_allowed: bool = False
    source_verified: bool = False
    evidence: tuple[str, ...] = ()
    proof_failures: tuple[str, ...] = ()
    profile_states: dict[RunKey, dict[str, set[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set))
    )
    command_resolutions: dict[RunKey, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    command_states: dict[RunKey, dict[str, str]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    @property
    def proven(self) -> bool:
        return self.source_verified and not self.proof_failures

    def permits(self, key: RunKey | None) -> bool:
        return (
            self.proven
            and key is not None
            and key.experiment in self.eligible_experiments
        )

    def note_state(self, key: RunKey, location: str, state: str) -> None:
        self.profile_states[key][location].add(state)


@dataclass
class AuditReport:
    release_root: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_errors: int = 0
    summaries: dict[str, int | str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.total_errors == 0

    def add_error(self, message: str) -> None:
        self.total_errors += 1
        if len(self.errors) < MAX_REPORTED_ERRORS:
            self.errors.append(message)

    def add_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    @property
    def suppressed_errors(self) -> int:
        return self.total_errors - len(self.errors)


@dataclass(frozen=True)
class SealedRelease:
    release_root: Path
    manifest_path: Path
    report: AuditReport
    copied_bytes: int
    clone_files: int
    copied_files: int


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value!r}")


def _strict_json_float(value: str) -> Decimal:
    return _bounded_decimal(value, "JSON number")


def _strict_json_int(value: str) -> int:
    return _parse_int(value, "JSON integer")


def _strict_json_loads(text: str, source: Path | str) -> object:
    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            parse_float=_strict_json_float,
            parse_int=_strict_json_int,
        )
    except (json.JSONDecodeError, DecimalException, ValueError, OverflowError) as exc:
        raise ReleaseAuditInputError(f"invalid JSON in {source}: {exc}") from exc


def _load_json(path: Path, report: AuditReport) -> dict | None:
    if not path.is_file():
        report.add_error(f"missing required JSON file: {path}")
        return None
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"), path)
    except (OSError, UnicodeError, ReleaseAuditInputError) as exc:
        report.add_error(str(exc))
        return None
    if not isinstance(value, dict):
        report.add_error(f"JSON root must be an object: {path}")
        return None
    return value


def _bounded_decimal(value: object, label: str) -> Decimal:
    text = str(value)
    if not text or text != text.strip():
        raise ReleaseAuditInputError(f"{label} is not a canonical number: {value!r}")
    if len(text) > MAX_NUMERIC_TEXT_LENGTH:
        raise ReleaseAuditInputError(f"{label} is too long: {len(text)} characters")
    try:
        number = Decimal(text)
    except (DecimalException, ValueError, OverflowError) as exc:
        raise ReleaseAuditInputError(f"{label} is not numeric: {value!r}") from exc
    if not number.is_finite():
        raise ReleaseAuditInputError(f"{label} is nonfinite: {value!r}")
    digits = number.as_tuple().digits
    try:
        adjusted = number.adjusted()
    except (DecimalException, ValueError, OverflowError) as exc:
        raise ReleaseAuditInputError(
            f"{label} has an invalid decimal exponent: {value!r}"
        ) from exc
    if len(digits) > MAX_DECIMAL_DIGITS:
        raise ReleaseAuditInputError(
            f"{label} has too many significant digits: {len(digits)}"
        )
    if not MIN_DECIMAL_ADJUSTED <= adjusted <= MAX_DECIMAL_ADJUSTED:
        raise ReleaseAuditInputError(
            f"{label} exponent is outside the audited range: {adjusted}"
        )
    return number


def _canonical_number(value: object) -> str:
    number = _bounded_decimal(value, "number")
    if number == 0:
        return "0"
    try:
        rendered = format(number, "f")
    except (DecimalException, ValueError, OverflowError) as exc:
        raise ReleaseAuditInputError(f"cannot canonicalize number: {value!r}") from exc
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _parse_int(value: object, label: str) -> int:
    number = _bounded_decimal(value, label)
    try:
        is_integral = number == number.to_integral_value()
    except (DecimalException, ValueError, OverflowError) as exc:
        raise ReleaseAuditInputError(f"{label} is not an integer: {value!r}") from exc
    if not is_integral:
        raise ReleaseAuditInputError(f"{label} is not an integer: {value!r}")
    try:
        return int(number)
    except (DecimalException, ValueError, OverflowError) as exc:
        raise ReleaseAuditInputError(f"{label} is not an integer: {value!r}") from exc


def _parse_canonical_int(value: object, label: str) -> int:
    text = str(value)
    if len(text) > MAX_INTEGER_TEXT_LENGTH:
        raise ReleaseAuditInputError(f"{label} is too long: {len(text)} characters")
    if not re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text):
        raise ReleaseAuditInputError(f"{label} is not a canonical integer: {value!r}")
    try:
        return int(text)
    except (ValueError, OverflowError) as exc:
        raise ReleaseAuditInputError(
            f"{label} is not a bounded integer: {value!r}"
        ) from exc


def _finite_metric(value: object, label: str) -> float:
    decimal_value = _bounded_decimal(value, label)
    if decimal_value < 0:
        raise ReleaseAuditInputError(f"{label} is negative: {value!r}")
    try:
        number = float(decimal_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReleaseAuditInputError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ReleaseAuditInputError(f"{label} is nonfinite: {value!r}")
    return number


def _run_key(row: Mapping[str, object], source: str) -> RunKey:
    missing = [
        field
        for field in ("experiment", "algo", "resolution", "wiggle", "seed")
        if row.get(field) in (None, "")
    ]
    if missing:
        raise ReleaseAuditInputError(
            f"{source} is missing run-key fields: {', '.join(missing)}"
        )
    return RunKey(
        experiment=str(row["experiment"]).strip().lower(),
        algo=str(row["algo"]).strip().lower(),
        resolution=_canonical_number(row["resolution"]),
        wiggle=_canonical_number(row["wiggle"]),
        seed=_parse_int(row["seed"], f"{source} seed"),
    )


def _safe_release_path(root: Path, raw_path: object, label: str) -> Path:
    value = str(raw_path or "")
    pure = PurePosixPath(value)
    if not value or pure.is_absolute() or ".." in pure.parts:
        raise ReleaseAuditInputError(f"{label} is not release-relative: {value!r}")
    path = root.joinpath(*pure.parts)
    try:
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ReleaseAuditInputError(
            f"{label} escapes the release root: {value!r}"
        ) from exc
    return path


def _lexical_absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_release_symlinks(root: Path, report: AuditReport) -> bool:
    """Reject links and special nodes before any release content is trusted."""
    try:
        root_stat = os.lstat(root)
    except OSError as exc:
        report.add_error(f"could not inspect release root {root}: {exc}")
        return False
    if stat.S_ISLNK(root_stat.st_mode):
        report.add_error(f"release root is a symbolic link: {root}")
        return False
    if not stat.S_ISDIR(root_stat.st_mode):
        report.add_error(f"release root is not a directory: {root}")
        return False

    valid = True
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in list(directory_names):
            path = directory_path / name
            try:
                node_stat = os.lstat(path)
            except OSError as exc:
                report.add_error(f"could not inspect release path {path}: {exc}")
                directory_names.remove(name)
                valid = False
                continue
            if stat.S_ISLNK(node_stat.st_mode):
                report.add_error(f"release contains a symbolic link: {path}")
                directory_names.remove(name)
                valid = False
            elif not stat.S_ISDIR(node_stat.st_mode):
                report.add_error(f"release contains a non-directory node: {path}")
                directory_names.remove(name)
                valid = False
        for name in file_names:
            path = directory_path / name
            try:
                node_stat = os.lstat(path)
            except OSError as exc:
                report.add_error(f"could not inspect release path {path}: {exc}")
                valid = False
                continue
            if stat.S_ISLNK(node_stat.st_mode):
                report.add_error(f"release contains a symbolic link: {path}")
                valid = False
            elif not stat.S_ISREG(node_stat.st_mode):
                report.add_error(f"release contains a non-regular file: {path}")
                valid = False
    return valid


def _require_headers(
    fieldnames: Sequence[str] | None,
    required: Iterable[str],
    path: Path,
    report: AuditReport,
) -> bool:
    if fieldnames is None:
        report.add_error(f"CSV has no header: {path}")
        return False
    duplicates = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
    if duplicates:
        report.add_error(f"CSV {path} has duplicate columns: {', '.join(duplicates)}")
        return False
    missing = sorted(set(required) - set(fieldnames))
    if missing:
        report.add_error(f"CSV {path} is missing columns: {', '.join(missing)}")
        return False
    return True


def _validate_csv_row_shape(
    row: Mapping[object, object],
    path: Path,
    row_number: int,
    report: AuditReport,
) -> bool:
    valid = True
    trailing = row.get(None)
    if trailing:
        report.add_error(
            f"CSV {path}:{row_number} has trailing fields beyond the header: "
            f"{_diagnostic_value(trailing)}"
        )
        valid = False
    missing = sorted(
        str(field_name)
        for field_name, value in row.items()
        if field_name is not None and value is None
    )
    if missing:
        report.add_error(
            f"CSV {path}:{row_number} has missing trailing columns: "
            f"{', '.join(missing)}"
        )
        valid = False
    return valid


def _read_csv_rows(
    path: Path,
    required_headers: Iterable[str],
    report: AuditReport,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        report.add_error(f"missing required CSV file: {path}")
        return [], []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
            if not _require_headers(fieldnames, required_headers, path, report):
                return fieldnames, []
            rows: list[dict[str, str]] = []
            for row_number, row in enumerate(reader, start=2):
                if _validate_csv_row_shape(row, path, row_number, report):
                    rows.append(row)
            return fieldnames, rows
    except (OSError, UnicodeError, csv.Error) as exc:
        report.add_error(f"could not read CSV {path}: {exc}")
        return [], []


def _option_values(tokens: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == option:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ReleaseAuditInputError(f"command option {option} has no value")
            values.append(tokens[index + 1])
            index += 2
            continue
        prefix = f"{option}="
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
        index += 1
    return values


def _command_tokens(command: object) -> list[str]:
    if not isinstance(command, str) or not command.strip():
        raise ReleaseAuditInputError("command is absent or empty")
    try:
        return shlex.split(command)
    except ValueError as exc:
        raise ReleaseAuditInputError(f"command cannot be parsed: {exc}") from exc


def _manifest_command_tokens(
    manifest: Mapping[str, object],
    *,
    allow_legacy_string: bool,
) -> list[str]:
    argv = manifest.get("argv")
    if argv is None:
        if not allow_legacy_string:
            raise ReleaseAuditInputError(
                "argv is required; legacy command-string parsing is restricted to "
                "the frozen 505aefa schema-1 release with no spaces"
            )
        schema_version = manifest.get("schema_version")
        if (
            type(schema_version) is not int
            or schema_version != LEGACY_COMMAND_SCHEMA_VERSION
        ):
            raise ReleaseAuditInputError(
                "legacy command-string schema is not the frozen schema version 1"
            )
        command = manifest.get("command")
        tokens = _command_tokens(command)
        if command != " ".join(tokens) or any(
            not token or any(character.isspace() for character in token)
            for token in tokens
        ):
            raise ReleaseAuditInputError(
                "legacy command is not a canonical no-space serialization"
            )
        return tokens
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(token, str) or "\x00" in token for token in argv)
    ):
        raise ReleaseAuditInputError("argv is not a non-empty JSON string array")
    command = manifest.get("command")
    if command not in (None, ""):
        command_tokens = _command_tokens(command)
        if argv != command_tokens:
            raise ReleaseAuditInputError(
                "argv does not exactly match the tokenized command string"
            )
    return list(argv)


def _allow_legacy_command_strings(
    target_commit: str,
    source_verified: bool,
    historical_root: PurePosixPath | None,
) -> bool:
    return bool(
        source_verified
        and target_commit == LEGACY_COMMAND_SOURCE_COMMIT
        and historical_root is not None
        and not any(character.isspace() for character in str(historical_root))
    )


def _historical_repository_root(raw_root: object) -> PurePosixPath:
    if not isinstance(raw_root, str) or not raw_root:
        raise ReleaseAuditInputError(
            "environment repository.root is not an absolute path"
        )
    root = PurePosixPath(raw_root)
    if (
        not root.is_absolute()
        or str(root) != raw_root
        or "." in root.parts
        or ".." in root.parts
    ):
        raise ReleaseAuditInputError(
            "environment repository.root is not a canonical absolute POSIX path"
        )
    return root


def _command_invokes_driver(
    tokens: Sequence[str],
    driver_module: str,
    historical_root: PurePosixPath | None,
) -> bool:
    expected_path = driver_module.replace(".", "/") + ".py"
    if not tokens or historical_root is None:
        return False
    raw_executable = tokens[0]
    executable = PurePosixPath(raw_executable)
    if (
        "\x00" in raw_executable
        or str(executable) != raw_executable
        or "." in executable.parts
        or ".." in executable.parts
    ):
        return False
    if executable.is_absolute():
        try:
            executable = executable.relative_to(historical_root)
        except ValueError:
            return False
    return executable.as_posix() == expected_path


def _build_rescue_profile_inheritance(
    config: Mapping[str, object],
    global_profiles_verified: bool,
    historical_root: PurePosixPath | None,
    legacy_command_strings_allowed: bool,
) -> RescueProfileInheritance:
    production = config.get("production_method")
    expected = ""
    if isinstance(production, dict):
        expected = str(production.get("rescue_profile") or "")
    failures: list[str] = []
    if not expected:
        failures.append("production rescue_profile is empty")
    if not global_profiles_verified:
        failures.append(
            "source provenance or controller production-profile pin is unverified"
        )

    benchmarks = config.get("benchmarks")
    configured_inheritance_experiments: frozenset[str] = frozenset()
    driver_modules: dict[str, str] = {}
    if not isinstance(benchmarks, Mapping):
        failures.append("resolved config benchmarks are unavailable")
    else:
        configured_inheritance_experiments = frozenset(
            str(experiment).lower()
            for experiment in benchmarks
            if str(experiment).lower() in RESCUE_INHERITANCE_EXPERIMENTS
        )
        for configured_experiment in sorted(benchmarks):
            experiment = str(configured_experiment).lower()
            benchmark = benchmarks.get(experiment)
            driver = benchmark.get("driver") if isinstance(benchmark, Mapping) else None
            expected_driver = PRODUCTION_DRIVER_MODULES.get(experiment)
            if not isinstance(driver, str) or driver != expected_driver:
                failures.append(
                    f"benchmark driver is not the reviewed production driver for "
                    f"{experiment}"
                )
                continue
            driver_modules[experiment] = driver

    evidence = (
        "approved Git commit object",
        "byte-exact archived tracked source tree",
        "reviewed production launcher/controller/driver/reconstruction fingerprints",
        "per-run child command bound to the reviewed driver",
    )
    return RescueProfileInheritance(
        value=expected,
        eligible_experiments=configured_inheritance_experiments,
        driver_modules=driver_modules,
        historical_root=historical_root,
        legacy_command_strings_allowed=legacy_command_strings_allowed,
        source_verified=global_profiles_verified,
        evidence=evidence,
        proof_failures=tuple(dict.fromkeys(failures)),
    )


def _production_context(
    production: Mapping[str, object], report: AuditReport
) -> dict[str, str]:
    expected: dict[str, str] = {}
    for context_field, config_field in PRODUCTION_CONTEXT_CONFIG_FIELDS.items():
        value = production.get(config_field)
        if value in (None, ""):
            report.add_error(
                f"resolved config production_method.{config_field} is empty"
            )
            expected[context_field] = ""
        else:
            expected[context_field] = str(value)
    return expected


def _check_production_context(
    row: Mapping[str, object],
    production_context: Mapping[str, str],
    label: str,
    report: AuditReport,
    *,
    key: RunKey | None = None,
    inheritance: RescueProfileInheritance | None = None,
    inheritance_location: str = "context",
) -> None:
    for field_name, expected in production_context.items():
        actual = row.get(field_name)
        if actual == expected:
            if field_name == "rescue_profile" and key is not None and inheritance:
                inheritance.note_state(key, inheritance_location, "explicit")
            continue
        if field_name == "rescue_profile" and actual in (None, ""):
            if inheritance is not None and inheritance.permits(key):
                assert key is not None
                inheritance.note_state(key, inheritance_location, "omitted")
                continue
            if inheritance is None:
                proof = "no inheritance proof was constructed"
            elif (
                key is not None
                and key.experiment not in inheritance.eligible_experiments
            ):
                proof = f"{key.experiment} requires an explicit rescue profile"
            else:
                proof = "; ".join(inheritance.proof_failures) or "proof is incomplete"
            report.add_error(
                f"{label} rescue_profile omission cannot be inherited: {proof}"
            )
            continue
        report.add_error(
            f"{label} {field_name} differs from production: "
            f"{actual!r} != {expected!r}"
        )


def _check_child_manifest_command(
    manifest: Mapping[str, object],
    key: RunKey,
    production_context: Mapping[str, str],
    inheritance: RescueProfileInheritance,
    label: str,
    report: AuditReport,
) -> None:
    parameters = manifest.get("parameters")
    if not isinstance(parameters, Mapping):
        report.add_error(f"{label} parameters are unavailable for command binding")
        return
    try:
        tokens = _manifest_command_tokens(
            manifest,
            allow_legacy_string=inheritance.legacy_command_strings_allowed,
        )
    except ReleaseAuditInputError as exc:
        report.add_error(f"{label} command evidence is invalid: {exc}")
        return

    driver_module = inheritance.driver_modules.get(key.experiment)
    if not driver_module or not _command_invokes_driver(
        tokens, driver_module, inheritance.historical_root
    ):
        report.add_error(
            f"{label} does not invoke reviewed driver {driver_module!r} for "
            f"{key.display()}"
        )

    for field_name, option in PRODUCTION_COMMAND_OPTIONS.items():
        expected = production_context.get(field_name, "")
        actual = parameters.get(field_name)
        try:
            values = _option_values(tokens, option)
        except ReleaseAuditInputError as exc:
            report.add_error(f"{label} {option} evidence is invalid: {exc}")
            continue
        rescue_omission = (
            field_name == "rescue_profile"
            and actual in (None, "")
            and inheritance.permits(key)
        )
        if rescue_omission:
            if values not in ([], [expected]):
                report.add_error(
                    f"{label} {option} conflicts with inherited production profile: "
                    f"{values!r} not in {([], [expected])!r}"
                )
                continue
            inheritance.command_resolutions[key].add(label)
            inheritance.command_states[key][label] = "explicit" if values else "omitted"
            continue
        if actual != expected:
            report.add_error(
                f"{label} parameter {field_name} is not the production value: "
                f"{actual!r} != {expected!r}"
            )
        if values != [expected]:
            report.add_error(
                f"{label} command does not contain exactly one matching {option}: "
                f"{values!r} != {[expected]!r}"
            )


def _values_match(actual: object, expected: object, *, numeric: bool = False) -> bool:
    if numeric:
        try:
            return _canonical_number(actual) == _canonical_number(expected)
        except ReleaseAuditInputError:
            return False
    return type(actual) is type(expected) and actual == expected


def _declared_child_config_path(benchmark: Mapping[str, object]) -> str:
    raw = benchmark.get("config")
    if not isinstance(raw, str):
        raise ReleaseAuditInputError("benchmark config path is unavailable")
    pure = PurePosixPath(raw)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or len(pure.parts) != 3
        or pure.parts[:2] != ("config", "static")
        or pure.suffix != ".yaml"
    ):
        raise ReleaseAuditInputError(f"benchmark config path is unsafe: {raw!r}")
    return PurePosixPath(*pure.parts[1:]).with_suffix("").as_posix()


def _check_parameter_value(
    parameters: Mapping[str, object],
    field_name: str,
    expected: object,
    label: str,
    report: AuditReport,
    *,
    numeric: bool = False,
) -> None:
    actual = parameters.get(field_name)
    if not _values_match(actual, expected, numeric=numeric):
        report.add_error(
            f"{label} parameter {field_name} differs from resolved production "
            f"configuration: {actual!r} != {expected!r}"
        )


def _check_exact_command_option(
    tokens: Sequence[str],
    option: str,
    expected: object,
    label: str,
    report: AuditReport,
    *,
    numeric: bool = False,
) -> None:
    try:
        values = _option_values(tokens, option)
    except ReleaseAuditInputError as exc:
        report.add_error(f"{label} {option} evidence is invalid: {exc}")
        return
    if len(values) != 1 or not _values_match(values[0], str(expected), numeric=numeric):
        report.add_error(
            f"{label} command does not bind {option} to the resolved run value: "
            f"{values!r} != {[str(expected)]!r}"
        )


def _check_child_manifest_contract(
    manifest: Mapping[str, object],
    key: RunKey,
    config: Mapping[str, object],
    save_name: str,
    inheritance: RescueProfileInheritance,
    label: str,
    report: AuditReport,
) -> None:
    parameters = manifest.get("parameters")
    benchmarks = config.get("benchmarks")
    grid = config.get("benchmark_grid")
    if not isinstance(parameters, Mapping):
        report.add_error(f"{label} parameters are unavailable")
        return
    if not isinstance(benchmarks, Mapping) or not isinstance(grid, Mapping):
        report.add_error(f"{label} cannot be bound to an invalid resolved config")
        return
    benchmark = benchmarks.get(key.experiment)
    if not isinstance(benchmark, Mapping):
        report.add_error(f"{label} has no resolved benchmark configuration")
        return
    try:
        config_value = _declared_child_config_path(benchmark)
        trials = _parse_int(
            grid.get("trials_per_setting"), "benchmark_grid.trials_per_setting"
        )
        grid_seed = _parse_int(grid.get("seed"), "benchmark_grid.seed")
    except ReleaseAuditInputError as exc:
        report.add_error(f"{label} cannot resolve child parameters: {exc}")
        return

    expected_mesh = {"perturbed Cartesian quadrilaterals": "perturbed_quads"}.get(
        grid.get("mesh_type")
    )
    if expected_mesh is None:
        report.add_error(
            f"{label} resolved mesh_type is not the frozen production mesh: "
            f"{grid.get('mesh_type')!r}"
        )
        return
    expected_fix_boundary = 1 if grid.get("fix_boundary_nodes") is True else 0
    common_values = {
        "config": (config_value, False),
        "resolution": (key.resolution, True),
        "perturb_wiggle": (key.wiggle, True),
        "perturb_seed": (grid_seed, True),
        "mesh_type": (expected_mesh, False),
        "perturb_fix_boundary": (expected_fix_boundary, True),
        "perturb_type": (None, False),
        "perturb_max_tries": (None, False),
        "case_indices": (None, False),
        "do_c0": (False, False),
        "random_seed": (BENCHMARK_RANDOM_SEEDS[key.experiment], True),
        BENCHMARK_COUNT_PARAMETERS[key.experiment]: (trials, True),
    }
    for field_name, (expected, numeric) in common_values.items():
        _check_parameter_value(
            parameters, field_name, expected, label, report, numeric=numeric
        )
    actual_algo = str(parameters.get("facet_algo", "")).lower()
    if actual_algo != key.algo:
        report.add_error(
            f"{label} parameter facet_algo differs from run key: "
            f"{actual_algo!r} != {key.algo!r}"
        )

    geometry_parameters: dict[str, tuple[object, bool]] = {}
    if key.experiment == "circles":
        geometry_parameters["radius"] = (benchmark.get("radius"), True)
    elif key.experiment == "ellipses":
        geometry_parameters["major_axis"] = (30, True)
    elif key.experiment == "zalesak":
        geometry_parameters.update(
            {
                "radius": (benchmark.get("radius"), True),
                "slot_width": (benchmark.get("slot_width"), True),
                "slot_top_rel": (
                    benchmark.get("slot_top_relative_to_center"),
                    True,
                ),
                "arc_failure_fallback": ("local_linear", False),
            }
        )
    for field_name, (expected, numeric) in geometry_parameters.items():
        if expected is None:
            report.add_error(
                f"{label} resolved geometry parameter {field_name} is absent"
            )
            continue
        _check_parameter_value(
            parameters, field_name, expected, label, report, numeric=numeric
        )

    try:
        tokens = _manifest_command_tokens(
            manifest,
            allow_legacy_string=inheritance.legacy_command_strings_allowed,
        )
    except ReleaseAuditInputError:
        return
    command_contract = {
        "--config": (config_value, False),
        "--resolution": (key.resolution, True),
        "--facet_algo": (parameters.get("facet_algo"), False),
        "--save_name": (save_name, False),
        "--mesh_type": (expected_mesh, False),
        "--perturb_wiggle": (key.wiggle, True),
        "--perturb_seed": (grid_seed, True),
        "--perturb_fix_boundary": (expected_fix_boundary, True),
        f"--{BENCHMARK_COUNT_PARAMETERS[key.experiment]}": (trials, True),
    }
    for option, (expected, numeric) in command_contract.items():
        _check_exact_command_option(
            tokens, option, expected, label, report, numeric=numeric
        )


def _iter_jsonl(path: Path, report: AuditReport) -> Iterator[tuple[int, dict]]:
    if not path.is_file():
        report.add_error(f"missing required JSONL file: {path}")
        return
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                try:
                    value = _strict_json_loads(raw_line, f"{path}:{line_number}")
                except ReleaseAuditInputError as exc:
                    report.add_error(str(exc))
                    continue
                if not isinstance(value, dict):
                    report.add_error(
                        f"JSONL row must be an object: {path}:{line_number}"
                    )
                    continue
                yield line_number, value
    except (OSError, UnicodeError) as exc:
        report.add_error(f"could not read JSONL {path}: {exc}")


def _expected_grid(
    config: Mapping[str, object], report: AuditReport
) -> tuple[set[RunKey], int]:
    try:
        grid = config["benchmark_grid"]
        benchmarks = config["benchmarks"]
        if not isinstance(grid, dict) or not isinstance(benchmarks, dict):
            raise TypeError("benchmark_grid and benchmarks must be objects")
        wiggles = list(grid["wiggles"])
        seed = _parse_int(grid["seed"], "benchmark_grid.seed")
        trials = _parse_int(
            grid["trials_per_setting"], "benchmark_grid.trials_per_setting"
        )
    except (KeyError, TypeError, ReleaseAuditInputError) as exc:
        report.add_error(f"invalid benchmark grid in resolved config: {exc}")
        return set(), 0

    expected: set[RunKey] = set()
    for experiment, raw_benchmark in benchmarks.items():
        if not isinstance(raw_benchmark, dict):
            report.add_error(f"benchmark {experiment!r} is not an object")
            continue
        try:
            resolution_key = str(raw_benchmark["resolutions"])
            resolutions = list(grid[resolution_key])
            methods = list(raw_benchmark["methods"])
        except (KeyError, TypeError) as exc:
            report.add_error(f"invalid benchmark {experiment!r}: {exc}")
            continue
        computed = len(resolutions) * len(wiggles) * len(methods)
        try:
            configured = _parse_int(
                raw_benchmark["planned_runs"], f"{experiment}.planned_runs"
            )
        except (KeyError, ReleaseAuditInputError) as exc:
            report.add_error(f"invalid benchmark {experiment!r}: {exc}")
            continue
        if computed != configured:
            report.add_error(
                f"benchmark {experiment} computes {computed} runs but config records "
                f"{configured}"
            )
        for resolution in resolutions:
            for wiggle in wiggles:
                for method in methods:
                    key = RunKey(
                        str(experiment).lower(),
                        str(method).lower(),
                        _canonical_number(resolution),
                        _canonical_number(wiggle),
                        seed,
                    )
                    if key in expected:
                        report.add_error(
                            f"duplicate configured run key: {key.display()}"
                        )
                    expected.add(key)
    return expected, trials


def _check_exact_counts(
    config: Mapping[str, object],
    expected_runs: int,
    trials: int,
    required_runs: int,
    required_cases: int,
    report: AuditReport,
) -> None:
    computed_cases = expected_runs * trials
    if expected_runs != required_runs:
        report.add_error(
            f"resolved config defines {expected_runs} runs; exactly {required_runs} required"
        )
    if computed_cases != required_cases:
        report.add_error(
            f"resolved config defines {computed_cases} cases; exactly {required_cases} required"
        )
    try:
        totals = config["planned_totals"]
        configured_runs = _parse_int(totals["runs"], "planned_totals.runs")
        configured_cases = _parse_int(totals["cases"], "planned_totals.cases")
    except (KeyError, TypeError, ReleaseAuditInputError) as exc:
        report.add_error(f"invalid planned_totals in resolved config: {exc}")
        return
    if configured_runs != required_runs:
        report.add_error(
            f"planned_totals.runs is {configured_runs}; exactly {required_runs} required"
        )
    if configured_cases != required_cases:
        report.add_error(
            f"planned_totals.cases is {configured_cases}; exactly {required_cases} required"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trusted_git_executable() -> Path:
    for candidate in TRUSTED_GIT_CANDIDATES:
        try:
            node = os.lstat(candidate)
        except OSError:
            continue
        if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
            continue
        if hasattr(os, "getuid") and node.st_uid != 0:
            continue
        if node.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            continue
        if not os.access(candidate, os.X_OK):
            continue
        return candidate
    raise ReleaseAuditInputError(
        "no absolute, regular, non-group/world-writable Git executable was found "
        f"in the fixed trust set: {[str(path) for path in TRUSTED_GIT_CANDIDATES]!r}"
    )


def _scrubbed_git_environment() -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_COUNT": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _run_git(
    repository: Path,
    arguments: Sequence[str],
    *,
    input_data: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    executable = _trusted_git_executable()
    environment = _scrubbed_git_environment()
    try:
        return subprocess.run(
            [
                str(executable),
                "--no-replace-objects",
                "-C",
                str(repository),
                *arguments,
            ],
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseAuditInputError(f"could not execute Git: {exc}") from exc


def _git_sha1(object_type: str, data: bytes) -> str:
    header = f"{object_type} {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _find_git_repository(root: Path, target_commit: str) -> Path:
    candidates = (root, Path(__file__).resolve().parents[1])
    repositories: list[Path] = []
    for candidate in candidates:
        result = _run_git(candidate, ("rev-parse", "--show-toplevel"))
        if result.returncode != 0:
            continue
        try:
            repository = Path(result.stdout.decode("utf-8").strip()).resolve()
        except UnicodeError:
            continue
        if repository not in repositories:
            repositories.append(repository)
    for repository in repositories:
        result = _run_git(
            repository,
            ("rev-parse", "--verify", "--quiet", f"{target_commit}^{{commit}}"),
        )
        if result.returncode == 0:
            resolved = result.stdout.decode("ascii", errors="strict").strip()
            if resolved == target_commit:
                commit = _run_git(repository, ("cat-file", "commit", target_commit))
                if (
                    commit.returncode == 0
                    and _git_sha1("commit", commit.stdout) == target_commit
                ):
                    return repository
    raise ReleaseAuditInputError(
        f"target_commit does not exist as an exact Git commit object: {target_commit!r}"
    )


def _git_tree_blobs(repository: Path, target_commit: str) -> dict[str, GitBlob]:
    commit = _run_git(repository, ("cat-file", "commit", target_commit))
    if commit.returncode != 0 or _git_sha1("commit", commit.stdout) != target_commit:
        raise ReleaseAuditInputError(
            "could not independently verify the exact target commit object"
        )
    first_line = commit.stdout.split(b"\n", 1)[0]
    if not re.fullmatch(rb"tree [0-9a-f]{40}", first_line):
        raise ReleaseAuditInputError("target commit has no canonical tree linkage")
    tree_id = first_line[5:].decode("ascii")
    tree_object = _run_git(repository, ("cat-file", "tree", tree_id))
    if tree_object.returncode != 0 or _git_sha1("tree", tree_object.stdout) != tree_id:
        raise ReleaseAuditInputError(
            "target commit's root tree object failed independent hash verification"
        )
    result = _run_git(
        repository,
        ("ls-tree", "-r", "-z", "--full-tree", tree_id),
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseAuditInputError(
            f"could not enumerate target_commit tree: {detail}"
        )

    entries: list[tuple[str, str, str]] = []
    paths: set[str] = set()
    try:
        records = result.stdout.split(b"\0")
        if records[-1] != b"":
            raise ReleaseAuditInputError("Git ls-tree output is not NUL terminated")
        for record in records[:-1]:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
            pure = PurePosixPath(path)
            if (
                not path
                or pure.is_absolute()
                or ".." in pure.parts
                or str(pure) != path
            ):
                raise ReleaseAuditInputError(
                    f"unsafe path in target_commit tree: {path!r}"
                )
            if path in paths:
                raise ReleaseAuditInputError(
                    f"duplicate path in target_commit tree: {path!r}"
                )
            if object_type != "blob" or mode not in {"100644", "100755"}:
                raise ReleaseAuditInputError(
                    f"unsupported tracked source entry {path!r}: "
                    f"mode={mode!r}, type={object_type!r}"
                )
            if not re.fullmatch(r"[0-9a-f]{40}", object_id):
                raise ReleaseAuditInputError(
                    f"invalid blob object ID for tracked source entry {path!r}"
                )
            paths.add(path)
            entries.append((path, mode, object_id))
    except (UnicodeError, ValueError) as exc:
        raise ReleaseAuditInputError(
            f"could not parse Git ls-tree output: {exc}"
        ) from exc

    request = b"".join(object_id.encode("ascii") + b"\n" for _, _, object_id in entries)
    blobs = _run_git(repository, ("cat-file", "--batch"), input_data=request)
    if blobs.returncode != 0:
        detail = blobs.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseAuditInputError(f"could not read target_commit blobs: {detail}")

    members: dict[str, GitBlob] = {}
    cursor = 0
    try:
        for path, mode, expected_object_id in entries:
            line_end = blobs.stdout.index(b"\n", cursor)
            header = blobs.stdout[cursor:line_end].decode("ascii")
            object_id, object_type, raw_size = header.split(" ")
            if object_id != expected_object_id or object_type != "blob":
                raise ReleaseAuditInputError(
                    f"Git cat-file returned the wrong object for {path!r}"
                )
            size = int(raw_size)
            if size < 0:
                raise ReleaseAuditInputError(
                    f"Git cat-file returned a negative size for {path!r}"
                )
            data_start = line_end + 1
            data_end = data_start + size
            if data_end >= len(blobs.stdout) or blobs.stdout[data_end] != 0x0A:
                raise ReleaseAuditInputError(
                    f"Git cat-file returned a truncated blob for {path!r}"
                )
            data = blobs.stdout[data_start:data_end]
            if _git_sha1("blob", data) != object_id:
                raise ReleaseAuditInputError(
                    f"Git blob hash verification failed for {path!r}"
                )
            members[path] = GitBlob(mode, object_id, data)
            cursor = data_end + 1
    except (UnicodeError, ValueError) as exc:
        raise ReleaseAuditInputError(
            f"could not parse Git cat-file output: {exc}"
        ) from exc
    if cursor != len(blobs.stdout):
        raise ReleaseAuditInputError("Git cat-file returned unexpected trailing data")
    return members


def _tar_padded_size(size: int) -> int:
    return ((size + TAR_BLOCK_SIZE - 1) // TAR_BLOCK_SIZE) * TAR_BLOCK_SIZE


def _source_tar_decompressed_budget(expected: Mapping[str, GitBlob]) -> int:
    canonical_members = sum(
        TAR_BLOCK_SIZE + _tar_padded_size(len(blob.data)) for blob in expected.values()
    )
    per_member_metadata = len(expected) * (
        TAR_BLOCK_SIZE + _tar_padded_size(MAX_TAR_METADATA_BYTES_PER_MEMBER)
    )
    return (
        canonical_members
        + per_member_metadata
        + max(TAR_RECORD_SIZE, MAX_TAR_GLOBAL_METADATA_BYTES)
    )


def _validate_single_gzip_member(compressed_stream, byte_budget: int) -> int:
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    decompressed_bytes = 0
    while True:
        compressed = compressed_stream.read(GZIP_VALIDATION_CHUNK_SIZE)
        if not compressed:
            break
        pending = compressed
        while pending:
            maximum_output = min(
                GZIP_VALIDATION_CHUNK_SIZE,
                byte_budget - decompressed_bytes + 1,
            )
            try:
                output = decompressor.decompress(pending, maximum_output)
            except zlib.error as exc:
                raise SourceSnapshotFormatError(
                    f"gzip source snapshot CRC/trailer validation failed: {exc}"
                ) from exc
            decompressed_bytes += len(output)
            if decompressed_bytes > byte_budget:
                raise DecompressedSnapshotBudgetExceeded(
                    "decompressed source snapshot exceeds Git-derived byte budget "
                    f"of {byte_budget} bytes"
                )
            pending = decompressor.unconsumed_tail
            if decompressor.eof:
                if decompressor.unused_data or pending or compressed_stream.read(1):
                    raise SourceSnapshotFormatError(
                        "gzip source snapshot contains multiple members or trailing "
                        "compressed data"
                    )
                return decompressed_bytes
    raise SourceSnapshotFormatError(
        "gzip source snapshot is truncated or missing its CRC/trailer"
    )


def _canonical_tar_end(data_end: int) -> int:
    minimum_end = data_end + 2 * TAR_BLOCK_SIZE
    return ((minimum_end + TAR_RECORD_SIZE - 1) // TAR_RECORD_SIZE) * TAR_RECORD_SIZE


def _read_exact_tar_bytes(
    stream: _DecompressedByteBudgetStream,
    offset: int,
    size: int,
    description: str,
) -> bytes:
    stream.seek(offset)
    data = stream.read(size)
    if len(data) != size:
        raise SourceSnapshotFormatError(
            f"source tar is truncated while reading {description} at offset {offset}"
        )
    return data


def _validate_zero_tar_padding(
    stream: _DecompressedByteBudgetStream,
    offset: int,
    size: int,
    description: str,
) -> None:
    if size == 0:
        return
    padding = _read_exact_tar_bytes(stream, offset, size, description)
    for relative_offset, value in enumerate(padding):
        if value:
            raise SourceSnapshotFormatError(
                f"source tar contains nonzero {description} byte at offset "
                f"{offset + relative_offset}"
            )


def _validate_tar_member_padding(
    stream: _DecompressedByteBudgetStream,
    members: Sequence[tarfile.TarInfo],
    encoding: str,
    errors: str,
) -> int:
    cursor = 0
    for member in members:
        member_header_offset = member.offset_data - TAR_BLOCK_SIZE
        if member_header_offset < cursor:
            raise SourceSnapshotFormatError(
                f"source tar member layout overlaps before {member.name!r}"
            )

        while cursor < member_header_offset:
            header = _read_exact_tar_bytes(
                stream,
                cursor,
                TAR_BLOCK_SIZE,
                f"extension header before {member.name!r}",
            )
            try:
                extension = tarfile.TarInfo.frombuf(header, encoding, errors)
            except tarfile.HeaderError as exc:
                raise SourceSnapshotFormatError(
                    f"invalid source tar extension header at offset {cursor}: {exc}"
                ) from exc
            if extension.type not in TAR_EXTENSION_TYPES:
                raise SourceSnapshotFormatError(
                    "source tar contains an unexpected physical record before "
                    f"{member.name!r} at offset {cursor}"
                )
            if extension.size < 0:
                raise SourceSnapshotFormatError(
                    f"source tar extension has a negative size at offset {cursor}"
                )
            if extension.size > MAX_TAR_METADATA_BYTES_PER_MEMBER:
                raise SourceSnapshotFormatError(
                    "source tar extension metadata size exceeds per-extension "
                    f"limit at offset {cursor}: {extension.size} > "
                    f"{MAX_TAR_METADATA_BYTES_PER_MEMBER}"
                )
            extension_data_offset = cursor + TAR_BLOCK_SIZE
            extension_padded_end = extension_data_offset + _tar_padded_size(
                extension.size
            )
            if extension_padded_end > member_header_offset:
                raise SourceSnapshotFormatError(
                    "source tar extension overlaps the following member header "
                    f"before {member.name!r}"
                )
            _validate_zero_tar_padding(
                stream,
                extension_data_offset + extension.size,
                extension_padded_end - extension_data_offset - extension.size,
                "extension-member padding",
            )
            cursor = extension_padded_end

        if cursor != member_header_offset:
            raise SourceSnapshotFormatError(
                f"source tar member header is misaligned for {member.name!r}"
            )

        member_padded_end = member.offset_data + _tar_padded_size(member.size)
        _validate_zero_tar_padding(
            stream,
            member.offset_data + member.size,
            member_padded_end - member.offset_data - member.size,
            f"padding after member {member.name!r}",
        )
        cursor = member_padded_end
    return cursor


def _drain_and_validate_tar_termination(
    stream: _DecompressedByteBudgetStream,
    data_end: int,
    preflight_size: int,
) -> None:
    expected_end = _canonical_tar_end(data_end)
    stream.seek(data_end)
    trailing_bytes = 0
    has_nonzero_data = False
    while True:
        chunk = stream.read(GZIP_VALIDATION_CHUNK_SIZE)
        if not chunk:
            break
        trailing_bytes += len(chunk)
        has_nonzero_data = has_nonzero_data or any(chunk)
    actual_end = data_end + trailing_bytes
    if actual_end != preflight_size:
        raise SourceSnapshotFormatError(
            "gzip source snapshot decompressed size changed between validation passes: "
            f"{actual_end} != {preflight_size}"
        )
    if has_nonzero_data:
        raise SourceSnapshotFormatError(
            "source tar contains nonzero trailing decompressed data after the final "
            "member"
        )
    if actual_end != expected_end:
        raise SourceSnapshotFormatError(
            "source tar is missing or exceeds its canonical two-zero-block, "
            f"record-aligned terminator: {actual_end} != {expected_end}"
        )


def _read_bounded_source_snapshot(
    snapshot_path: Path,
    expected: Mapping[str, GitBlob],
    report: AuditReport,
) -> tuple[dict[str, bytes], dict[str, int]]:
    members: dict[str, bytes] = {}
    modes: dict[str, int] = {}
    maximum_members = len(expected)
    maximum_total_bytes = sum(len(blob.data) for blob in expected.values())
    declared_total = 0
    member_count = 0
    metadata_errors = report.total_errors
    decompressed_budget = _source_tar_decompressed_budget(expected)
    try:
        with snapshot_path.open("rb") as compressed_stream:
            preflight_size = _validate_single_gzip_member(
                compressed_stream, decompressed_budget
            )
            compressed_stream.seek(0)
            with gzip.GzipFile(fileobj=compressed_stream, mode="rb") as gzip_stream:
                bounded_stream = _DecompressedByteBudgetStream(
                    gzip_stream, decompressed_budget
                )
                with tarfile.open(fileobj=bounded_stream, mode="r:") as archive:
                    validated_members: list[tarfile.TarInfo] = []
                    seen_paths: set[str] = set()
                    for member in archive:
                        member_count += 1
                        declared_total += max(member.size, 0)
                        if member_count > maximum_members:
                            report.add_error(
                                "source snapshot member count exceeds target_commit "
                                f"tree bound: {member_count} > {maximum_members}"
                            )
                            break
                        if declared_total > maximum_total_bytes:
                            report.add_error(
                                "source snapshot total uncompressed bytes exceed "
                                f"target_commit tree bound: {declared_total} > "
                                f"{maximum_total_bytes}"
                            )
                            break
                        pure = PurePosixPath(member.name)
                        if (
                            not member.name
                            or pure.is_absolute()
                            or ".." in pure.parts
                            or str(pure) != member.name
                        ):
                            report.add_error(
                                f"unsafe path in source snapshot: {member.name!r}"
                            )
                            continue
                        if not member.isfile():
                            report.add_error(
                                f"non-regular entry in source snapshot: {member.name!r}"
                            )
                            continue
                        if member.name in seen_paths:
                            report.add_error(
                                f"duplicate file in source snapshot: {member.name}"
                            )
                            continue
                        seen_paths.add(member.name)
                        expected_blob = expected.get(member.name)
                        if expected_blob is None:
                            report.add_error(
                                "source snapshot contains an unbudgeted path absent "
                                f"from target_commit: {member.name!r}"
                            )
                            continue
                        expected_size = len(expected_blob.data)
                        if member.size != expected_size:
                            report.add_error(
                                "source snapshot member size exceeds or differs from "
                                f"target_commit bound for {member.name}: "
                                f"{member.size} != {expected_size}"
                            )
                            continue
                        expected_mode = int(expected_blob.mode[-3:], 8)
                        actual_mode = member.mode
                        if actual_mode != expected_mode:
                            report.add_error(
                                "source snapshot complete mode differs from "
                                f"target_commit for {member.name}: "
                                f"{actual_mode:o} != {expected_mode:o}"
                            )
                            continue
                        validated_members.append(member)

                    if report.total_errors != metadata_errors:
                        return {}, {}
                    if not validated_members:
                        raise SourceSnapshotFormatError(
                            "source tar contains no validated tracked-file members"
                        )

                    data_end = _validate_tar_member_padding(
                        bounded_stream,
                        validated_members,
                        archive.encoding,
                        archive.errors,
                    )

                    for member in validated_members:
                        expected_blob = expected[member.name]
                        expected_size = len(expected_blob.data)
                        stream = archive.extractfile(member)
                        if stream is None:
                            report.add_error(
                                f"could not read source snapshot member: {member.name}"
                            )
                            continue
                        data = stream.read(expected_size + 1)
                        if len(data) != expected_size:
                            report.add_error(
                                "source snapshot member read length differs from "
                                f"declared size for {member.name}"
                            )
                            continue
                        members[member.name] = data
                        modes[member.name] = member.mode
                    _drain_and_validate_tar_termination(
                        bounded_stream, data_end, preflight_size
                    )
    except (EOFError, OSError, tarfile.TarError) as exc:
        report.add_error(f"could not read source snapshot: {exc}")
    return members, modes


def _report_path_set_difference(
    report: AuditReport,
    label: str,
    paths: set[str],
) -> None:
    if not paths:
        return
    sample = ", ".join(repr(path) for path in sorted(paths)[:10])
    suffix = f" and {len(paths) - 10} more" if len(paths) > 10 else ""
    report.add_error(f"{label}: {sample}{suffix}")


def _check_archived_benchmark_configs(
    config: Mapping[str, object],
    snapshot_members: Mapping[str, bytes],
    report: AuditReport,
) -> None:
    benchmarks = config.get("benchmarks")
    grid = config.get("benchmark_grid")
    numerics = config.get("numerics")
    if not all(isinstance(value, Mapping) for value in (benchmarks, grid, numerics)):
        report.add_error(
            "resolved benchmark_grid/benchmarks/numerics cannot bind archived configs"
        )
        return
    assert isinstance(benchmarks, Mapping)
    assert isinstance(grid, Mapping)
    assert isinstance(numerics, Mapping)
    if grid.get("perturbation_type") != "random":
        report.add_error("resolved perturbation_type is not the production 'random'")
    if grid.get("fix_boundary_nodes") is not True:
        report.add_error("resolved fix_boundary_nodes is not true")

    for experiment, raw_benchmark in sorted(benchmarks.items()):
        label = f"benchmark {experiment} archived config"
        if not isinstance(raw_benchmark, Mapping):
            report.add_error(f"{label} declaration is not an object")
            continue
        relative = raw_benchmark.get("config")
        if not isinstance(relative, str):
            report.add_error(f"{label} path is unavailable")
            continue
        try:
            _declared_child_config_path(raw_benchmark)
        except ReleaseAuditInputError as exc:
            report.add_error(f"{label}: {exc}")
            continue
        data = snapshot_members.get(relative)
        if data is None:
            report.add_error(f"{label} is absent from the source snapshot: {relative}")
            continue
        try:
            parsed = yaml.safe_load(data.decode("utf-8"))
        except (UnicodeError, yaml.YAMLError) as exc:
            report.add_error(f"{label} cannot be parsed: {exc}")
            continue
        if not isinstance(parsed, Mapping):
            report.add_error(f"{label} root is not an object")
            continue
        mesh = parsed.get("MESH")
        geoms = parsed.get("GEOMS")
        if not isinstance(mesh, Mapping) or not isinstance(geoms, Mapping):
            report.add_error(f"{label} lacks MESH or GEOMS settings")
            continue
        perturb = mesh.get("PERTURB")
        if not isinstance(perturb, Mapping):
            report.add_error(f"{label} lacks MESH.PERTURB settings")
            continue
        for actual, expected, field_name in (
            (mesh.get("GRID_SIZE"), grid.get("grid_size"), "MESH.GRID_SIZE"),
            (perturb.get("SEED"), grid.get("seed"), "MESH.PERTURB.SEED"),
            (
                geoms.get("THRESHOLD"),
                numerics.get("mesh_fraction_threshold"),
                "GEOMS.THRESHOLD",
            ),
        ):
            if not _values_match(actual, expected, numeric=True):
                report.add_error(
                    f"{label} {field_name} differs from resolved configuration: "
                    f"{actual!r} != {expected!r}"
                )
        if perturb.get("FIX_BOUNDARY") is not grid.get("fix_boundary_nodes"):
            report.add_error(
                f"{label} MESH.PERTURB.FIX_BOUNDARY differs from resolved configuration"
            )
        if geoms.get("DO_C0") is not False:
            report.add_error(f"{label} GEOMS.DO_C0 is not false")


def _check_source_provenance(
    root: Path,
    config: dict,
    report: AuditReport,
) -> tuple[str, dict[str, bytes], bool, PurePosixPath | None]:
    initial_errors = report.total_errors
    target_commit = str(config.get("source", {}).get("target_commit", ""))
    target_branch = str(config.get("source", {}).get("target_branch", ""))
    if config.get("status") != "frozen":
        report.add_error("resolved config status is not 'frozen'")
    if config.get("launch_approved") is not True:
        report.add_error("resolved config launch_approved is not true")
    if not target_commit:
        report.add_error("resolved config source.target_commit is empty")
    elif not re.fullmatch(r"[0-9a-f]{40}", target_commit):
        report.add_error("resolved config source.target_commit is not a full SHA-1")
    elif target_commit != FINAL_SOURCE_COMMIT:
        report.add_error(
            f"resolved config target_commit is not the approved production commit: "
            f"{target_commit!r} != {FINAL_SOURCE_COMMIT!r}"
        )

    state_path = root / "diagnostics" / "source_state.json"
    state = _load_json(state_path, report)
    snapshot_path = root / "diagnostics" / "source_snapshot.tar.gz"
    environment = _load_json(root / "environment.json", report)
    historical_root: PurePosixPath | None = None
    if state:
        if state.get("source_commit") != target_commit:
            report.add_error("source_state commit does not match resolved config")
        if state.get("source_branch") != target_branch:
            report.add_error("source_state branch does not match resolved config")
        if state.get("source_dirty") is not False or state.get("source_status"):
            report.add_error("source_state does not record a clean source tree")
        if snapshot_path.is_file():
            try:
                actual_digest = _sha256(snapshot_path)
            except OSError as exc:
                report.add_error(f"could not hash source snapshot: {exc}")
            else:
                if state.get("snapshot_sha256") != actual_digest:
                    report.add_error(
                        "source snapshot SHA-256 does not match source_state"
                    )

    if environment:
        repository = environment.get("repository")
        if not isinstance(repository, dict):
            report.add_error("environment.json has no repository object")
        else:
            if repository.get("commit") != target_commit:
                report.add_error("environment commit does not match resolved config")
            if repository.get("branch") != target_branch:
                report.add_error("environment branch does not match resolved config")
            if repository.get("source_dirty") is not False:
                report.add_error("environment records a dirty source tree")
            try:
                historical_root = _historical_repository_root(repository.get("root"))
            except ReleaseAuditInputError as exc:
                report.add_error(str(exc))

    git_members: dict[str, GitBlob] = {}
    if re.fullmatch(r"[0-9a-f]{40}", target_commit):
        try:
            git_repository = _find_git_repository(root, target_commit)
            git_members = _git_tree_blobs(git_repository, target_commit)
        except (ReleaseAuditInputError, UnicodeError) as exc:
            report.add_error(str(exc))
    expected_snapshot = {
        path: blob
        for path, blob in git_members.items()
        if PurePosixPath(path).parts[0] not in SNAPSHOT_EXCLUDED_ROOTS
    }

    snapshot_members: dict[str, bytes] = {}
    if snapshot_path.is_file() and expected_snapshot:
        snapshot_members, _ = _read_bounded_source_snapshot(
            snapshot_path, expected_snapshot, report
        )

    if state:
        excluded_roots = state.get("excluded_roots")
        if (
            not isinstance(excluded_roots, list)
            or set(excluded_roots) != SNAPSHOT_EXCLUDED_ROOTS
            or len(excluded_roots) != len(SNAPSHOT_EXCLUDED_ROOTS)
        ):
            report.add_error(
                "source_state excluded_roots does not match the fixed generated-root "
                "policy"
            )
    if state:
        try:
            recorded_file_count = _parse_int(
                state.get("snapshot_file_count"), "source_state snapshot_file_count"
            )
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
        else:
            if recorded_file_count != len(snapshot_members):
                report.add_error(
                    "source snapshot file count does not match source_state"
                )

    original_config_bytes = snapshot_members.get("submission/submission_config.json")
    if original_config_bytes is None:
        report.add_error("source snapshot lacks submission/submission_config.json")
    else:
        try:
            original_config = _strict_json_loads(
                original_config_bytes.decode("utf-8"),
                "source snapshot submission/submission_config.json",
            )
        except (UnicodeError, ReleaseAuditInputError) as exc:
            report.add_error(str(exc))
        else:
            if not isinstance(original_config, dict):
                report.add_error("source snapshot submission config is not an object")
            else:
                expected_resolved = copy.deepcopy(original_config)
                expected_resolved["status"] = "frozen"
                expected_resolved.setdefault("source", {})[
                    "target_commit"
                ] = target_commit
                if expected_resolved != config:
                    report.add_error(
                        "resolved config differs from the snapshotted config beyond "
                        "status and source.target_commit"
                    )

    _check_archived_benchmark_configs(config, snapshot_members, report)

    if environment:
        fingerprints = environment.get("input_fingerprints")
        if not isinstance(fingerprints, list):
            report.add_error("environment input_fingerprints is not a list")
        else:
            seen_fingerprints: set[str] = set()
            for index, fingerprint in enumerate(fingerprints):
                if not isinstance(fingerprint, dict):
                    report.add_error(
                        f"environment fingerprint {index} is not an object"
                    )
                    continue
                relative = str(fingerprint.get("path", ""))
                if relative in seen_fingerprints:
                    report.add_error(f"duplicate environment fingerprint: {relative}")
                    continue
                seen_fingerprints.add(relative)
                data = snapshot_members.get(relative)
                if data is None:
                    report.add_error(
                        f"environment fingerprint path is absent from source snapshot: "
                        f"{relative}"
                    )
                    continue
                digest = hashlib.sha256(data).hexdigest()
                if fingerprint.get("sha256") != digest:
                    report.add_error(
                        f"environment fingerprint digest mismatch: {relative}"
                    )
                try:
                    recorded_size = _parse_int(
                        fingerprint.get("size_bytes"),
                        f"environment fingerprint size for {relative}",
                    )
                except ReleaseAuditInputError as exc:
                    report.add_error(str(exc))
                else:
                    if recorded_size != len(data):
                        report.add_error(
                            f"environment fingerprint size mismatch: {relative}"
                        )
            if "submission/submission_config.json" not in seen_fingerprints:
                report.add_error(
                    "environment lacks a fingerprint for submission/submission_config.json"
                )

    _report_path_set_difference(
        report,
        "source snapshot is missing tracked files from target_commit",
        set(expected_snapshot) - set(snapshot_members),
    )
    _report_path_set_difference(
        report,
        "source snapshot contains files not tracked by target_commit",
        set(snapshot_members) - set(expected_snapshot),
    )
    for path in sorted(set(expected_snapshot) & set(snapshot_members)):
        if snapshot_members[path] != expected_snapshot[path].data:
            report.add_error(
                f"source snapshot bytes differ from target_commit for {path}"
            )

    for path, expected_digest in PRODUCTION_SOURCE_SHA256.items():
        data = snapshot_members.get(path)
        if data is None:
            report.add_error(
                f"source snapshot lacks production fingerprint file: {path}"
            )
            continue
        actual_digest = hashlib.sha256(data).hexdigest()
        if actual_digest != expected_digest:
            report.add_error(
                f"production source fingerprint mismatch for {path}: "
                f"{actual_digest} != {expected_digest}"
            )

    return (
        target_commit,
        snapshot_members,
        report.total_errors == initial_errors,
        historical_root,
    )


def _check_controller(
    root: Path,
    required_runs: int,
    required_cases: int,
    production_context: Mapping[str, str],
    historical_root: PurePosixPath | None,
    legacy_command_strings_allowed: bool,
    report: AuditReport,
) -> tuple[dict | None, bool]:
    manifest = _load_json(root / "sweep_manifest.json", report)
    profiles_verified = bool(manifest)
    if manifest:
        expected_values = {
            "planned_run_count": required_runs,
            "planned_case_count": required_cases,
            "successful_run_count": required_runs,
            "failure_count": 0,
        }
        if manifest.get("status") != "completed":
            report.add_error(
                f"controller status is {manifest.get('status')!r}, not 'completed'"
            )
        for field_name, expected in expected_values.items():
            try:
                actual = _parse_int(manifest.get(field_name), field_name)
            except ReleaseAuditInputError as exc:
                report.add_error(str(exc))
                continue
            if actual != expected:
                report.add_error(
                    f"controller {field_name} is {actual}; expected {expected}"
                )
        failures = manifest.get("failures")
        if failures != []:
            report.add_error("controller manifest failures list is not empty")
        parameters = manifest.get("parameters")
        if not isinstance(parameters, Mapping):
            report.add_error("controller manifest parameters are unavailable")
            profiles_verified = False
        try:
            tokens = _manifest_command_tokens(
                manifest,
                allow_legacy_string=legacy_command_strings_allowed,
            )
        except ReleaseAuditInputError as exc:
            report.add_error(f"controller command evidence is invalid: {exc}")
            tokens = []
            profiles_verified = False
        if tokens and not _command_invokes_driver(
            tokens, PRODUCTION_CONTROLLER_MODULE, historical_root
        ):
            report.add_error(
                "controller command does not invoke the reviewed sweep controller"
            )
            profiles_verified = False
        for field_name, option in PRODUCTION_COMMAND_OPTIONS.items():
            expected = production_context.get(field_name, "")
            actual = (
                parameters.get(field_name) if isinstance(parameters, Mapping) else None
            )
            if actual != expected:
                report.add_error(
                    f"controller parameter {field_name} differs from production: "
                    f"{actual!r} != {expected!r}"
                )
                profiles_verified = False
            try:
                values = _option_values(tokens, option) if tokens else []
            except ReleaseAuditInputError as exc:
                report.add_error(f"controller {option} evidence is invalid: {exc}")
                profiles_verified = False
                continue
            if values != [expected]:
                report.add_error(
                    f"controller command does not contain exactly one matching "
                    f"{option}: {values!r} != {[expected]!r}"
                )
                profiles_verified = False

    _, failure_rows = _read_csv_rows(
        root / "failures.csv",
        ("experiment", "algo", "resolution", "wiggle", "seed", "save_name"),
        report,
    )
    if failure_rows:
        report.add_error(
            f"failures.csv contains {len(failure_rows)} controller failures"
        )
    return manifest, profiles_verified


def _check_context_source(
    row: Mapping[str, object],
    target_commit: str,
    target_branch: str,
    label: str,
    report: AuditReport,
) -> None:
    if row.get("source_commit") != target_commit:
        report.add_error(f"{label} source_commit does not match final source commit")
    if row.get("source_branch") != target_branch:
        report.add_error(f"{label} source_branch does not match final source branch")


def _check_inventory(
    root: Path,
    expected_runs: set[RunKey],
    trials: int,
    target_commit: str,
    target_branch: str,
    production_context: Mapping[str, str],
    inheritance: RescueProfileInheritance,
    report: AuditReport,
) -> dict[RunKey, dict[str, str]]:
    path = root / "diagnostics" / "run_inventory.csv"
    required = set(RUN_CONTEXT_FIELDS) | {
        "run_bundle",
        "case_geometry_rows",
        "case_metrics_rows",
        "cell_metrics_rows",
        "merge_events_rows",
        "unresolved_plic_fallbacks_rows",
    }
    _, rows = _read_csv_rows(path, required, report)
    inventory: dict[RunKey, dict[str, str]] = {}
    save_names: set[str] = set()
    bundle_paths: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        label = f"{path}:{row_number}"
        try:
            key = _run_key(row, label)
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        if key in inventory:
            report.add_error(f"duplicate run key in inventory: {key.display()}")
            continue
        inventory[key] = row
        if key not in expected_runs:
            report.add_error(f"unexpected run key in inventory: {key.display()}")
        save_name = row.get("save_name", "")
        if not save_name:
            report.add_error(f"inventory run has empty save_name: {key.display()}")
        elif save_name in save_names:
            report.add_error(f"duplicate inventory save_name: {save_name}")
        save_names.add(save_name)
        bundle_value = row.get("run_bundle", "")
        if bundle_value in bundle_paths:
            report.add_error(f"duplicate inventory run_bundle: {bundle_value}")
        bundle_paths.add(bundle_value)
        try:
            bundle = _safe_release_path(root, bundle_value, f"{label} run_bundle")
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
        else:
            expected_bundle = root / "raw_runs" / str(save_name)
            if bundle != expected_bundle:
                report.add_error(
                    f"inventory bundle for {key.display()} must be "
                    f"raw_runs/{save_name}"
                )
            if not bundle.is_dir():
                report.add_error(f"inventory raw bundle is missing: {bundle}")
        _check_context_source(row, target_commit, target_branch, label, report)
        _check_production_context(
            row,
            production_context,
            f"inventory {key.display()}",
            report,
            key=key,
            inheritance=inheritance,
            inheritance_location="inventory",
        )
        for count_field in (
            "case_geometry_rows",
            "case_metrics_rows",
            "cell_metrics_rows",
            "merge_events_rows",
            "unresolved_plic_fallbacks_rows",
        ):
            try:
                count = _parse_int(row.get(count_field), f"{label} {count_field}")
            except ReleaseAuditInputError as exc:
                report.add_error(str(exc))
                continue
            if count < 0:
                report.add_error(f"{label} {count_field} is negative")
            if (
                count_field in {"case_geometry_rows", "case_metrics_rows"}
                and count != trials
            ):
                report.add_error(f"{label} {count_field} is {count}; expected {trials}")

    missing = expected_runs - set(inventory)
    for key in sorted(missing):
        report.add_error(f"missing run key from inventory: {key.display()}")
    report.summaries["inventory_runs"] = len(rows)
    return inventory


def _check_consolidated_run_manifests(
    root: Path,
    config: Mapping[str, object],
    expected_runs: set[RunKey],
    inventory: Mapping[RunKey, Mapping[str, str]],
    target_commit: str,
    target_branch: str,
    production_context: Mapping[str, str],
    inheritance: RescueProfileInheritance,
    report: AuditReport,
) -> dict[RunKey, dict]:
    path = root / "diagnostics" / "run_manifests.jsonl"
    seen: set[RunKey] = set()
    save_names: set[str] = set()
    manifests: dict[RunKey, dict] = {}
    for line_number, row in _iter_jsonl(path, report):
        label = f"{path}:{line_number}"
        try:
            key = _run_key(row, label)
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        if key in seen:
            report.add_error(f"duplicate run key in run_manifests: {key.display()}")
        else:
            manifests[key] = row
        seen.add(key)
        if key not in expected_runs:
            report.add_error(f"unexpected run manifest key: {key.display()}")
        save_name = str(row.get("save_name", ""))
        if save_name in save_names:
            report.add_error(f"duplicate save_name in run_manifests: {save_name}")
        save_names.add(save_name)
        inventory_row = inventory.get(key)
        if inventory_row and save_name != inventory_row.get("save_name"):
            report.add_error(
                f"run manifest save_name disagrees with inventory: {key.display()}"
            )
        _check_context_source(row, target_commit, target_branch, label, report)
        _check_production_context(
            row,
            production_context,
            f"consolidated run manifest {key.display()}",
            report,
            key=key,
            inheritance=inheritance,
            inheritance_location="consolidated run-manifest context",
        )
        manifest = row.get("manifest")
        if not isinstance(manifest, dict):
            report.add_error(f"{label} has no nested manifest object")
            continue
        if manifest.get("source_commit") != target_commit:
            report.add_error(f"nested run manifest commit mismatch: {key.display()}")
        if manifest.get("source_branch") != target_branch:
            report.add_error(f"nested run manifest branch mismatch: {key.display()}")
        if str(manifest.get("experiment", "")).lower() != key.experiment:
            report.add_error(
                f"nested run manifest experiment mismatch: {key.display()}"
            )
        parameters = manifest.get("parameters")
        if not isinstance(parameters, dict):
            report.add_error(f"nested run manifest parameters missing: {key.display()}")
            continue
        _check_production_context(
            parameters,
            production_context,
            f"nested run manifest parameters {key.display()}",
            report,
            key=key,
            inheritance=inheritance,
            inheritance_location="nested run-manifest parameters",
        )
        _check_child_manifest_command(
            manifest,
            key,
            production_context,
            inheritance,
            "consolidated child manifest",
            report,
        )
        _check_child_manifest_contract(
            manifest,
            key,
            config,
            save_name,
            inheritance,
            "consolidated child manifest",
            report,
        )
        try:
            nested_key = RunKey(
                key.experiment,
                str(parameters.get("facet_algo", "")).lower(),
                _canonical_number(parameters.get("resolution")),
                _canonical_number(parameters.get("perturb_wiggle")),
                _parse_int(parameters.get("perturb_seed"), "perturb_seed"),
            )
        except ReleaseAuditInputError as exc:
            report.add_error(f"invalid nested run manifest for {key.display()}: {exc}")
        else:
            if nested_key != key:
                report.add_error(f"nested run manifest key mismatch: {key.display()}")

    for key in sorted(expected_runs - seen):
        report.add_error(f"missing consolidated run manifest: {key.display()}")
    return manifests


def _case_key(row: Mapping[str, object], source: str) -> tuple[RunKey, int]:
    key = _run_key(row, source)
    if row.get("case_index") in (None, ""):
        raise ReleaseAuditInputError(f"{source} has no case_index")
    return key, _parse_int(row["case_index"], f"{source} case_index")


def _require_numeric_close(
    actual: object,
    expected: object,
    label: str,
    report: AuditReport,
    *,
    tolerance: float = 1e-12,
) -> None:
    try:
        actual_number = float(_bounded_decimal(actual, label))
        expected_number = float(_bounded_decimal(expected, f"{label} expected"))
    except (ReleaseAuditInputError, ValueError, OverflowError) as exc:
        report.add_error(str(exc))
        return
    if not math.isclose(actual_number, expected_number, rel_tol=0.0, abs_tol=tolerance):
        report.add_error(
            f"{label} differs from the resolved benchmark geometry: "
            f"{actual!r} != {expected!r}"
        )


def _point_coordinates(
    value: object, label: str, report: AuditReport
) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        report.add_error(f"{label} is not a two-coordinate point")
        return None
    coordinates: list[float] = []
    for coordinate_index, coordinate in enumerate(value):
        try:
            number = float(_bounded_decimal(coordinate, f"{label}[{coordinate_index}]"))
        except (ReleaseAuditInputError, ValueError, OverflowError) as exc:
            report.add_error(str(exc))
            return None
        coordinates.append(number)
    return coordinates[0], coordinates[1]


def _check_point(value: object, label: str, report: AuditReport) -> None:
    _point_coordinates(value, label, report)


def _check_point_in_unit_center_box(
    value: object, label: str, report: AuditReport
) -> tuple[float, float] | None:
    point = _point_coordinates(value, label, report)
    if point is not None and not all(
        50.0 <= coordinate <= 51.0 for coordinate in point
    ):
        report.add_error(f"{label} is outside the frozen [50, 51]^2 sampling box")
    return point


def _require_point_close(
    actual: object,
    expected: tuple[float, float],
    label: str,
    report: AuditReport,
    *,
    tolerance: float = 1e-10,
    relative_tolerance: float = 1e-12,
) -> None:
    point = _point_coordinates(actual, label, report)
    if point is None:
        return
    for coordinate_index, (actual_coordinate, expected_coordinate) in enumerate(
        zip(point, expected)
    ):
        if not math.isclose(
            actual_coordinate,
            expected_coordinate,
            rel_tol=relative_tolerance,
            abs_tol=tolerance,
        ):
            report.add_error(
                f"{label}[{coordinate_index}] differs from the resolved benchmark "
                f"geometry: {actual_coordinate!r} != {expected_coordinate!r}"
            )


def _rotated_point(
    point: tuple[float, float], center: tuple[float, float], theta: float
) -> tuple[float, float]:
    x = point[0] - center[0]
    y = point[1] - center[1]
    return (
        x * math.cos(theta) - y * math.sin(theta) + center[0],
        x * math.sin(theta) + y * math.cos(theta) + center[1],
    )


def _check_case_geometry_contract(
    row: Mapping[str, object],
    key: RunKey,
    case_index: int,
    trials: int,
    config: Mapping[str, object],
    label: str,
    report: AuditReport,
) -> None:
    benchmarks = config.get("benchmarks")
    if not isinstance(benchmarks, Mapping):
        report.add_error(f"{label} cannot resolve benchmark geometry")
        return
    benchmark = benchmarks.get(key.experiment)
    if not isinstance(benchmark, Mapping):
        report.add_error(f"{label} has no benchmark geometry declaration")
        return
    expected_type = BENCHMARK_GEOMETRY_TYPES.get(key.experiment)
    if row.get("geometry_type") != expected_type:
        report.add_error(
            f"{label} geometry_type differs from benchmark: "
            f"{row.get('geometry_type')!r} != {expected_type!r}"
        )

    if key.experiment == "lines":
        expected_angle = case_index * 2.0 * math.pi / trials
        _require_numeric_close(
            row.get("angle"), expected_angle, f"{label} angle", report
        )
        left = _check_point_in_unit_center_box(
            row.get("p_left"), f"{label} p_left", report
        )
        if left is not None:
            expected_right = (
                left[0] + 0.2,
                left[1] + math.tan(expected_angle) * 0.2,
            )
            _require_point_close(
                row.get("p_right"), expected_right, f"{label} p_right", report
            )
    elif key.experiment == "circles":
        _require_numeric_close(
            row.get("radius"), benchmark.get("radius"), f"{label} radius", report
        )
        _check_point_in_unit_center_box(row.get("center"), f"{label} center", report)
    elif key.experiment == "ellipses":
        denominator = max(trials - 1, 1)
        aspect_ratio = 1.5 + 1.5 * case_index / denominator
        _require_numeric_close(
            row.get("aspect_ratio"), aspect_ratio, f"{label} aspect_ratio", report
        )
        _require_numeric_close(row.get("major_axis"), 30, f"{label} major_axis", report)
        _require_numeric_close(
            row.get("minor_axis"), 30 / aspect_ratio, f"{label} minor_axis", report
        )
        _check_point_in_unit_center_box(row.get("center"), f"{label} center", report)
    elif key.experiment == "squares":
        denominator = max(trials - 1, 1)
        side_length = 10.0 + 20.0 * case_index / denominator
        _require_numeric_close(
            row.get("side_length"), side_length, f"{label} side_length", report
        )
        center = _check_point_in_unit_center_box(
            row.get("center"), f"{label} center", report
        )
        vertices = row.get("vertices")
        if not isinstance(vertices, list) or len(vertices) != 4:
            report.add_error(f"{label} does not contain four square vertices")
        else:
            for vertex_index, vertex in enumerate(vertices):
                _check_point(vertex, f"{label} vertices[{vertex_index}]", report)
    elif key.experiment == "zalesak":
        for field_name, config_name in (
            ("radius", "radius"),
            ("slot_width", "slot_width"),
            ("slot_top_rel", "slot_top_relative_to_center"),
        ):
            _require_numeric_close(
                row.get(field_name),
                benchmark.get(config_name),
                f"{label} {field_name}",
                report,
            )
        center = _check_point_in_unit_center_box(
            row.get("center"), f"{label} center", report
        )
        vertices = row.get("slot_vertices")
        if not isinstance(vertices, list) or len(vertices) != 4:
            report.add_error(f"{label} does not contain four slot vertices")
        else:
            for vertex_index, vertex in enumerate(vertices):
                _check_point(vertex, f"{label} slot_vertices[{vertex_index}]", report)

    if key.experiment in {"ellipses", "squares", "zalesak"}:
        try:
            theta = float(_bounded_decimal(row.get("theta"), f"{label} theta"))
        except (ReleaseAuditInputError, ValueError, OverflowError) as exc:
            report.add_error(str(exc))
        else:
            if not 0.0 <= theta < math.pi / 2:
                report.add_error(f"{label} theta is outside [0, pi/2): {theta}")
            elif (
                key.experiment == "squares"
                and center is not None
                and isinstance(vertices, list)
                and len(vertices) == 4
            ):
                half_side = side_length / 2.0
                unrotated = (
                    (center[0] - half_side, center[1] - half_side),
                    (center[0] + half_side, center[1] - half_side),
                    (center[0] + half_side, center[1] + half_side),
                    (center[0] - half_side, center[1] + half_side),
                )
                for vertex_index, (vertex, expected_vertex) in enumerate(
                    zip(vertices, unrotated)
                ):
                    _require_point_close(
                        vertex,
                        _rotated_point(expected_vertex, center, theta),
                        f"{label} vertices[{vertex_index}]",
                        report,
                    )
            elif (
                key.experiment == "zalesak"
                and center is not None
                and isinstance(vertices, list)
                and len(vertices) == 4
            ):
                try:
                    radius = float(_bounded_decimal(benchmark.get("radius"), "radius"))
                    slot_width = float(
                        _bounded_decimal(benchmark.get("slot_width"), "slot_width")
                    )
                    slot_top = float(
                        _bounded_decimal(
                            benchmark.get("slot_top_relative_to_center"), "slot_top"
                        )
                    )
                except (ReleaseAuditInputError, ValueError, OverflowError) as exc:
                    report.add_error(str(exc))
                    return
                half_width = slot_width / 2.0
                unrotated = (
                    (center[0] - half_width, center[1] - radius - 1.0e-6),
                    (center[0] + half_width, center[1] - radius - 1.0e-6),
                    (center[0] + half_width, center[1] + slot_top),
                    (center[0] - half_width, center[1] + slot_top),
                )
                for vertex_index, (vertex, expected_vertex) in enumerate(
                    zip(vertices, unrotated)
                ):
                    _require_point_close(
                        vertex,
                        _rotated_point(expected_vertex, center, theta),
                        f"{label} slot_vertices[{vertex_index}]",
                        report,
                    )


def _check_case_metrics(
    root: Path,
    expected_runs: set[RunKey],
    trials: int,
    target_commit: str,
    target_branch: str,
    production_context: Mapping[str, str],
    inheritance: RescueProfileInheritance,
    report: AuditReport,
) -> dict[tuple[RunKey, str], list[float]]:
    path = root / "diagnostics" / "case_metrics.csv"
    all_metrics = {
        metric for metrics in METRICS_BY_EXPERIMENT.values() for metric in metrics
    }
    required = (
        set(RUN_CONTEXT_FIELDS)
        | {
            "case_index",
            "num_mixed_cells",
            "num_final_missing_cells",
        }
        | all_metrics
    )
    _, rows = _read_csv_rows(path, required, report)
    seen: set[tuple[RunKey, int]] = set()
    values: dict[tuple[RunKey, str], list[float]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=2):
        label = f"{path}:{row_number}"
        try:
            key, case_index = _case_key(row, label)
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        case_key = (key, case_index)
        if case_key in seen:
            report.add_error(
                f"duplicate case key in case_metrics: {key.display()}/case={case_index}"
            )
            continue
        seen.add(case_key)
        if key not in expected_runs or not 0 <= case_index < trials:
            report.add_error(
                f"unexpected case key in case_metrics: {key.display()}/case={case_index}"
            )
        _check_context_source(row, target_commit, target_branch, label, report)
        _check_production_context(
            row,
            production_context,
            label,
            report,
            key=key,
            inheritance=inheritance,
            inheritance_location="consolidated case metrics",
        )
        try:
            mixed_cells = _parse_int(
                row.get("num_mixed_cells"), f"{label} num_mixed_cells"
            )
            missing_cells = _parse_int(
                row.get("num_final_missing_cells"), f"{label} num_final_missing_cells"
            )
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
        else:
            if mixed_cells <= 0:
                report.add_error(f"{label} has no mixed cells")
            if missing_cells != 0:
                report.add_error(
                    f"{label} reports {missing_cells} final missing facets"
                )
        for metric in METRICS_BY_EXPERIMENT.get(key.experiment, ()):
            try:
                value = _finite_metric(row.get(metric), f"{label} {metric}")
            except ReleaseAuditInputError as exc:
                report.add_error(str(exc))
            else:
                values[(key, metric)].append(value)

    expected_cases = {
        (run_key, case_index)
        for run_key in expected_runs
        for case_index in range(trials)
    }
    for key, case_index in sorted(expected_cases - seen):
        report.add_error(f"missing case_metrics key: {key.display()}/case={case_index}")
    report.summaries["case_metric_rows"] = len(rows)
    return values


def _check_case_geometry(
    root: Path,
    config: Mapping[str, object],
    expected_runs: set[RunKey],
    trials: int,
    target_commit: str,
    target_branch: str,
    production_context: Mapping[str, str],
    inheritance: RescueProfileInheritance,
    report: AuditReport,
) -> dict[tuple[RunKey, int], dict]:
    path = root / "diagnostics" / "case_geometry.jsonl"
    seen: set[tuple[RunKey, int]] = set()
    geometry: dict[tuple[RunKey, int], dict] = {}
    case_references: dict[tuple[str, int], object] = {}
    row_count = 0
    for line_number, row in _iter_jsonl(path, report):
        row_count += 1
        label = f"{path}:{line_number}"
        try:
            key, case_index = _case_key(row, label)
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        case_key = (key, case_index)
        if case_key in seen:
            report.add_error(
                f"duplicate case key in case_geometry: {key.display()}/case={case_index}"
            )
        else:
            geometry[case_key] = row
        seen.add(case_key)
        if key not in expected_runs or not 0 <= case_index < trials:
            report.add_error(
                f"unexpected case key in case_geometry: {key.display()}/case={case_index}"
            )
        if not row.get("geometry_type"):
            report.add_error(f"{label} has no geometry_type")
        _check_case_geometry_contract(
            row, key, case_index, trials, config, label, report
        )
        geometry_payload = {
            field_name: value
            for field_name, value in row.items()
            if field_name not in RUN_CONTEXT_FIELDS
        }
        try:
            normalized_payload = _normalize_json_reconciliation_value(
                geometry_payload, f"{label} geometry payload"
            )
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
        else:
            reference_key = (key.experiment, case_index)
            reference = case_references.setdefault(reference_key, normalized_payload)
            difference = _first_json_difference(normalized_payload, reference)
            if difference:
                report.add_error(
                    f"case geometry is inconsistent across settings for "
                    f"{key.experiment}/case={case_index} at {difference}"
                )
        _check_context_source(row, target_commit, target_branch, label, report)
        _check_production_context(
            row,
            production_context,
            label,
            report,
            key=key,
            inheritance=inheritance,
            inheritance_location="consolidated case geometry",
        )
    expected_cases = {
        (run_key, case_index)
        for run_key in expected_runs
        for case_index in range(trials)
    }
    for key, case_index in sorted(expected_cases - seen):
        report.add_error(
            f"missing case_geometry key: {key.display()}/case={case_index}"
        )
    report.summaries["case_geometry_rows"] = row_count
    report.summaries["case_geometry_reference_cases"] = len(case_references)
    return geometry


def _canonical_cell_id(row: Mapping[str, object], label: str) -> str:
    raw_cell_id = row.get("cell_id")
    cell_x = _parse_canonical_int(row.get("cell_x"), f"{label} cell_x")
    cell_y = _parse_canonical_int(row.get("cell_y"), f"{label} cell_y")
    if cell_x < 0 or cell_y < 0:
        raise ReleaseAuditInputError(
            f"{label} cell coordinates must be nonnegative: {cell_x},{cell_y}"
        )
    canonical = f"{cell_x},{cell_y}"
    if raw_cell_id != canonical:
        raise ReleaseAuditInputError(
            f"{label} cell_id is not canonical for cell_x/cell_y: "
            f"{raw_cell_id!r} != {canonical!r}"
        )
    return canonical


def _validate_cell_diagnostic_row(
    row: Mapping[str, object],
    label: str,
    expected_fallback_policy: str,
    report: AuditReport,
) -> tuple[int, int, str] | None:
    try:
        _canonical_cell_id(row, label)
    except ReleaseAuditInputError as exc:
        report.add_error(str(exc))
    for field_name in sorted(CELL_INTEGER_FIELDS):
        value = row.get(field_name)
        if value in (None, ""):
            continue
        try:
            _parse_canonical_int(value, f"{label} {field_name}")
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))

    facet_class = row.get("final_facet_class", "")
    if facet_class in ("", "missing"):
        report.add_error(f"{label} has no final reconstructed facet")
    if not row.get("facet_geometry_json"):
        report.add_error(f"{label} has no facet geometry metadata")

    construction_path = row.get("construction_path")
    fallback_policy = row.get("fallback_policy")
    if not isinstance(construction_path, str) or not construction_path:
        report.add_error(f"{label} has no construction_path provenance")
        construction_path = ""
    if fallback_policy is None:
        report.add_error(f"{label} has no fallback_policy provenance field")
        fallback_policy = ""
    elif not isinstance(fallback_policy, str):
        report.add_error(f"{label} fallback_policy is not text")
        fallback_policy = str(fallback_policy)

    if construction_path == "plic_fallback":
        if fallback_policy != expected_fallback_policy:
            report.add_error(
                f"{label} plic_fallback policy differs from production: "
                f"{fallback_policy!r} != {expected_fallback_policy!r}"
            )
    elif fallback_policy:
        report.add_error(
            f"{label} nonfallback construction_path {construction_path!r} "
            f"cannot claim fallback_policy {fallback_policy!r}"
        )

    try:
        case_index = _parse_canonical_int(row.get("case_index"), f"{label} case_index")
        merge_id = _parse_canonical_int(row.get("merge_id"), f"{label} merge_id")
    except ReleaseAuditInputError as exc:
        report.add_error(str(exc))
        return None
    if case_index < 0 or merge_id < 0:
        report.add_error(
            f"{label} case_index and merge_id must be nonnegative: "
            f"{case_index},{merge_id}"
        )
        return None
    return case_index, merge_id, construction_path


def _line_facet_signature(
    raw_geometry: object, label: str, report: AuditReport
) -> tuple[tuple[str, str], tuple[str, str]] | None:
    if isinstance(raw_geometry, str):
        try:
            geometry = _strict_json_loads(raw_geometry, label)
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            return None
    else:
        geometry = raw_geometry
    if not isinstance(geometry, Mapping):
        report.add_error(f"{label} facet geometry is not an object")
        return None
    geometry_class = geometry.get("class", geometry.get("kind"))
    geometry_name = geometry.get("name", geometry.get("source_name"))
    if geometry_class not in {"linear", "line"} or geometry_name != "LVIRA":
        report.add_error(
            f"{label} fallback facet identity is not linear/LVIRA: "
            f"{geometry_class!r}/{geometry_name!r}"
        )
        return None
    points: list[tuple[str, str]] = []
    for field_name in ("p_left", "p_right"):
        point = geometry.get(field_name)
        if not isinstance(point, list) or len(point) != 2:
            report.add_error(f"{label} {field_name} is not a two-coordinate point")
            return None
        try:
            points.append((_canonical_number(point[0]), _canonical_number(point[1])))
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            return None
    if points[0] == points[1]:
        report.add_error(f"{label} fallback line has coincident endpoints")
        return None
    return tuple(sorted(points))  # type: ignore[return-value]


def _count_csv_rows(
    path: Path,
    required_headers: Iterable[str],
    report: AuditReport,
    *,
    validate_cell_rows: bool = False,
    valid_cases: set[int] | None = None,
    expected_values: Mapping[str, str] | None = None,
    expected_fallback_policy: str = "",
) -> int:
    if not path.is_file():
        report.add_error(f"missing required CSV file: {path}")
        return 0
    count = 0
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not _require_headers(reader.fieldnames, required_headers, path, report):
                return 0
            for row_number, row in enumerate(reader, start=2):
                if not _validate_csv_row_shape(row, path, row_number, report):
                    continue
                count += 1
                for field_name, expected in (expected_values or {}).items():
                    actual = row.get(field_name)
                    if actual != expected:
                        report.add_error(
                            f"{path}:{row_number} {field_name} differs from "
                            f"production: {actual!r} != {expected!r}"
                        )
                if valid_cases is not None:
                    try:
                        case_index = _parse_int(
                            row.get("case_index"), f"{path}:{row_number} case_index"
                        )
                    except ReleaseAuditInputError as exc:
                        report.add_error(str(exc))
                    else:
                        if case_index not in valid_cases:
                            report.add_error(
                                f"{path}:{row_number} references unexpected case {case_index}"
                            )
                if validate_cell_rows:
                    _validate_cell_diagnostic_row(
                        row,
                        f"{path}:{row_number}",
                        expected_fallback_policy,
                        report,
                    )
    except (OSError, UnicodeError, csv.Error) as exc:
        report.add_error(f"could not read CSV {path}: {exc}")
    return count


def _check_consolidated_table_counts(
    root: Path,
    inventory: Mapping[RunKey, Mapping[str, str]],
    production_context: Mapping[str, str],
    report: AuditReport,
) -> None:
    diagnostics = root / "diagnostics"
    table_specs = {
        "cell_metrics_rows": (
            diagnostics / "cell_metrics.csv",
            (
                "case_index",
                "cell_id",
                "cell_x",
                "cell_y",
                "merge_id",
                "final_facet_class",
                "construction_path",
                "fallback_policy",
                "facet_geometry_json",
            ),
            True,
        ),
        "merge_events_rows": (
            diagnostics / "merge_events.csv",
            ("case_index", "event_order", "event_kind"),
            False,
        ),
        "unresolved_plic_fallbacks_rows": (
            diagnostics / "unresolved_plic_fallbacks.csv",
            ("case_index", "merge_id", "policy"),
            False,
        ),
    }
    for count_field, (path, headers, validate_cells) in table_specs.items():
        expected = 0
        for key, row in inventory.items():
            try:
                expected += _parse_int(
                    row.get(count_field), f"{key.display()} {count_field}"
                )
            except ReleaseAuditInputError:
                continue
        actual = _count_csv_rows(
            path,
            set(RUN_CONTEXT_FIELDS) | set(headers),
            report,
            validate_cell_rows=validate_cells,
            expected_values=(
                {"policy": production_context.get("plic_fallback", "")}
                if path.name == "unresolved_plic_fallbacks.csv"
                else None
            ),
            expected_fallback_policy=(
                production_context.get("plic_fallback", "") if validate_cells else ""
            ),
        )
        if actual != expected:
            report.add_error(
                f"consolidated {path.name} has {actual} rows; inventory records {expected}"
            )
        report.summaries[count_field] = actual


def _diagnostic_value(value: object, limit: int = 160) -> str:
    rendered = repr(value)
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


def _normalized_relative_json_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseAuditInputError(f"{label} is not a nonempty relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ReleaseAuditInputError(f"{label} is not a safe relative path: {value!r}")
    normalized = str(pure)
    if normalized in ("", "."):
        raise ReleaseAuditInputError(f"{label} is not a file path: {value!r}")
    return normalized


def _normalize_json_reconciliation_value(
    value: object,
    label: str,
    path: tuple[str, ...] = (),
) -> object:
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ReleaseAuditInputError(
                    f"{label} contains a non-string JSON key at {'.'.join(path) or '$'}"
                )
            nested_path = (*path, key)
            nested_label = f"{label} at {'.'.join(nested_path)}"
            if key in JSON_RELATIVE_PATH_FIELDS or (path and path[-1] == "artifacts"):
                normalized[key] = _normalized_relative_json_path(nested, nested_label)
            else:
                normalized[key] = _normalize_json_reconciliation_value(
                    nested, label, nested_path
                )
        return normalized
    if isinstance(value, list):
        return [
            _normalize_json_reconciliation_value(nested, label, (*path, str(index)))
            for index, nested in enumerate(value)
        ]
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, (int, float, Decimal)):
        return ("number", _canonical_number(value))
    if isinstance(value, str):
        return ("string", value)
    raise ReleaseAuditInputError(
        f"{label} contains an unsupported JSON value at {'.'.join(path) or '$'}"
    )


def _first_json_difference(actual: object, expected: object, path: str = "$") -> str:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        for key in sorted(set(actual) | set(expected)):
            child_path = f"{path}.{key}"
            if key not in actual or key not in expected:
                return child_path
            difference = _first_json_difference(actual[key], expected[key], child_path)
            if difference:
                return difference
        return ""
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            difference = _first_json_difference(
                actual_item, expected_item, f"{path}[{index}]"
            )
            if difference:
                return difference
        return ""
    if actual == expected:
        return ""
    return path


def _reconcile_json_value(
    label: str,
    consolidated: object,
    raw: object,
    report: AuditReport,
) -> None:
    try:
        normalized_consolidated = _normalize_json_reconciliation_value(
            consolidated, f"consolidated {label}"
        )
        normalized_raw = _normalize_json_reconciliation_value(raw, f"raw {label}")
    except ReleaseAuditInputError as exc:
        report.add_error(str(exc))
        return
    difference = _first_json_difference(normalized_consolidated, normalized_raw)
    if not difference:
        return
    report.add_error(
        f"raw/consolidated JSON value mismatch for {label} at {difference}: "
        f"consolidated={_diagnostic_value(consolidated)}, "
        f"raw={_diagnostic_value(raw)}"
    )


def _csv_values_equal(actual: object, expected: object) -> bool:
    actual_text = str(actual)
    expected_text = str(expected)
    if actual_text == expected_text:
        return True
    # Consolidation copies CSV text verbatim except for recomputed case summaries.
    # Accept equivalent decimal spellings, but no epsilon-sized scientific drift.
    try:
        return _canonical_number(actual_text) == _canonical_number(expected_text)
    except ReleaseAuditInputError:
        return False


def _reconciliation_key(
    row: Mapping[str, object],
    key_fields: Sequence[str],
    label: str,
) -> tuple[object, ...]:
    values: list[object] = []
    for field_name in key_fields:
        value = row.get(field_name)
        if value in (None, ""):
            raise ReleaseAuditInputError(f"{label} has empty key field {field_name}")
        if field_name == "cell_id":
            value = _canonical_cell_id(row, label)
        elif field_name in INTEGER_KEY_FIELDS:
            value = _parse_int(value, f"{label} {field_name}")
        else:
            value = str(value)
        values.append(value)
    return tuple(values)


def _format_reconciliation_key(key_fields: Sequence[str], key: Sequence[object]) -> str:
    return ",".join(
        f"{field_name}={value}" for field_name, value in zip(key_fields, key)
    )


def _index_reconciliation_rows(
    rows: Iterable[Mapping[str, object]],
    key_fields: Sequence[str],
    label: str,
    report: AuditReport,
) -> dict[tuple[object, ...], Mapping[str, object]]:
    indexed: dict[tuple[object, ...], Mapping[str, object]] = {}
    for row_number, row in enumerate(rows, start=1):
        row_label = f"{label} row {row_number}"
        try:
            key = _reconciliation_key(row, key_fields, row_label)
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        if key in indexed:
            report.add_error(
                f"duplicate reconciliation key in {label}: "
                f"{_format_reconciliation_key(key_fields, key)}"
            )
            continue
        indexed[key] = row
    return indexed


def _diagnostic_numeric_value(value: object, default: int = 0) -> int:
    try:
        return _parse_canonical_int(value, "diagnostic integer")
    except ReleaseAuditInputError:
        return default


def _cell_row_priority(row: Mapping[str, object]) -> tuple[bool, bool, int, int]:
    return (
        row.get("construction_path") == "plic_fallback",
        row.get("final_facet_class") not in (None, "", "missing"),
        _diagnostic_numeric_value(row.get("event_count")),
        _diagnostic_numeric_value(row.get("merge_id"), default=-1),
    )


def _deduplicate_raw_cell_rows(
    rows: Iterable[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    selected: dict[tuple[object, object], Mapping[str, object]] = {}
    for row in rows:
        key = (row.get("case_index", ""), row.get("cell_id", ""))
        current = selected.get(key)
        if current is None or _cell_row_priority(row) > _cell_row_priority(current):
            selected[key] = row
    return list(selected.values())


def _cell_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, int | float]:
    class_counts: dict[str, int] = defaultdict(int)
    merge_ids: set[object] = set()
    merged_ids: set[object] = set()
    for row in rows:
        class_counts[str(row.get("final_facet_class", ""))] += 1
        merge_id = row.get("merge_id", "")
        merge_ids.add(merge_id)
        if _diagnostic_numeric_value(row.get("is_merged")):
            merged_ids.add(merge_id)

    def count(field_name: str) -> int:
        return sum(_diagnostic_numeric_value(row.get(field_name)) for row in rows)

    def fraction(value: int) -> float:
        return value / len(rows) if rows else 0.0

    merged_cells = count("is_merged")
    fallback_cells = sum(
        row.get("construction_path") == "plic_fallback" for row in rows
    )
    used_circular = count("used_circular")
    used_linear_corner = count("used_linear_corner")
    used_curved_corner = count("used_curved_corner")
    used_curved_corner_rescue = count("used_curved_corner_rescue")
    return {
        "num_mixed_cells": len(rows),
        "num_merge_components": len(merge_ids),
        "num_merged_cells": merged_cells,
        "num_merged_components": len(merged_ids),
        "num_plic_fallback_cells": fallback_cells,
        "num_used_circular_cells": used_circular,
        "num_used_linear_corner_cells": used_linear_corner,
        "num_used_curved_corner_cells": used_curved_corner,
        "num_used_curved_corner_rescue_cells": used_curved_corner_rescue,
        "num_final_linear_cells": class_counts["linear"],
        "num_final_circular_cells": class_counts["circular"],
        "num_final_linear_corner_cells": class_counts["linear_corner"],
        "num_final_curved_corner_cells": class_counts["curved_corner"],
        "num_final_missing_cells": class_counts["missing"],
        "fraction_merged_cells": fraction(merged_cells),
        "fraction_plic_fallback_cells": fraction(fallback_cells),
        "fraction_used_circular_cells": fraction(used_circular),
        "fraction_used_linear_corner_cells": fraction(used_linear_corner),
        "fraction_used_curved_corner_cells": fraction(used_curved_corner),
        "fraction_used_curved_corner_rescue_cells": fraction(used_curved_corner_rescue),
        "fraction_final_linear_cells": fraction(class_counts["linear"]),
        "fraction_final_circular_cells": fraction(class_counts["circular"]),
        "fraction_final_linear_corner_cells": fraction(class_counts["linear_corner"]),
        "fraction_final_curved_corner_cells": fraction(class_counts["curved_corner"]),
    }


def _summaries_from_cell_rows(
    key: RunKey,
    rows: Sequence[Mapping[str, object]],
    report: AuditReport,
) -> dict[tuple[RunKey, int], dict[str, int | float]]:
    rows_by_case: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=1):
        try:
            case_index = _parse_int(
                row.get("case_index"),
                f"raw cell row {row_number} for {key.display()} case_index",
            )
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        rows_by_case[case_index].append(row)
    return {
        (key, case_index): _cell_summary(case_rows)
        for case_index, case_rows in rows_by_case.items()
    }


def _repair_raw_case_rows(
    key: RunKey,
    rows: Sequence[Mapping[str, object]],
    summaries: Mapping[tuple[RunKey, int], Mapping[str, int | float]],
    report: AuditReport,
) -> list[Mapping[str, object]]:
    repaired: list[Mapping[str, object]] = []
    for row_number, row in enumerate(rows, start=1):
        copied = dict(row)
        try:
            case_index = _parse_int(
                row.get("case_index"),
                f"raw case row {row_number} for {key.display()} case_index",
            )
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            repaired.append(copied)
            continue
        summary = summaries.get((key, case_index))
        if summary is None:
            report.add_error(
                f"raw case row has no reconciled cell summary: "
                f"{key.display()}/case={case_index}"
            )
        else:
            for field_name, value in summary.items():
                if field_name in copied:
                    copied[field_name] = value
        repaired.append(copied)
    return repaired


def _check_consolidated_row_context(
    row: Mapping[str, object],
    key: RunKey,
    inventory_row: Mapping[str, str],
    production_context: Mapping[str, str],
    inheritance: RescueProfileInheritance,
    table_name: str,
    reported: set[tuple[RunKey, str, str]],
    report: AuditReport,
) -> None:
    _check_production_context(
        row,
        production_context,
        f"consolidated {table_name} context for {key.display()}",
        report,
        key=key,
        inheritance=inheritance,
        inheritance_location=f"consolidated {table_name} context",
    )
    for field_name in RUN_CONTEXT_FIELDS:
        if field_name in production_context:
            continue
        expected = inventory_row.get(field_name, "")
        actual = row.get(field_name)
        if actual == expected:
            continue
        token = (key, field_name, str(actual))
        if token in reported:
            continue
        reported.add(token)
        report.add_error(
            f"consolidated {table_name} context mismatch for {key.display()}: "
            f"{field_name}={actual!r}, expected {expected!r}"
        )


def _reconcile_run_rows(
    root: Path,
    table_name: str,
    key_fields: Sequence[str],
    consolidated_fields: Sequence[str],
    consolidated_rows: Sequence[Mapping[str, object]],
    key: RunKey,
    inventory_row: Mapping[str, str],
    case_summaries: dict[tuple[RunKey, int], dict[str, int | float]],
    report: AuditReport,
) -> int:
    try:
        bundle = _safe_release_path(
            root,
            inventory_row.get("run_bundle"),
            f"inventory bundle for {key.display()}",
        )
    except ReleaseAuditInputError as exc:
        report.add_error(str(exc))
        return 0
    raw_path = bundle / "metrics" / table_name
    raw_fields, raw_rows = _read_csv_rows(raw_path, key_fields, report)
    data_fields = [
        field_name
        for field_name in consolidated_fields
        if field_name not in RUN_CONTEXT_FIELDS
    ]
    if raw_fields != data_fields:
        report.add_error(
            f"raw/consolidated schema mismatch for {key.display()}/{table_name}: "
            f"raw={raw_fields!r}, consolidated={data_fields!r}"
        )

    expected_rows: Sequence[Mapping[str, object]] = raw_rows
    if table_name == "cell_metrics.csv":
        expected_rows = _deduplicate_raw_cell_rows(raw_rows)
        case_summaries.update(_summaries_from_cell_rows(key, expected_rows, report))
    elif table_name == "case_metrics.csv":
        expected_rows = _repair_raw_case_rows(key, raw_rows, case_summaries, report)

    consolidated_index = _index_reconciliation_rows(
        consolidated_rows,
        key_fields,
        f"consolidated {key.display()}/{table_name}",
        report,
    )
    raw_index = _index_reconciliation_rows(
        expected_rows,
        key_fields,
        f"raw {key.display()}/{table_name}",
        report,
    )
    for row_key in sorted(raw_index.keys() - consolidated_index.keys()):
        report.add_error(
            f"missing consolidated row for {key.display()}/{table_name}/"
            f"{_format_reconciliation_key(key_fields, row_key)}"
        )
    for row_key in sorted(consolidated_index.keys() - raw_index.keys()):
        report.add_error(
            f"unexpected consolidated row for {key.display()}/{table_name}/"
            f"{_format_reconciliation_key(key_fields, row_key)}"
        )

    shared_fields = [
        field_name for field_name in data_fields if field_name in raw_fields
    ]
    for row_key in sorted(consolidated_index.keys() & raw_index.keys()):
        actual = consolidated_index[row_key]
        expected = raw_index[row_key]
        for field_name in shared_fields:
            if _csv_values_equal(
                actual.get(field_name, ""), expected.get(field_name, "")
            ):
                continue
            report.add_error(
                f"raw/consolidated value mismatch for {key.display()}/{table_name}/"
                f"{_format_reconciliation_key(key_fields, row_key)} column "
                f"{field_name}: consolidated="
                f"{_diagnostic_value(actual.get(field_name, ''))}, raw="
                f"{_diagnostic_value(expected.get(field_name, ''))}"
            )
    return len(consolidated_rows)


def _reconcile_consolidated_table(
    root: Path,
    table_name: str,
    key_fields: Sequence[str],
    expected_runs: set[RunKey],
    inventory: Mapping[RunKey, Mapping[str, str]],
    production_context: Mapping[str, str],
    inheritance: RescueProfileInheritance,
    case_summaries: dict[tuple[RunKey, int], dict[str, int | float]],
    report: AuditReport,
) -> None:
    path = root / "diagnostics" / table_name
    if not path.is_file():
        report.add_error(f"missing required CSV file: {path}")
        return
    seen_groups: set[RunKey] = set()
    reported_context: set[tuple[RunKey, str, str]] = set()
    total_rows = 0

    def reconcile_group(
        key: RunKey | None,
        rows: list[Mapping[str, object]],
        fieldnames: Sequence[str],
    ) -> None:
        nonlocal total_rows
        if key is None or not rows:
            return
        total_rows += len(rows)
        if key in seen_groups:
            report.add_error(
                f"consolidated {table_name} contains noncontiguous blocks for "
                f"{key.display()}"
            )
        seen_groups.add(key)
        inventory_row = inventory.get(key)
        if inventory_row is None:
            report.add_error(
                f"consolidated {table_name} has rows for an unindexed run: "
                f"{key.display()}"
            )
            return
        for row in rows:
            _check_consolidated_row_context(
                row,
                key,
                inventory_row,
                production_context,
                inheritance,
                table_name,
                reported_context,
                report,
            )
        _reconcile_run_rows(
            root,
            table_name,
            key_fields,
            fieldnames,
            rows,
            key,
            inventory_row,
            case_summaries,
            report,
        )

    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
            required = set(RUN_CONTEXT_FIELDS) | set(key_fields)
            if not _require_headers(fieldnames, required, path, report):
                return
            current_key: RunKey | None = None
            current_rows: list[Mapping[str, object]] = []
            for row_number, row in enumerate(reader, start=2):
                if not _validate_csv_row_shape(row, path, row_number, report):
                    continue
                label = f"{path}:{row_number}"
                try:
                    key = _run_key(row, label)
                except ReleaseAuditInputError as exc:
                    report.add_error(str(exc))
                    continue
                if current_key is not None and key != current_key:
                    reconcile_group(current_key, current_rows, fieldnames)
                    current_rows = []
                current_key = key
                current_rows.append(row)
            reconcile_group(current_key, current_rows, fieldnames)
    except (OSError, UnicodeError, csv.Error) as exc:
        report.add_error(f"could not reconcile CSV {path}: {exc}")
        return

    count_field = table_name.removesuffix(".csv") + "_rows"
    for key in sorted(expected_runs):
        row = inventory.get(key)
        if row is None:
            continue
        try:
            expected_count = _parse_int(
                row.get(count_field), f"{key.display()} {count_field}"
            )
        except ReleaseAuditInputError:
            continue
        if expected_count and key not in seen_groups:
            report.add_error(
                f"consolidated {table_name} is missing the run block for "
                f"{key.display()}"
            )
    report.summaries[f"reconciled_{table_name.removesuffix('.csv')}_rows"] = total_rows


def _reconcile_consolidated_tables(
    root: Path,
    expected_runs: set[RunKey],
    inventory: Mapping[RunKey, Mapping[str, str]],
    production_context: Mapping[str, str],
    inheritance: RescueProfileInheritance,
    report: AuditReport,
) -> None:
    case_summaries: dict[tuple[RunKey, int], dict[str, int | float]] = {}
    for table_name, key_fields in RECONCILIATION_TABLES:
        _reconcile_consolidated_table(
            root,
            table_name,
            key_fields,
            expected_runs,
            inventory,
            production_context,
            inheritance,
            case_summaries,
            report,
        )


def _reconcile_jsonl_records(
    consolidated_manifests: Mapping[RunKey, Mapping[str, object]],
    raw_manifests: Mapping[RunKey, Mapping[str, object]],
    consolidated_geometry: Mapping[tuple[RunKey, int], Mapping[str, object]],
    raw_geometry: Mapping[tuple[RunKey, int], Mapping[str, object]],
    report: AuditReport,
) -> None:
    for key in sorted(raw_manifests.keys() - consolidated_manifests.keys()):
        report.add_error(
            f"missing consolidated run_manifests.jsonl row for {key.display()}"
        )
    for key in sorted(consolidated_manifests.keys() - raw_manifests.keys()):
        report.add_error(
            f"unexpected consolidated run_manifests.jsonl row for {key.display()}"
        )
    for key in sorted(consolidated_manifests.keys() & raw_manifests.keys()):
        _reconcile_json_value(
            f"run_manifests.jsonl/{key.display()}",
            consolidated_manifests[key].get("manifest"),
            raw_manifests[key],
            report,
        )

    for key, case_index in sorted(raw_geometry.keys() - consolidated_geometry.keys()):
        report.add_error(
            "missing consolidated case_geometry.jsonl row for "
            f"{key.display()}/case={case_index}"
        )
    for key, case_index in sorted(consolidated_geometry.keys() - raw_geometry.keys()):
        report.add_error(
            "unexpected consolidated case_geometry.jsonl row for "
            f"{key.display()}/case={case_index}"
        )
    shared_geometry = consolidated_geometry.keys() & raw_geometry.keys()
    for key, case_index in sorted(shared_geometry):
        consolidated_payload = {
            field_name: value
            for field_name, value in consolidated_geometry[(key, case_index)].items()
            if field_name not in RUN_CONTEXT_FIELDS
        }
        _reconcile_json_value(
            f"case_geometry.jsonl/{key.display()}/case={case_index}",
            consolidated_payload,
            raw_geometry[(key, case_index)],
            report,
        )

    report.summaries["reconciled_run_manifests_rows"] = len(
        consolidated_manifests.keys() & raw_manifests.keys()
    )
    report.summaries["reconciled_case_geometry_rows"] = len(shared_geometry)


def _check_raw_case_rows(
    bundle: Path,
    key: RunKey,
    trials: int,
    report: AuditReport,
) -> tuple[int, int, list[dict]]:
    metrics_path = bundle / "metrics" / "case_metrics.csv"
    required_metrics = METRICS_BY_EXPERIMENT.get(key.experiment, ())
    _, metric_rows = _read_csv_rows(
        metrics_path,
        {"case_index", "num_final_missing_cells", *required_metrics},
        report,
    )
    seen_metrics: set[int] = set()
    for row_number, row in enumerate(metric_rows, start=2):
        label = f"{metrics_path}:{row_number}"
        try:
            case_index = _parse_int(row.get("case_index"), f"{label} case_index")
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        if case_index in seen_metrics:
            report.add_error(
                f"duplicate raw case metric index in {bundle}: {case_index}"
            )
        seen_metrics.add(case_index)
        if not 0 <= case_index < trials:
            report.add_error(
                f"unexpected raw case metric index in {bundle}: {case_index}"
            )
        for metric in required_metrics:
            try:
                _finite_metric(row.get(metric), f"{label} {metric}")
            except ReleaseAuditInputError as exc:
                report.add_error(str(exc))
        try:
            missing = _parse_int(
                row.get("num_final_missing_cells"), f"{label} num_final_missing_cells"
            )
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
        else:
            if missing != 0:
                report.add_error(f"{label} reports {missing} final missing facets")
    if seen_metrics != set(range(trials)):
        report.add_error(f"raw case_metrics coverage is incomplete in {bundle}")

    geometry_path = bundle / "metrics" / "case_geometry.jsonl"
    geometry_rows: list[dict] = []
    seen_geometry: set[int] = set()
    for line_number, row in _iter_jsonl(geometry_path, report):
        geometry_rows.append(row)
        try:
            case_index = _parse_int(
                row.get("case_index"), f"{geometry_path}:{line_number} case_index"
            )
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        if case_index in seen_geometry:
            report.add_error(
                f"duplicate raw case geometry index in {bundle}: {case_index}"
            )
        seen_geometry.add(case_index)
        if not 0 <= case_index < trials:
            report.add_error(
                f"unexpected raw case geometry index in {bundle}: {case_index}"
            )
    if seen_geometry != set(range(trials)):
        report.add_error(f"raw case_geometry coverage is incomplete in {bundle}")
    return len(metric_rows), len(geometry_rows), geometry_rows


def _require_nonempty_file(path: Path, label: str, report: AuditReport) -> None:
    if not path.is_file():
        report.add_error(f"missing {label}: {path}")
        return
    try:
        if path.stat().st_size == 0:
            report.add_error(f"empty {label}: {path}")
    except OSError as exc:
        report.add_error(f"could not inspect {label} {path}: {exc}")


def _finalize_rescue_profile_inheritance(
    inheritance: RescueProfileInheritance, report: AuditReport
) -> None:
    inherited_runs: set[RunKey] = set()
    required_command_evidence = {
        "consolidated child manifest",
        "raw child manifest",
    }
    for key, locations in sorted(inheritance.profile_states.items()):
        states: set[str] = set()
        for location, location_states in sorted(locations.items()):
            states.update(location_states)
            if len(location_states) > 1:
                report.add_error(
                    f"rescue_profile omission pattern is mixed within {location} "
                    f"for {key.display()}: {sorted(location_states)!r}"
                )
        if "omitted" not in states:
            continue
        inherited_runs.add(key)
        if "explicit" in states:
            report.add_error(
                f"rescue_profile omission pattern is inconsistent across provenance "
                f"layers for {key.display()}"
            )
        if not inheritance.permits(key):
            report.add_error(
                f"rescue_profile omission is not eligible for authoritative "
                f"inheritance: {key.display()}"
            )
        missing_evidence = (
            required_command_evidence - inheritance.command_resolutions.get(key, set())
        )
        if missing_evidence:
            report.add_error(
                f"rescue_profile omission lacks child-command evidence for "
                f"{key.display()}: {sorted(missing_evidence)!r}"
            )
        command_states = set(inheritance.command_states.get(key, {}).values())
        if len(command_states) > 1:
            report.add_error(
                f"rescue_profile command-option omission pattern is inconsistent "
                f"across child manifests for {key.display()}"
            )
    report.summaries["rescue_profile_inherited_runs"] = len(inherited_runs)
    if not inherited_runs:
        return
    locations = sorted(
        {
            location
            for key in inherited_runs
            for location, states in inheritance.profile_states[key].items()
            if "omitted" in states
        }
    )
    evidence = "; ".join(inheritance.evidence)
    report.add_warning(
        f"Audited rescue_profile={inheritance.value!r} by inheritance for "
        f"{len(inherited_runs)} non-Zalesak runs across {', '.join(locations)}. Evidence: "
        f"{evidence}."
    )


def _check_raw_cell_fallback_provenance(
    bundle: Path,
    key: RunKey,
    trials: int,
    expected_policy: str,
    report: AuditReport,
) -> tuple[int, int, int]:
    metrics = bundle / "metrics"
    cell_path = metrics / "cell_metrics.csv"
    _, cell_rows = _read_csv_rows(
        cell_path,
        {
            "case_index",
            "cell_id",
            "cell_x",
            "cell_y",
            "merge_id",
            "merge_component_size",
            "final_facet_class",
            "final_facet_name",
            "construction_path",
            "fallback_policy",
            "facet_geometry_json",
        },
        report,
    )
    component_paths: dict[tuple[int, int], set[str]] = defaultdict(set)
    component_cells: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    component_sizes: dict[tuple[int, int], set[int]] = defaultdict(set)
    component_facets: dict[
        tuple[int, int], set[tuple[tuple[str, str], tuple[str, str]]]
    ] = defaultdict(set)
    fallback_components: set[tuple[int, int]] = set()
    for row_number, row in enumerate(cell_rows, start=2):
        label = f"{cell_path}:{row_number}"
        component = _validate_cell_diagnostic_row(row, label, expected_policy, report)
        if component is None:
            continue
        case_index, merge_id, construction_path = component
        if not 0 <= case_index < trials:
            report.add_error(f"{label} references unexpected case {case_index}")
        component_key = (case_index, merge_id)
        component_paths[component_key].add(construction_path)
        try:
            cell = (
                _parse_canonical_int(row.get("cell_x"), f"{label} cell_x"),
                _parse_canonical_int(row.get("cell_y"), f"{label} cell_y"),
            )
            component_size = _parse_canonical_int(
                row.get("merge_component_size"), f"{label} merge_component_size"
            )
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        component_cells[component_key].add(cell)
        component_sizes[component_key].add(component_size)
        if construction_path == "plic_fallback":
            fallback_components.add(component_key)
            if row.get("final_facet_class") != "linear":
                report.add_error(f"{label} fallback final_facet_class is not 'linear'")
            if row.get("final_facet_name") != expected_policy:
                report.add_error(
                    f"{label} fallback final_facet_name differs from production: "
                    f"{row.get('final_facet_name')!r} != {expected_policy!r}"
                )
            signature = _line_facet_signature(
                row.get("facet_geometry_json"), f"{label} facet_geometry_json", report
            )
            if signature is not None:
                component_facets[component_key].add(signature)

    merge_path = metrics / "merge_events.csv"
    _, merge_rows = _read_csv_rows(
        merge_path,
        {
            "case_index",
            "event_order",
            "merge_id",
            "member_cells_json",
            "stage",
            "event_kind",
            "fallback_policy",
            "fallback_reason",
            "previous_facet_class",
            "previous_facet_name",
            "facet_class",
            "facet_name",
        },
        report,
    )
    merge_fallback_events: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for row_number, row in enumerate(merge_rows, start=2):
        if row.get("event_kind") != "plic_fallback":
            continue
        label = f"{merge_path}:{row_number}"
        try:
            case_index = _parse_canonical_int(
                row.get("case_index"), f"{label} case_index"
            )
            merge_id = _parse_canonical_int(row.get("merge_id"), f"{label} merge_id")
            member_value = _strict_json_loads(
                str(row.get("member_cells_json", "")),
                f"{label} member_cells_json",
            )
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        event_key = (case_index, merge_id)
        if event_key in merge_fallback_events:
            report.add_error(
                f"duplicate plic_fallback merge event for {key.display()}/"
                f"case={case_index}/merge_id={merge_id}"
            )
            continue
        members: set[tuple[int, int]] = set()
        if not isinstance(member_value, list) or not member_value:
            report.add_error(f"{label} member_cells_json is not a non-empty array")
        else:
            for member_index, member in enumerate(member_value):
                if not isinstance(member, list) or len(member) != 2:
                    report.add_error(
                        f"{label} member_cells_json[{member_index}] is not a cell pair"
                    )
                    continue
                try:
                    cell = (
                        _parse_int(member[0], f"{label} member cell x"),
                        _parse_int(member[1], f"{label} member cell y"),
                    )
                except ReleaseAuditInputError as exc:
                    report.add_error(str(exc))
                    continue
                if cell in members:
                    report.add_error(f"{label} contains duplicate member cell {cell}")
                members.add(cell)
        merge_fallback_events[event_key] = members
        identity = (
            row.get("fallback_policy"),
            row.get("facet_class"),
            row.get("facet_name"),
        )
        if identity != (expected_policy, "linear", expected_policy):
            report.add_error(
                f"{label} fallback merge-event identity differs from production: "
                f"{identity!r} != {(expected_policy, 'linear', expected_policy)!r}"
            )
        if not row.get("fallback_reason"):
            report.add_error(f"{label} has no fallback_reason")

    fallback_path = metrics / "unresolved_plic_fallbacks.csv"
    _, fallback_rows = _read_csv_rows(
        fallback_path,
        {"case_index", "merge_id", "policy", "setting", "facet_name", "num_vertices"},
        report,
    )
    fallback_events: set[tuple[int, int]] = set()
    for row_number, row in enumerate(fallback_rows, start=2):
        label = f"{fallback_path}:{row_number}"
        try:
            case_index = _parse_canonical_int(
                row.get("case_index"), f"{label} case_index"
            )
            merge_id = _parse_canonical_int(row.get("merge_id"), f"{label} merge_id")
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        if not 0 <= case_index < trials or merge_id < 0:
            report.add_error(
                f"{label} has unexpected case_index/merge_id: "
                f"{case_index},{merge_id}"
            )
        event_key = (case_index, merge_id)
        if event_key in fallback_events:
            report.add_error(
                "duplicate unresolved fallback event for "
                f"{key.display()}/case={case_index}/merge_id={merge_id}"
            )
        fallback_events.add(event_key)
        policy = row.get("policy")
        if policy != expected_policy:
            report.add_error(
                f"{label} fallback event policy differs from production: "
                f"{policy!r} != {expected_policy!r}"
            )
        if str(row.get("setting", "")).lower() != key.algo:
            report.add_error(
                f"{label} fallback setting does not match run algorithm: "
                f"{row.get('setting')!r} != {key.algo!r}"
            )
        if row.get("facet_name") != expected_policy:
            report.add_error(
                f"{label} fallback facet_name differs from production: "
                f"{row.get('facet_name')!r} != {expected_policy!r}"
            )
        try:
            num_vertices = _parse_canonical_int(
                row.get("num_vertices"), f"{label} num_vertices"
            )
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
        else:
            if num_vertices < 3:
                report.add_error(f"{label} fallback polygon has fewer than 3 vertices")

    for case_index, merge_id in sorted(fallback_components):
        paths = component_paths[(case_index, merge_id)]
        if paths != {"plic_fallback"}:
            report.add_error(
                "fallback component has inconsistent construction paths for "
                f"{key.display()}/case={case_index}/merge_id={merge_id}: "
                f"{sorted(paths)!r}"
            )
        members = component_cells[(case_index, merge_id)]
        sizes = component_sizes[(case_index, merge_id)]
        if sizes != {len(members)}:
            report.add_error(
                f"fallback component size disagrees with member cells for "
                f"{key.display()}/case={case_index}/merge_id={merge_id}: "
                f"{sorted(sizes)!r} != {[len(members)]!r}"
            )
        facets = component_facets[(case_index, merge_id)]
        if len(facets) != 1:
            report.add_error(
                f"fallback component does not have one exact LVIRA facet geometry for "
                f"{key.display()}/case={case_index}/merge_id={merge_id}"
            )
    for case_index, merge_id in sorted(fallback_components - fallback_events):
        report.add_error(
            "plic_fallback component has no unresolved fallback event for "
            f"{key.display()}/case={case_index}/merge_id={merge_id}"
        )
    for case_index, merge_id in sorted(fallback_events - fallback_components):
        report.add_error(
            "unresolved fallback event has no plic_fallback cell component for "
            f"{key.display()}/case={case_index}/merge_id={merge_id}"
        )
    for case_index, merge_id in sorted(
        fallback_components - set(merge_fallback_events)
    ):
        report.add_error(
            "plic_fallback component has no plic_fallback merge event for "
            f"{key.display()}/case={case_index}/merge_id={merge_id}"
        )
    for case_index, merge_id in sorted(
        set(merge_fallback_events) - fallback_components
    ):
        report.add_error(
            "plic_fallback merge event has no plic_fallback cell component for "
            f"{key.display()}/case={case_index}/merge_id={merge_id}"
        )
    for event_key in sorted(fallback_components & set(merge_fallback_events)):
        if merge_fallback_events[event_key] != component_cells[event_key]:
            report.add_error(
                f"plic_fallback merge-event member cells disagree with cell provenance "
                f"for {key.display()}/case={event_key[0]}/merge_id={event_key[1]}: "
                f"{sorted(merge_fallback_events[event_key])!r} != "
                f"{sorted(component_cells[event_key])!r}"
            )

    expected_metadata_facets: dict[
        int, Counter[tuple[tuple[str, str], tuple[str, str]]]
    ] = defaultdict(Counter)
    for (case_index, merge_id), facets in component_facets.items():
        if (case_index, merge_id) in fallback_components and len(facets) == 1:
            expected_metadata_facets[case_index][next(iter(facets))] += 1
    for case_index, expected_facets in sorted(expected_metadata_facets.items()):
        metadata_path = (
            bundle
            / "vtk"
            / "reconstructed"
            / "facets"
            / f"{case_index}.facet_metadata.json"
        )
        metadata = _load_json(metadata_path, report)
        if metadata is None:
            continue
        primitives = metadata.get("primitives")
        if not isinstance(primitives, list):
            report.add_error(f"{metadata_path} primitives is not an array")
            continue
        actual_facets: Counter[tuple[tuple[str, str], tuple[str, str]]] = Counter()
        for primitive_index, primitive in enumerate(primitives):
            if not isinstance(primitive, Mapping):
                continue
            if (
                primitive.get("kind") != "line"
                or primitive.get("source_name") != expected_policy
            ):
                continue
            signature = _line_facet_signature(
                primitive,
                f"{metadata_path} primitive {primitive_index}",
                report,
            )
            if signature is not None:
                actual_facets[signature] += 1
        for signature, expected_count in expected_facets.items():
            if actual_facets[signature] < expected_count:
                report.add_error(
                    f"saved facet metadata lacks fallback LVIRA geometry for "
                    f"{key.display()}/case={case_index}: {signature!r}"
                )
    return len(cell_rows), len(merge_rows), len(fallback_rows)


def _check_raw_bundle(
    root: Path,
    bundle: Path,
    key: RunKey,
    config: Mapping[str, object],
    inventory_row: Mapping[str, str],
    trials: int,
    target_commit: str,
    target_branch: str,
    production_context: Mapping[str, str],
    inheritance: RescueProfileInheritance,
    report: AuditReport,
) -> tuple[dict | None, list[dict]]:
    for path in bundle.rglob("*"):
        if path.is_symlink():
            report.add_error(f"raw bundle contains a symbolic link: {path}")
        if path.is_dir() and path.name == "plt":
            report.add_error(f"raw bundle contains temporary raster previews: {path}")

    manifest_path = bundle / "run_manifest.json"
    manifest = _load_json(manifest_path, report)
    if manifest:
        if manifest.get("source_commit") != target_commit:
            report.add_error(f"raw run manifest commit mismatch: {key.display()}")
        if manifest.get("source_branch") != target_branch:
            report.add_error(f"raw run manifest branch mismatch: {key.display()}")
        if str(manifest.get("experiment", "")).lower() != key.experiment:
            report.add_error(f"raw run manifest experiment mismatch: {key.display()}")
        parameters = manifest.get("parameters")
        if not isinstance(parameters, dict):
            report.add_error(f"raw run manifest parameters missing: {key.display()}")
        else:
            _check_production_context(
                parameters,
                production_context,
                f"raw run manifest parameters {key.display()}",
                report,
                key=key,
                inheritance=inheritance,
                inheritance_location="raw run-manifest parameters",
            )
            try:
                manifest_key = RunKey(
                    key.experiment,
                    str(parameters.get("facet_algo", "")).lower(),
                    _canonical_number(parameters.get("resolution")),
                    _canonical_number(parameters.get("perturb_wiggle")),
                    _parse_int(parameters.get("perturb_seed"), "perturb_seed"),
                )
            except ReleaseAuditInputError as exc:
                report.add_error(f"invalid raw run manifest for {key.display()}: {exc}")
            else:
                if manifest_key != key:
                    report.add_error(f"raw run manifest key mismatch: {key.display()}")
        _check_child_manifest_command(
            manifest,
            key,
            production_context,
            inheritance,
            "raw child manifest",
            report,
        )
        _check_child_manifest_contract(
            manifest,
            key,
            config,
            str(inventory_row.get("save_name", "")),
            inheritance,
            "raw child manifest",
            report,
        )
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            report.add_error(f"raw run manifest artifacts missing: {key.display()}")
        else:
            for artifact_name, expected_relative in REQUIRED_RAW_ARTIFACTS.items():
                if artifacts.get(artifact_name) != expected_relative:
                    report.add_error(
                        f"raw artifact mapping {artifact_name!r} is inconsistent for "
                        f"{key.display()}"
                    )

    for artifact_name, relative in REQUIRED_RAW_ARTIFACTS.items():
        _require_nonempty_file(bundle / relative, artifact_name, report)

    raw_case_count, raw_geometry_count, geometry_rows = _check_raw_case_rows(
        bundle, key, trials, report
    )
    raw_cell_count, raw_merge_count, raw_fallback_count = (
        _check_raw_cell_fallback_provenance(
            bundle,
            key,
            trials,
            production_context.get("plic_fallback", ""),
            report,
        )
    )
    actual_counts = {
        "case_metrics_rows": raw_case_count,
        "case_geometry_rows": raw_geometry_count,
        "cell_metrics_rows": raw_cell_count,
        "merge_events_rows": raw_merge_count,
        "unresolved_plic_fallbacks_rows": raw_fallback_count,
    }
    for field_name, actual in actual_counts.items():
        try:
            recorded = _parse_int(
                inventory_row.get(field_name),
                f"inventory {field_name} for {key.display()}",
            )
        except ReleaseAuditInputError:
            continue
        if actual != recorded:
            report.add_error(
                f"raw {field_name} for {key.display()} is {actual}; inventory records "
                f"{recorded}"
            )

    for metric in METRICS_BY_EXPERIMENT.get(key.experiment, ()):
        _require_nonempty_file(bundle / "metrics" / f"{metric}.txt", metric, report)

    for geometry in geometry_rows:
        try:
            case_index = _parse_int(
                geometry.get("case_index"), f"raw geometry case_index in {bundle}"
            )
        except ReleaseAuditInputError:
            continue
        for truth_field in ("truth_vtp", "truth_metadata"):
            if truth_field not in geometry:
                continue
            try:
                truth_path = _safe_release_path(
                    bundle, geometry[truth_field], f"{key.display()} {truth_field}"
                )
            except ReleaseAuditInputError as exc:
                report.add_error(str(exc))
            else:
                _require_nonempty_file(truth_path, truth_field, report)
        reconstructed = bundle / "vtk" / "reconstructed"
        _require_nonempty_file(
            reconstructed / "facets" / f"{case_index}.vtp",
            "reconstructed facets",
            report,
        )
        _require_nonempty_file(
            reconstructed / "facets" / f"{case_index}.facet_metadata.json",
            "reconstructed facet metadata",
            report,
        )
        _require_nonempty_file(
            reconstructed / "mixed_cells" / f"{case_index}.vtp",
            "reconstructed mixed cells",
            report,
        )
    return manifest, geometry_rows


def _check_raw_bundles(
    root: Path,
    config: Mapping[str, object],
    expected_runs: set[RunKey],
    inventory: Mapping[RunKey, Mapping[str, str]],
    trials: int,
    target_commit: str,
    target_branch: str,
    production_context: Mapping[str, str],
    inheritance: RescueProfileInheritance,
    report: AuditReport,
) -> tuple[dict[RunKey, dict], dict[tuple[RunKey, int], dict]]:
    raw_manifests: dict[RunKey, dict] = {}
    raw_geometry: dict[tuple[RunKey, int], dict] = {}
    raw_root = root / "raw_runs"
    if not raw_root.is_dir():
        report.add_error(f"missing raw bundle directory: {raw_root}")
        return raw_manifests, raw_geometry
    children = list(raw_root.iterdir())
    staging = [path for path in children if path.name.startswith(".")]
    for path in staging:
        report.add_error(f"raw bundle root contains temporary/staging path: {path}")
    raw_dirs = {
        path.name: path
        for path in children
        if path.is_dir() and not path.name.startswith(".")
    }
    non_dirs = [path for path in children if not path.is_dir()]
    for path in non_dirs:
        report.add_error(f"unexpected non-directory in raw_runs: {path}")

    inventory_names = {
        str(row.get("save_name", ""))
        for row in inventory.values()
        if row.get("save_name")
    }
    if set(raw_dirs) != inventory_names:
        for name in sorted(inventory_names - set(raw_dirs)):
            report.add_error(f"inventory raw bundle directory is missing: {name}")
        for name in sorted(set(raw_dirs) - inventory_names):
            report.add_error(f"unindexed raw bundle directory: {name}")
    if len(raw_dirs) != len(expected_runs):
        report.add_error(
            f"raw_runs contains {len(raw_dirs)} bundles; expected {len(expected_runs)}"
        )

    for key in sorted(expected_runs):
        inventory_row = inventory.get(key)
        if inventory_row is None:
            continue
        bundle = raw_dirs.get(str(inventory_row.get("save_name", "")))
        if bundle is None:
            continue
        manifest, geometry_rows = _check_raw_bundle(
            root,
            bundle,
            key,
            config,
            inventory_row,
            trials,
            target_commit,
            target_branch,
            production_context,
            inheritance,
            report,
        )
        if manifest is not None:
            raw_manifests[key] = manifest
        for row_number, geometry in enumerate(geometry_rows, start=1):
            try:
                case_index = _parse_int(
                    geometry.get("case_index"),
                    f"raw geometry row {row_number} for {key.display()} case_index",
                )
            except ReleaseAuditInputError:
                continue
            raw_geometry.setdefault((key, case_index), geometry)
    report.summaries["raw_bundles"] = len(raw_dirs)
    return raw_manifests, raw_geometry


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ReleaseAuditInputError("cannot aggregate an empty metric series")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _aggregate_stat(values: Sequence[float], stat: str) -> float:
    if stat == "mean":
        return statistics.fmean(values)
    if stat == "median":
        return statistics.median(values)
    if stat == "p25":
        return _percentile(values, 0.25)
    if stat == "p75":
        return _percentile(values, 0.75)
    raise ReleaseAuditInputError(f"unknown aggregate statistic: {stat}")


def _check_aggregate_metrics(
    root: Path,
    expected_runs: set[RunKey],
    case_values: Mapping[tuple[RunKey, str], Sequence[float]],
    report: AuditReport,
) -> None:
    path = root / "perturbed_sweep.csv"
    _, rows = _read_csv_rows(
        path,
        {
            "experiment",
            "algo",
            "resolution",
            "wiggle",
            "seed",
            "metric_key",
            "metric_value",
            "save_name",
        },
        report,
    )
    expected_keys = {
        (run_key, f"{metric}_{stat}")
        for run_key in expected_runs
        for metric in METRICS_BY_EXPERIMENT.get(run_key.experiment, ())
        for stat in AGGREGATE_STATS
    }
    seen: set[tuple[RunKey, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        label = f"{path}:{row_number}"
        try:
            run_key = _run_key(row, label)
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        metric_key = str(row.get("metric_key", ""))
        key = (run_key, metric_key)
        if key in seen:
            report.add_error(
                f"duplicate aggregate key: {run_key.display()}/{metric_key}"
            )
            continue
        seen.add(key)
        if key not in expected_keys:
            report.add_error(
                f"unexpected aggregate key: {run_key.display()}/{metric_key}"
            )
            continue
        try:
            value = _finite_metric(row.get("metric_value"), f"{label} metric_value")
        except ReleaseAuditInputError as exc:
            report.add_error(str(exc))
            continue
        metric = ""
        stat = ""
        for suffix in AGGREGATE_STATS:
            token = f"_{suffix}"
            if metric_key.endswith(token):
                metric = metric_key[: -len(token)]
                stat = suffix
                break
        values = case_values.get((run_key, metric), ())
        if not values:
            report.add_error(
                f"aggregate row has no complete case series: {run_key.display()}/{metric}"
            )
            continue
        expected_value = _aggregate_stat(values, stat)
        if not math.isclose(value, expected_value, rel_tol=1e-12, abs_tol=1e-15):
            report.add_error(
                f"aggregate value mismatch for {run_key.display()}/{metric_key}: "
                f"{value} != {expected_value}"
            )

    for run_key, metric_key in sorted(expected_keys - seen):
        report.add_error(f"missing aggregate key: {run_key.display()}/{metric_key}")
    report.summaries["aggregate_rows"] = len(rows)


def audit_final_release(
    release_root: Path,
    *,
    required_runs: int = FINAL_RUN_COUNT,
    required_cases: int = FINAL_CASE_COUNT,
) -> AuditReport:
    """Audit a release without modifying it and return every detected failure."""
    root = _lexical_absolute(release_root)
    report = AuditReport(root)
    if not _reject_release_symlinks(root, report):
        return report
    for relative in REQUIRED_RELEASE_FILES:
        if not (root / relative).is_file():
            report.add_error(f"missing required release file: {relative}")

    config = _load_json(root / "submission_config.resolved.json", report)
    if config is None:
        return report
    expected_runs, trials = _expected_grid(config, report)
    _check_exact_counts(
        config,
        len(expected_runs),
        trials,
        required_runs,
        required_cases,
        report,
    )
    target_commit, _, source_verified, historical_root = _check_source_provenance(
        root, config, report
    )
    target_branch = str(config.get("source", {}).get("target_branch", ""))
    production = config.get("production_method", {})
    if not isinstance(production, dict):
        report.add_error("resolved config production_method is not an object")
        production = {}
    production_context = _production_context(production, report)
    legacy_command_strings_allowed = _allow_legacy_command_strings(
        target_commit, source_verified, historical_root
    )

    _, controller_profiles_verified = _check_controller(
        root,
        required_runs,
        required_cases,
        production_context,
        historical_root,
        legacy_command_strings_allowed,
        report,
    )
    inheritance = _build_rescue_profile_inheritance(
        config,
        source_verified and controller_profiles_verified,
        historical_root,
        legacy_command_strings_allowed,
    )
    inventory = _check_inventory(
        root,
        expected_runs,
        trials,
        target_commit,
        target_branch,
        production_context,
        inheritance,
        report,
    )
    consolidated_manifests = _check_consolidated_run_manifests(
        root,
        config,
        expected_runs,
        inventory,
        target_commit,
        target_branch,
        production_context,
        inheritance,
        report,
    )
    case_values = _check_case_metrics(
        root,
        expected_runs,
        trials,
        target_commit,
        target_branch,
        production_context,
        inheritance,
        report,
    )
    consolidated_geometry = _check_case_geometry(
        root,
        config,
        expected_runs,
        trials,
        target_commit,
        target_branch,
        production_context,
        inheritance,
        report,
    )
    _check_consolidated_table_counts(root, inventory, production_context, report)
    raw_manifests, raw_geometry = _check_raw_bundles(
        root,
        config,
        expected_runs,
        inventory,
        trials,
        target_commit,
        target_branch,
        production_context,
        inheritance,
        report,
    )
    _reconcile_jsonl_records(
        consolidated_manifests,
        raw_manifests,
        consolidated_geometry,
        raw_geometry,
        report,
    )
    _reconcile_consolidated_tables(
        root,
        expected_runs,
        inventory,
        production_context,
        inheritance,
        report,
    )
    _finalize_rescue_profile_inheritance(inheritance, report)
    _check_aggregate_metrics(root, expected_runs, case_values, report)
    report.summaries["expected_runs"] = len(expected_runs)
    report.summaries["expected_cases"] = len(expected_runs) * trials
    return report


def _manifest_path(root: Path, relative_path: Path | str) -> Path:
    pure = PurePosixPath(str(relative_path))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ReleaseAuditInputError(
            f"SHA-256 manifest path must be release-relative: {relative_path!r}"
        )
    path = root.joinpath(*pure.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ReleaseAuditInputError(
            "SHA-256 manifest path escapes release root"
        ) from exc
    return path


def _release_files(root: Path, excluded: set[Path]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ReleaseAuditInputError(f"release contains a symbolic link: {path}")
        if not path.is_file() or path in excluded:
            continue
        relative = path.relative_to(root).as_posix()
        if "\n" in relative or "\r" in relative:
            raise ReleaseAuditInputError(
                f"release filename cannot be represented safely: {relative!r}"
            )
        files.append((relative, path))
    files.sort(key=lambda item: item[0])
    return files


def _private_seal_parent(destination: Path) -> Path:
    parent = destination.parent
    try:
        parent_stat = os.lstat(parent)
    except OSError as exc:
        raise ReleaseAuditInputError(
            f"sealed-release parent cannot be inspected: {parent}: {exc}"
        ) from exc
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise ReleaseAuditInputError(
            f"sealed-release parent must be a real directory: {parent}"
        )
    if hasattr(os, "getuid") and parent_stat.st_uid != os.getuid():
        raise ReleaseAuditInputError(
            f"sealed-release parent is not owned by the current user: {parent}"
        )
    if stat.S_IMODE(parent_stat.st_mode) & 0o077:
        raise ReleaseAuditInputError(
            f"sealed-release parent must be private (mode 0700 or stricter): {parent}"
        )
    if destination.exists() or destination.is_symlink():
        raise ReleaseAuditInputError(
            f"sealed-release destination already exists: {destination}"
        )
    return parent


def _darwin_clonefile(source: Path, destination: Path) -> bool:
    if sys.platform != "darwin":
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    clonefile = getattr(libc, "clonefile", None)
    if clonefile is None:
        return False
    clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    clonefile.restype = ctypes.c_int
    result = clonefile(os.fsencode(source), os.fsencode(destination), 0)
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error in {
        errno.ENOTSUP,
        errno.EXDEV,
        errno.EINVAL,
        errno.ENOSYS,
        errno.EPERM,
    }:
        return False
    raise OSError(error, os.strerror(error), str(source))


def _clone_or_copy_file(source: Path, destination: Path) -> bool:
    source_stat = os.lstat(source)
    if not stat.S_ISREG(source_stat.st_mode):
        raise ReleaseAuditInputError(f"seal source is not a regular file: {source}")
    cloned = _darwin_clonefile(source, destination)
    if not cloned:
        shutil.copyfile(source, destination, follow_symlinks=False)
    destination_stat = os.lstat(destination)
    if not stat.S_ISREG(destination_stat.st_mode):
        raise ReleaseAuditInputError(
            f"sealed snapshot copy is not a regular file: {destination}"
        )
    os.chmod(destination, stat.S_IMODE(source_stat.st_mode) & 0o777)
    return cloned


def _copy_release_tree(source: Path, destination: Path) -> tuple[int, int, int]:
    source_report = AuditReport(source)
    if not _reject_release_symlinks(source, source_report):
        raise ReleaseAuditInputError("; ".join(source_report.errors))
    destination.mkdir(mode=0o700)
    clone_files = 0
    copied_files = 0
    copied_bytes = 0
    for directory, directory_names, file_names in os.walk(source, followlinks=False):
        source_directory = Path(directory)
        relative_directory = source_directory.relative_to(source)
        target_directory = destination / relative_directory
        for name in sorted(directory_names):
            (target_directory / name).mkdir(mode=0o700)
        for name in sorted(file_names):
            source_path = source_directory / name
            target_path = target_directory / name
            source_stat = os.lstat(source_path)
            copied_bytes += source_stat.st_size
            if _clone_or_copy_file(source_path, target_path):
                clone_files += 1
            else:
                copied_files += 1
    return copied_bytes, clone_files, copied_files


def _release_digest_map(
    root: Path, manifest_path: Path
) -> tuple[list[tuple[str, str]], dict[str, tuple[int, int, int]]]:
    records: list[tuple[str, str]] = []
    identities: dict[str, tuple[int, int, int]] = {}
    for relative, path in _release_files(root, {manifest_path}):
        before = os.lstat(path)
        digest = _sha256(path)
        after = os.lstat(path)
        before_identity = (before.st_size, before.st_mtime_ns, before.st_ino)
        after_identity = (after.st_size, after.st_mtime_ns, after.st_ino)
        if before_identity != after_identity:
            raise ReleaseAuditInputError(
                f"sealed snapshot file changed while hashing: {relative}"
            )
        records.append((relative, digest))
        identities[relative] = after_identity
    return records, identities


def _write_manifest_exclusive(
    manifest_path: Path, records: Sequence[tuple[str, str]]
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        manifest_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o400,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for relative, digest in records:
                stream.write(f"{digest}  {relative}\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        manifest_path.unlink(missing_ok=True)
        raise


def _make_snapshot_read_only(root: Path) -> None:
    directories: list[Path] = []
    for directory, _, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directories.append(directory_path)
        for name in file_names:
            os.chmod(directory_path / name, 0o400)
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        os.chmod(directory, 0o500)


def _atomic_publish_noreplace(staging: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename_exclusive = getattr(libc, "renamex_np", None)
        if rename_exclusive is None:
            raise ReleaseAuditInputError("renamex_np is unavailable for atomic sealing")
        rename_exclusive.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(
            os.fsencode(staging), os.fsencode(destination), 0x00000004
        )
    elif sys.platform.startswith("linux"):
        rename_exclusive = getattr(libc, "renameat2", None)
        if rename_exclusive is None:
            raise ReleaseAuditInputError("renameat2 is unavailable for atomic sealing")
        rename_exclusive.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(
            -100, os.fsencode(staging), -100, os.fsencode(destination), 1
        )
    else:
        raise ReleaseAuditInputError(
            f"atomic no-replace directory publication is unsupported on {sys.platform}"
        )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def seal_release_snapshot(
    release_root: Path,
    sealed_release_output: Path,
    manifest_relative_path: Path | str = DEFAULT_SHA256_MANIFEST,
    *,
    required_runs: int = FINAL_RUN_COUNT,
    required_cases: int = FINAL_CASE_COUNT,
) -> SealedRelease:
    """Copy, audit, hash, and atomically publish one private read-only snapshot."""
    source = _lexical_absolute(release_root)
    destination = _lexical_absolute(sealed_release_output)
    if destination == source or source in destination.parents:
        raise ReleaseAuditInputError(
            "sealed-release destination must not be the source or lie inside it"
        )
    parent = _private_seal_parent(destination)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.sealing-", dir=parent))
    os.chmod(staging, 0o700)
    published = False
    try:
        staging.rmdir()
        copied_bytes, clone_files, copied_files = _copy_release_tree(source, staging)
        manifest_path = _manifest_path(staging, manifest_relative_path)
        if manifest_path.exists() or manifest_path.is_symlink():
            raise ReleaseAuditInputError(
                f"source release already contains the target ledger: "
                f"{manifest_relative_path}"
            )

        before_records, before_identities = _release_digest_map(staging, manifest_path)
        report = audit_final_release(
            staging, required_runs=required_runs, required_cases=required_cases
        )
        if not report.ok:
            raise ReleaseAuditInputError(
                f"private snapshot audit failed with {report.total_errors} errors: "
                + "; ".join(report.errors[:10])
            )
        after_records, after_identities = _release_digest_map(staging, manifest_path)
        if before_records != after_records or before_identities != after_identities:
            raise ReleaseAuditInputError(
                "private snapshot changed between its cryptographic reads around audit"
            )
        _write_manifest_exclusive(manifest_path, after_records)
        verification_errors = verify_sha256_manifest(staging, manifest_relative_path)
        if verification_errors:
            raise ReleaseAuditInputError(
                "sealed snapshot ledger verification failed: "
                + "; ".join(verification_errors[:10])
            )
        _make_snapshot_read_only(staging)
        verification_errors = verify_sha256_manifest(staging, manifest_relative_path)
        if verification_errors:
            raise ReleaseAuditInputError(
                "sealed snapshot changed after permissions were sealed: "
                + "; ".join(verification_errors[:10])
            )
        _atomic_publish_noreplace(staging, destination)
        published = True
        published_manifest = _manifest_path(destination, manifest_relative_path)
        return SealedRelease(
            destination,
            published_manifest,
            report,
            copied_bytes,
            clone_files,
            copied_files,
        )
    finally:
        if not published and staging.exists():
            for directory, directory_names, file_names in os.walk(
                staging, topdown=False, followlinks=False
            ):
                directory_path = Path(directory)
                os.chmod(directory_path, 0o700)
                for name in file_names:
                    path = directory_path / name
                    if not path.is_symlink():
                        os.chmod(path, 0o600)
                for name in directory_names:
                    path = directory_path / name
                    if not path.is_symlink():
                        os.chmod(path, 0o700)
            shutil.rmtree(staging)


def generate_sha256_manifest(
    release_root: Path,
    manifest_relative_path: Path | str = DEFAULT_SHA256_MANIFEST,
    *,
    sealed_release_output: Path | None = None,
    required_runs: int = FINAL_RUN_COUNT,
    required_cases: int = FINAL_CASE_COUNT,
) -> Path:
    """Seal a private snapshot; direct writes into a live release are forbidden."""
    if sealed_release_output is None:
        raise ReleaseAuditInputError(
            "live manifest generation is forbidden; provide sealed_release_output"
        )
    sealed = seal_release_snapshot(
        release_root,
        sealed_release_output,
        manifest_relative_path,
        required_runs=required_runs,
        required_cases=required_cases,
    )
    return sealed.manifest_path


def verify_sha256_manifest(
    release_root: Path,
    manifest_relative_path: Path | str = DEFAULT_SHA256_MANIFEST,
) -> list[str]:
    """Return manifest verification failures, including incomplete file coverage."""
    root = _lexical_absolute(release_root)
    errors: list[str] = []
    structure_report = AuditReport(root)
    if not _reject_release_symlinks(root, structure_report):
        return structure_report.errors
    try:
        manifest_path = _manifest_path(root, manifest_relative_path)
    except ReleaseAuditInputError as exc:
        return [str(exc)]
    if not manifest_path.is_file():
        return [f"SHA-256 manifest is missing: {manifest_path}"]
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return [f"could not read SHA-256 manifest: {exc}"]

    records: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if len(line) < 67 or line[64:66] != "  ":
            errors.append(f"invalid SHA-256 manifest line {line_number}")
            continue
        digest, relative = line[:64].lower(), line[66:]
        if any(character not in "0123456789abcdef" for character in digest):
            errors.append(f"invalid SHA-256 digest on line {line_number}")
            continue
        pure = PurePosixPath(relative)
        if not relative or pure.is_absolute() or ".." in pure.parts:
            errors.append(f"unsafe SHA-256 path on line {line_number}: {relative!r}")
            continue
        if relative in seen_paths:
            errors.append(f"duplicate SHA-256 path: {relative}")
            continue
        seen_paths.add(relative)
        records.append((relative, digest))
    record_paths = [relative for relative, _ in records]
    if record_paths != sorted(record_paths):
        errors.append("SHA-256 manifest paths are not sorted")

    try:
        release_files = _release_files(root, {manifest_path})
    except ReleaseAuditInputError as exc:
        errors.append(str(exc))
        return errors
    actual_paths = {relative for relative, _ in release_files}
    manifest_paths = set(record_paths)
    for relative in sorted(actual_paths - manifest_paths):
        errors.append(f"file is absent from SHA-256 manifest: {relative}")
    for relative in sorted(manifest_paths - actual_paths):
        errors.append(f"manifest path is absent from release: {relative}")

    file_lookup = dict(release_files)
    for relative, expected_digest in records:
        path = file_lookup.get(relative)
        if path is None:
            continue
        try:
            actual_digest = _sha256(path)
        except OSError as exc:
            errors.append(f"could not hash {relative}: {exc}")
            continue
        if actual_digest != expected_digest:
            errors.append(f"SHA-256 mismatch: {relative}")
    return errors


def _print_report(report: AuditReport) -> None:
    print(f"Release root: {report.release_root}")
    for key in sorted(report.summaries):
        print(f"{key}: {report.summaries[key]}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    if report.ok:
        print("FINAL RELEASE AUDIT PASSED")
        return
    print(f"FINAL RELEASE AUDIT FAILED ({report.total_errors} errors)")
    for error in report.errors:
        print(f"- {error}")
    if report.suppressed_errors:
        print(f"- ... {report.suppressed_errors} additional errors suppressed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_root", type=Path)
    parser.add_argument(
        "--write-sha256-manifest",
        action="store_true",
        help="seal a private snapshot and publish it with a sorted SHA-256 ledger",
    )
    parser.add_argument(
        "--sealed-release-output",
        type=Path,
        help=(
            "new destination for --write-sha256-manifest; its existing parent must "
            "be private and the destination must not exist"
        ),
    )
    parser.add_argument(
        "--verify-sha256-manifest",
        action="store_true",
        help="verify exact manifest coverage and every recorded digest",
    )
    parser.add_argument(
        "--sha256-manifest",
        default=DEFAULT_SHA256_MANIFEST,
        help="release-relative manifest path (default: SHA256SUMS)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.write_sha256_manifest:
        if args.sealed_release_output is None:
            report = AuditReport(_lexical_absolute(args.release_root))
            report.add_error("--write-sha256-manifest requires --sealed-release-output")
            _print_report(report)
            return 1
        try:
            sealed = seal_release_snapshot(
                args.release_root,
                args.sealed_release_output,
                args.sha256_manifest,
            )
        except (OSError, ReleaseAuditInputError) as exc:
            report = AuditReport(_lexical_absolute(args.release_root))
            report.add_error(f"could not write SHA-256 manifest: {exc}")
        else:
            report = sealed.report
            report.release_root = sealed.release_root
            report.summaries["sealed_bytes"] = sealed.copied_bytes
            report.summaries["sealed_clone_files"] = sealed.clone_files
            report.summaries["sealed_copied_files"] = sealed.copied_files
            print(f"Published sealed release: {sealed.release_root}")
            print(f"Wrote SHA-256 manifest: {sealed.manifest_path}")
    else:
        if args.sealed_release_output is not None:
            report = AuditReport(_lexical_absolute(args.release_root))
            report.add_error(
                "--sealed-release-output is only valid with --write-sha256-manifest"
            )
        else:
            report = audit_final_release(args.release_root)
    if report.ok and args.verify_sha256_manifest:
        for error in verify_sha256_manifest(report.release_root, args.sha256_manifest):
            report.add_error(error)
    _print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
