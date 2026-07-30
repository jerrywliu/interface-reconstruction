#!/usr/bin/env python3
"""Capture the software and host environment used for a result release."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import locale
import os
import platform
import re
import subprocess
import sys
import sysconfig
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


REPO = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
GENERATED_ROOTS = frozenset({"logs", "output", "plots", "results", "tmp"})
REPRODUCIBILITY_ENVIRONMENT_VARIABLES = (
    "BLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MPLBACKEND",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
    "SOURCE_DATE_EPOCH",
    "VECLIB_MAXIMUM_THREADS",
)
FINGERPRINT_PATHS = (
    "requirements.txt",
    "submission/submission_config.json",
    "config/base.yaml",
)
FINGERPRINT_GLOBS = ("config/static/*.yaml",)
EXACT_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^;\s]+)"
    r"(?:\s*;\s*(?P<marker>.+))?$"
)


def _normalise_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _command_result(args: list[str], cwd: Optional[Path] = None) -> dict:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {
            "command": args,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": args,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _git_value(repo_root: Path, *args: str) -> str:
    result = _command_result(["git", *args], cwd=repo_root)
    return result["stdout"] if result["returncode"] == 0 else ""


def _status_path(status_line: str) -> str:
    path = status_line[3:] if len(status_line) >= 4 else status_line
    return path.split(" -> ")[-1].strip('"')


def capture_git_state(repo_root: Path) -> dict:
    status = _git_value(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    source_status = []
    for line in status:
        path = Path(_status_path(line))
        if path.parts and path.parts[0] not in GENERATED_ROOTS:
            source_status.append(line)

    submodules = _git_value(repo_root, "submodule", "status").splitlines()
    return {
        "commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "tree": _git_value(repo_root, "rev-parse", "HEAD^{tree}"),
        "branch": _git_value(repo_root, "branch", "--show-current"),
        "describe": _git_value(repo_root, "describe", "--always", "--tags", "--dirty"),
        "commit_timestamp": _git_value(repo_root, "show", "-s", "--format=%cI", "HEAD"),
        "dirty": bool(status),
        "source_dirty": bool(source_status),
        "status": status,
        "source_status": source_status,
        "generated_roots_ignored_for_source_dirty": sorted(GENERATED_ROOTS),
        "submodules": submodules,
    }


def capture_installed_distributions() -> list[dict[str, str]]:
    installed: dict[str, dict[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        normalised_name = _normalise_distribution_name(raw_name)
        installed[normalised_name] = {
            "name": raw_name,
            "version": distribution.version,
        }
    return [installed[name] for name in sorted(installed)]


def parse_requirements(requirements_path: Path) -> dict:
    requirements = []
    unparsed = []
    if not requirements_path.is_file():
        return {"path": str(requirements_path), "requirements": [], "unparsed": []}

    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = EXACT_REQUIREMENT_RE.fullmatch(line)
        if match is None:
            unparsed.append(line)
            continue
        requirements.append(
            {
                "name": match.group("name"),
                "version": match.group("version"),
                "marker": match.group("marker") or "",
                "raw": line,
            }
        )
    return {
        "path": str(requirements_path),
        "requirements": requirements,
        "unparsed": unparsed,
    }


def compare_declared_and_installed(
    declared: Iterable[dict[str, str]], installed: Iterable[dict[str, str]]
) -> dict:
    declared = list(declared)
    installed_by_name = {
        _normalise_distribution_name(item["name"]): item["version"]
        for item in installed
    }
    missing = []
    version_mismatches = []
    for requirement in declared:
        name = requirement["name"]
        installed_version = installed_by_name.get(_normalise_distribution_name(name))
        if installed_version is None:
            missing.append(name)
        elif installed_version != requirement["version"]:
            version_mismatches.append(
                {
                    "name": name,
                    "declared": requirement["version"],
                    "installed": installed_version,
                }
            )
    return {
        "declared_count": len(declared),
        "installed_count": len(installed_by_name),
        "missing": missing,
        "version_mismatches": version_mismatches,
    }


def _module_show_config(module) -> str:
    show_config = getattr(module, "show_config", None)
    if show_config is None:
        return ""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        show_config()
    return output.getvalue().strip()


def capture_scientific_stack() -> dict:
    captures = {}
    module_names = ("numpy", "scipy", "matplotlib", "shapely", "vtk")
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            record = {"version": getattr(module, "__version__", "")}
            if module_name in {"numpy", "scipy"}:
                record["build_configuration"] = _module_show_config(module)
            elif module_name == "matplotlib":
                record["backend"] = module.get_backend()
                try:
                    ft2font = importlib.import_module("matplotlib.ft2font")
                    record["freetype_version"] = getattr(
                        ft2font, "__freetype_version__", ""
                    )
                except Exception as exc:  # pragma: no cover - optional build detail
                    record["freetype_error"] = repr(exc)
            elif module_name == "shapely":
                try:
                    geos = importlib.import_module("shapely.geos")
                    record["geos_version"] = getattr(geos, "geos_version_string", "")
                except Exception as exc:  # pragma: no cover - version-dependent API
                    record["geos_error"] = repr(exc)
            elif module_name == "vtk":
                record["vtk_version"] = module.vtkVersion.GetVTKVersion()
            captures[module_name] = record
        except Exception as exc:
            captures[module_name] = {"import_error": repr(exc)}
    return captures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_input_fingerprints(repo_root: Path) -> list[dict]:
    paths = {repo_root / relative_path for relative_path in FINGERPRINT_PATHS}
    for pattern in FINGERPRINT_GLOBS:
        paths.update(repo_root.glob(pattern))
    fingerprints = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        fingerprints.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return fingerprints


def capture_environment(
    repo_root: Path = REPO,
    *,
    include_scientific_stack: bool = True,
    include_pip_check: bool = True,
) -> dict:
    repo_root = repo_root.resolve()
    installed = capture_installed_distributions()
    parsed_requirements = parse_requirements(repo_root / "requirements.txt")
    requirement_comparison = compare_declared_and_installed(
        parsed_requirements["requirements"], installed
    )
    requirement_comparison["unparsed"] = parsed_requirements["unparsed"]

    uname = platform.uname()
    runtime = {
        "python_version": platform.python_version(),
        "python_version_detail": sys.version,
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "python_compiler": platform.python_compiler(),
        "python_build": list(platform.python_build()),
        "python_abi": sysconfig.get_config_var("SOABI") or "",
        "sysconfig_platform": sysconfig.get_platform(),
        "byteorder": sys.byteorder,
        "filesystem_encoding": sys.getfilesystemencoding(),
        "preferred_encoding": locale.getpreferredencoding(False),
    }
    system = {
        "platform": platform.platform(),
        "uname": uname._asdict(),
        "mac_version": list(platform.mac_ver()),
        "libc_version": list(platform.libc_ver()),
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "root": str(repo_root),
            **capture_git_state(repo_root),
        },
        "runtime": runtime,
        "system": system,
        "environment_variables": {
            name: os.environ[name]
            for name in REPRODUCIBILITY_ENVIRONMENT_VARIABLES
            if name in os.environ
        },
        "declared_requirements": parsed_requirements,
        "installed_distributions": installed,
        "requirement_comparison": requirement_comparison,
        "input_fingerprints": capture_input_fingerprints(repo_root),
    }
    if include_scientific_stack:
        record["scientific_stack"] = capture_scientific_stack()
    if include_pip_check:
        record["pip_check"] = _command_result(
            [sys.executable, "-m", "pip", "check"], cwd=repo_root
        )
    return record


def write_environment_capture(output_path: Path, record: dict) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary_path.replace(output_path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="destination JSON file, normally inside a release result root",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO,
        help="repository to fingerprint (default: this repository)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    record = capture_environment(args.repo_root)
    write_environment_capture(args.output, record)
    comparison = record["requirement_comparison"]
    print(f"Wrote environment capture: {args.output.resolve()}")
    print(
        "Declared-environment differences: "
        f"{len(comparison['missing'])} missing, "
        f"{len(comparison['version_mismatches'])} version mismatches"
    )
    print(
        "Git source state: "
        f"{'dirty' if record['repository']['source_dirty'] else 'clean'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
