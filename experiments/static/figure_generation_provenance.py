"""Shared provenance for dedicated paper-figure generation commands."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from main.structs.meshes.merge_mesh import MergeMesh


REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_STATUS_ROOTS = {"logs", "output", "plots", "results", "tmp"}
RELEASE_ANCHOR_FILES = (
    "submission_config.resolved.json",
    "sweep_manifest.json",
    "perturbed_sweep.csv",
    "SHA256SUMS",
)


def frozen_reconstruction_profile(
    *,
    plic_fallback: str = "LVIRA",
    corner_behavior_profile: str = MergeMesh.default_corner_behavior_profile,
    rescue_profile: str = MergeMesh.default_rescue_profile,
) -> dict[str, str]:
    if plic_fallback not in {"Youngs", "ELVIRA", "LVIRA"}:
        raise ValueError(f"Unsupported PLIC fallback: {plic_fallback}")
    if corner_behavior_profile not in MergeMesh.corner_behavior_profiles:
        raise ValueError(
            f"Unsupported corner behavior profile: {corner_behavior_profile}"
        )
    if rescue_profile not in MergeMesh.rescue_profiles:
        raise ValueError(f"Unsupported rescue profile: {rescue_profile}")
    return {
        "plic_fallback": plic_fallback,
        "corner_behavior_profile": corner_behavior_profile,
        "rescue_profile": rescue_profile,
    }


def reconstruction_cli_args(experiment: str, profile: dict[str, str]) -> list[str]:
    args = [
        "--plic_fallback",
        profile["plic_fallback"],
        "--corner_behavior_profile",
        profile["corner_behavior_profile"],
    ]
    if experiment == "zalesak":
        args.extend(["--rescue_profile", profile["rescue_profile"]])
    return args


def _git_output(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip()


def generation_provenance(
    *,
    profile: dict[str, str],
    profile_application: str,
) -> dict:
    status = []
    for line in _git_output(["status", "--short"]).splitlines():
        path = line[3:].split(" -> ")[-1]
        parts = Path(path).parts
        if parts and parts[0] in EXCLUDED_STATUS_ROOTS:
            continue
        status.append(line)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git_output(["rev-parse", "HEAD"]),
        "source_branch": _git_output(["branch", "--show-current"]),
        "source_dirty": bool(status),
        "source_status": status,
        "reconstruction_profile": dict(profile),
        "profile_application": profile_application,
    }


def vector_figure_artifacts(review_png: Path) -> dict[str, str]:
    review_png = Path(review_png).resolve()
    return {
        "pdf": str(review_png.with_suffix(".pdf")),
        "png_review_300dpi": str(review_png),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _producer_generation(payload: Mapping[str, object]) -> Mapping[str, object] | None:
    direct = payload.get("generation_provenance")
    if isinstance(direct, dict):
        return direct
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        nested = metadata.get("generation_provenance")
        if isinstance(nested, dict):
            return nested
    return None


def _release_profile(config: Mapping[str, object]) -> dict[str, str]:
    production = config.get("production_method")
    if not isinstance(production, dict):
        raise ValueError("Resolved release config lacks production_method")
    profile = {
        "plic_fallback": production.get("unresolved_orientation_fallback"),
        "corner_behavior_profile": production.get("corner_behavior_profile"),
        "rescue_profile": production.get("rescue_profile"),
    }
    if any(not isinstance(value, str) or not value for value in profile.values()):
        raise ValueError(
            "Resolved release config has an incomplete reconstruction profile"
        )
    return profile


def _parse_release_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"Malformed SHA256SUMS line {line_number}")
        digest, relative = parts
        if relative in checksums:
            raise ValueError(f"Duplicate SHA256SUMS path: {relative}")
        checksums[relative] = digest
    return checksums


def release_figure_anchor(release_root: Path) -> dict:
    root = Path(release_root).resolve()
    if not root.is_dir():
        raise ValueError(f"Final release root is not a directory: {root}")
    required = {name: root / name for name in RELEASE_ANCHOR_FILES}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Final release lacks required files: {', '.join(missing)}")

    config = _load_json_object(required["submission_config.resolved.json"])
    sweep = _load_json_object(required["sweep_manifest.json"])
    source = config.get("source")
    if not isinstance(source, dict):
        raise ValueError("Resolved release config lacks source provenance")
    source_commit = source.get("target_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("Resolved release config lacks a full target commit")
    if sweep.get("status") != "completed":
        raise ValueError("Final release sweep manifest is not completed")

    checksums = _parse_release_checksums(required["SHA256SUMS"])
    for relative in RELEASE_ANCHOR_FILES[:-1]:
        actual = file_sha256(required[relative])
        if checksums.get(relative) != actual:
            raise ValueError(f"Final release checksum does not prove {relative}")

    return {
        "root": str(root),
        "name": root.name,
        "source_commit": source_commit,
        "reconstruction_profile": _release_profile(config),
        "artifacts": {
            relative: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for relative, path in required.items()
        },
    }


def artifact_provenance(path: Path, role: str, release_root: Path) -> dict:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"Provenance input does not exist: {resolved}")
    root = Path(release_root).resolve()
    try:
        release_relative_path = resolved.relative_to(root).as_posix()
    except ValueError:
        release_relative_path = None
    return {
        "role": role,
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "release_relative_path": release_relative_path,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
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


def write_authoritative_figure_provenance(
    path: Path,
    *,
    generator: str,
    release_root: Path,
    generation: Mapping[str, object],
    producer_manifest: Path,
    inputs: Sequence[tuple[str, Path]],
    outputs: Mapping[str, Path],
) -> dict:
    """Bind generated candidates to a clean source, release, inputs, and hashes."""
    anchor = release_figure_anchor(release_root)
    source_commit = generation.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("Generator provenance lacks a full source commit")
    if generation.get("source_dirty") is not False or generation.get("source_status"):
        raise ValueError("Authoritative figure generation requires a clean source tree")
    if generation.get("reconstruction_profile") != anchor["reconstruction_profile"]:
        raise ValueError("Generator reconstruction profile differs from final release")
    if not outputs:
        raise ValueError("Authoritative figure provenance requires candidate outputs")

    producer_manifest = Path(producer_manifest).resolve()
    producer_payload = _load_json_object(producer_manifest)
    producer_generation = _producer_generation(producer_payload)
    if not isinstance(producer_generation, Mapping):
        raise ValueError("Producer manifest lacks generation provenance")
    for key in (
        "source_commit",
        "source_dirty",
        "source_status",
        "reconstruction_profile",
    ):
        if producer_generation.get(key) != generation.get(key):
            raise ValueError(f"Producer and figure provenance disagree on {key}")
    input_records = [
        artifact_provenance(producer_manifest, "producer_manifest", release_root)
    ]
    input_records.extend(
        artifact_provenance(input_path, role, release_root)
        for role, input_path in inputs
    )
    output_records = []
    seen_ids = set()
    for candidate_id, output_path in outputs.items():
        if candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate ID: {candidate_id}")
        seen_ids.add(candidate_id)
        resolved = Path(output_path).resolve()
        if not resolved.is_file() or resolved.suffix.lower() != ".pdf":
            raise ValueError(f"Candidate output is not a PDF: {resolved}")
        output_records.append(
            {
                "candidate_id": candidate_id,
                "path": str(resolved),
                "sha256": file_sha256(resolved),
            }
        )

    manifest = {
        "schema_version": 1,
        "manifest_type": "final_figure_generation",
        "status": "completed",
        "generator": generator,
        "generation_provenance": dict(generation),
        "release": anchor,
        "inputs": input_records,
        "outputs": output_records,
    }
    _atomic_write_json(path, manifest)
    return manifest
