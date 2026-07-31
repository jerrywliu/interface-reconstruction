"""Shared provenance for dedicated paper-figure generation commands."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from main.structs.meshes.merge_mesh import MergeMesh


REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_STATUS_ROOTS = {"logs", "output", "plots", "results", "tmp"}


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
