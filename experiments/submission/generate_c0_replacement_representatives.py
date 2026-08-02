"""Generate and audit globally continuous appendix C0 representatives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.static import generate_section6_maintext_figures as figures


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "submission" / "c0_global_representatives_20260801"
)
CONTINUITY_TOLERANCE = 1.0e-8
CONSERVATION_TOLERANCE = 1.0e-10

ELLIPSE_RUNS = {
    "linear": "c0_replacement_ellipse_linear_n32_w010_s0",
    "linear+C0": "ellipse_joint_c0_posthoc_n32_w010_20260801",
    "circular": "c0_replacement_ellipse_circular_n32_w010_s0",
}
ELLIPSE_SUMMARY = (
    REPO_ROOT
    / "results"
    / "submission"
    / "ellipse_joint_c0_posthoc_n32_w010_20260801"
    / "case_summary.csv"
)
ZALESAK_RUN = (
    "c0_continuous_case_review_20260801_"
    "perturb_sweep_zalesak_circularplusc0_r1p0_w0p1_s0"
)
ZALESAK_PDF = (
    REPO_ROOT
    / "results"
    / "submission"
    / "c0_continuous_case_review_20260801"
    / "representative_cases"
    / "zalesak_appendix_c0_representative_clean.pdf"
)


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _generation_provenance() -> dict[str, Any]:
    status = _git_output("status", "--short").splitlines()
    source_status = [
        line
        for line in status
        if not line[3:].startswith(("plots/", "results/", "tmp/"))
    ]
    return {
        "source_commit": _git_output("rev-parse", "HEAD"),
        "source_branch": _git_output("branch", "--show-current"),
        "source_dirty": bool(source_status),
        "source_status": source_status,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_case(path: Path, case_index: int) -> dict[str, str]:
    with path.open(newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if int(row["case_index"]) == case_index
        ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one case {case_index} row in {path}, got {len(rows)}"
        )
    return rows[0]


def _facet_paths(run_name: str, case_index: int) -> tuple[Path, Path]:
    root = REPO_ROOT / "plots" / run_name / "vtk" / "reconstructed" / "facets"
    return root / f"{case_index}.vtp", root / f"{case_index}.facet_metadata.json"


def _endpoint_continuity(metadata_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    endpoints = []
    for primitive in metadata["primitives"]:
        endpoints.extend((primitive["p_left"], primitive["p_right"]))
    points = np.asarray(endpoints, dtype=float)
    if len(points) < 4:
        raise ValueError(f"Too few facet endpoints in {metadata_path}")
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    partner_gaps = np.min(distances, axis=1)
    return {
        "primitive_count": len(metadata["primitives"]),
        "endpoint_count": len(points),
        "max_endpoint_partner_gap": float(np.max(partner_gaps)),
        "mean_endpoint_partner_gap": float(np.mean(partner_gaps)),
        "endpoints_above_tolerance": int(
            np.count_nonzero(partner_gaps > CONTINUITY_TOLERANCE)
        ),
        "globally_continuous": bool(np.max(partner_gaps) <= CONTINUITY_TOLERANCE),
    }


def _run_provenance(
    run_name: str,
    case_index: int,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    run_root = REPO_ROOT / "plots" / run_name
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parameters = manifest["parameters"]
    for key, value in expected.items():
        if parameters.get(key) != value:
            raise ValueError(
                f"{run_name} has {key}={parameters.get(key)!r}; expected {value!r}"
            )
    case_indices = parameters.get("case_indices")
    if case_indices is not None:
        if isinstance(case_indices, str):
            case_indices = [
                int(item.strip()) for item in case_indices.split(",") if item.strip()
            ]
        if case_index not in case_indices:
            raise ValueError(f"{run_name} does not contain case {case_index}")
    vtp_path, metadata_path = _facet_paths(run_name, case_index)
    for path in (vtp_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "run_name": run_name,
        "source_commit": manifest["source_commit"],
        "parameters": parameters,
        "run_manifest": str(manifest_path.resolve()),
        "run_manifest_sha256": _sha256(manifest_path),
        "facet_vtp": str(vtp_path.resolve()),
        "facet_vtp_sha256": _sha256(vtp_path),
        "facet_metadata": str(metadata_path.resolve()),
        "facet_metadata_sha256": _sha256(metadata_path),
    }


def _ellipse_audit(case_index: int) -> dict[str, Any]:
    summary = _csv_case(ELLIPSE_SUMMARY, case_index)
    _, metadata_path = _facet_paths(ELLIPSE_RUNS["linear+C0"], case_index)
    continuity = _endpoint_continuity(metadata_path)
    maximum_gap = float(summary["max_gap_after"])
    maximum_area_residual = float(summary["max_relative_area_residual_after"])
    if int(summary["continuous_after"]) != 1 or maximum_gap > CONTINUITY_TOLERANCE:
        raise ValueError(f"Ellipse case {case_index} is not globally continuous")
    if not continuity["globally_continuous"]:
        raise ValueError(f"Ellipse case {case_index} fails the exported-endpoint audit")
    if maximum_area_residual > CONSERVATION_TOLERANCE:
        raise ValueError(f"Ellipse case {case_index} fails the conservation guard")
    return {
        **continuity,
        "eligible_joins": int(summary["eligible_joins"]),
        "max_topology_join_gap": maximum_gap,
        "mean_topology_join_gap": float(summary["mean_gap_after"]),
        "max_relative_zone_area_residual": maximum_area_residual,
        "hausdorff": float(summary["hausdorff_after"]),
        "facet_gap": float(summary["facet_gap_metric_after"]),
        "joint_components_solved": int(summary["components_solved"]),
        "joint_components_failed": int(summary["components_failed"]),
    }


def _zalesak_audit(case_index: int) -> dict[str, Any]:
    run_root = REPO_ROOT / "plots" / ZALESAK_RUN
    metrics = _csv_case(run_root / "metrics" / "case_metrics.csv", case_index)
    _, metadata_path = _facet_paths(ZALESAK_RUN, case_index)
    continuity = _endpoint_continuity(metadata_path)
    if not continuity["globally_continuous"]:
        raise ValueError(f"Zalesak case {case_index} fails the exported-endpoint audit")
    area_error = float(metrics["area_error"])
    if area_error > CONSERVATION_TOLERANCE:
        raise ValueError(f"Zalesak case {case_index} fails the conservation guard")
    return {
        **continuity,
        "mean_topology_join_gap": float(metrics["facet_gap"]),
        "max_topology_join_gap": continuity["max_endpoint_partner_gap"],
        "global_relative_area_error": area_error,
        "hausdorff": float(metrics["hausdorff"]),
        "facet_gap": float(metrics["facet_gap"]),
    }


def generate(
    output_dir: Path, *, ellipse_case: int, zalesak_case: int
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ellipse_sources = {
        label: _run_provenance(
            run_name,
            ellipse_case,
            expected={
                "facet_algo": "linear" if label != "circular" else "circular",
                "do_c0": label == "linear+C0",
                "resolution": 0.32,
                "perturb_wiggle": 0.1,
                "perturb_seed": 0,
                "plic_fallback": "LVIRA",
                "corner_behavior_profile": "pre_f8_corner",
            },
        )
        for label, run_name in ELLIPSE_RUNS.items()
    }
    zalesak_source = _run_provenance(
        ZALESAK_RUN,
        zalesak_case,
        expected={
            "facet_algo": "circular",
            "do_c0": True,
            "resolution": 1.0,
            "perturb_wiggle": 0.1,
            "perturb_seed": 0,
            "plic_fallback": "LVIRA",
            "corner_behavior_profile": "pre_f8_corner",
            "rescue_profile": "exact_linear_support_only",
        },
    )

    ellipse_png = output_dir / "ellipses_appendix_connected_c0_representative_clean.png"
    figures._generate_representative_figure(
        exp_name="ellipses",
        spec={
            "resolution": 0.32,
            "wiggle": 0.1,
            "seed": 0,
            "case_index": ellipse_case,
            "methods": [
                ("linear", "Ours (linear)"),
                ("linear+C0", r"Ours (linear, connected $C^0$)"),
                ("circular", "Ours (circular)"),
            ],
            "source_runs": ELLIPSE_RUNS,
            "min_span": 66.0,
            "margin_frac": 0.12,
            "inset": {"kind": "ellipse_max_curvature", "half_span": 5.0},
            "show_main_endpoints": False,
            "show_inset_endpoints": True,
        },
        out_path=ellipse_png,
    )
    ellipse_pdf = ellipse_png.with_suffix(".pdf")
    if not ZALESAK_PDF.is_file():
        raise FileNotFoundError(ZALESAK_PDF)

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": _generation_provenance(),
        "criteria": {
            "maximum_endpoint_gap": CONTINUITY_TOLERANCE,
            "maximum_relative_conservation_residual": CONSERVATION_TOLERANCE,
        },
        "ellipse": {
            "case_index": ellipse_case,
            "N": 32,
            "perturbation": 0.1,
            "mesh_seed": 0,
            "correction": "connected-component joint C0 postprocessor",
            "audit": _ellipse_audit(ellipse_case),
            "sources": ellipse_sources,
            "pdf": str(ellipse_pdf.resolve()),
            "pdf_sha256": _sha256(ellipse_pdf),
        },
        "zalesak": {
            "case_index": zalesak_case,
            "N": 100,
            "perturbation": 0.1,
            "mesh_seed": 0,
            "correction": "one-pass guarded C0 correction",
            "audit": _zalesak_audit(zalesak_case),
            "source": zalesak_source,
            "pdf": str(ZALESAK_PDF.resolve()),
            "pdf_sha256": _sha256(ZALESAK_PDF),
            "disposition": "retained without regeneration after audit",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {**payload, "manifest": str(manifest_path.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ellipse-case", type=int, default=9)
    parser.add_argument("--zalesak-case", type=int, default=22)
    args = parser.parse_args(argv)
    result = generate(
        args.output_dir.resolve(),
        ellipse_case=args.ellipse_case,
        zalesak_case=args.zalesak_case,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
