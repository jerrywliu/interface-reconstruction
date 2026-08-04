"""Generate and audit the approved globally continuous appendix C0 examples."""

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
from experiments.static.figure_generation_provenance import (
    frozen_reconstruction_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "submission" / "c0_global_representatives_20260801"
)
CONTINUITY_TOLERANCE = 1.0e-8
CONSERVATION_TOLERANCE = 1.0e-10
ELLIPSE_CASE = 9
ZALESAK_CASE = 22

# These defaults preserve the reviewed local diagnostic workflow. The final
# publication orchestrator supplies attested run names from its private sweep.
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
ZALESAK_RUNS = {
    "circular": (
        "c0_continuous_case_review_20260801_"
        "perturb_sweep_zalesak_circular_r1p0_w0p1_s0"
    ),
    "circular+C0": (
        "c0_continuous_case_review_20260801_"
        "perturb_sweep_zalesak_circularplusc0_r1p0_w0p1_s0"
    ),
    "circular+corner": (
        "c0_continuous_case_review_20260801_"
        "perturb_sweep_zalesak_circularpluscorner_r1p0_w0p1_s0"
    ),
}
ZALESAK_RUN = ZALESAK_RUNS["circular+C0"]


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
        "reconstruction_profile": frozen_reconstruction_profile(),
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


def _facet_paths(plots_root: Path, run_name: str, case_index: int) -> tuple[Path, Path]:
    root = plots_root / run_name / "vtk" / "reconstructed" / "facets"
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
    plots_root: Path,
    run_name: str,
    case_index: int,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    run_root = plots_root / run_name
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
    vtp_path, metadata_path = _facet_paths(plots_root, run_name, case_index)
    case_metrics = run_root / "metrics" / "case_metrics.csv"
    case_geometry = run_root / "metrics" / "case_geometry.jsonl"
    mesh_path = run_root / "vtk" / "mesh.vtk"
    for path in (
        manifest_path,
        case_metrics,
        case_geometry,
        mesh_path,
        vtp_path,
        metadata_path,
    ):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    return {
        "run_name": run_name,
        "source_commit": manifest["source_commit"],
        "parameters": parameters,
        "files": {
            "run_manifest": {
                "relative_path": "run_manifest.json",
                "sha256": _sha256(manifest_path),
            },
            "case_geometry": {
                "relative_path": "metrics/case_geometry.jsonl",
                "sha256": _sha256(case_geometry),
            },
            "case_metrics": {
                "relative_path": "metrics/case_metrics.csv",
                "sha256": _sha256(case_metrics),
            },
            "mesh": {
                "relative_path": "vtk/mesh.vtk",
                "sha256": _sha256(mesh_path),
            },
            "facet_vtp": {
                "relative_path": f"vtk/reconstructed/facets/{case_index}.vtp",
                "sha256": _sha256(vtp_path),
            },
            "facet_metadata": {
                "relative_path": (
                    f"vtk/reconstructed/facets/{case_index}.facet_metadata.json"
                ),
                "sha256": _sha256(metadata_path),
            },
        },
    }


def _ellipse_audit(
    plots_root: Path,
    summary_path: Path,
    connected_run: str,
    case_index: int,
) -> dict[str, Any]:
    summary = _csv_case(summary_path, case_index)
    _, metadata_path = _facet_paths(plots_root, connected_run, case_index)
    continuity = _endpoint_continuity(metadata_path)
    maximum_gap = float(summary["max_gap_after"])
    maximum_area_residual = float(summary["max_relative_area_residual_after"])
    if int(summary["continuous_after"]) != 1 or maximum_gap > CONTINUITY_TOLERANCE:
        raise ValueError(f"Ellipse case {case_index} is not globally continuous")
    if not continuity["globally_continuous"]:
        raise ValueError(f"Ellipse case {case_index} fails the exported-endpoint audit")
    if maximum_area_residual > CONSERVATION_TOLERANCE:
        raise ValueError(f"Ellipse case {case_index} fails the conservation guard")
    if int(summary["components_failed"]) != 0:
        raise ValueError(f"Ellipse case {case_index} has an unsolved joint component")
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


def _zalesak_audit(
    plots_root: Path, guarded_run: str, case_index: int
) -> dict[str, Any]:
    run_root = plots_root / guarded_run
    metrics = _csv_case(run_root / "metrics" / "case_metrics.csv", case_index)
    _, metadata_path = _facet_paths(plots_root, guarded_run, case_index)
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


def _generate_variants(
    *,
    output_dir: Path,
    experiment: str,
    spec: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    artifacts = {}
    for variant, show_main_endpoints in (
        ("with_endpoints", True),
        ("clean", False),
    ):
        png_path = output_dir / f"{experiment}_appendix_c0_representative_{variant}.png"
        figures._generate_representative_figure(
            exp_name=experiment,
            spec=figures._endpoint_visibility_spec(
                spec, show_main_endpoints=show_main_endpoints
            ),
            out_path=png_path,
        )
        pdf_path = png_path.with_suffix(".pdf")
        artifacts[variant] = {
            "pdf": pdf_path.name,
            "pdf_sha256": _sha256(pdf_path),
            "png_review_300dpi": png_path.name,
            "png_review_300dpi_sha256": _sha256(png_path),
        }
    return artifacts


def generate(
    output_dir: Path,
    *,
    plots_root: Path,
    ellipse_runs: Mapping[str, str],
    ellipse_summary: Path,
    zalesak_runs: Mapping[str, str],
    ellipse_case: int,
    zalesak_case: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_root = plots_root.resolve()
    figures.PLOTS_ROOT = plots_root
    if ellipse_case != ELLIPSE_CASE or zalesak_case != ZALESAK_CASE:
        raise ValueError(
            f"Approved representatives are ellipse {ELLIPSE_CASE} and "
            f"Zalesak {ZALESAK_CASE}"
        )
    if set(ellipse_runs) != {"linear", "linear+C0", "circular"}:
        raise ValueError("Ellipse source-run labels differ from the approved panel")
    if set(zalesak_runs) != {"circular", "circular+C0", "circular+corner"}:
        raise ValueError("Zalesak source-run labels differ from the approved panel")

    ellipse_sources = {
        label: _run_provenance(
            plots_root,
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
        for label, run_name in ellipse_runs.items()
    }
    zalesak_sources = {
        label: _run_provenance(
            plots_root,
            run_name,
            zalesak_case,
            expected={
                "facet_algo": (
                    "circular+corner" if label == "circular+corner" else "circular"
                ),
                "do_c0": label == "circular+C0",
                "resolution": 1.0,
                "perturb_wiggle": 0.1,
                "perturb_seed": 0,
                "plic_fallback": "LVIRA",
                "corner_behavior_profile": "pre_f8_corner",
                "rescue_profile": "exact_linear_support_only",
            },
        )
        for label, run_name in zalesak_runs.items()
    }

    ellipse_outputs = _generate_variants(
        output_dir=output_dir,
        experiment="ellipses",
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
            "source_runs": dict(ellipse_runs),
            "min_span": 66.0,
            "margin_frac": 0.12,
            "inset": {"kind": "ellipse_max_curvature", "half_span": 5.0},
            "show_inset_endpoints": True,
        },
    )
    zalesak_outputs = _generate_variants(
        output_dir=output_dir,
        experiment="zalesak",
        spec={
            "resolution": 1.0,
            "wiggle": 0.1,
            "seed": 0,
            "case_index": zalesak_case,
            "methods": [
                ("circular", "Ours (circular)"),
                ("circular+C0", "Ours (circular, C0)"),
                ("circular+corner", "Ours (circular+corner)"),
            ],
            "source_runs": dict(zalesak_runs),
            "min_span": 42.0,
            "margin_frac": 0.12,
            "inset": {"kind": "zalesak_corner", "zoom": 3.0},
            "show_inset_endpoints": True,
        },
    )

    payload = {
        "schema_version": 1,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_provenance": _generation_provenance(),
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
            "postprocessor_case_summary_sha256": _sha256(ellipse_summary),
            "audit": _ellipse_audit(
                plots_root,
                ellipse_summary,
                ellipse_runs["linear+C0"],
                ellipse_case,
            ),
            "sources": ellipse_sources,
            "outputs": ellipse_outputs,
        },
        "zalesak": {
            "case_index": zalesak_case,
            "N": 100,
            "perturbation": 0.1,
            "mesh_seed": 0,
            "correction": "one-pass guarded C0 correction",
            "audit": _zalesak_audit(
                plots_root, zalesak_runs["circular+C0"], zalesak_case
            ),
            "sources": zalesak_sources,
            "outputs": zalesak_outputs,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {**payload, "manifest": str(manifest_path.resolve())}


def _run_map(raw: Sequence[str], expected_labels: Sequence[str]) -> dict[str, str]:
    parsed = {}
    for item in raw:
        label, separator, run_name = item.partition("=")
        if not separator or not label or not run_name or label in parsed:
            raise ValueError(f"Invalid or duplicate LABEL=RUN source: {item}")
        parsed[label] = run_name
    if set(parsed) != set(expected_labels):
        raise ValueError(
            f"Run labels must be {sorted(expected_labels)}, got {sorted(parsed)}"
        )
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plots-root", type=Path, default=REPO_ROOT / "plots")
    parser.add_argument("--ellipse-summary", type=Path, default=ELLIPSE_SUMMARY)
    parser.add_argument(
        "--ellipse-run",
        action="append",
        default=[],
        metavar="LABEL=RUN",
        help="repeat for linear, linear+C0, and circular",
    )
    parser.add_argument(
        "--zalesak-run",
        action="append",
        default=[],
        metavar="LABEL=RUN",
        help="repeat for circular, circular+C0, and circular+corner",
    )
    parser.add_argument("--ellipse-case", type=int, default=ELLIPSE_CASE)
    parser.add_argument("--zalesak-case", type=int, default=ZALESAK_CASE)
    args = parser.parse_args(argv)
    ellipse_runs = (
        _run_map(args.ellipse_run, ELLIPSE_RUNS) if args.ellipse_run else ELLIPSE_RUNS
    )
    zalesak_runs = (
        _run_map(args.zalesak_run, ZALESAK_RUNS) if args.zalesak_run else ZALESAK_RUNS
    )
    result = generate(
        args.output_dir.resolve(),
        plots_root=args.plots_root,
        ellipse_runs=ellipse_runs,
        ellipse_summary=args.ellipse_summary.resolve(),
        zalesak_runs=zalesak_runs,
        ellipse_case=args.ellipse_case,
        zalesak_case=args.zalesak_case,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
