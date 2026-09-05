"""Run both frozen Cartesian bare-PCIC variants on the project benchmarks."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments.baselines.external_runner import (
    ExternalCellContext,
    UnsupportedExternalCell,
    UnresolvedExternalCell,
    run_external_static_baseline,
)
from experiments.baselines.project_benchmarks import (
    DEFAULT_BENCHMARKS,
    DEFAULT_CASE_INDICES,
    DEFAULT_RESOLUTIONS,
    DOMAIN_SIZE,
    canonical_benchmark_cases,
)
from experiments.baselines.project_smoke import (
    CASE_FIELDS,
    CELL_FIELDS,
    aggregate_case_rows,
    append_csv,
    sampled_directed_hausdorff,
    sampled_symmetric_hausdorff,
    write_csv,
    write_json,
)
from main.algos.baselines.external_method_adapters import adapt_pcic_cell
from main.algos.baselines.external_metrics import (
    conservation_metrics,
    shared_edge_gap_metrics,
)
from main.algos.baselines.pcic import (
    PCICCircle,
    PCICError,
    PCICUnsupportedGeometry,
    reconstruct_bare_pcic_cartesian_cell,
    source_variant_for_correction,
)
from main.structs.facets.linear_facet import LinearFacet
from main.structs.polys.base_polygon import BasePolygon


METHOD = "PCIC"
CORRECTIONS = ("translate_center", "adjust_radius")
BOUNDARY_POLICIES = ("unsupported", "zero_exterior")

PCIC_CASE_FIELDS = CASE_FIELDS + (
    "correction",
    "reconstructed_components",
    "reconstructed_primitives",
    "multi_component_cells",
    "boundary_7x7_unsupported_cells",
    "straight_limit_fallback_cells",
    "component_count_histogram_json",
    "fallback_reason_counts_json",
    "unsupported_reason_counts_json",
    "unresolved_reason_counts_json",
)

PCIC_CELL_FIELDS = CELL_FIELDS + (
    "correction",
    "phase",
    "component_count",
    "primitive_count",
    "boundary_crossings",
    "source_center_x",
    "source_center_y",
    "source_radius",
    "corrected_center_x",
    "corrected_center_y",
    "corrected_radius",
    "curvature_absolute_error_max",
    "outcome_reason",
    "exact_parameters_json",
)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete_7x7_block(mesh, x: int, y: int):
    if x < 3 or y < 3 or x + 3 >= len(mesh.polys) or y + 3 >= len(mesh.polys[0]):
        return None
    return tuple(
        tuple(mesh.polys[x + dx][y + dy] for dy in range(-3, 4)) for dx in range(-3, 4)
    )


def _cartesian_cell_spacing(mesh) -> tuple[float, float]:
    if len(mesh.polys) < 2 or len(mesh.polys[0]) < 2:
        raise ValueError("PCIC ghost padding requires at least a 2-by-2 Cartesian mesh")
    x0 = float(np.mean([point[0] for point in mesh.polys[0][0].points]))
    x1 = float(np.mean([point[0] for point in mesh.polys[1][0].points]))
    y0 = float(np.mean([point[1] for point in mesh.polys[0][0].points]))
    y1 = float(np.mean([point[1] for point in mesh.polys[0][1].points]))
    return x1 - x0, y1 - y0


def _zero_exterior_ghost(mesh, x: int, y: int) -> BasePolygon:
    nx = len(mesh.polys)
    ny = len(mesh.polys[0])
    source_x = min(max(x, 0), nx - 1)
    source_y = min(max(y, 0), ny - 1)
    spacing_x, spacing_y = _cartesian_cell_spacing(mesh)
    shift_x = (x - source_x) * spacing_x
    shift_y = (y - source_y) * spacing_y
    source = mesh.polys[source_x][source_y]
    ghost = BasePolygon(
        [[point[0] + shift_x, point[1] + shift_y] for point in source.points]
    )
    ghost.setFraction(0.0)
    ghost.setFractionTolerance(source.getFractionTolerance())
    return ghost


def _pcic_7x7_block(mesh, x: int, y: int, boundary_policy: str):
    if boundary_policy not in BOUNDARY_POLICIES:
        raise ValueError(
            f"Unknown PCIC boundary policy {boundary_policy!r}; "
            f"expected one of {BOUNDARY_POLICIES}"
        )
    block = _complete_7x7_block(mesh, x, y)
    if block is not None or boundary_policy == "unsupported":
        return block
    nx = len(mesh.polys)
    ny = len(mesh.polys[0])
    return tuple(
        tuple(
            mesh.polys[x + dx][y + dy]
            if 0 <= x + dx < nx and 0 <= y + dy < ny
            else _zero_exterior_ghost(mesh, x + dx, y + dy)
            for dy in range(-3, 4)
        )
        for dx in range(-3, 4)
    )


def _pcic_method(mesh, correction: str, boundary_policy: str = "unsupported"):
    variant = source_variant_for_correction(correction)

    def method(context: ExternalCellContext):
        x, y = context.cell_index
        block = _pcic_7x7_block(mesh, x, y, boundary_policy)
        if block is None:
            raise UnsupportedExternalCell(
                "PCIC requires a complete Cartesian 7x7 predictor halo",
                {
                    "boundary_halo_limitation": True,
                    "required_halo": "7x7",
                    "reason_code": "incomplete_7x7_boundary_halo",
                },
            )
        try:
            fitted = reconstruct_bare_pcic_cartesian_cell(
                block,
                correction=correction,
                phase="infer_from_plic",
                center_translation_root_policy="nearest_bracket",
            )
        except PCICUnsupportedGeometry as error:
            raise UnsupportedExternalCell(
                str(error),
                {
                    "boundary_halo_limitation": "complete halo" in str(error),
                    "required_halo": "7x7",
                    "reason_code": type(error).__name__,
                },
            ) from error
        except (PCICError, RuntimeError, ValueError, ArithmeticError) as error:
            raise UnresolvedExternalCell(
                str(error),
                {
                    "exception_type": type(error).__name__,
                    "reason_code": type(error).__name__,
                },
            ) from error

        record = adapt_pcic_cell(
            fitted,
            cell_index=context.cell_index,
            polygon=context.polygon_points,
            target_phase_area=context.target_phase_area,
            source_variant=variant,
        )
        diagnostics = dict(record.diagnostics)
        diagnostics.update(
            {
                "correction": correction,
                "required_halo": "7x7",
                "boundary_policy": boundary_policy,
                "phase_policy": "infer_from_plic",
                "center_translation_root_policy": "nearest_bracket",
            }
        )
        if isinstance(fitted, PCICCircle):
            diagnostics.update(
                {
                    "source_center": list(fitted.source_center),
                    "source_radius": fitted.source_radius,
                    "corrected_center": list(fitted.center),
                    "corrected_radius": fitted.radius,
                    "component_pairing_status": fitted.component_pairing_status,
                }
            )
        elif isinstance(fitted, LinearFacet):
            diagnostics.update(
                {
                    "reason_code": "published_straight_line_limit",
                    "fallback_endpoints": [list(fitted.pLeft), list(fitted.pRight)],
                }
            )
        return replace(record, diagnostics=diagnostics)

    return method


def _reason(diagnostics: Mapping[str, Any]) -> str:
    return str(
        diagnostics.get("reason_code")
        or diagnostics.get("reason")
        or diagnostics.get("message")
        or "unspecified"
    )


def _curvature_samples(record, case) -> list[tuple[float, float, float]]:
    values = []
    for primitive in record.primitives():
        estimate = (
            0.0 if primitive.kind == "line" else 1.0 / abs(float(primitive.radius))
        )
        truth = float(case.true_curvature_at(primitive.point(0.5)))
        values.append((estimate, truth, abs(estimate - truth)))
    return values


def _cell_rows(result, case, resolution: int, correction: str) -> list:
    rows = []
    for (x, y), record in sorted(result.cells.items()):
        diagnostics = dict(record.diagnostics)
        samples = _curvature_samples(record, case)
        estimate = float(np.mean([item[0] for item in samples])) if samples else None
        truth = float(np.mean([item[1] for item in samples])) if samples else None
        errors = [item[2] for item in samples]
        reconstructed_area = None
        area_residual = None
        try:
            reconstructed_area = record.exact_phase_area()
            area_residual = abs(reconstructed_area - record.target_phase_area)
        except RuntimeError:
            pass

        source_center = diagnostics.get("source_center")
        corrected_center = diagnostics.get("corrected_center")
        exact_parameters = {
            key: diagnostics[key]
            for key in (
                "phase",
                "source_center",
                "source_radius",
                "corrected_center",
                "corrected_radius",
                "boundary_crossings",
                "component_count",
                "fallback_endpoints",
            )
            if key in diagnostics
        }
        rows.append(
            {
                "method": METHOD,
                "variant": result.source_variant,
                "benchmark": case.benchmark,
                "case_index": case.case_index,
                "cells_per_side": resolution,
                "cell_x": x,
                "cell_y": y,
                "status": record.status.value,
                "target_phase_area": record.target_phase_area,
                "reconstructed_phase_area": reconstructed_area,
                "absolute_area_residual": area_residual,
                "curvature_estimate": estimate,
                "true_curvature": truth,
                "curvature_absolute_error": float(np.mean(errors)) if errors else None,
                "curvature_absolute_error_max": max(errors) if errors else None,
                "objective": None,
                "optimizer_success": None,
                "ghf_method": None,
                "correction": correction,
                "phase": diagnostics.get("phase"),
                "component_count": len(record.components),
                "primitive_count": len(record.primitives()),
                "boundary_crossings": diagnostics.get("boundary_crossings"),
                "source_center_x": source_center[0] if source_center else None,
                "source_center_y": source_center[1] if source_center else None,
                "source_radius": diagnostics.get("source_radius"),
                "corrected_center_x": corrected_center[0] if corrected_center else None,
                "corrected_center_y": corrected_center[1] if corrected_center else None,
                "corrected_radius": diagnostics.get("corrected_radius"),
                "outcome_reason": _reason(diagnostics),
                "exact_parameters_json": json.dumps(exact_parameters, sort_keys=True),
                "diagnostics_json": json.dumps(diagnostics, sort_keys=True),
            }
        )
    return rows


def _counter_json(rows: Sequence[Mapping[str, Any]], status: str) -> str:
    counts = Counter(
        str(row["outcome_reason"]) for row in rows if str(row["status"]) == status
    )
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def run_case(
    case,
    resolution: int,
    correction: str,
    output_directory: Path,
    boundary_policy: str = "unsupported",
) -> Dict[str, Any]:
    mesh = case.build_mesh(resolution)
    case.initialize_fractions(mesh)
    cell_size = DOMAIN_SIZE / resolution
    variant = source_variant_for_correction(correction)
    started = time.perf_counter()
    result = run_external_static_baseline(
        mesh,
        _pcic_method(mesh, correction, boundary_policy),
        source_method=METHOD,
        source_variant=variant,
        config={
            "benchmark": case.benchmark,
            "case_index": case.case_index,
            "cells_per_side": resolution,
            "cell_size": cell_size,
            "correction": correction,
            "phase_policy": "infer_from_plic",
            "lls_overcrowded_radius_scale": 0.5,
            "center_translation_root_policy": "nearest_bracket",
            "required_halo": "7x7",
            "boundary_policy": boundary_policy,
        },
    )
    runtime = time.perf_counter() - started
    cell_rows = _cell_rows(result, case, resolution, correction)
    gaps = shared_edge_gap_metrics(result)
    conservation = conservation_metrics(result)
    truth = case.truth_primitives()
    sample_spacing = min(cell_size / 128.0, 2.5e-3)
    hausdorff = sampled_symmetric_hausdorff(result, truth, spacing=sample_spacing)
    reconstruction_to_truth = sampled_directed_hausdorff(
        result, truth, spacing=sample_spacing
    )
    status = result.metadata["status_counts"]
    curvature_errors = np.asarray(
        [
            row["curvature_absolute_error"]
            for row in cell_rows
            if row["curvature_absolute_error"] is not None
        ],
        dtype=float,
    )
    component_histogram = Counter(
        int(row["component_count"])
        for row in cell_rows
        if row["status"] in ("reconstructed", "paper_fallback")
    )
    row = {
        "method": METHOD,
        "variant": variant,
        "correction": correction,
        "benchmark": case.benchmark,
        "case_index": case.case_index,
        "random_seed": case.random_seed,
        "cells_per_side": resolution,
        "cell_size": cell_size,
        "runtime_seconds": runtime,
        "mixed_cells": len(result.cells),
        "reconstructed_cells": status["reconstructed"],
        "paper_fallback_cells": status["paper_fallback"],
        "unsupported_cells": status["unsupported"],
        "unresolved_cells": status["unresolved"],
        "optimizer_failures": 0,
        "reconstructed_components": sum(
            int(item["component_count"]) for item in cell_rows
        ),
        "reconstructed_primitives": sum(
            int(item["primitive_count"]) for item in cell_rows
        ),
        "multi_component_cells": sum(
            int(item["component_count"]) > 1 for item in cell_rows
        ),
        "boundary_7x7_unsupported_cells": sum(
            item["outcome_reason"] == "incomplete_7x7_boundary_halo"
            for item in cell_rows
        ),
        "straight_limit_fallback_cells": status["paper_fallback"],
        "component_count_histogram_json": json.dumps(
            {str(key): value for key, value in sorted(component_histogram.items())}
        ),
        "fallback_reason_counts_json": _counter_json(cell_rows, "paper_fallback"),
        "unsupported_reason_counts_json": _counter_json(cell_rows, "unsupported"),
        "unresolved_reason_counts_json": _counter_json(cell_rows, "unresolved"),
        "sampled_hausdorff": hausdorff,
        "sampled_reconstruction_to_truth": reconstruction_to_truth,
        "shared_edge_gap_mean": gaps["mean"],
        "shared_edge_gap_max": gaps["max"],
        "unmatched_crossings": gaps["unmatched_count"],
        "conservation_mean_absolute_residual": conservation["mean_absolute_residual"],
        "conservation_max_absolute_residual": conservation["max_absolute_residual"],
        "curvature_estimator_mean_absolute_error": (
            float(np.mean(curvature_errors)) if len(curvature_errors) else math.nan
        ),
        "curvature_estimator_median_absolute_error": (
            float(np.median(curvature_errors)) if len(curvature_errors) else math.nan
        ),
        "curvature_estimator_max_absolute_error": (
            float(np.max(curvature_errors)) if len(curvature_errors) else math.nan
        ),
        "curvature_estimator_samples": len(curvature_errors),
        "parameters_json": json.dumps(case.to_dict(), sort_keys=True),
        "status_message": (
            "complete"
            if not status["unsupported"] and not status["unresolved"]
            else "partial"
        ),
    }
    stem = f"{correction}_{case.benchmark}_N{resolution}_case{case.case_index:02d}"
    (output_directory / "geometry").mkdir(parents=True, exist_ok=True)
    result.to_json(output_directory / "geometry" / f"{stem}.json")
    write_csv(output_directory / "cells" / f"{stem}.csv", cell_rows, PCIC_CELL_FIELDS)
    return row


def aggregate_pcic_rows(rows: Sequence[Mapping[str, Any]]) -> list:
    summary = []
    for correction in CORRECTIONS:
        subset = [row for row in rows if row["correction"] == correction]
        for item in aggregate_case_rows(subset):
            group = [
                row
                for row in subset
                if row["benchmark"] == item["benchmark"]
                and int(row["cells_per_side"]) == int(item["cells_per_side"])
            ]
            item.update(
                {
                    "method": METHOD,
                    "variant": source_variant_for_correction(correction),
                    "correction": correction,
                    "reconstructed_components_total": sum(
                        int(row["reconstructed_components"]) for row in group
                    ),
                    "reconstructed_primitives_total": sum(
                        int(row["reconstructed_primitives"]) for row in group
                    ),
                    "multi_component_cells_total": sum(
                        int(row["multi_component_cells"]) for row in group
                    ),
                    "boundary_7x7_unsupported_cells_total": sum(
                        int(row["boundary_7x7_unsupported_cells"]) for row in group
                    ),
                    "straight_limit_fallback_cells_total": sum(
                        int(row["straight_limit_fallback_cells"]) for row in group
                    ),
                }
            )
            item["fallback_fraction"] = item["paper_fallback_cells_total"] / max(
                1, item["mixed_cells_total"]
            )
            summary.append(item)
    return summary


def plot_pcic_all_benchmarks(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    benchmarks = tuple(
        name
        for name in DEFAULT_BENCHMARKS
        if any(row["benchmark"] == name for row in summary)
    )
    styles = {
        "translate_center": ("#2166ac", "o", "center translation"),
        "adjust_radius": ("#b2182b", "s", "radius adjustment"),
    }
    fig, axes = plt.subplots(len(benchmarks), 3, figsize=(10.2, 2.25 * len(benchmarks)))
    if len(benchmarks) == 1:
        axes = np.asarray([axes])
    for row_index, benchmark in enumerate(benchmarks):
        for correction in CORRECTIONS:
            color, marker, label = styles[correction]
            rows = sorted(
                (
                    row
                    for row in summary
                    if row["benchmark"] == benchmark and row["correction"] == correction
                ),
                key=lambda row: int(row["cells_per_side"]),
            )
            resolutions = [int(row["cells_per_side"]) for row in rows]
            geometry = [row["sampled_reconstruction_to_truth_median"] for row in rows]
            gaps = [row["shared_edge_gap_mean_median"] for row in rows]
            curvature = [
                row["curvature_estimator_median_absolute_error_median"] for row in rows
            ]
            fallback = [100.0 * float(row["fallback_fraction"]) for row in rows]
            unavailable = [100.0 * float(row["unresolved_fraction"]) for row in rows]

            axes[row_index, 0].plot(
                resolutions, geometry, color=color, marker=marker, label=label
            )
            if all(value is not None and value > 0.0 for value in gaps):
                axes[row_index, 0].plot(
                    resolutions,
                    gaps,
                    color=color,
                    marker=marker,
                    linestyle="--",
                    alpha=0.65,
                )
            if all(value is not None and value > 0.0 for value in curvature):
                axes[row_index, 1].plot(
                    resolutions, curvature, color=color, marker=marker, label=label
                )
            axes[row_index, 2].plot(
                resolutions,
                fallback,
                color=color,
                marker=marker,
                label=f"{label}: fallback",
            )
            axes[row_index, 2].plot(
                resolutions,
                unavailable,
                color=color,
                marker=marker,
                linestyle="--",
                alpha=0.7,
                label=f"{label}: unsupported/unresolved",
            )

        axes[row_index, 0].set_ylabel(benchmark.capitalize())
        axes[row_index, 0].set_yscale("log")
        if axes[row_index, 1].lines:
            axes[row_index, 1].set_yscale("log")
        for axis in axes[row_index]:
            axis.set_xscale("log", base=2)
            resolutions = sorted(
                {
                    int(row["cells_per_side"])
                    for row in summary
                    if row["benchmark"] == benchmark
                }
            )
            axis.set_xticks(resolutions, [str(value) for value in resolutions])
            axis.grid(True, which="both", alpha=0.25)
        axes[row_index, 2].set_ylim(bottom=0.0)

    axes[0, 0].set_title("Geometry error (solid) and facet gap (dashed)")
    axes[0, 1].set_title("Unsigned local curvature error")
    axes[0, 2].set_title("Outcome incidence (%)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 1].legend(frameon=False, fontsize=7)
    axes[0, 2].legend(frameon=False, fontsize=6, ncol=2)
    for axis in axes[-1]:
        axis.set_xlabel("Cells per side")
    fig.suptitle("Bare PCIC Cartesian matched five-case smoke", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(
    output_directory: Path,
    benchmarks: Sequence[str],
    resolutions: Sequence[int],
    case_indices: Sequence[int],
    corrections: Sequence[str] = CORRECTIONS,
    boundary_policy: str = "unsupported",
) -> list:
    wall_started = time.perf_counter()
    output_directory.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    implementation_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("project_benchmarks.py"),
        Path(__file__).with_name("project_smoke.py"),
        repository / "main/algos/baselines/pcic.py",
        repository / "main/algos/baselines/external_method_adapters.py",
    )
    manifest = {
        "study": "bare-PCIC five-benchmark Cartesian matched mini resolution study",
        "method": METHOD,
        "variants": [source_variant_for_correction(value) for value in corrections],
        "corrections": list(corrections),
        "git_head_at_launch": _git_head(),
        "implementation_sha256": {
            str(path.relative_to(repository)): _sha256(path)
            for path in implementation_paths
        },
        "benchmarks": list(benchmarks),
        "resolutions": list(resolutions),
        "case_indices": list(case_indices),
        "mesh": "uniform axis-aligned Cartesian",
        "frozen_policies": {
            "phase": "infer from oriented central PLIC",
            "lls_overcrowded_radius_scale": 0.5,
            "center_translation_root": "nearest conservative bracket",
            "multi_arc_chord": "PLIC proximity/alignment; preserve all paired arcs",
        },
        "boundary_policy": (
            "unsupported when the complete 7x7 predictor halo is unavailable"
            if boundary_policy == "unsupported"
            else "known empty exterior phase represented by zero-volume-fraction Cartesian ghost cells"
        ),
        "curvature_metric": "unsigned absolute local error |1/|R|-kappa_truth| at each returned arc midpoint; straight-limit fallbacks use zero curvature",
        "hausdorff_metric": "symmetric sampled point-cloud smoke diagnostic with spacing min(h/128, 2.5e-3)",
        "geometry_error_metric": "sampled reconstruction-to-truth maximum with the same spacing; does not penalize missing unsupported truth segments",
        "tuning": "none; frozen policies were selected before project benchmark results",
    }
    write_json(output_directory / "run_manifest.json", manifest)
    case_csv = output_directory / "case_results.csv"
    rows = []
    for correction in corrections:
        for benchmark in benchmarks:
            for case in canonical_benchmark_cases(benchmark, case_indices):
                for resolution in resolutions:
                    print(
                        f"RUN correction={correction} benchmark={benchmark} "
                        f"case={case.case_index} N={resolution}",
                        flush=True,
                    )
                    row = run_case(
                        case,
                        resolution,
                        correction,
                        output_directory,
                        boundary_policy=boundary_policy,
                    )
                    rows.append(row)
                    append_csv(case_csv, row, PCIC_CASE_FIELDS)
                    print(
                        "RESULT "
                        f"reconstructed={row['reconstructed_cells']} "
                        f"fallback={row['paper_fallback_cells']} "
                        f"unresolved={row['unresolved_cells']} "
                        f"unsupported={row['unsupported_cells']} "
                        f"runtime={row['runtime_seconds']:.2f}s",
                        flush=True,
                    )
    summary = aggregate_pcic_rows(rows)
    manifest["wall_runtime_seconds"] = time.perf_counter() - wall_started
    manifest["timed_reconstruction_seconds"] = sum(
        float(row["runtime_seconds"]) for row in rows
    )
    write_json(output_directory / "run_manifest.json", manifest)
    write_json(
        output_directory / "summary.json", {"manifest": manifest, "rows": summary}
    )
    if summary:
        write_csv(output_directory / "summary.csv", summary, tuple(summary[0].keys()))
        plot_pcic_all_benchmarks(
            summary, output_directory / "pcic_both_variants_all_benchmarks_summary.pdf"
        )
    return rows


def _parse_csv(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/baselines/results/pcic_project_smoke"),
    )
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument(
        "--resolutions", default=",".join(map(str, DEFAULT_RESOLUTIONS))
    )
    parser.add_argument("--cases", default=",".join(map(str, DEFAULT_CASE_INDICES)))
    parser.add_argument("--corrections", default=",".join(CORRECTIONS))
    parser.add_argument(
        "--boundary-policy",
        choices=BOUNDARY_POLICIES,
        default="unsupported",
    )
    args = parser.parse_args()
    run(
        args.output,
        _parse_csv(args.benchmarks, str),
        _parse_csv(args.resolutions, int),
        _parse_csv(args.cases, int),
        _parse_csv(args.corrections, str),
        args.boundary_policy,
    )


if __name__ == "__main__":
    main()
