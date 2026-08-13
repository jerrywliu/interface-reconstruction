"""Run frozen Cartesian QUASI on the five canonical project benchmarks."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

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
from main.algos.baselines.external_geometry import (
    ExternalBaselineResult,
    ExternalReconstructionStatus,
)
from main.algos.baselines.external_method_adapters import adapt_quasi_cell
from main.algos.baselines.external_metrics import (
    conservation_metrics,
    shared_edge_gap_metrics,
)
from main.algos.baselines.quasi import DEFAULT_QUASI_POLICY, reconstruct_quasi


METHOD = "QUASI"
VARIANT = "QUASI (frozen Cartesian port)"

QUASI_CASE_FIELDS = CASE_FIELDS + (
    "conservative_fallback_cells",
    "unresolved_events",
    "joins",
    "c1_updates",
    "c1_misses",
    "curvature_updates",
    "vertex_jumps",
    "sweeps_completed",
    "sweep_converged",
    "policy_json",
    "unresolved_json",
)

QUASI_CELL_FIELDS = CELL_FIELDS + (
    "bulge",
    "chord_length",
    "conservative_fallback",
    "fallback_reasons_json",
)

_CELL_PATTERN = re.compile(r"\((\d+)\s*,\s*(\d+)\)")


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


def _fallback_reasons_by_cell(
    messages: Iterable[str],
) -> Dict[Tuple[int, int], list[str]]:
    """Map QUASI's frozen human-readable fallback events to involved cells."""

    reasons: Dict[Tuple[int, int], list[str]] = {}
    for message in messages:
        for x, y in _CELL_PATTERN.findall(message):
            reasons.setdefault((int(x), int(y)), []).append(str(message))
    return reasons


def _external_result(
    mesh, quasi_result, case, resolution: int
) -> ExternalBaselineResult:
    fallback_reasons = _fallback_reasons_by_cell(quasi_result.unresolved)
    cells = {}
    for x, column in enumerate(mesh.polys):
        for y, polygon in enumerate(column):
            if not polygon.isMixed(tolerance=1.0e-10):
                continue
            cell_index = (x, y)
            facet = quasi_result.facets.get(cell_index)
            if facet is None:
                record = adapt_quasi_cell(
                    None,
                    cell_index=cell_index,
                    polygon=polygon.points,
                    target_phase_area=polygon.getArea(),
                    unresolved_reason="QUASI returned no facet for a mixed cell",
                )
                cells[cell_index] = replace(
                    record,
                    diagnostics={
                        **dict(record.diagnostics),
                        "missing_facet": True,
                    },
                )
                continue
            record = adapt_quasi_cell(
                facet,
                cell_index=cell_index,
                polygon=polygon.points,
                target_phase_area=polygon.getArea(),
            )
            reasons = fallback_reasons.get(cell_index, ())
            diagnostics = {
                **dict(record.diagnostics),
                "bulge": facet.bulge,
                "chord_length": facet.chord_length,
                "curvature_midpoint": facet.curvature(0.5),
                "conservative_fallback": bool(reasons),
                "fallback_reasons": list(reasons),
            }
            if reasons:
                record = replace(
                    record,
                    status=ExternalReconstructionStatus.PAPER_FALLBACK,
                    diagnostics=diagnostics,
                )
            else:
                record = replace(record, diagnostics=diagnostics)
            cells[cell_index] = record

    status_counts = {status.value: 0 for status in ExternalReconstructionStatus}
    for record in cells.values():
        status_counts[record.status.value] += 1
    return ExternalBaselineResult(
        source_method=METHOD,
        source_variant=VARIANT,
        cells=cells,
        metadata={
            "mesh_class": "axis_aligned_cartesian",
            "mesh_shape": [len(mesh.polys), len(mesh.polys[0])],
            "config": {
                "benchmark": case.benchmark,
                "case_index": case.case_index,
                "cells_per_side": resolution,
                "cell_size": DOMAIN_SIZE / resolution,
            },
            "status_counts": status_counts,
            "quasi": {
                "policy": quasi_result.policy,
                "unresolved": list(quasi_result.unresolved),
                "joins": len(quasi_result.joins),
                "c1_updates": quasi_result.c1_updates,
                "c1_misses": quasi_result.c1_misses,
                "curvature_updates": quasi_result.curvature_updates,
                "vertex_jumps": quasi_result.vertex_jumps,
                "sweeps_completed": quasi_result.sweeps_completed,
                "converged": quasi_result.converged,
            },
        },
    )


def _cell_rows(result, quasi_result, case, resolution: int) -> list[Dict[str, Any]]:
    fallback_reasons = _fallback_reasons_by_cell(quasi_result.unresolved)
    rows = []
    for (x, y), record in sorted(result.cells.items()):
        facet = quasi_result.facets.get((x, y))
        curvature = None
        true_curvature = None
        curvature_error = None
        reconstructed_area = None
        if facet is not None:
            curvature = abs(float(facet.curvature(0.5)))
            true_curvature = abs(float(case.true_curvature_at(facet.midpoint)))
            curvature_error = abs(curvature - true_curvature)
            reconstructed_area = record.exact_phase_area()
        reasons = fallback_reasons.get((x, y), ())
        rows.append(
            {
                "method": METHOD,
                "variant": VARIANT,
                "benchmark": case.benchmark,
                "case_index": case.case_index,
                "cells_per_side": resolution,
                "cell_x": x,
                "cell_y": y,
                "status": record.status.value,
                "target_phase_area": record.target_phase_area,
                "reconstructed_phase_area": reconstructed_area,
                "absolute_area_residual": (
                    None
                    if reconstructed_area is None
                    else abs(reconstructed_area - record.target_phase_area)
                ),
                "curvature_estimate": curvature,
                "true_curvature": true_curvature,
                "curvature_absolute_error": curvature_error,
                "objective": None,
                "optimizer_success": None,
                "ghf_method": None,
                "diagnostics_json": json.dumps(record.diagnostics, sort_keys=True),
                "bulge": None if facet is None else facet.bulge,
                "chord_length": None if facet is None else facet.chord_length,
                "conservative_fallback": bool(reasons),
                "fallback_reasons_json": json.dumps(list(reasons), sort_keys=True),
            }
        )
    return rows


def run_case(case, resolution: int, output_directory: Path) -> Dict[str, Any]:
    mesh = case.build_mesh(resolution)
    case.initialize_fractions(mesh)
    cell_size = DOMAIN_SIZE / resolution
    started = time.perf_counter()
    quasi_result = reconstruct_quasi(mesh, policy=DEFAULT_QUASI_POLICY)
    runtime = time.perf_counter() - started
    result = _external_result(mesh, quasi_result, case, resolution)
    cell_rows = _cell_rows(result, quasi_result, case, resolution)

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
    fallback_cells = sum(row["conservative_fallback"] for row in cell_rows)
    row = {
        "method": METHOD,
        "variant": VARIANT,
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
            if not quasi_result.unresolved and quasi_result.converged
            else "complete_with_reported_fallback"
        ),
        "conservative_fallback_cells": fallback_cells,
        "unresolved_events": len(quasi_result.unresolved),
        "joins": len(quasi_result.joins),
        "c1_updates": quasi_result.c1_updates,
        "c1_misses": quasi_result.c1_misses,
        "curvature_updates": quasi_result.curvature_updates,
        "vertex_jumps": quasi_result.vertex_jumps,
        "sweeps_completed": quasi_result.sweeps_completed,
        "sweep_converged": quasi_result.converged,
        "policy_json": json.dumps(quasi_result.policy, sort_keys=True),
        "unresolved_json": json.dumps(quasi_result.unresolved, sort_keys=True),
    }

    stem = f"{case.benchmark}_N{resolution}_case{case.case_index:02d}"
    (output_directory / "geometry").mkdir(parents=True, exist_ok=True)
    result.to_json(output_directory / "geometry" / f"{stem}.json")
    write_csv(
        output_directory / "cells" / f"{stem}.csv",
        cell_rows,
        QUASI_CELL_FIELDS,
    )
    return row


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    summary = aggregate_case_rows(rows)
    source_groups: Dict[Tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        source_groups.setdefault(
            (str(row["benchmark"]), int(row["cells_per_side"])), []
        ).append(row)
    for item in summary:
        group = source_groups[(item["benchmark"], item["cells_per_side"])]
        for field in (
            "conservative_fallback_cells",
            "unresolved_events",
            "joins",
            "c1_updates",
            "c1_misses",
            "curvature_updates",
            "vertex_jumps",
            "sweeps_completed",
        ):
            values = np.asarray([float(row[field]) for row in group])
            item[field + "_total"] = int(np.sum(values))
            item[field + "_median"] = float(np.median(values))
        item["sweep_converged_cases"] = int(
            sum(_as_bool(row["sweep_converged"]) for row in group)
        )
        item["sweep_converged_fraction"] = item["sweep_converged_cases"] / len(group)
    return summary


def _positive(values: Sequence[Any]) -> list[float]:
    return [
        float(value) if value is not None and float(value) > 0.0 else math.nan
        for value in values
    ]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _plot_all_benchmarks(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    benchmarks = tuple(
        benchmark
        for benchmark in DEFAULT_BENCHMARKS
        if any(row["benchmark"] == benchmark for row in summary)
    )
    fig, axes = plt.subplots(len(benchmarks), 3, figsize=(10.0, 2.15 * len(benchmarks)))
    if len(benchmarks) == 1:
        axes = np.asarray([axes])
    for index, benchmark in enumerate(benchmarks):
        rows = sorted(
            (row for row in summary if row["benchmark"] == benchmark),
            key=lambda row: int(row["cells_per_side"]),
        )
        resolutions = [int(row["cells_per_side"]) for row in rows]
        geometry, curvature, operations = axes[index]
        for field, label, marker in (
            ("sampled_reconstruction_to_truth_median", "geometry", "o"),
            ("shared_edge_gap_mean_median", "facet gap", "s"),
        ):
            geometry.plot(
                resolutions,
                _positive([row[field] for row in rows]),
                marker=marker,
                label=label,
            )
        geometry.set_yscale("log")
        geometry.set_ylabel(benchmark.capitalize())
        geometry.grid(True, which="both", alpha=0.25)
        if index == 0:
            geometry.set_title("Geometry / continuity")
            geometry.legend(frameon=False, fontsize=7)

        curvature.plot(
            resolutions,
            _positive(
                [
                    row["curvature_estimator_median_absolute_error_median"]
                    for row in rows
                ]
            ),
            marker="^",
            color="#b24745",
        )
        curvature.set_yscale("log")
        curvature.grid(True, which="both", alpha=0.25)
        if index == 0:
            curvature.set_title("Unsigned local curvature error")

        operations.plot(
            resolutions,
            [row["c1_updates_total"] / row["mixed_cells_total"] for row in rows],
            marker="o",
            label="C1 updates / mixed cell",
            color="#357266",
        )
        operations.plot(
            resolutions,
            [row["c1_misses_total"] / row["mixed_cells_total"] for row in rows],
            marker="s",
            label="C1 misses / mixed cell",
            color="#d08c60",
        )
        operations.set_yscale("log")
        operations.text(
            0.98,
            0.95,
            "converged: "
            + ", ".join(
                f"{100.0 * row['sweep_converged_fraction']:.0f}%" for row in rows
            )
            + "\nfallback cells: "
            + ", ".join(str(row["conservative_fallback_cells_total"]) for row in rows)
            + "\ncurvature updates: "
            + ", ".join(str(row["curvature_updates_total"]) for row in rows)
            + "; vertex jumps: "
            + ", ".join(str(row["vertex_jumps_total"]) for row in rows),
            transform=operations.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color="#444444",
        )
        operations.grid(True, which="both", alpha=0.25)
        if index == 0:
            operations.set_title("Frozen sweep diagnostics")
            operations.legend(frameon=False, fontsize=6.8, loc="upper left")

        for axis in (geometry, curvature, operations):
            axis.set_xscale("log", base=2)
            axis.set_xticks(resolutions, [str(value) for value in resolutions])
    for axis in axes[-1]:
        axis.set_xlabel("Cells per side")
    fig.suptitle("QUASI frozen Cartesian five-case smoke", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _existing_keys(path: Path) -> set[Tuple[str, int, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (row["benchmark"], int(row["case_index"]), int(row["cells_per_side"]))
            for row in csv.DictReader(stream)
        }


def _read_rows(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run(
    output_directory: Path,
    benchmarks: Sequence[str],
    resolutions: Sequence[int],
    case_indices: Sequence[int],
) -> list[Mapping[str, Any]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    implementation_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("project_benchmarks.py"),
        Path(__file__).with_name("project_smoke.py"),
        repository / "main/algos/baselines/quasi.py",
        repository / "main/algos/baselines/quasi_roots.py",
        repository / "main/algos/baselines/external_method_adapters.py",
    )
    manifest = {
        "study": "QUASI five-benchmark Cartesian mini resolution study",
        "method": METHOD,
        "variant": VARIANT,
        "git_head_at_launch": _git_head(),
        "implementation_sha256": {
            str(path.relative_to(repository)): _sha256(path)
            for path in implementation_paths
        },
        "benchmarks": list(benchmarks),
        "resolutions": list(resolutions),
        "case_indices": list(case_indices),
        "mesh": "uniform axis-aligned Cartesian",
        "policy": {
            "max_sweeps": DEFAULT_QUASI_POLICY.max_sweeps,
            "convergence_tolerance": DEFAULT_QUASI_POLICY.convergence_tolerance,
            "target_fraction_lower": DEFAULT_QUASI_POLICY.target_fraction_lower,
            "target_fraction_upper": DEFAULT_QUASI_POLICY.target_fraction_upper,
            "fallback": DEFAULT_QUASI_POLICY.fallback,
        },
        "curvature_metric": "unsigned midpoint curvature error |abs(kappa_quadratic)-abs(kappa_truth)| per mixed cell",
        "hausdorff_metric": "symmetric sampled point-cloud diagnostic with spacing min(h/128, 2.5e-3)",
        "geometry_error_metric": "sampled reconstruction-to-truth maximum with the same spacing",
        "resume_policy": "completed benchmark/case/resolution keys in case_results.csv are skipped",
    }
    write_json(output_directory / "run_manifest.json", manifest)
    case_csv = output_directory / "case_results.csv"
    completed = _existing_keys(case_csv)
    for benchmark in benchmarks:
        for case in canonical_benchmark_cases(benchmark, case_indices):
            for resolution in resolutions:
                key = (benchmark, case.case_index, resolution)
                if key in completed:
                    print(
                        f"SKIP benchmark={benchmark} case={case.case_index} N={resolution}",
                        flush=True,
                    )
                    continue
                print(
                    f"RUN benchmark={benchmark} case={case.case_index} N={resolution}",
                    flush=True,
                )
                row = run_case(case, resolution, output_directory)
                append_csv(case_csv, row, QUASI_CASE_FIELDS)
                completed.add(key)
                print(
                    "RESULT "
                    f"mixed={row['mixed_cells']} fallback={row['conservative_fallback_cells']} "
                    f"events={row['unresolved_events']} converged={row['sweep_converged']} "
                    f"runtime={row['runtime_seconds']:.2f}s",
                    flush=True,
                )
    rows = _read_rows(case_csv)
    summary = _aggregate(rows)
    write_json(
        output_directory / "summary.json", {"manifest": manifest, "rows": summary}
    )
    if summary:
        write_csv(output_directory / "summary.csv", summary, tuple(summary[0].keys()))
        _plot_all_benchmarks(
            summary, output_directory / "quasi_all_benchmarks_summary.pdf"
        )
    return rows


def _parse_csv(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/baselines/results/quasi_project_smoke"),
    )
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument(
        "--resolutions", default=",".join(map(str, DEFAULT_RESOLUTIONS))
    )
    parser.add_argument("--cases", default=",".join(map(str, DEFAULT_CASE_INDICES)))
    args = parser.parse_args()
    run(
        args.output,
        _parse_csv(args.benchmarks, str),
        _parse_csv(args.resolutions, int),
        _parse_csv(args.cases, int),
    )


if __name__ == "__main__":
    main()
