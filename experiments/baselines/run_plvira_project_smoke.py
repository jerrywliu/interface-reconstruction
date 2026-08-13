"""Run operational Cartesian PLVIRA on the five project benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, Sequence

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
    plot_all_benchmarks,
    sampled_directed_hausdorff,
    sampled_symmetric_hausdorff,
    write_csv,
    write_json,
)
from main.algos.baselines.external_method_adapters import adapt_plvira_cell
from main.algos.baselines.external_metrics import (
    conservation_metrics,
    shared_edge_gap_metrics,
)
from main.algos.baselines.plvira import reconstruct_plvira
from main.algos.baselines.plvira_ghf import GHFStencilError


METHOD = "PLVIRA"
VARIANT = "PLVIRA"


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[2]
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plvira_method(mesh, cell_size: float):
    cartesian_fractions = np.asarray(mesh.getFractions(), dtype=float).T

    def method(context: ExternalCellContext):
        if not context.complete_3x3:
            raise UnsupportedExternalCell(
                "PLVIRA requires a complete 3x3 target stencil",
                {"boundary_halo_limitation": True},
            )
        x, y = context.cell_index
        stencil = context.stencil_3x3
        polygons = tuple(
            tuple(stencil[column][row].points for column in range(3))
            for row in range(3)
        )
        fractions = tuple(
            tuple(float(stencil[column][row].getFraction()) for column in range(3))
            for row in range(3)
        )
        try:
            interface = reconstruct_plvira(
                polygons,
                fractions,
                cartesian_fractions=cartesian_fractions,
                target_index=(y, x),
                cell_size=cell_size,
            )
        except GHFStencilError as error:
            raise UnsupportedExternalCell(
                str(error), {"boundary_halo_limitation": True}
            ) from error
        except (RuntimeError, ValueError, ArithmeticError) as error:
            raise UnresolvedExternalCell(
                str(error), {"exception_type": type(error).__name__}
            ) from error
        return adapt_plvira_cell(
            interface,
            cell_index=context.cell_index,
            polygon=context.polygon_points,
            target_phase_area=context.target_phase_area,
        )

    return method


def _cell_rows(result, case, resolution: int) -> list:
    rows = []
    for (x, y), record in sorted(result.cells.items()):
        diagnostics = dict(record.diagnostics)
        curvature = diagnostics.get("curvature")
        if curvature is None and record.primitives():
            curvature = getattr(record.primitives()[0], "curvature", None)
        true_curvature = None
        curvature_error = None
        if curvature is not None and record.primitives():
            true_curvature = case.true_curvature_at(record.primitives()[0].point(0.5))
            curvature_error = abs(float(curvature) - float(true_curvature))
        reconstructed_area = None
        area_residual = None
        try:
            reconstructed_area = record.exact_phase_area()
            area_residual = abs(reconstructed_area - record.target_phase_area)
        except RuntimeError:
            pass
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
                "absolute_area_residual": area_residual,
                "curvature_estimate": curvature,
                "true_curvature": true_curvature,
                "curvature_absolute_error": curvature_error,
                "objective": diagnostics.get("objective"),
                "optimizer_success": diagnostics.get("optimizer_success"),
                "ghf_method": diagnostics.get("ghf_method"),
                "diagnostics_json": json.dumps(diagnostics, sort_keys=True),
            }
        )
    return rows


def run_case(case, resolution: int, output_directory: Path) -> Dict[str, Any]:
    mesh = case.build_mesh(resolution)
    case.initialize_fractions(mesh)
    cell_size = DOMAIN_SIZE / resolution
    started = time.perf_counter()
    result = run_external_static_baseline(
        mesh,
        _plvira_method(mesh, cell_size),
        source_method=METHOD,
        source_variant=VARIANT,
        config={
            "benchmark": case.benchmark,
            "case_index": case.case_index,
            "cells_per_side": resolution,
            "cell_size": cell_size,
            "curvature_source": "cartesian-ghf",
        },
    )
    runtime = time.perf_counter() - started
    cell_rows = _cell_rows(result, case, resolution)
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
        "optimizer_failures": sum(
            item["optimizer_success"] is False for item in cell_rows
        ),
        "sampled_hausdorff": hausdorff,
        "sampled_reconstruction_to_truth": reconstruction_to_truth,
        "shared_edge_gap_mean": gaps["mean"],
        "shared_edge_gap_max": gaps["max"],
        "unmatched_crossings": gaps["unmatched_count"],
        "conservation_mean_absolute_residual": conservation["mean_absolute_residual"],
        "conservation_max_absolute_residual": conservation["max_absolute_residual"],
        "curvature_estimator_mean_absolute_error": float(np.mean(curvature_errors)) if len(curvature_errors) else math.nan,
        "curvature_estimator_median_absolute_error": float(np.median(curvature_errors)) if len(curvature_errors) else math.nan,
        "curvature_estimator_max_absolute_error": float(np.max(curvature_errors)) if len(curvature_errors) else math.nan,
        "curvature_estimator_samples": len(curvature_errors),
        "parameters_json": json.dumps(case.to_dict(), sort_keys=True),
        "status_message": "complete" if not status["unsupported"] and not status["unresolved"] else "partial",
    }
    stem = f"{case.benchmark}_N{resolution}_case{case.case_index:02d}"
    (output_directory / "geometry").mkdir(parents=True, exist_ok=True)
    result.to_json(output_directory / "geometry" / f"{stem}.json")
    write_csv(output_directory / "cells" / f"{stem}.csv", cell_rows, CELL_FIELDS)
    return row


def run(
    output_directory: Path,
    benchmarks: Sequence[str],
    resolutions: Sequence[int],
    case_indices: Sequence[int],
) -> list:
    output_directory.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    implementation_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("project_benchmarks.py"),
        Path(__file__).with_name("project_smoke.py"),
        repository / "main/algos/baselines/plvira.py",
        repository / "main/algos/baselines/plvira_ghf.py",
        repository / "main/algos/baselines/external_method_adapters.py",
    )
    manifest = {
        "study": "PLVIRA five-benchmark Cartesian mini resolution study",
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
        "curvature_source": "Cartesian generalized height function",
        "hausdorff_metric": "symmetric sampled point-cloud diagnostic with spacing min(h/128, 2.5e-3); not the adaptive native metric",
        "geometry_error_metric": "sampled reconstruction-to-truth maximum with the same spacing; does not penalize unsupported boundary truth segments",
        "boundary_policy": "unsupported when PLVIRA/GHF requires cells outside the available halo",
    }
    write_json(output_directory / "run_manifest.json", manifest)
    case_csv = output_directory / "case_results.csv"
    rows = []
    for benchmark in benchmarks:
        for case in canonical_benchmark_cases(benchmark, case_indices):
            for resolution in resolutions:
                print(
                    f"RUN benchmark={benchmark} case={case.case_index} N={resolution}",
                    flush=True,
                )
                row = run_case(case, resolution, output_directory)
                rows.append(row)
                append_csv(case_csv, row, CASE_FIELDS)
                print(
                    "RESULT "
                    f"mixed={row['mixed_cells']} unresolved={row['unresolved_cells']} "
                    f"unsupported={row['unsupported_cells']} runtime={row['runtime_seconds']:.2f}s",
                    flush=True,
                )
    summary = aggregate_case_rows(rows)
    write_json(output_directory / "summary.json", {"manifest": manifest, "rows": summary})
    if summary:
        write_csv(output_directory / "summary.csv", summary, tuple(summary[0].keys()))
        plot_all_benchmarks(summary, output_directory / "plvira_all_benchmarks_summary.pdf")
    return rows


def _parse_csv(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/baselines/results/plvira_project_smoke"),
    )
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--resolutions", default=",".join(map(str, DEFAULT_RESOLUTIONS)))
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
