#!/usr/bin/env python3
"""Replay finalized circular ellipse facets with the common native metrics."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.baselines.project_benchmarks import (
    DOMAIN_SIZE,
    ProjectBenchmarkCase,
    canonical_benchmark_cases,
)
from experiments.baselines.project_smoke import write_json
from main.algos.baselines.external_metrics import (
    directed_hausdorff_external,
    geometric_curvature_error_external,
)
from main.algos.baselines.project_facet_adapter import (
    external_primitives_from_facet_metadata,
)


DEFAULT_SOURCE = Path(
    "results/static/submission_static_20260731_012430_505aefa45432.sealed/raw_runs"
)
DEFAULT_OUTPUT = Path(
    "experiments/baselines/results/graph_circular_ellipse_common_metrics_20260814"
)
DEFAULT_RESOLUTIONS = (32, 64, 128)
DEFAULT_CASE_INDICES = (0, 1, 2, 3, 4)
GEOMETRY_ABSOLUTE_TOLERANCE = 1.0e-12
CASE_FIELDS = (
    "method",
    "variant",
    "benchmark",
    "case_index",
    "cells_per_side",
    "cell_size",
    "source_run",
    "source_commit",
    "canonical_geometry_match",
    "primitive_count",
    "arc_count",
    "line_count",
    "native_symmetric_hausdorff",
    "native_reconstruction_to_truth",
    "native_truth_to_reconstruction",
    "geometric_curvature_mean_absolute_error",
    "geometric_curvature_rms_error",
    "geometric_curvature_max_absolute_error",
    "geometric_curvature_relative_l1_error",
    "geometric_curvature_reconstructed_length",
    "geometric_curvature_quadrature_samples",
)
SUMMARY_METRICS = (
    "native_symmetric_hausdorff",
    "native_reconstruction_to_truth",
    "native_truth_to_reconstruction",
    "geometric_curvature_mean_absolute_error",
    "geometric_curvature_rms_error",
    "geometric_curvature_relative_l1_error",
)
ORDER_METRICS = (
    "native_symmetric_hausdorff",
    "geometric_curvature_mean_absolute_error",
)


def _load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["case_index"])] = row
    return rows


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _resolution_from_manifest(manifest: Mapping[str, Any]) -> int:
    return round(DOMAIN_SIZE * float(manifest["parameters"]["resolution"]))


def discover_finalized_runs(
    source_root: Path,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
) -> dict[int, Path]:
    """Find the unique finalized Cartesian circular run at each requested N."""

    requested = {int(value) for value in resolutions}
    found: dict[int, Path] = {}
    pattern = "*_perturb_sweep_ellipses_circular_r*_w0p0_s0"
    for run_dir in sorted(source_root.glob(pattern)):
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parameters = manifest["parameters"]
        resolution = _resolution_from_manifest(manifest)
        if resolution not in requested:
            continue
        expected = {
            "facet_algo": "circular",
            "mesh_type": "perturbed_quads",
            "perturb_wiggle": 0.0,
            "perturb_seed": 0,
            "plic_fallback": "LVIRA",
            "random_seed": 42,
            "num_ellipses": 25,
        }
        for key, expected_value in expected.items():
            if parameters.get(key) != expected_value:
                raise ValueError(
                    f"{run_dir.name} has {key}={parameters.get(key)!r}; "
                    f"expected {expected_value!r}"
                )
        if resolution in found:
            raise ValueError(f"duplicate finalized ellipse run at N={resolution}")
        found[resolution] = run_dir

    missing = sorted(requested - set(found))
    if missing:
        raise FileNotFoundError(
            f"missing finalized Cartesian circular ellipse runs at N={missing}"
        )
    return found


def verify_canonical_ellipse_geometry(
    saved: Mapping[str, Any],
    canonical: ProjectBenchmarkCase,
    *,
    absolute_tolerance: float = GEOMETRY_ABSOLUTE_TOLERANCE,
) -> None:
    """Fail if a saved case is not the exact canonical seed-42 ellipse case."""

    if saved.get("geometry_type") != "ellipse":
        raise ValueError("saved geometry is not an ellipse")
    if int(saved.get("case_index", -1)) != canonical.case_index:
        raise ValueError("saved and canonical case indices differ")
    expected = canonical.parameters
    scalar_fields = ("major_axis", "minor_axis", "aspect_ratio", "theta")
    for field in scalar_fields:
        if not math.isclose(
            float(saved[field]),
            float(expected[field]),
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ):
            raise ValueError(
                f"case {canonical.case_index} {field} differs from canonical geometry"
            )
    for axis, (actual, target) in enumerate(zip(saved["center"], expected["center"])):
        if not math.isclose(
            float(actual),
            float(target),
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        ):
            raise ValueError(
                f"case {canonical.case_index} center[{axis}] differs from canonical geometry"
            )


def _evaluate_case(run_dir_value: str, case_index: int) -> dict[str, Any]:
    run_dir = Path(run_dir_value)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    resolution = _resolution_from_manifest(manifest)
    saved_geometry = _load_jsonl(run_dir / "metrics" / "case_geometry.jsonl")[
        case_index
    ]
    canonical = canonical_benchmark_cases("ellipses", (case_index,))[0]
    verify_canonical_ellipse_geometry(saved_geometry, canonical)

    metadata_path = (
        run_dir
        / "vtk"
        / "reconstructed"
        / "facets"
        / f"{case_index}.facet_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reconstruction = external_primitives_from_facet_metadata(metadata)
    truth = canonical.truth_primitives()
    reconstruction_to_truth = directed_hausdorff_external(reconstruction, truth)
    truth_to_reconstruction = directed_hausdorff_external(truth, reconstruction)
    curvature = geometric_curvature_error_external(reconstruction, truth)
    kinds = [primitive.kind for primitive in reconstruction]
    return {
        "method": "Ours",
        "variant": "graph-coordinated circular",
        "benchmark": "ellipses",
        "case_index": case_index,
        "cells_per_side": resolution,
        "cell_size": DOMAIN_SIZE / resolution,
        "source_run": run_dir.name,
        "source_commit": manifest["source_commit"],
        "canonical_geometry_match": True,
        "primitive_count": len(reconstruction),
        "arc_count": kinds.count("arc"),
        "line_count": kinds.count("line"),
        "native_symmetric_hausdorff": max(
            reconstruction_to_truth, truth_to_reconstruction
        ),
        "native_reconstruction_to_truth": reconstruction_to_truth,
        "native_truth_to_reconstruction": truth_to_reconstruction,
        "geometric_curvature_mean_absolute_error": curvature["mean_absolute_error"],
        "geometric_curvature_rms_error": curvature["rms_error"],
        "geometric_curvature_max_absolute_error": curvature["max_absolute_error"],
        "geometric_curvature_relative_l1_error": curvature["relative_l1_error"],
        "geometric_curvature_reconstructed_length": curvature["reconstructed_length"],
        "geometric_curvature_quadrature_samples": curvature["quadrature_samples"],
    }


def _fit_order(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    if len(rows) < 3:
        raise ValueError("an observed-order fit requires at least three resolutions")
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    cell_sizes = np.asarray([float(row["cell_size"]) for row in rows], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        return math.nan
    return float(np.polyfit(np.log(cell_sizes), np.log(values), 1)[0])


def summarize_case_results(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    for resolution in sorted({int(row["cells_per_side"]) for row in rows}):
        selected = [row for row in rows if int(row["cells_per_side"]) == resolution]
        result: dict[str, Any] = {
            "method": "Ours",
            "variant": "graph-coordinated circular",
            "benchmark": "ellipses",
            "cells_per_side": resolution,
            "cell_size": DOMAIN_SIZE / resolution,
            "case_count": len(selected),
            "primitive_count": sum(int(row["primitive_count"]) for row in selected),
            "arc_count": sum(int(row["arc_count"]) for row in selected),
            "line_count": sum(int(row["line_count"]) for row in selected),
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in selected], dtype=float)
            result[f"{metric}_median"] = float(np.median(values))
            result[f"{metric}_q25"] = float(np.quantile(values, 0.25))
            result[f"{metric}_q75"] = float(np.quantile(values, 0.75))
        summary.append(result)

    for metric in ORDER_METRICS:
        median_field = f"{metric}_median"
        fit_order = _fit_order(summary, median_field)
        previous = None
        for row in summary:
            row[f"{metric}_fit_order"] = fit_order
            row[f"{metric}_order_from_previous"] = (
                math.log(float(previous[median_field]) / float(row[median_field]))
                / math.log(float(previous["cell_size"]) / float(row["cell_size"]))
                if previous is not None
                else math.nan
            )
            previous = row

    case_orders = []
    for case_index in sorted({int(row["case_index"]) for row in rows}):
        selected = sorted(
            (row for row in rows if int(row["case_index"]) == case_index),
            key=lambda row: int(row["cells_per_side"]),
        )
        case_orders.append(
            {
                "case_index": case_index,
                "resolution_count": len(selected),
                "native_symmetric_hausdorff_fit_order": _fit_order(
                    selected, "native_symmetric_hausdorff"
                ),
                "geometric_curvature_mean_absolute_error_fit_order": _fit_order(
                    selected, "geometric_curvature_mean_absolute_error"
                ),
            }
        )
    return summary, case_orders


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolutions", type=_parse_ints, default=DEFAULT_RESOLUTIONS)
    parser.add_argument(
        "--case-indices", type=_parse_ints, default=DEFAULT_CASE_INDICES
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_indices = tuple(int(value) for value in args.case_indices)
    canonical_benchmark_cases("ellipses", case_indices)
    runs = discover_finalized_runs(args.source_root, args.resolutions)
    jobs = [
        (runs[resolution], case_index)
        for resolution in sorted(runs)
        for case_index in case_indices
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(_evaluate_case, str(run_dir), case_index): (
                run_dir,
                case_index,
            )
            for run_dir, case_index in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"completed {completed}/{len(futures)}", flush=True)

    rows.sort(key=lambda row: (row["cells_per_side"], row["case_index"]))
    summary, case_orders = summarize_case_results(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "case_results.csv", rows, CASE_FIELDS)
    _write_csv(args.output / "summary.csv", summary, tuple(summary[0]))
    _write_csv(args.output / "case_orders.csv", case_orders, tuple(case_orders[0]))
    write_json(
        args.output / "manifest.json",
        {
            "source_root": str(args.source_root),
            "source_runs": [runs[resolution].name for resolution in sorted(runs)],
            "source_commits": sorted({row["source_commit"] for row in rows}),
            "method": "Ours (graph-coordinated circular)",
            "benchmark": "canonical Cartesian ellipses",
            "cells_per_side": sorted(runs),
            "case_indices": list(case_indices),
            "case_count": len(rows),
            "canonical_geometry_absolute_tolerance": GEOMETRY_ABSOLUTE_TOLERANCE,
            "geometry_source": "exact schema-v2 writeFacets primitive metadata",
            "geometry_metric": (
                "partition-invariant native symmetric Hausdorff distance with "
                "projected target endpoints and bounded interval optimization"
            ),
            "curvature_metric": (
                "arc-length-weighted mean absolute geometric-curvature error "
                "against nearest analytic ellipse truth"
            ),
        },
    )


if __name__ == "__main__":
    main()
