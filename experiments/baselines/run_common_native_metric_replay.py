"""Replay saved external-baseline geometry with common native observables."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments.baselines.project_benchmarks import (
    DEFAULT_BENCHMARKS,
    canonical_benchmark_cases,
)
from experiments.baselines.project_smoke import write_csv, write_json
from main.algos.baselines.external_geometry import ExternalBaselineResult
from main.algos.baselines.external_metrics import (
    directed_hausdorff_external,
    geometric_curvature_error_external,
)


DEFAULT_INPUTS = (
    Path("experiments/baselines/results/plvira_project_smoke_20260813_final"),
    Path("experiments/baselines/results/pcic_project_smoke_20260813_final"),
    Path("experiments/baselines/results/quasi_project_smoke_20260813_qa_fixed"),
)
SMOOTH_BENCHMARKS = {"circles", "ellipses"}
CASE_FIELDS = (
    "method",
    "variant",
    "benchmark",
    "case_index",
    "cells_per_side",
    "cell_size",
    "geometry_file",
    "metric_status",
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


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _case_payload(path: Path) -> Dict[str, Any]:
    result = ExternalBaselineResult.from_json(path)
    config = result.metadata["config"]
    benchmark = str(config["benchmark"])
    case_index = int(config["case_index"])
    cells_per_side = int(config["cells_per_side"])
    case = canonical_benchmark_cases(benchmark, (case_index,))[0]
    truth = case.truth_primitives()
    active_primitives = sum(
        len(record.primitives()) for record in result.cells.values()
    )
    if active_primitives:
        reconstruction_to_truth = directed_hausdorff_external(result, truth)
        truth_to_reconstruction = directed_hausdorff_external(truth, result)
        symmetric = max(reconstruction_to_truth, truth_to_reconstruction)
        metric_status = "complete"
    else:
        reconstruction_to_truth = None
        truth_to_reconstruction = None
        symmetric = None
        metric_status = "no active reconstructed geometry"
    curvature = (
        geometric_curvature_error_external(result, truth)
        if benchmark in SMOOTH_BENCHMARKS and active_primitives
        else {}
    )
    return {
        "method": result.source_method,
        "variant": result.source_variant,
        "benchmark": benchmark,
        "case_index": case_index,
        "cells_per_side": cells_per_side,
        "cell_size": float(config["cell_size"]),
        "geometry_file": str(path),
        "active_primitives": active_primitives,
        "metric_status": metric_status,
        "native_symmetric_hausdorff": symmetric,
        "native_reconstruction_to_truth": reconstruction_to_truth,
        "native_truth_to_reconstruction": truth_to_reconstruction,
        "geometric_curvature_mean_absolute_error": curvature.get(
            "mean_absolute_error", None
        ),
        "geometric_curvature_rms_error": curvature.get("rms_error"),
        "geometric_curvature_max_absolute_error": curvature.get(
            "max_absolute_error", None
        ),
        "geometric_curvature_relative_l1_error": curvature.get(
            "relative_l1_error", None
        ),
        "geometric_curvature_reconstructed_length": curvature.get(
            "reconstructed_length", None
        ),
        "geometric_curvature_quadrature_samples": curvature.get(
            "quadrature_samples", 0
        ),
    }


def _case_cache_path(output: Path, geometry_path: Path) -> Path:
    parent = geometry_path.parent.parent.name
    return output / "cases" / f"{_slug(parent)}_{geometry_path.stem}.json"


def _run_case(path_text: str, output_text: str) -> Dict[str, Any]:
    path = Path(path_text)
    output = Path(output_text)
    cache = _case_cache_path(output, path)
    if cache.exists():
        try:
            row = json.loads(cache.read_text(encoding="utf-8"))
            row.setdefault("active_primitives", None)
            row.setdefault("metric_status", "complete")
            return row
        except (json.JSONDecodeError, ValueError):
            cache.unlink()
    row = _case_payload(path)
    write_json(cache, row)
    return row


def _finite_median(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = np.asarray(
        [float(row[field]) if row[field] is not None else math.nan for row in rows],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else math.nan


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    groups: Dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["method"]),
            str(row["variant"]),
            str(row["benchmark"]),
            int(row["cells_per_side"]),
        )
        groups.setdefault(key, []).append(row)
    summary = []
    metric_fields = (
        "native_symmetric_hausdorff",
        "native_reconstruction_to_truth",
        "native_truth_to_reconstruction",
        "geometric_curvature_mean_absolute_error",
        "geometric_curvature_rms_error",
        "geometric_curvature_relative_l1_error",
    )
    for (method, variant, benchmark, resolution), group in sorted(groups.items()):
        item: Dict[str, Any] = {
            "method": method,
            "variant": variant,
            "benchmark": benchmark,
            "cells_per_side": resolution,
            "cases": len(group),
        }
        for field in metric_fields:
            item[field + "_median"] = _finite_median(group, field)
        summary.append(item)

    by_curve: Dict[tuple[str, str, str], list[Dict[str, Any]]] = {}
    for row in summary:
        by_curve.setdefault(
            (row["method"], row["variant"], row["benchmark"]), []
        ).append(row)
    for group in by_curve.values():
        group.sort(key=lambda row: row["cells_per_side"])
        previous = None
        for row in group:
            for field in (
                "native_symmetric_hausdorff_median",
                "geometric_curvature_mean_absolute_error_median",
            ):
                value = float(row[field])
                key = field.removesuffix("_median") + "_order"
                row[key] = (
                    math.log(float(previous[field]) / value, 2.0)
                    if previous is not None
                    and math.isfinite(value)
                    and value > 0.0
                    and math.isfinite(float(previous[field]))
                    and float(previous[field]) > 0.0
                    else math.nan
                )
            previous = row
    return summary


def _label(method: str, variant: str) -> str:
    if method == "PCIC":
        if "center translation" in variant:
            return "PCIC center translation"
        if "radius adjustment" in variant:
            return "PCIC radius adjustment"
    return method


def _plot_geometry(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    fig, axes = plt.subplots(3, 2, figsize=(8.0, 8.6), sharex=True)
    axes = axes.ravel()
    styles = ("o", "s", "^", "D")
    curves = sorted({(row["method"], row["variant"]) for row in summary})
    for axis, benchmark in zip(axes, DEFAULT_BENCHMARKS):
        for (method, variant), marker in zip(curves, styles):
            rows = sorted(
                (
                    row
                    for row in summary
                    if row["benchmark"] == benchmark
                    and row["method"] == method
                    and row["variant"] == variant
                ),
                key=lambda row: row["cells_per_side"],
            )
            if not rows:
                continue
            axis.plot(
                [row["cells_per_side"] for row in rows],
                [row["native_symmetric_hausdorff_median"] for row in rows],
                marker=marker,
                label=_label(method, variant),
            )
        axis.set_title(benchmark.capitalize())
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks((32, 64, 128), ("32", "64", "128"))
        axis.grid(True, which="both", alpha=0.25)
    axes[5].axis("off")
    axes[0].legend(frameon=False, fontsize=7)
    for axis in axes[::2]:
        axis.set_ylabel("Native symmetric Hausdorff")
    for axis in axes[4:5]:
        axis.set_xlabel("Cells per side")
    fig.suptitle("Common native-geometry replay", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_smooth(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), sharex=True)
    styles = ("o", "s", "^", "D")
    curves = sorted({(row["method"], row["variant"]) for row in summary})
    for row_index, benchmark in enumerate(("circles", "ellipses")):
        for (method, variant), marker in zip(curves, styles):
            rows = sorted(
                (
                    row
                    for row in summary
                    if row["benchmark"] == benchmark
                    and row["method"] == method
                    and row["variant"] == variant
                ),
                key=lambda row: row["cells_per_side"],
            )
            if not rows:
                continue
            x = [row["cells_per_side"] for row in rows]
            axes[row_index, 0].plot(
                x,
                [row["native_symmetric_hausdorff_median"] for row in rows],
                marker=marker,
                label=_label(method, variant),
            )
            axes[row_index, 1].plot(
                x,
                [row["geometric_curvature_mean_absolute_error_median"] for row in rows],
                marker=marker,
                label=_label(method, variant),
            )
        axes[row_index, 0].set_ylabel(
            benchmark.capitalize() + "\nNative symmetric Hausdorff"
        )
        axes[row_index, 1].set_ylabel("Arc-length weighted |curvature error|")
    for axis in axes.ravel():
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks((32, 64, 128), ("32", "64", "128"))
        axis.grid(True, which="both", alpha=0.25)
    axes[0, 0].set_title("Geometry")
    axes[0, 1].set_title("Geometric curvature")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[1, 0].set_xlabel("Cells per side")
    axes[1, 1].set_xlabel("Cells per side")
    fig.suptitle("Common smooth-interface observables", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _input_geometry(paths: Iterable[Path]) -> list[Path]:
    return sorted(
        geometry for path in paths for geometry in (path / "geometry").glob("*.json")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    geometry_paths = _input_geometry(args.inputs)
    if not geometry_paths:
        raise SystemExit("no saved geometry files found")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(_run_case, str(path), str(args.output)): path
            for path in geometry_paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"completed {completed}/{len(futures)}", flush=True)

    rows.sort(
        key=lambda row: (
            row["method"],
            row["variant"],
            row["benchmark"],
            row["cells_per_side"],
            row["case_index"],
        )
    )
    write_csv(args.output / "case_results.csv", rows, CASE_FIELDS)
    summary = _aggregate(rows)
    summary_fields = tuple(summary[0])
    write_csv(args.output / "summary.csv", summary, summary_fields)
    write_json(
        args.output / "manifest.json",
        {
            "inputs": [str(path) for path in args.inputs],
            "case_count": len(rows),
            "geometry_metric": (
                "symmetric supremum of exact/native point-to-curve distances; "
                "source intervals use projected target endpoints and bounded optimization"
            ),
            "curvature_metric": (
                "arc-length-weighted mean absolute geometric-curvature error on "
                "native reconstructed primitives; nearest analytic truth branch"
            ),
            "curvature_benchmarks": sorted(SMOOTH_BENCHMARKS),
        },
    )
    _plot_geometry(summary, args.output / "native_geometry_all_methods.pdf")
    _plot_smooth(summary, args.output / "smooth_observables_all_methods.pdf")


if __name__ == "__main__":
    main()
