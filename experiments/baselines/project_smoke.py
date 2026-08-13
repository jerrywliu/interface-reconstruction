"""Shared result schema and reporting utilities for project baseline smokes."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from main.algos.baselines.external_geometry import ExternalBaselineResult


CASE_FIELDS = (
    "method",
    "variant",
    "benchmark",
    "case_index",
    "random_seed",
    "cells_per_side",
    "cell_size",
    "runtime_seconds",
    "mixed_cells",
    "reconstructed_cells",
    "paper_fallback_cells",
    "unsupported_cells",
    "unresolved_cells",
    "optimizer_failures",
    "sampled_hausdorff",
    "sampled_reconstruction_to_truth",
    "shared_edge_gap_mean",
    "shared_edge_gap_max",
    "unmatched_crossings",
    "conservation_mean_absolute_residual",
    "conservation_max_absolute_residual",
    "curvature_estimator_mean_absolute_error",
    "curvature_estimator_median_absolute_error",
    "curvature_estimator_max_absolute_error",
    "curvature_estimator_samples",
    "parameters_json",
    "status_message",
)

CELL_FIELDS = (
    "method",
    "variant",
    "benchmark",
    "case_index",
    "cells_per_side",
    "cell_x",
    "cell_y",
    "status",
    "target_phase_area",
    "reconstructed_phase_area",
    "absolute_area_residual",
    "curvature_estimate",
    "true_curvature",
    "curvature_absolute_error",
    "objective",
    "optimizer_success",
    "ghf_method",
    "diagnostics_json",
)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sample_primitives(geometry: Any, spacing: float) -> np.ndarray:
    if isinstance(geometry, ExternalBaselineResult):
        primitives = tuple(
            primitive
            for record in geometry.cells.values()
            for primitive in record.primitives()
        )
    else:
        primitives = tuple(geometry)
    points = []
    for primitive in primitives:
        count = max(2, int(math.ceil(primitive.length() / spacing)) + 1)
        points.extend(primitive.sample(count))
    return np.asarray(points, dtype=float)


def sampled_symmetric_hausdorff(
    reconstruction: ExternalBaselineResult,
    truth: Iterable[Any],
    *,
    spacing: float,
) -> float:
    """Return a reproducible sampled Hausdorff diagnostic.

    This deliberately does not claim the adaptive native-geometry accuracy of
    ``symmetric_hausdorff_external``.  It is the inexpensive common smoke-test
    metric used before the full 25-case baseline sweep.
    """

    reconstructed_points = _sample_primitives(reconstruction, spacing)
    truth_points = _sample_primitives(tuple(truth), spacing)
    if not len(reconstructed_points) or not len(truth_points):
        return float("inf")
    truth_tree = cKDTree(truth_points)
    reconstructed_tree = cKDTree(reconstructed_points)
    return max(
        float(np.max(truth_tree.query(reconstructed_points, k=1)[0])),
        float(np.max(reconstructed_tree.query(truth_points, k=1)[0])),
    )


def sampled_directed_hausdorff(
    source: Any,
    target: Iterable[Any],
    *,
    spacing: float,
) -> float:
    """Return the sampled maximum distance from ``source`` to ``target``."""

    source_points = _sample_primitives(source, spacing)
    target_points = _sample_primitives(tuple(target), spacing)
    if not len(source_points) or not len(target_points):
        return float("inf")
    return float(np.max(cKDTree(target_points).query(source_points, k=1)[0]))


def aggregate_case_rows(rows: Sequence[Mapping[str, Any]]) -> list:
    grouped: Dict[Tuple[str, str, str, int], list] = {}
    for row in rows:
        key = (
            str(row["method"]),
            str(row["variant"]),
            str(row["benchmark"]),
            int(row["cells_per_side"]),
        )
        grouped.setdefault(key, []).append(row)
    summary = []
    for (method, variant, benchmark, resolution), group in sorted(grouped.items()):
        item: Dict[str, Any] = {
            "method": method,
            "variant": variant,
            "benchmark": benchmark,
            "cells_per_side": resolution,
            "cases": len(group),
        }
        for field in (
            "sampled_hausdorff",
            "sampled_reconstruction_to_truth",
            "shared_edge_gap_mean",
            "shared_edge_gap_max",
            "conservation_max_absolute_residual",
            "curvature_estimator_median_absolute_error",
            "runtime_seconds",
        ):
            values = np.asarray([float(row[field]) for row in group], dtype=float)
            finite = values[np.isfinite(values)]
            item[field + "_median"] = float(np.median(finite)) if len(finite) else None
            item[field + "_max"] = float(np.max(finite)) if len(finite) else None
            item[field + "_nonfinite_cases"] = int(len(values) - len(finite))
        for field in (
            "mixed_cells",
            "reconstructed_cells",
            "paper_fallback_cells",
            "unsupported_cells",
            "unresolved_cells",
            "optimizer_failures",
            "unmatched_crossings",
        ):
            item[field + "_total"] = int(sum(int(row[field]) for row in group))
        item["unresolved_fraction"] = (
            item["unresolved_cells_total"] + item["unsupported_cells_total"]
        ) / max(1, item["mixed_cells_total"])
        summary.append(item)
    return summary


def plot_all_benchmarks(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9})
    available = {str(row["benchmark"]) for row in summary}
    preferred = ("lines", "squares", "circles", "ellipses", "zalesak")
    benchmarks = tuple(name for name in preferred if name in available)
    fig, axes = plt.subplots(len(benchmarks), 2, figsize=(9.0, 2.55 * len(benchmarks)))
    if len(benchmarks) == 1:
        axes = np.asarray([axes])
    for index, benchmark in enumerate(benchmarks):
        rows = sorted(
            (row for row in summary if row["benchmark"] == benchmark),
            key=lambda row: int(row["cells_per_side"]),
        )
        resolutions = [int(row["cells_per_side"]) for row in rows]
        left, right = axes[index]
        for field, label, marker in (
            (
                "sampled_reconstruction_to_truth_median",
                "reconstruction-to-truth",
                "o",
            ),
            ("shared_edge_gap_mean_median", "facet gap", "s"),
        ):
            values = [row[field] for row in rows]
            if all(value is not None and value > 0.0 for value in values):
                left.plot(resolutions, values, marker=marker, label=label)
        left.set_xscale("log", base=2)
        left.set_yscale("log")
        left.set_xticks(resolutions, [str(value) for value in resolutions])
        left.set_ylabel(benchmark.capitalize())
        left.grid(True, which="both", alpha=0.25)
        if index == 0:
            left.legend(frameon=False, fontsize=8)

        curvature = [row["curvature_estimator_median_absolute_error_median"] for row in rows]
        if all(value is not None and value > 0.0 for value in curvature):
            right.plot(resolutions, curvature, marker="^", color="#b24745", label="GHF curvature error")
            right.set_yscale("log")
        unresolved = [100.0 * float(row["unresolved_fraction"]) for row in rows]
        right.text(
            0.98,
            0.82,
            "unsupported/unresolved: "
            + ", ".join(f"{value:.1f}%" for value in unresolved),
            transform=right.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#555555",
        )
        if index == 0:
            right.legend(frameon=False, fontsize=8, loc="upper left")
        right.set_xscale("log", base=2)
        right.set_xticks(resolutions, [str(value) for value in resolutions])
        right.grid(True, which="both", alpha=0.25)
        right.set_ylabel("Curvature estimator error")
    axes[-1, 0].set_xlabel("Cells per side")
    axes[-1, 1].set_xlabel("Cells per side")
    fig.suptitle("PLVIRA Cartesian five-case smoke", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "CASE_FIELDS",
    "CELL_FIELDS",
    "aggregate_case_rows",
    "append_csv",
    "plot_all_benchmarks",
    "sampled_directed_hausdorff",
    "sampled_symmetric_hausdorff",
    "write_csv",
    "write_json",
]
