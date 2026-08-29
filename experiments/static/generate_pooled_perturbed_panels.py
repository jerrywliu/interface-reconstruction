#!/usr/bin/env python3
"""Generate paper-facing perturbed-mesh panels from pooled case-level rows."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from experiments.plotting import add_convergence_order_triangle
from experiments.static.run_perturbed_sweeps import (
    PERTURBATION_AXIS_LABEL,
    RESOLUTION_AXIS_LABEL,
    _draw_method_curves,
    _merge_legend_entries,
    _metric_label,
    _save_figure,
)


mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

DEFAULT_SOURCE = (
    REPO_ROOT
    / "results"
    / "static"
    / "submission_static_20260731_012430_505aefa45432.sealed"
)
DEFAULT_CURVATURE = (
    REPO_ROOT
    / "results"
    / "submission"
    / "perturbed_native_curvature_panels_20260822"
    / "case_curvature_comparison.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "submission" / "perturbed_pooled_panels_20260828"
)
METRICS = ("hausdorff", "facet_gap", "curvature_error", "tangent_error")
CURVED_EXPERIMENTS = ("circles", "ellipses")
ORDER_TRIANGLES = {
    "circles": {"hausdorff": 2.0, "facet_gap": 2.0},
    "ellipses": {"facet_gap": 3.0, "curvature_error": 1.0},
}
OUTPUT_NAMES = {
    "lines": "line_reconstruction_perturbed_all_methods_2x2.png",
    "squares": "square_reconstruction_perturbed_all_methods_2x2.png",
    "circles": "circle_reconstruction_perturbed_all_methods_5x2_axes.png",
    "ellipses": "ellipse_reconstruction_perturbed_all_methods_5x2_axes.png",
    "zalesak": "zalesak_reconstruction_perturbed_all_methods_2x2.png",
}


def _native_curvature_lookup(path: Path) -> dict[tuple, float]:
    values = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (
                row["experiment"],
                row["algo"],
                float(row["resolution"]),
                float(row["wiggle"]),
                int(row["seed"]),
                int(row["case_index"]),
            )
            values[key] = float(row["native_curvature_mae"])
    return values


def load_case_index(case_metrics: Path, curvature_metrics: Path) -> dict:
    native_curvature = _native_curvature_lookup(curvature_metrics)
    data = {}
    used_native = set()
    with case_metrics.open(newline="") as stream:
        for row in csv.DictReader(stream):
            experiment = row["experiment"]
            algo = row["algo"]
            resolution = float(row["resolution"])
            wiggle = float(row["wiggle"])
            case_key = (
                experiment,
                algo,
                resolution,
                wiggle,
                int(row["seed"]),
                int(row["case_index"]),
            )
            for metric in METRICS:
                raw = row.get(metric)
                if metric == "curvature_error" and experiment in CURVED_EXPERIMENTS:
                    if case_key not in native_curvature:
                        raise KeyError(f"missing native curvature row: {case_key}")
                    value = native_curvature[case_key]
                    used_native.add(case_key)
                elif raw in (None, ""):
                    continue
                else:
                    value = float(raw)
                data.setdefault(experiment, {}).setdefault(algo, {}).setdefault(
                    metric, {}
                ).setdefault(resolution, {}).setdefault(wiggle, {}).setdefault(
                    "value", []
                ).append(value)
    if used_native != set(native_curvature):
        missing = set(native_curvature) - used_native
        raise ValueError(f"unused native curvature rows: {len(missing)}")
    return data


def _pooled_curves(exp_data: dict, metric: str, axis: str) -> dict:
    if axis == "resolution":
        x_values = sorted(
            {
                resolution
                for algo_data in exp_data.values()
                for resolution in algo_data.get(metric, {})
            }
        )
    elif axis == "wiggle":
        x_values = sorted(
            {
                wiggle
                for algo_data in exp_data.values()
                for wiggle_map in algo_data.get(metric, {}).values()
                for wiggle in wiggle_map
            }
        )
    else:
        raise ValueError(f"unknown pooled axis: {axis}")

    curves = {}
    for algo, algo_data in exp_data.items():
        metric_data = algo_data.get(metric, {})
        if not metric_data:
            continue
        medians = []
        p25 = []
        p75 = []
        for x_value in x_values:
            if axis == "resolution":
                samples = [
                    value
                    for stats in metric_data.get(x_value, {}).values()
                    for value in stats.get("value", [])
                ]
            else:
                samples = [
                    value
                    for wiggle_map in metric_data.values()
                    for value in wiggle_map.get(x_value, {}).get("value", [])
                ]
            values = np.asarray([v for v in samples if np.isfinite(v)], dtype=float)
            if values.size == 0:
                medians.append(float("nan"))
                p25.append(float("nan"))
                p75.append(float("nan"))
            else:
                medians.append(float(np.median(values)))
                p25.append(float(np.percentile(values, 25)))
                p75.append(float(np.percentile(values, 75)))
        median_array = np.asarray(medians, dtype=float)
        if np.any(np.isfinite(median_array)):
            curves[algo] = {
                "x_values": np.asarray(x_values, dtype=float),
                "median": median_array,
                "p25": np.asarray(p25, dtype=float),
                "p75": np.asarray(p75, dtype=float),
            }
    return curves


def _plot_grid(
    data: dict,
    experiment: str,
    metrics: tuple[str, ...],
    output: Path,
) -> None:
    exp_data = data[experiment]
    rows = len(metrics)
    figure_size = (12, 9.5) if rows == 2 else (14, 16.8)
    fig, axes = plt.subplots(rows, 2, figsize=figure_size)
    if rows == 1:
        axes = np.asarray([axes])
    legend_entries = {}

    for row_index, metric in enumerate(metrics):
        curves_by_axis = {
            "wiggle": _pooled_curves(exp_data, metric, "wiggle"),
            "resolution": _pooled_curves(exp_data, metric, "resolution"),
        }
        metric_label = (
            "Curvature MAE" if metric == "curvature_error" else _metric_label(metric)
        )
        for column, axis_name in enumerate(("wiggle", "resolution")):
            axis = axes[row_index, column]
            _draw_method_curves(
                axis,
                curves_by_axis[axis_name],
                metric,
                x_label=(
                    PERTURBATION_AXIS_LABEL
                    if axis_name == "wiggle"
                    else RESOLUTION_AXIS_LABEL
                ),
                x_mode="perturbation" if axis_name == "wiggle" else "resolution",
                exp_name=experiment,
            )
            axis.set_title(
                f"{metric_label} vs "
                f"{'Perturbation Magnitude' if axis_name == 'wiggle' else 'Cells per Side'}",
                fontsize=11.5,
                fontweight="bold",
            )
            if metric == "curvature_error":
                axis.set_ylabel("Curvature MAE", fontsize=11)
            _merge_legend_entries(legend_entries, axis)

        left, right = axes[row_index]
        y_min = min(left.get_ylim()[0], right.get_ylim()[0])
        y_max = max(left.get_ylim()[1], right.get_ylim()[1])
        left.set_ylim(y_min, y_max)
        right.set_ylim(y_min, y_max)
        order = ORDER_TRIANGLES.get(experiment, {}).get(metric)
        if order is not None:
            right.set_xscale("log")
            add_convergence_order_triangle(
                right,
                order,
                anchor=(0.72, 0.68),
                width=0.11,
                trend="decreasing",
                fontsize=7.5,
            )

    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            loc="lower center",
            ncol=min(6, len(legend_entries)),
            fontsize=8.5 if rows > 2 else 9,
            frameon=True,
            bbox_to_anchor=(0.5, -0.004),
        )
    title = f"{experiment.title()} Reconstruction on Perturbed Cartesian Meshes"
    fig.suptitle(title, fontsize=15 if rows > 2 else 14, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0.045, 1, 0.96], h_pad=1.8, w_pad=1.4)
    _save_figure(fig, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--curvature", type=Path, default=DEFAULT_CURVATURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_case_index(
        args.source / "diagnostics" / "case_metrics.csv", args.curvature
    )
    for experiment in ("lines", "squares", "zalesak"):
        _plot_grid(
            data,
            experiment,
            ("hausdorff", "facet_gap"),
            args.output / OUTPUT_NAMES[experiment],
        )
    for experiment in CURVED_EXPERIMENTS:
        _plot_grid(data, experiment, METRICS, args.output / OUTPUT_NAMES[experiment])


if __name__ == "__main__":
    main()
