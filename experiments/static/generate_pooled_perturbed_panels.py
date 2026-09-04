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

import matplotlib.pyplot as plt
import numpy as np

from experiments.plotting import (
    PAPER_METRIC_LABELS,
    add_convergence_order_triangle,
    apply_paper_metric_axis_style,
    apply_paper_serif_style,
    draw_log_axis_break_marks,
    paper_markers_by_label,
)
from experiments.static.run_perturbed_sweeps import (
    DISPLAY_LABELS,
    PERTURBATION_AXIS_LABEL,
    RESOLUTION_AXIS_LABEL,
    _draw_method_curves,
    _merge_legend_entries,
    _save_figure,
)


apply_paper_serif_style()

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
ORDER_TRIANGLE_ANCHORS = {
    "circles": {
        "hausdorff": (0.72, 0.78),
        "facet_gap": (0.72, 0.78),
    },
    "ellipses": {"facet_gap": (0.65, 0.36)},
}
OUTPUT_NAMES = {
    "lines": "line_reconstruction_perturbed_all_methods_2x2.png",
    "squares": "square_reconstruction_perturbed_all_methods_2x2.png",
    "circles": "circle_reconstruction_perturbed_all_methods_5x2_axes.png",
    "ellipses": "ellipse_reconstruction_perturbed_all_methods_5x2_axes.png",
    "zalesak": "zalesak_reconstruction_perturbed_all_methods_2x2.png",
}
MARKERS_BY_LABEL = paper_markers_by_label(DISPLAY_LABELS)
BREAK_SPECS = {
    "squares": {
        "metrics": {"hausdorff", "facet_gap"},
        "lower": (2.0e-11, 2.0e-8),
        "upper": (3.0e-3, 1.2),
    },
    "circles": {
        "metrics": {"hausdorff", "facet_gap", "curvature_error"},
        "lower": (5.0e-11, 1.0e-8),
        "upper": (5.0e-5, 3.0e-1),
    },
    "zalesak": {
        "metrics": {"hausdorff", "facet_gap"},
        "lower": (1.0e-10, 4.0e-8),
        "upper": (3.0e-3, 1.2),
    },
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
                ).append(
                    value
                )
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
    if experiment in BREAK_SPECS:
        _plot_broken_grid(data, experiment, metrics, output)
        return

    exp_data = data[experiment]
    rows = len(metrics)
    figure_size = (7.05, 5.35) if rows == 2 else (7.05, 9.5)
    fig, axes = plt.subplots(rows, 2, figsize=figure_size, sharex="col")
    if rows == 1:
        axes = np.asarray([axes])
    legend_entries = {}

    for row_index, metric in enumerate(metrics):
        curves_by_axis = {
            "wiggle": _pooled_curves(exp_data, metric, "wiggle"),
            "resolution": _pooled_curves(exp_data, metric, "resolution"),
        }
        for column, axis_name in enumerate(("wiggle", "resolution")):
            axis = axes[row_index, column]
            x_mode = "perturbation" if axis_name == "wiggle" else "resolution"
            _draw_method_curves(
                axis,
                curves_by_axis[axis_name],
                metric,
                x_label=(
                    PERTURBATION_AXIS_LABEL
                    if axis_name == "wiggle"
                    else RESOLUTION_AXIS_LABEL
                ),
                x_mode=x_mode,
                exp_name=experiment,
            )
            apply_paper_metric_axis_style(
                axis,
                metric,
                x_mode,
                markers_by_label=MARKERS_BY_LABEL,
            )
            axis.set_title(
                ("Perturbation sweep" if axis_name == "wiggle" else "Resolution study")
                if row_index == 0
                else ""
            )
            if row_index == rows - 1:
                axis.set_xlabel(
                    (
                        r"Perturbation magnitude, $w$"
                        if axis_name == "wiggle"
                        else r"Cells per side, $N$"
                    )
                )
            else:
                axis.set_xlabel("")
            _merge_legend_entries(legend_entries, axis)

        left, right = axes[row_index]
        y_min = min(left.get_ylim()[0], right.get_ylim()[0])
        y_max = max(left.get_ylim()[1], right.get_ylim()[1])
        left.set_ylim(y_min, y_max)
        right.set_ylim(y_min, y_max)
        order = ORDER_TRIANGLES.get(experiment, {}).get(metric)
        if order is not None:
            add_convergence_order_triangle(
                right,
                order,
                anchor=ORDER_TRIANGLE_ANCHORS.get(experiment, {}).get(
                    metric, (0.72, 0.68)
                ),
                width=0.11,
                trend="decreasing",
                fontsize=7.5,
            )

    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            loc="upper center",
            ncol=min(3, len(legend_entries)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.005),
            columnspacing=0.9,
            handletextpad=0.4,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.90], h_pad=0.9, w_pad=0.8)
    _save_figure(fig, output)
    plt.close(fig)


def _plot_broken_grid(
    data: dict,
    experiment: str,
    metrics: tuple[str, ...],
    output: Path,
) -> None:
    exp_data = data[experiment]
    spec = BREAK_SPECS[experiment]
    row_specs = []
    height_ratios = []
    metric_rows = {}
    for metric_index, metric in enumerate(metrics):
        if metric in spec["metrics"]:
            upper_row = len(row_specs)
            row_specs.extend(((metric, "upper"), (metric, "lower")))
            height_ratios.extend((1.45, 0.48))
            metric_rows[metric] = (upper_row, upper_row + 1)
        else:
            row = len(row_specs)
            row_specs.append((metric, "full"))
            height_ratios.append(1.45)
            metric_rows[metric] = (row,)
        if metric_index != len(metrics) - 1:
            row_specs.append((None, "spacer"))
            height_ratios.append(0.20)

    figure_size = (7.05, 6.55) if len(metrics) == 2 else (7.05, 9.9)
    fig = plt.figure(figsize=figure_size)
    grid = fig.add_gridspec(
        len(row_specs),
        2,
        height_ratios=height_ratios,
        hspace=0.08,
        wspace=0.27,
    )
    axes_by_metric = {}
    legend_entries = {}

    for row, (metric, band) in enumerate(row_specs):
        if metric is None:
            continue
        axes_by_metric.setdefault(metric, {})[band] = []
        for column, axis_name in enumerate(("wiggle", "resolution")):
            share_axis = None
            if band == "lower":
                share_axis = axes_by_metric[metric]["upper"][column]
            axis = fig.add_subplot(grid[row, column], sharex=share_axis)
            curves = _pooled_curves(exp_data, metric, axis_name)
            x_mode = "perturbation" if axis_name == "wiggle" else "resolution"
            y_window = spec[band] if band in {"upper", "lower"} else None
            _draw_method_curves(
                axis,
                curves,
                metric,
                x_label="",
                x_mode=x_mode,
                exp_name=experiment,
                y_window=y_window,
            )
            apply_paper_metric_axis_style(
                axis,
                metric,
                x_mode,
                markers_by_label=MARKERS_BY_LABEL,
            )
            axis.set_ylabel("")
            if y_window is not None:
                axis.set_ylim(*y_window)
            if metric == metrics[0] and band in {"upper", "full"}:
                axis.set_title(
                    "Perturbation sweep"
                    if axis_name == "wiggle"
                    else "Resolution study"
                )
            final_band = metric == metrics[-1] and band in {"lower", "full"}
            if final_band:
                axis.set_xlabel(
                    r"Perturbation magnitude, $w$"
                    if axis_name == "wiggle"
                    else r"Cells per side, $N$"
                )
            else:
                axis.set_xlabel("")
            _merge_legend_entries(legend_entries, axis)
            axes_by_metric[metric][band].append(axis)

    for metric in metrics:
        bands = axes_by_metric[metric]
        if "upper" in bands:
            for column in range(2):
                draw_log_axis_break_marks(
                    bands["upper"][column],
                    bands["lower"][column],
                )
        else:
            left, right = bands["full"]
            y_min = min(left.get_ylim()[0], right.get_ylim()[0])
            y_max = max(left.get_ylim()[1], right.get_ylim()[1])
            left.set_ylim(y_min, y_max)
            right.set_ylim(y_min, y_max)

        order = ORDER_TRIANGLES.get(experiment, {}).get(metric)
        if order is not None:
            target = bands.get("upper", bands.get("full"))[1]
            add_convergence_order_triangle(
                target,
                order,
                anchor=ORDER_TRIANGLE_ANCHORS.get(experiment, {}).get(
                    metric, (0.72, 0.68)
                ),
                width=0.11,
                trend="decreasing",
                fontsize=7.5,
            )

    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            loc="upper center",
            ncol=min(3, len(legend_entries)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.998),
            columnspacing=0.9,
            handletextpad=0.4,
        )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.07, top=0.90)
    fig.canvas.draw()
    for metric in metrics:
        bands = axes_by_metric[metric]
        left_axes = [axes[0] for axes in bands.values()]
        bottom = min(axis.get_position().y0 for axis in left_axes)
        top = max(axis.get_position().y1 for axis in left_axes)
        fig.text(
            0.016,
            0.5 * (bottom + top),
            PAPER_METRIC_LABELS.get(metric, metric.replace("_", " ").title()),
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.5,
        )
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
