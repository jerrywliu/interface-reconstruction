#!/usr/bin/env python3
"""
Run perturbed-quad sweeps for static experiments.
Sweeps resolution, perturbation amplitude (wiggle), and seed.
Writes a CSV with per-run metrics and (optionally) sends aggregate plots to Slack.
"""

import argparse
import csv
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from util.io.slack import load_slack_env, send_results_to_slack


STATIC_GRID_SIZE = 100.0
LINEAR_ALGOS = ["Youngs", "ELVIRA", "LVIRA", "safe_linear", "linear"]
LINE_ALGOS = LINEAR_ALGOS
CIRCLE_ALGOS = LINEAR_ALGOS + ["safe_circle", "circular"]
ELLIPSE_ALGOS = LINEAR_ALGOS + ["safe_circle", "circular"]
SQUARE_ALGOS = LINEAR_ALGOS + ["linear+corner", "safe_circle", "circular"]
ZALESAK_ALGOS = LINEAR_ALGOS + ["safe_circle", "circular", "circular+corner"]

METHOD_ORDER = [
    "Youngs",
    "ELVIRA",
    "LVIRA",
    "safe_linear",
    "linear",
    "linear+C0",
    "safe_linear_corner",
    "linear+corner",
    "safe_circle",
    "circular",
    "circular+C0",
    "circular+corner",
    "circular+corner+C0",
]

DEFAULT_RESOLUTIONS = [0.32, 0.50, 0.64, 1.00, 1.28, 1.50]
DEFAULT_RESOLUTIONS_SHORT = [0.50, 0.64, 1.00, 1.28, 1.50]
DEFAULT_WIGGLES = [0.0, 0.05, 0.1, 0.2, 0.3]
DEFAULT_SEEDS = [0]

DISPLAY_LABELS = {
    "Youngs": "Youngs",
    "ELVIRA": "ELVIRA",
    "LVIRA": "LVIRA",
    "safe_linear": "Ours (safe linear)",
    "linear": "Ours (linear)",
    "linear+C0": "Ours (linear, C0)",
    "linear+corner": "Ours (linear+corner)",
    "safe_circle": "Ours (safe circular)",
    "circular": "Ours (circular)",
    "circular+C0": "Ours (circular, C0)",
    "circular+corner": "Ours (circular+corner)",
    "circular+corner+C0": "Ours (circular+corner, C0)",
}

METHOD_STYLES = {
    "Youngs": {"color": "#6c757d", "linestyle": "-", "linewidth": 1.8},
    "ELVIRA": {"color": "#495057", "linestyle": "--", "linewidth": 1.8},
    "LVIRA": {"color": "#212529", "linestyle": "-.", "linewidth": 1.8},
    "safe_linear": {"color": "#74a9cf", "linestyle": "--", "linewidth": 1.9},
    "linear": {"color": "#1d4ed8", "linestyle": "-", "linewidth": 2.3},
    "linear+C0": {"color": "#1d4ed8", "linestyle": "--", "linewidth": 2.3},
    "linear+corner": {"color": "#0f766e", "linestyle": "-", "linewidth": 2.4},
    "safe_circle": {"color": "#f59e0b", "linestyle": "--", "linewidth": 1.9},
    "circular": {"color": "#d97706", "linestyle": "-", "linewidth": 2.3},
    "circular+C0": {"color": "#d97706", "linestyle": "--", "linewidth": 2.3},
    "circular+corner": {"color": "#b91c1c", "linestyle": "-", "linewidth": 2.5},
    "circular+corner+C0": {"color": "#b91c1c", "linestyle": "--", "linewidth": 2.5},
}

METRIC_LABELS = {
    "hausdorff": "Hausdorff",
    "facet_gap": "Facet Gap",
    "curvature_error": "Curvature Error",
    "tangent_error": "Tangent Error",
    "area_error": "Area Error",
}

PERTURBATION_AXIS_LABEL = "Perturbation magnitude"
RESOLUTION_AXIS_LABEL = "Cells per side, N"


EXPERIMENTS = [
    {
        "name": "circles",
        "module": "experiments.static.circles",
        "config": "static/circle",
        "num_arg": "--num_circles",
        "algorithms": CIRCLE_ALGOS,
    },
    {
        "name": "ellipses",
        "module": "experiments.static.ellipses",
        "config": "static/ellipse",
        "num_arg": "--num_ellipses",
        "algorithms": ELLIPSE_ALGOS,
    },
    {
        "name": "lines",
        "module": "experiments.static.lines",
        "config": "static/line",
        "num_arg": "--num_lines",
        "algorithms": LINE_ALGOS,
    },
    {
        "name": "squares",
        "module": "experiments.static.squares",
        "config": "static/square",
        "num_arg": "--num_squares",
        "algorithms": SQUARE_ALGOS,
    },
    {
        "name": "zalesak",
        "module": "experiments.static.zalesak",
        "config": "static/zalesak",
        "num_arg": "--num_cases",
        "algorithms": ZALESAK_ALGOS,
    },
]


def _parse_list(value, cast=float):
    if value is None:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [cast(p) for p in parts]


def _parse_str_list(value):
    if value is None:
        return []
    return [p.strip().lower() for p in value.split(",") if p.strip()]


def _filter_algos(algos, selected_algos):
    if not selected_algos:
        return algos
    return [algo for algo in algos if algo.lower() in selected_algos]


def _safe_mean(values):
    if not values:
        return float("nan")
    mean_val = float(np.mean(np.array(values)))
    return mean_val


def _safe_stats(values):
    if not values:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
        }
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def _metric_stats(metric_name, values):
    stats = _safe_stats(values)
    return {f"{metric_name}_{key}": value for key, value in stats.items()}


def _display_method_label(algo):
    return DISPLAY_LABELS.get(algo, algo)


def _metric_label(metric):
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _resolution_to_cells_per_side(resolutions):
    return STATIC_GRID_SIZE * np.asarray(resolutions, dtype=float)


def _solver_floor_curve(exp_name, metric, x_mode, raw_x_values):
    if metric != "hausdorff" or x_mode != "resolution":
        return None

    if exp_name == "lines":
        epsilon = 1e-10
        label = r"approx. line-fit floor ($10^{-10}h$)"
    elif exp_name == "circles":
        epsilon = 1e-10
        label = r"approx. arc-fit floor ($10^{-10}h$)"
    else:
        return None

    resolutions = np.asarray(raw_x_values, dtype=float)
    floor = epsilon / resolutions
    return {
        "x_values": _resolution_to_cells_per_side(resolutions),
        "y_values": np.maximum(floor, 1e-14),
        "label": label,
    }


def _merge_legend_entries(legend_entries, ax):
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_") and label not in legend_entries:
            legend_entries[label] = handle
    return legend_entries


def _make_save_name(exp_name, algo, resolution, wiggle, seed):
    res_tag = str(resolution).replace(".", "p")
    wiggle_tag = str(wiggle).replace(".", "p")
    algo_tag = algo.lower().replace("+", "plus")
    return f"perturb_sweep_{exp_name}_{algo_tag}_r{res_tag}_w{wiggle_tag}_s{seed}"


def _algo_tag(algo):
    return algo.lower().replace("+", "plus")


def _run_subprocess(cmd, log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return result.returncode


def _read_metric_values(path):
    if not Path(path).exists():
        return []
    values = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values.append(line)
    return values


def _parse_numeric_values(path):
    values = []
    for line in _read_metric_values(path):
        try:
            values.append(float(line))
        except ValueError:
            pass
    return values


def _parse_labeled_values(path):
    values = []
    for line in _read_metric_values(path):
        try:
            values.append(float(line.split("_")[-1]))
        except ValueError:
            pass
    return values


def _collect_metrics(exp_name, save_name):
    metrics_dir = Path("plots") / save_name / "metrics"
    if exp_name in ["circles", "ellipses"]:
        curvature = _parse_numeric_values(metrics_dir / "curvature_error.txt")
        gaps = _parse_numeric_values(metrics_dir / "facet_gap.txt")
        hausdorff = _parse_numeric_values(metrics_dir / "hausdorff.txt")
        tangent = _parse_numeric_values(metrics_dir / "tangent_error.txt")
        curvature_proxy = _parse_numeric_values(
            metrics_dir / "curvature_proxy_error.txt"
        )
        metrics = {}
        metrics.update(_metric_stats("curvature_error", curvature))
        metrics.update(_metric_stats("facet_gap", gaps))
        metrics.update(_metric_stats("hausdorff", hausdorff))
        metrics.update(_metric_stats("tangent_error", tangent))
        metrics.update(_metric_stats("curvature_proxy_error", curvature_proxy))
        return metrics
    if exp_name == "lines":
        hausdorff = _parse_labeled_values(metrics_dir / "hausdorff.txt")
        gaps = _parse_labeled_values(metrics_dir / "facet_gap.txt")
        metrics = {}
        metrics.update(_metric_stats("hausdorff", hausdorff))
        metrics.update(_metric_stats("facet_gap", gaps))
        return metrics
    if exp_name == "squares":
        area = _parse_numeric_values(metrics_dir / "area_error.txt")
        gaps = _parse_numeric_values(metrics_dir / "facet_gap.txt")
        hausdorff = _parse_numeric_values(metrics_dir / "hausdorff.txt")
        metrics = {}
        metrics.update(_metric_stats("area_error", area))
        metrics.update(_metric_stats("facet_gap", gaps))
        metrics.update(_metric_stats("hausdorff", hausdorff))
        return metrics
    if exp_name == "zalesak":
        area = _parse_numeric_values(metrics_dir / "area_error.txt")
        gaps = _parse_numeric_values(metrics_dir / "facet_gap.txt")
        hausdorff = _parse_numeric_values(metrics_dir / "hausdorff.txt")
        metrics = {}
        metrics.update(_metric_stats("area_error", area))
        metrics.update(_metric_stats("facet_gap", gaps))
        metrics.update(_metric_stats("hausdorff", hausdorff))
        return metrics
    return {}


def _sample_indices(count, sample_count):
    if sample_count <= 0 or count <= 0:
        return []
    if count <= sample_count:
        return list(range(count))
    positions = np.linspace(0, count - 1, sample_count)
    indices = sorted({int(round(pos)) for pos in positions})
    if len(indices) < sample_count:
        for idx in range(count):
            if idx not in indices:
                indices.append(idx)
            if len(indices) >= sample_count:
                break
        indices = sorted(indices)
    return indices


def _build_aggregate_plot(save_name, sample_count=5):
    if sample_count <= 0:
        return None, []
    base_dir = Path("plots") / save_name / "plt"
    areas_dir = base_dir / "areas"
    partial_dir = base_dir / "partial_areas"
    if not areas_dir.exists() or not partial_dir.exists():
        return None, []

    area_files = sorted(areas_dir.glob("*.png"))
    partial_files = sorted(partial_dir.glob("*.png"))
    count = min(len(area_files), len(partial_files))
    if count == 0:
        return None, []

    indices = _sample_indices(count, sample_count)
    if not indices:
        return None, []

    try:
        from PIL import Image
    except Exception:
        return None, indices

    first = Image.open(area_files[indices[0]])
    tile_w, tile_h = first.size
    first.close()

    pad = 10
    cols = len(indices)
    rows = 2
    width = cols * tile_w + (cols + 1) * pad
    height = rows * tile_h + (rows + 1) * pad

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))

    for col, idx in enumerate(indices):
        area_path = area_files[idx]
        partial_path = partial_files[idx]

        area_img = Image.open(area_path)
        partial_img = Image.open(partial_path)

        if area_img.size != (tile_w, tile_h):
            area_img = area_img.resize((tile_w, tile_h))
        if partial_img.size != (tile_w, tile_h):
            partial_img = partial_img.resize((tile_w, tile_h))

        x = pad + col * (tile_w + pad)
        y_top = pad
        y_bottom = pad * 2 + tile_h
        canvas.paste(area_img, (x, y_top))
        canvas.paste(partial_img, (x, y_bottom))

        area_img.close()
        partial_img.close()

    out_dir = Path("results") / "static" / "aggregates"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{save_name}_aggregate.png"
    canvas.save(out_path)
    return str(out_path), indices


def _load_sweep_rows(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    algos = {row.get("algo") for row in rows}
    if "ELVIRA" not in algos and "LVIRA" in algos:
        for row in rows:
            if row.get("algo") == "LVIRA":
                row["algo"] = "ELVIRA"
    return rows


def _build_metric_index(rows):
    def _split_metric_key(metric_key):
        for suffix in ("_mean", "_median", "_p25", "_p75"):
            if metric_key.endswith(suffix):
                return metric_key[: -len(suffix)], suffix[1:]
        return metric_key, "value"

    data = {}
    for row in rows:
        try:
            exp = row["experiment"]
            algo = row["algo"]
            metric, stat = _split_metric_key(row["metric_key"])
            res = float(row["resolution"])
            wiggle = float(row["wiggle"])
            value = float(row["metric_value"])
        except (KeyError, ValueError):
            continue

        data.setdefault(exp, {}).setdefault(algo, {}).setdefault(metric, {}).setdefault(
            res, {}
        ).setdefault(wiggle, {}).setdefault(stat, []).append(value)

    return data


def _plot_metric_vs_wiggle(exp, algo, metric, res_map, out_dir):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    min_error = 1e-14

    def _aggregate(values):
        if not values:
            return float("nan")
        arr = np.asarray(values, dtype=float)
        return float(np.median(arr))

    for res in sorted(res_map.keys()):
        wiggle_map = res_map[res]
        wiggles = sorted(wiggle_map.keys())
        medians = []
        p25 = []
        p75 = []
        for w in wiggles:
            stats = wiggle_map[w]
            if not stats:
                medians.append(float("nan"))
                p25.append(float("nan"))
                p75.append(float("nan"))
                continue

            med = (
                _aggregate(stats.get("median"))
                if "median" in stats
                else _aggregate(stats.get("mean"))
                if "mean" in stats
                else _aggregate(stats.get("value"))
            )
            q25 = _aggregate(stats.get("p25")) if "p25" in stats else med
            q75 = _aggregate(stats.get("p75")) if "p75" in stats else med

            medians.append(max(med, min_error))
            p25.append(max(q25, min_error))
            p75.append(max(q75, min_error))

        cells_per_side = int(round(STATIC_GRID_SIZE * float(res)))
        plt.plot(
            wiggles,
            medians,
            marker="o",
            linewidth=2.0,
            label=f"N={cells_per_side}",
        )
        plt.fill_between(wiggles, p25, p75, alpha=0.08)

    metric_label = _metric_label(metric)
    plt.xlabel(PERTURBATION_AXIS_LABEL, fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    plt.title(
        f"{exp} {_display_method_label(algo)} ({metric_label})",
        fontsize=14,
        fontweight="bold",
    )
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, frameon=True)
    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{exp}_{_algo_tag(algo)}_{metric}.png"
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out_path)


def _extract_stat_value(stats):
    for key in ("median", "mean", "value"):
        values = stats.get(key)
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        return float(np.median(arr))
    return float("nan")


def _ordered_methods(methods):
    ordered = [algo for algo in METHOD_ORDER if algo in methods]
    for algo in sorted(methods):
        if algo not in ordered:
            ordered.append(algo)
    return ordered


def _build_method_curves(exp_data, metric):
    metric_wiggles = set()
    for algo_data in exp_data.values():
        res_map = algo_data.get(metric, {})
        for wiggle_map in res_map.values():
            metric_wiggles.update(wiggle_map.keys())
    wiggles = sorted(metric_wiggles)
    if not wiggles:
        return {}

    curves = {}
    for algo, algo_data in exp_data.items():
        res_map = algo_data.get(metric, {})
        if not res_map:
            continue
        medians = []
        p25 = []
        p75 = []
        for wiggle in wiggles:
            values = []
            for wiggle_map in res_map.values():
                stats = wiggle_map.get(wiggle)
                if not stats:
                    continue
                value = _extract_stat_value(stats)
                if not math.isnan(value):
                    values.append(value)
            if not values:
                medians.append(float("nan"))
                p25.append(float("nan"))
                p75.append(float("nan"))
                continue
            arr = np.asarray(values, dtype=float)
            medians.append(float(np.median(arr)))
            p25.append(float(np.percentile(arr, 25)))
            p75.append(float(np.percentile(arr, 75)))
        medians_arr = np.asarray(medians, dtype=float)
        if not np.any(np.isfinite(medians_arr)):
            continue
        curves[algo] = {
            "x_values": np.asarray(wiggles, dtype=float),
            "median": medians_arr,
            "p25": np.asarray(p25, dtype=float),
            "p75": np.asarray(p75, dtype=float),
        }
    return curves


def _build_method_curves_by_resolution(exp_data, metric):
    metric_resolutions = set()
    for algo_data in exp_data.values():
        res_map = algo_data.get(metric, {})
        metric_resolutions.update(res_map.keys())
    resolutions = sorted(metric_resolutions)
    if not resolutions:
        return {}

    curves = {}
    for algo, algo_data in exp_data.items():
        res_map = algo_data.get(metric, {})
        if not res_map:
            continue
        medians = []
        p25 = []
        p75 = []
        for resolution in resolutions:
            wiggle_map = res_map.get(resolution, {})
            values = []
            for stats in wiggle_map.values():
                if not stats:
                    continue
                value = _extract_stat_value(stats)
                if not math.isnan(value):
                    values.append(value)
            if not values:
                medians.append(float("nan"))
                p25.append(float("nan"))
                p75.append(float("nan"))
                continue
            arr = np.asarray(values, dtype=float)
            medians.append(float(np.median(arr)))
            p25.append(float(np.percentile(arr, 25)))
            p75.append(float(np.percentile(arr, 75)))

        medians_arr = np.asarray(medians, dtype=float)
        if not np.any(np.isfinite(medians_arr)):
            continue
        curves[algo] = {
            "x_values": np.asarray(resolutions, dtype=float),
            "median": medians_arr,
            "p25": np.asarray(p25, dtype=float),
            "p75": np.asarray(p75, dtype=float),
        }

    return curves


def _draw_method_curves(ax, curves, metric, x_label, x_mode, exp_name=None):
    min_error = 1e-14
    for algo in _ordered_methods(curves.keys()):
        series = curves[algo]
        x_values = np.asarray(series["x_values"], dtype=float)
        medians = series["median"]
        p25 = series["p25"]
        p75 = series["p75"]
        valid = np.isfinite(medians)
        if not np.any(valid):
            continue
        x_values_v = x_values[valid]
        if x_mode == "resolution":
            x_values_v = _resolution_to_cells_per_side(x_values_v)
        medians_v = np.maximum(medians[valid], min_error)
        p25_v = np.maximum(p25[valid], min_error)
        p75_v = np.maximum(p75[valid], min_error)
        style = METHOD_STYLES.get(algo, {})
        label = _display_method_label(algo)
        ax.plot(
            x_values_v,
            medians_v,
            marker="o",
            markersize=4.2,
            label=label,
            **style,
        )
        ax.fill_between(
            x_values_v,
            p25_v,
            p75_v,
            alpha=0.08,
            color=style.get("color", None),
            zorder=1,
        )

    floor_curve = _solver_floor_curve(
        exp_name=exp_name,
        metric=metric,
        x_mode=x_mode,
        raw_x_values=np.unique(
            np.concatenate(
                [np.asarray(series["x_values"], dtype=float) for series in curves.values()]
            )
        ),
    )
    if floor_curve is not None:
        ax.plot(
            floor_curve["x_values"],
            floor_curve["y_values"],
            color="#111827",
            linestyle=":",
            linewidth=1.6,
            label=floor_curve["label"],
            zorder=3,
        )

    metric_label = _metric_label(metric)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if x_mode == "resolution":
        tick_values = sorted(
            {
                int(round(val))
                for series in curves.values()
                for val in _resolution_to_cells_per_side(series["x_values"])
            }
        )
        ax.set_xticks(tick_values)


def _generate_experiment_method_summary_plots(data, exp_name, metric_candidates, out_dir):
    import matplotlib.pyplot as plt

    exp_data = data.get(exp_name, {})
    if not exp_data:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = []
    metric_curves = {}
    for metric in metric_candidates:
        curves = _build_method_curves(exp_data, metric)
        if not curves:
            continue
        metric_curves[metric] = curves

        fig, ax = plt.subplots(figsize=(8, 6))
        _draw_method_curves(
            ax,
            curves,
            metric,
            x_label=PERTURBATION_AXIS_LABEL,
            x_mode="perturbation",
            exp_name=exp_name,
        )
        ax.set_title(
            f"{exp_name.title()} Perturbed Sweep: All Methods ({metric.replace('_', ' ').title()})",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(fontsize=10, frameon=True)
        fig.tight_layout()

        out_path = out_dir / f"{exp_name}_all_methods_{metric}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        plots.append(str(out_path))

    metric_order = [metric for metric in metric_candidates if metric in metric_curves]
    if len(metric_order) >= 2:
        fig, axes = plt.subplots(1, len(metric_order), figsize=(6 * len(metric_order), 5.5))
        if len(metric_order) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metric_order):
            _draw_method_curves(
                ax,
                metric_curves[metric],
                metric,
                x_label=PERTURBATION_AXIS_LABEL,
                x_mode="perturbation",
                exp_name=exp_name,
            )
            ax.set_title(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, fontsize=9, frameon=True)
        fig.suptitle(
            f"{exp_name.title()} Perturbed Sweep: Method Comparison",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        combined_path = out_dir / f"{exp_name}_all_methods_combined.png"
        fig.savefig(combined_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        plots.append(str(combined_path))

    return plots


def _generate_lines_method_axes_grid_plot(data, out_dir):
    return _generate_two_metric_axes_grid_plot(
        data=data,
        out_dir=out_dir,
        exp_name="lines",
        metric_left="hausdorff",
        metric_right="facet_gap",
        figure_title="Lines Perturbed Sweep: Method Comparison",
        out_filename="lines_all_methods_2x2.png",
    )


def _generate_lines_method_summary_plots(data, out_dir):
    plots = _generate_experiment_method_summary_plots(
        data,
        "lines",
        ("hausdorff", "facet_gap"),
        out_dir,
    )
    plots.extend(_generate_lines_method_axes_grid_plot(data, out_dir))
    return plots


def _generate_two_metric_axes_grid_plot(
    data,
    out_dir,
    exp_name,
    metric_left,
    metric_right,
    figure_title,
    out_filename=None,
):
    import matplotlib.pyplot as plt

    exp_data = data.get(exp_name, {})
    if not exp_data:
        return []

    metrics = (metric_left, metric_right)
    wiggle_curves = {}
    resolution_curves = {}
    for metric in metrics:
        curves_w = _build_method_curves(exp_data, metric)
        if curves_w:
            wiggle_curves[metric] = curves_w
        curves_r = _build_method_curves_by_resolution(exp_data, metric)
        if curves_r:
            resolution_curves[metric] = curves_r

    if not wiggle_curves and not resolution_curves:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    def _metric_title(metric):
        return _metric_label(metric)

    subplot_defs = [
        (
            0,
            0,
            metric_left,
            wiggle_curves,
            PERTURBATION_AXIS_LABEL,
            f"{_metric_title(metric_left)} vs {PERTURBATION_AXIS_LABEL.title()}",
        ),
        (
            0,
            1,
            metric_left,
            resolution_curves,
            RESOLUTION_AXIS_LABEL,
            f"{_metric_title(metric_left)} vs Cells per Side",
        ),
        (
            1,
            0,
            metric_right,
            wiggle_curves,
            PERTURBATION_AXIS_LABEL,
            f"{_metric_title(metric_right)} vs {PERTURBATION_AXIS_LABEL.title()}",
        ),
        (
            1,
            1,
            metric_right,
            resolution_curves,
            RESOLUTION_AXIS_LABEL,
            f"{_metric_title(metric_right)} vs Cells per Side",
        ),
    ]

    legend_entries = {}
    for row, col, metric, curves_by_metric, x_label, title in subplot_defs:
        ax = axes[row][col]
        curves = curves_by_metric.get(metric)
        if not curves:
            ax.set_axis_off()
            continue
        _draw_method_curves(
            ax,
            curves,
            metric,
            x_label=x_label,
            x_mode="perturbation" if col == 0 else "resolution",
            exp_name=exp_name,
        )
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        _merge_legend_entries(legend_entries, ax)

    for row in range(2):
        row_axes = [axes[row][col] for col in range(2) if axes[row][col].axison]
        if row_axes:
            ymin = min(ax.get_ylim()[0] for ax in row_axes)
            ymax = max(ax.get_ylim()[1] for ax in row_axes)
            for ax in row_axes:
                ax.set_ylim(ymin, ymax)

    for col in range(2):
        col_axes = [axes[row][col] for row in range(2) if axes[row][col].axison]
        if col_axes:
            xmin = min(ax.get_xlim()[0] for ax in col_axes)
            xmax = max(ax.get_xlim()[1] for ax in col_axes)
            for ax in col_axes:
                ax.set_xlim(xmin, xmax)

    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            loc="lower center",
            ncol=min(6, len(legend_entries)),
            fontsize=9,
            frameon=True,
            bbox_to_anchor=(0.5, -0.01),
        )

    fig.suptitle(figure_title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_name = out_filename or f"{exp_name}_all_methods_2x2.png"
    out_path = out_dir / output_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [str(out_path)]


def _generate_circle_method_axes_grid_plot(data, out_dir):
    import matplotlib.pyplot as plt

    exp_data = data.get("circles", {})
    if not exp_data:
        return []

    metrics = (
        "hausdorff",
        "facet_gap",
        "curvature_error",
        "tangent_error",
    )

    wiggle_curves = {}
    resolution_curves = {}
    for metric in metrics:
        curves_w = _build_method_curves(exp_data, metric)
        if curves_w:
            wiggle_curves[metric] = curves_w
        curves_r = _build_method_curves_by_resolution(exp_data, metric)
        if curves_r:
            resolution_curves[metric] = curves_r

    available_metrics = [
        metric
        for metric in metrics
        if metric in wiggle_curves or metric in resolution_curves
    ]
    if not available_metrics:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = []

    for metric in available_metrics:
        metric_label = _metric_label(metric)
        curves_w = wiggle_curves.get(metric)
        if curves_w:
            fig, ax = plt.subplots(figsize=(8, 6))
            _draw_method_curves(
                ax,
                curves_w,
                metric,
                x_label=PERTURBATION_AXIS_LABEL,
                x_mode="perturbation",
                exp_name="circles",
            )
            ax.set_title(
                f"Circles All Methods: {metric_label} vs {PERTURBATION_AXIS_LABEL.title()}",
                fontsize=12.5,
                fontweight="bold",
            )
            ax.legend(fontsize=10, frameon=True)
            fig.tight_layout()
            out_path = out_dir / f"circles_all_methods_{metric}_vs_perturbation.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            plots.append(str(out_path))

        curves_r = resolution_curves.get(metric)
        if curves_r:
            fig, ax = plt.subplots(figsize=(8, 6))
            _draw_method_curves(
                ax,
                curves_r,
                metric,
                x_label=RESOLUTION_AXIS_LABEL,
                x_mode="resolution",
                exp_name="circles",
            )
            ax.set_title(
                f"Circles All Methods: {metric_label} vs Cells per Side",
                fontsize=12.5,
                fontweight="bold",
            )
            ax.legend(fontsize=10, frameon=True)
            fig.tight_layout()
            out_path = out_dir / f"circles_all_methods_{metric}_vs_resolution.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            plots.append(str(out_path))

    rows = len(available_metrics)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4.2 * rows))
    if rows == 1:
        axes = np.array([axes])

    legend_entries = {}
    for row, metric in enumerate(available_metrics):
        metric_label = _metric_label(metric)

        ax_w = axes[row][0]
        curves_w = wiggle_curves.get(metric)
        if curves_w:
            _draw_method_curves(
                ax_w,
                curves_w,
                metric,
                x_label=PERTURBATION_AXIS_LABEL,
                x_mode="perturbation",
                exp_name="circles",
            )
            ax_w.set_title(
                f"{metric_label} vs {PERTURBATION_AXIS_LABEL.title()}",
                fontsize=11.5,
                fontweight="bold",
            )
            _merge_legend_entries(legend_entries, ax_w)
        else:
            ax_w.set_axis_off()

        ax_r = axes[row][1]
        curves_r = resolution_curves.get(metric)
        if curves_r:
            _draw_method_curves(
                ax_r,
                curves_r,
                metric,
                x_label=RESOLUTION_AXIS_LABEL,
                x_mode="resolution",
                exp_name="circles",
            )
            ax_r.set_title(
                f"{metric_label} vs Cells per Side",
                fontsize=11.5,
                fontweight="bold",
            )
            _merge_legend_entries(legend_entries, ax_r)
        else:
            ax_r.set_axis_off()

    for row in range(rows):
        row_axes = [axes[row][col] for col in range(2) if axes[row][col].axison]
        if row_axes:
            ymin = min(ax.get_ylim()[0] for ax in row_axes)
            ymax = max(ax.get_ylim()[1] for ax in row_axes)
            for ax in row_axes:
                ax.set_ylim(ymin, ymax)

    for col in range(2):
        col_axes = [axes[row][col] for row in range(rows) if axes[row][col].axison]
        if col_axes:
            xmin = min(ax.get_xlim()[0] for ax in col_axes)
            xmax = max(ax.get_xlim()[1] for ax in col_axes)
            for ax in col_axes:
                ax.set_xlim(xmin, xmax)

    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            loc="lower center",
            ncol=min(6, len(legend_entries)),
            fontsize=9,
            frameon=True,
            bbox_to_anchor=(0.5, -0.005),
        )

    fig.suptitle(
        "Circles Perturbed Sweep: Methods Across Both X-Axes",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    grid_path = out_dir / "circles_all_methods_5x2_axes.png"
    fig.savefig(grid_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plots.append(str(grid_path))
    return plots


def _generate_ellipse_method_axes_grid_plot(data, out_dir):
    import matplotlib.pyplot as plt

    exp_data = data.get("ellipses", {})
    if not exp_data:
        return []

    metrics = (
        "hausdorff",
        "facet_gap",
        "curvature_error",
        "tangent_error",
    )

    wiggle_curves = {}
    resolution_curves = {}
    for metric in metrics:
        curves_w = _build_method_curves(exp_data, metric)
        if curves_w:
            wiggle_curves[metric] = curves_w
        curves_r = _build_method_curves_by_resolution(exp_data, metric)
        if curves_r:
            resolution_curves[metric] = curves_r

    available_metrics = [
        metric
        for metric in metrics
        if metric in wiggle_curves or metric in resolution_curves
    ]
    if not available_metrics:
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = []

    for metric in available_metrics:
        metric_label = _metric_label(metric)
        curves_w = wiggle_curves.get(metric)
        if curves_w:
            fig, ax = plt.subplots(figsize=(8, 6))
            _draw_method_curves(
                ax,
                curves_w,
                metric,
                x_label=PERTURBATION_AXIS_LABEL,
                x_mode="perturbation",
                exp_name="ellipses",
            )
            ax.set_title(
                f"Ellipses All Methods: {metric_label} vs {PERTURBATION_AXIS_LABEL.title()}",
                fontsize=12.5,
                fontweight="bold",
            )
            ax.legend(fontsize=10, frameon=True)
            fig.tight_layout()
            out_path = out_dir / f"ellipses_all_methods_{metric}_vs_perturbation.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            plots.append(str(out_path))

        curves_r = resolution_curves.get(metric)
        if curves_r:
            fig, ax = plt.subplots(figsize=(8, 6))
            _draw_method_curves(
                ax,
                curves_r,
                metric,
                x_label=RESOLUTION_AXIS_LABEL,
                x_mode="resolution",
                exp_name="ellipses",
            )
            ax.set_title(
                f"Ellipses All Methods: {metric_label} vs Cells per Side",
                fontsize=12.5,
                fontweight="bold",
            )
            ax.legend(fontsize=10, frameon=True)
            fig.tight_layout()
            out_path = out_dir / f"ellipses_all_methods_{metric}_vs_resolution.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            plots.append(str(out_path))

    rows = len(available_metrics)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4.2 * rows))
    if rows == 1:
        axes = np.array([axes])

    legend_entries = {}
    for row, metric in enumerate(available_metrics):
        metric_label = _metric_label(metric)

        ax_w = axes[row][0]
        curves_w = wiggle_curves.get(metric)
        if curves_w:
            _draw_method_curves(
                ax_w,
                curves_w,
                metric,
                x_label=PERTURBATION_AXIS_LABEL,
                x_mode="perturbation",
                exp_name="ellipses",
            )
            ax_w.set_title(
                f"{metric_label} vs {PERTURBATION_AXIS_LABEL.title()}",
                fontsize=11.5,
                fontweight="bold",
            )
            _merge_legend_entries(legend_entries, ax_w)
        else:
            ax_w.set_axis_off()

        ax_r = axes[row][1]
        curves_r = resolution_curves.get(metric)
        if curves_r:
            _draw_method_curves(
                ax_r,
                curves_r,
                metric,
                x_label=RESOLUTION_AXIS_LABEL,
                x_mode="resolution",
                exp_name="ellipses",
            )
            ax_r.set_title(
                f"{metric_label} vs Cells per Side",
                fontsize=11.5,
                fontweight="bold",
            )
            _merge_legend_entries(legend_entries, ax_r)
        else:
            ax_r.set_axis_off()

    for row in range(rows):
        row_axes = [axes[row][col] for col in range(2) if axes[row][col].axison]
        if row_axes:
            ymin = min(ax.get_ylim()[0] for ax in row_axes)
            ymax = max(ax.get_ylim()[1] for ax in row_axes)
            for ax in row_axes:
                ax.set_ylim(ymin, ymax)

    for col in range(2):
        col_axes = [axes[row][col] for row in range(rows) if axes[row][col].axison]
        if col_axes:
            xmin = min(ax.get_xlim()[0] for ax in col_axes)
            xmax = max(ax.get_xlim()[1] for ax in col_axes)
            for ax in col_axes:
                ax.set_xlim(xmin, xmax)

    if legend_entries:
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            loc="lower center",
            ncol=min(6, len(legend_entries)),
            fontsize=9,
            frameon=True,
            bbox_to_anchor=(0.5, -0.005),
        )

    fig.suptitle(
        "Ellipses Perturbed Sweep: Methods Across Both X-Axes",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    grid_path = out_dir / "ellipses_all_methods_5x2_axes.png"
    fig.savefig(grid_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plots.append(str(grid_path))
    return plots


def _generate_circle_method_summary_plots(data, out_dir):
    plots = _generate_experiment_method_summary_plots(
        data,
        "circles",
        ("hausdorff", "facet_gap", "curvature_error"),
        out_dir,
    )
    plots.extend(_generate_circle_method_axes_grid_plot(data, out_dir))
    return plots


def _generate_ellipse_method_summary_plots(data, out_dir):
    plots = _generate_experiment_method_summary_plots(
        data,
        "ellipses",
        ("hausdorff", "facet_gap", "curvature_error"),
        out_dir,
    )
    plots.extend(_generate_ellipse_method_axes_grid_plot(data, out_dir))
    return plots


def _generate_square_method_summary_plots(data, out_dir):
    plots = _generate_experiment_method_summary_plots(
        data,
        "squares",
        ("hausdorff", "facet_gap"),
        out_dir,
    )
    plots.extend(
        _generate_two_metric_axes_grid_plot(
            data=data,
            out_dir=out_dir,
            exp_name="squares",
            metric_left="hausdorff",
            metric_right="facet_gap",
            figure_title="Squares Perturbed Sweep: Method Comparison",
            out_filename="squares_all_methods_2x2.png",
        )
    )
    return plots


def _generate_zalesak_method_summary_plots(data, out_dir):
    plots = _generate_experiment_method_summary_plots(
        data,
        "zalesak",
        ("hausdorff", "facet_gap"),
        out_dir,
    )
    plots.extend(
        _generate_two_metric_axes_grid_plot(
            data=data,
            out_dir=out_dir,
            exp_name="zalesak",
            metric_left="hausdorff",
            metric_right="facet_gap",
            figure_title="Zalesak Perturbed Sweep: Method Comparison",
            out_filename="zalesak_all_methods_2x2.png",
        )
    )
    return plots


def _generate_summary_plots(csv_path, out_dir):
    rows = _load_sweep_rows(csv_path)
    data = _build_metric_index(rows)
    plots_by_exp = {}

    for exp, exp_data in data.items():
        for algo, algo_data in exp_data.items():
            for metric, res_map in algo_data.items():
                plot_path = _plot_metric_vs_wiggle(
                    exp, algo, metric, res_map, out_dir
                )
                plots_by_exp.setdefault(exp, []).append(plot_path)

    circle_summary_plots = _generate_circle_method_summary_plots(data, out_dir)
    if circle_summary_plots:
        plots_by_exp.setdefault("circles", []).extend(circle_summary_plots)

    ellipse_summary_plots = _generate_ellipse_method_summary_plots(data, out_dir)
    if ellipse_summary_plots:
        plots_by_exp.setdefault("ellipses", []).extend(ellipse_summary_plots)

    lines_summary_plots = _generate_lines_method_summary_plots(data, out_dir)
    if lines_summary_plots:
        plots_by_exp.setdefault("lines", []).extend(lines_summary_plots)

    squares_summary_plots = _generate_square_method_summary_plots(data, out_dir)
    if squares_summary_plots:
        plots_by_exp.setdefault("squares", []).extend(squares_summary_plots)

    zalesak_summary_plots = _generate_zalesak_method_summary_plots(data, out_dir)
    if zalesak_summary_plots:
        plots_by_exp.setdefault("zalesak", []).extend(zalesak_summary_plots)

    return plots_by_exp


def _filter_all_methods_summary_paths(plot_paths):
    return [
        path
        for path in plot_paths
        if "_all_methods_" in Path(path).name.lower()
    ]


def _send_results_to_slack_logged(message, file_paths):
    try:
        ok = send_results_to_slack(message, file_paths)
    except Exception as exc:
        print(f"[SLACK] failed: {message} ({exc})")
        return False

    if ok:
        print(f"[SLACK] sent: {message}")
    else:
        print(f"[SLACK] failed: {message}")
    return ok


def _notify_stage_summary(csv_path, out_dir, exp_name):
    plots_by_exp = _generate_summary_plots(csv_path, out_dir)
    plot_paths = plots_by_exp.get(exp_name, [])
    summary_paths = _filter_all_methods_summary_paths(plot_paths)
    if summary_paths:
        resolved_paths = [str(Path(path).resolve()) for path in summary_paths]
        return _send_results_to_slack_logged(
            f"Perturbed sweep stage complete: {exp_name}",
            resolved_paths,
        )
    return False


def main():
    parser = argparse.ArgumentParser(description="Run perturbed-quad sweeps for static experiments.")
    parser.add_argument("--circles", type=int, default=25, help="num circles")
    parser.add_argument("--ellipses", type=int, default=25, help="num ellipses")
    parser.add_argument("--lines", type=int, default=25, help="num lines")
    parser.add_argument("--squares", type=int, default=25, help="num squares")
    parser.add_argument("--zalesak", type=int, default=25, help="num zalesak cases")
    parser.add_argument(
        "--resolutions",
        type=str,
        default=None,
        help="comma-separated resolutions (default: match linear sweeps)",
    )
    parser.add_argument(
        "--wiggles",
        type=str,
        default=None,
        help="comma-separated wiggle amplitudes (default: 0,0.05,0.1,0.2,0.3)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="comma-separated seeds (default: 0)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated experiment names to run (e.g., lines,ellipses)",
    )
    parser.add_argument(
        "--algos",
        type=str,
        default=None,
        help="comma-separated algorithms to run (e.g., ELVIRA,LVIRA)",
    )
    parser.add_argument(
        "--aggregate_samples",
        type=int,
        default=0,
        help="number of sample cases in aggregate image (0 to disable)",
    )
    parser.add_argument("--notify", action="store_true", help="send aggregates to Slack")
    parser.add_argument("--dry_run", action="store_true", help="skip execution")
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="directory to write subprocess logs (default: logs/perturbed_sweeps/<timestamp>)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="output CSV path (default: results/static/perturbed_sweep_<timestamp>.csv)",
    )
    parser.add_argument(
        "--summary_dir",
        type=str,
        default=None,
        help="directory for generated summary plots (default: results/static/perturbed_plots)",
    )
    parser.add_argument(
        "--plot_from_csv",
        type=str,
        default=None,
        help="skip sweep execution and generate summary plots from an existing CSV",
    )
    parser.add_argument(
        "--collect_existing",
        action="store_true",
        help="skip execution and collect metrics from existing plots/perturb_sweep_* outputs",
    )

    args = parser.parse_args()

    load_slack_env()
    notify = args.notify or os.getenv("SLACK_NOTIFY", "").lower() in {"1", "true", "yes"}
    summary_dir = Path(
        args.summary_dir
        or os.path.join("results", "static", "perturbed_plots")
    ).resolve()

    if args.plot_from_csv:
        csv_path = Path(args.plot_from_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        plots_by_exp = _generate_summary_plots(str(csv_path), summary_dir)
        print(f"Generated perturbed summary plots from {csv_path}")
        if notify:
            for exp, plot_paths in plots_by_exp.items():
                summary_paths = _filter_all_methods_summary_paths(plot_paths)
                if summary_paths:
                    resolved_paths = [str(Path(path).resolve()) for path in summary_paths]
                    _send_results_to_slack_logged(
                        f"Perturbed sweep all-method summaries: {exp}",
                        resolved_paths,
                    )
        return

    resolutions_override = _parse_list(args.resolutions, float)
    wiggles = _parse_list(args.wiggles, float) or DEFAULT_WIGGLES
    seeds = _parse_list(args.seeds, int) or DEFAULT_SEEDS
    only_experiments = set(_parse_str_list(args.only))
    selected_algos = set(_parse_str_list(args.algos))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or os.path.join("logs", "perturbed_sweeps", stamp)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    out_csv = args.out_csv or os.path.join(
        "results", "static", f"perturbed_sweep_{stamp}.csv"
    )
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    failures = []
    stage_notified = set()

    with open(out_csv, "w", newline="") as csvfile:
        fieldnames = [
            "experiment",
            "algo",
            "resolution",
            "wiggle",
            "seed",
            "metric_key",
            "metric_value",
            "save_name",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        if args.collect_existing:
            print(f"Collecting existing perturbed sweep metrics at {datetime.now().isoformat()}")
        else:
            print(f"Perturbed sweeps started at {datetime.now().isoformat()}")
        print(f"Logging to {log_dir}")
        print(f"Writing CSV to {out_csv}")

        if args.dry_run:
            return

        for exp in EXPERIMENTS:
            if only_experiments and exp["name"] not in only_experiments:
                continue
            exp_algos = _filter_algos(exp["algorithms"], selected_algos)
            if not exp_algos:
                continue
            num_value = getattr(args, exp["name"], None)
            if resolutions_override:
                exp_resolutions = resolutions_override
            elif exp["name"] in {"squares", "zalesak"}:
                exp_resolutions = DEFAULT_RESOLUTIONS_SHORT
            else:
                exp_resolutions = DEFAULT_RESOLUTIONS

            for resolution in exp_resolutions:
                for wiggle in wiggles:
                    for seed in seeds:
                        for algo in exp_algos:
                            save_name = _make_save_name(
                                exp["name"], algo, resolution, wiggle, seed
                            )
                            cmd = [
                                sys.executable,
                                "-m",
                                exp["module"],
                                "--config",
                                exp["config"],
                                "--resolution",
                                str(resolution),
                                "--facet_algo",
                                algo,
                                "--save_name",
                                save_name,
                                "--mesh_type",
                                "perturbed_quads",
                                "--perturb_wiggle",
                                str(wiggle),
                                "--perturb_seed",
                                str(seed),
                                "--perturb_fix_boundary",
                                "1",
                            ]
                            if num_value is not None:
                                cmd += [exp["num_arg"], str(num_value)]

                            if args.collect_existing:
                                metrics = _collect_metrics(exp["name"], save_name)
                                if not metrics:
                                    print(
                                        f"[SKIP] Missing existing metrics for {exp['name']} {algo} r={resolution} w={wiggle} s={seed}"
                                    )
                                    continue
                            else:
                                log_path = Path(log_dir) / f"{save_name}.log"
                                code = _run_subprocess(cmd, log_path)
                                if code != 0:
                                    failures.append(
                                        {
                                            "experiment": exp["name"],
                                            "algo": algo,
                                            "resolution": resolution,
                                            "wiggle": wiggle,
                                            "seed": seed,
                                            "code": code,
                                        }
                                    )
                                    print(
                                        f"[ERROR] {exp['name']} {algo} r={resolution} w={wiggle} s={seed} failed (code {code})"
                                    )
                                    continue

                                metrics = _collect_metrics(exp["name"], save_name)
                            for key, value in metrics.items():
                                writer.writerow(
                                    {
                                        "experiment": exp["name"],
                                        "algo": algo,
                                        "resolution": resolution,
                                        "wiggle": wiggle,
                                        "seed": seed,
                                        "metric_key": key,
                                        "metric_value": value,
                                        "save_name": save_name,
                                    }
                                )

                            if notify:
                                agg_path, indices = _build_aggregate_plot(
                                    save_name, sample_count=args.aggregate_samples
                                )
                                if agg_path:
                                    _send_results_to_slack_logged(
                                        f"{exp['name']} {algo} r={resolution} w={wiggle} s={seed}: aggregate samples {indices}",
                                        [agg_path],
                                    )

            csvfile.flush()
            if notify:
                if _notify_stage_summary(out_csv, summary_dir, exp["name"]):
                    stage_notified.add(exp["name"])

    print("\n=== Perturbed sweep summary ===")
    print(f"Failures: {len(failures)}")
    for failure in failures:
        print(
            f"- {failure['experiment']} / {failure['algo']} / r={failure['resolution']} / w={failure['wiggle']} / s={failure['seed']} (code {failure['code']})"
        )

    plots_by_exp = _generate_summary_plots(out_csv, summary_dir)

    if notify:
        for exp, plot_paths in plots_by_exp.items():
            if exp in stage_notified:
                continue
            summary_paths = _filter_all_methods_summary_paths(plot_paths)
            if summary_paths:
                resolved_paths = [str(Path(path).resolve()) for path in summary_paths]
                _send_results_to_slack_logged(
                    f"Perturbed sweep all-method summaries: {exp}",
                    resolved_paths,
                )
        _send_results_to_slack_logged(
            f"Perturbed sweep complete. CSV: {out_csv}",
            [str(Path(out_csv).resolve())],
        )


if __name__ == "__main__":
    main()
