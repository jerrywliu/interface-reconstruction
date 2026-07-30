#!/usr/bin/env python3
"""Estimate ellipse convergence orders from case-indexed static metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_METRICS = (
    REPO_ROOT
    / "results/static/static_paper_simplified_default_20260717_212413"
    / "diagnostics/case_metrics.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/submission/convergence"

EXPERIMENT = "ellipses"
ALGORITHM = "circular"
GRID_SIZE = 100.0
EXPECTED_RESOLUTIONS = (32, 50, 64, 100, 128, 150)
METRICS = ("hausdorff", "tangent_error", "facet_gap")
METRIC_LABELS = {
    "hausdorff": "Hausdorff distance",
    "tangent_error": "Tangent error",
    "facet_gap": "Facet gap",
}
GEOMETRIC_METRICS = {"hausdorff", "facet_gap"}
SOLVER_TOLERANCE = 1.0e-10
PLOT_FLOOR = 1.0e-14
FIT_WINDOWS = (
    ("all_valid_non_floor", None),
    ("finest_five", 5),
    ("finest_four", 4),
)

mpl.rcParams.update(
    {
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_width(resolution: float) -> float:
    """Return physical cell width for the repository's resolution convention."""
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError(f"resolution must be positive and finite, got {resolution}")
    return 1.0 / resolution


def cells_per_side(resolution: float) -> int:
    return int(round(GRID_SIZE * resolution))


def numerical_floor(metric: str, h: float) -> float:
    """Return the metric-specific numerical floor used for fit exclusion.

    Geometric distances are compared with the reconstruction tolerance scaled by
    the cell width. Tangent error is dimensionless, so it uses the tolerance
    directly. The plotting floor prevents zero-valued logarithms.
    """
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError(f"h must be positive and finite, got {h}")
    if metric in GEOMETRIC_METRICS:
        return max(PLOT_FLOOR, SOLVER_TOLERANCE * h)
    if metric == "tangent_error":
        return max(PLOT_FLOOR, SOLVER_TOLERANCE)
    raise ValueError(f"unsupported metric: {metric}")


def fit_power_law(
    h_values: Sequence[float], error_values: Sequence[float]
) -> dict[str, float]:
    """Fit ``error = prefactor * h**order`` in natural-log coordinates."""
    h = np.asarray(h_values, dtype=float)
    error = np.asarray(error_values, dtype=float)
    if h.shape != error.shape:
        raise ValueError("h_values and error_values must have the same shape")
    valid = np.isfinite(h) & np.isfinite(error) & (h > 0.0) & (error > 0.0)
    if int(np.count_nonzero(valid)) < 2:
        raise ValueError("at least two positive finite points are required")

    log_h = np.log(h[valid])
    log_error = np.log(error[valid])
    order, intercept = np.polyfit(log_h, log_error, 1)
    fitted = intercept + order * log_h
    residual_sum = float(np.sum((log_error - fitted) ** 2))
    total_sum = float(np.sum((log_error - np.mean(log_error)) ** 2))
    if total_sum <= np.finfo(float).eps:
        r_squared = 1.0 if residual_sum <= np.finfo(float).eps else 0.0
    else:
        r_squared = 1.0 - residual_sum / total_sum

    return {
        "order": float(order),
        "intercept_log_e": float(intercept),
        "prefactor": float(math.exp(intercept)),
        "r_squared": float(r_squared),
    }


def select_fit_points(
    points: Sequence[Mapping[str, object]], max_points: int | None = None
) -> tuple[list[Mapping[str, object]], list[dict[str, object]]]:
    """Select non-floor points, retaining the finest ``max_points`` if set."""
    valid: list[Mapping[str, object]] = []
    excluded: list[dict[str, object]] = []
    for point in sorted(points, key=lambda item: int(item["N"])):
        error = float(point["median_error"])
        floor = float(point["numerical_floor"])
        if not math.isfinite(error) or error <= 0.0:
            excluded.append({"N": int(point["N"]), "reason": "invalid_or_nonpositive"})
        elif error <= floor:
            excluded.append({"N": int(point["N"]), "reason": "numerical_floor"})
        else:
            valid.append(point)

    if max_points is not None and len(valid) > max_points:
        outside_window = valid[:-max_points]
        valid = valid[-max_points:]
        excluded.extend(
            {"N": int(point["N"]), "reason": "outside_fit_window"}
            for point in outside_window
        )

    excluded.sort(key=lambda item: int(item["N"]))
    return valid, excluded


def load_case_metrics(path: Path, expected_cases: int = 25) -> dict[str, object]:
    grouped: dict[tuple[float, float, str], list[tuple[int, float]]] = defaultdict(list)
    source_rows: list[dict[str, str]] = []

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = {"experiment", "algo", "resolution", "wiggle", "case_index"}
        missing.update(METRICS)
        missing.difference_update(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        for row in reader:
            if row["experiment"] != EXPERIMENT or row["algo"] != ALGORITHM:
                continue
            source_rows.append(row)
            resolution = float(row["resolution"])
            wiggle = float(row["wiggle"])
            case_index = int(row["case_index"])
            for metric in METRICS:
                grouped[(wiggle, resolution, metric)].append(
                    (case_index, float(row[metric]))
                )

    if not source_rows:
        raise ValueError(f"no {EXPERIMENT}/{ALGORITHM} rows found in {path}")

    points: list[dict[str, object]] = []
    for (wiggle, resolution, metric), case_values in sorted(grouped.items()):
        case_indices = [case_index for case_index, _ in case_values]
        if len(case_indices) != len(set(case_indices)):
            raise ValueError(
                f"duplicate cases for w={wiggle}, resolution={resolution}, metric={metric}"
            )
        if expected_cases and len(case_values) != expected_cases:
            raise ValueError(
                f"expected {expected_cases} cases for w={wiggle}, "
                f"resolution={resolution}, metric={metric}; got {len(case_values)}"
            )
        values = np.asarray([value for _, value in case_values], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"nonfinite values for w={wiggle}, resolution={resolution}, metric={metric}"
            )
        h = cell_width(resolution)
        floor = numerical_floor(metric, h)
        median = float(np.median(values))
        points.append(
            {
                "metric": metric,
                "wiggle": wiggle,
                "mesh_family": "cartesian" if wiggle == 0.0 else "perturbed",
                "resolution": resolution,
                "N": cells_per_side(resolution),
                "h": h,
                "case_count": len(case_values),
                "median_error": median,
                "numerical_floor": floor,
                "at_or_below_floor": median <= floor,
            }
        )

    observed_resolutions = sorted({int(point["N"]) for point in points})
    if observed_resolutions != list(EXPECTED_RESOLUTIONS):
        raise ValueError(
            f"expected resolutions {list(EXPECTED_RESOLUTIONS)}, "
            f"got {observed_resolutions}"
        )

    context_fields = (
        "source_commit",
        "source_branch",
        "plic_fallback",
        "rescue_profile",
        "corner_behavior_profile",
    )
    context = {
        field: sorted({row.get(field, "") for row in source_rows})
        for field in context_fields
    }
    return {"points": points, "source_row_count": len(source_rows), "context": context}


def build_fit_rows(points: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[Mapping[str, object]]] = defaultdict(list)
    for point in points:
        grouped[(str(point["metric"]), float(point["wiggle"]))].append(point)

    rows: list[dict[str, object]] = []
    for (metric, wiggle), metric_points in sorted(grouped.items()):
        for window_name, max_points in FIT_WINDOWS:
            selected, excluded = select_fit_points(metric_points, max_points=max_points)
            if len(selected) < 2:
                raise ValueError(
                    f"insufficient points for metric={metric}, w={wiggle}, "
                    f"window={window_name}"
                )
            fit = fit_power_law(
                [float(point["h"]) for point in selected],
                [float(point["median_error"]) for point in selected],
            )
            rows.append(
                {
                    "metric": metric,
                    "wiggle": wiggle,
                    "mesh_family": "cartesian" if wiggle == 0.0 else "perturbed",
                    "fit_window": window_name,
                    "order": fit["order"],
                    "intercept_log_e": fit["intercept_log_e"],
                    "prefactor": fit["prefactor"],
                    "r_squared": fit["r_squared"],
                    "point_count": len(selected),
                    "included_N": [int(point["N"]) for point in selected],
                    "included_h": [float(point["h"]) for point in selected],
                    "included_median_error": [
                        float(point["median_error"]) for point in selected
                    ],
                    "excluded_points": excluded,
                }
            )
    return rows


def build_sensitivity_summary(
    fit_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in fit_rows:
        if float(row["wiggle"]) > 0.0:
            grouped[(str(row["metric"]), str(row["fit_window"]))].append(row)

    summaries = []
    for (metric, fit_window), rows in sorted(grouped.items()):
        orders = np.asarray([float(row["order"]) for row in rows], dtype=float)
        summaries.append(
            {
                "metric": metric,
                "fit_window": fit_window,
                "wiggles": [float(row["wiggle"]) for row in rows],
                "order_min": float(np.min(orders)),
                "order_median": float(np.median(orders)),
                "order_max": float(np.max(orders)),
            }
        )
    return summaries


def _list_string(values: Iterable[object]) -> str:
    return ";".join(str(value) for value in values)


def write_points_csv(path: Path, points: Sequence[Mapping[str, object]]) -> None:
    fieldnames = (
        "metric",
        "wiggle",
        "mesh_family",
        "resolution",
        "N",
        "h",
        "case_count",
        "median_error",
        "numerical_floor",
        "at_or_below_floor",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fieldnames} for row in points)


def write_fits_csv(path: Path, fit_rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = (
        "metric",
        "wiggle",
        "mesh_family",
        "fit_window",
        "order",
        "intercept_log_e",
        "prefactor",
        "r_squared",
        "point_count",
        "included_N",
        "included_h",
        "included_median_error",
        "excluded_N",
        "excluded_reasons",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in fit_rows:
            excluded = list(row["excluded_points"])
            writer.writerow(
                {
                    **{field: row[field] for field in fieldnames[:9]},
                    "included_N": _list_string(row["included_N"]),
                    "included_h": _list_string(row["included_h"]),
                    "included_median_error": _list_string(
                        row["included_median_error"]
                    ),
                    "excluded_N": _list_string(item["N"] for item in excluded),
                    "excluded_reasons": _list_string(
                        item["reason"] for item in excluded
                    ),
                }
            )


def _fit_lookup(
    fit_rows: Sequence[Mapping[str, object]], metric: str, wiggle: float, window: str
) -> Mapping[str, object]:
    return next(
        row
        for row in fit_rows
        if row["metric"] == metric
        and math.isclose(float(row["wiggle"]), wiggle)
        and row["fit_window"] == window
    )


def plot_convergence(
    path: Path,
    points: Sequence[Mapping[str, object]],
    fit_rows: Sequence[Mapping[str, object]],
) -> None:
    grouped: dict[tuple[str, float], list[Mapping[str, object]]] = defaultdict(list)
    for point in points:
        grouped[(str(point["metric"]), float(point["wiggle"]))].append(point)

    wiggle_styles = {
        0.0: {"color": "#111827", "marker": "s", "linewidth": 2.0, "label": "w = 0"},
        0.05: {"color": "#8da0b6", "marker": None, "linewidth": 1.0, "label": "w = 0.05"},
        0.1: {"color": "#00796b", "marker": "o", "linewidth": 2.0, "label": "w = 0.1"},
        0.2: {"color": "#d97706", "marker": None, "linewidth": 1.0, "label": "w = 0.2"},
        0.3: {"color": "#a33b75", "marker": None, "linewidth": 1.0, "label": "w = 0.3"},
    }
    metric_order = ("hausdorff", "tangent_error", "facet_gap")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.45), sharex=True)

    for axis_index, (ax, metric) in enumerate(zip(axes, metric_order)):
        for wiggle in sorted(wiggle_styles):
            metric_points = sorted(grouped[(metric, wiggle)], key=lambda row: int(row["N"]))
            n_values = np.asarray([int(row["N"]) for row in metric_points])
            errors = np.asarray([float(row["median_error"]) for row in metric_points])
            style = wiggle_styles[wiggle]
            ax.plot(
                n_values,
                errors,
                color=style["color"],
                marker=style["marker"],
                markersize=4.0,
                linewidth=style["linewidth"],
                alpha=1.0 if wiggle in (0.0, 0.1) else 0.72,
                label=style["label"] if axis_index == 0 else None,
                zorder=3 if wiggle in (0.0, 0.1) else 2,
            )

        reference_n = np.asarray(EXPECTED_RESOLUTIONS[2:], dtype=float)
        anchor = float(
            next(
                row["median_error"]
                for row in grouped[(metric, 0.1)]
                if int(row["N"]) == int(reference_n[0])
            )
        )
        for order, factor, linestyle in ((1, 4.0, ":"), (2, 1.8, "--"), (3, 0.8, "-.")):
            guide = anchor * factor * (reference_n[0] / reference_n) ** order
            ax.plot(
                reference_n,
                guide,
                color="#6b7280",
                linewidth=0.9,
                linestyle=linestyle,
                alpha=0.8,
                label=rf"$O(h^{order})$" if axis_index == 0 else None,
                zorder=1,
            )

        cart = _fit_lookup(fit_rows, metric, 0.0, "finest_five")
        perturbed = _fit_lookup(fit_rows, metric, 0.1, "finest_five")
        ax.text(
            0.04,
            0.05,
            rf"finest 5: $p_{{w=0}}={float(cart['order']):.2f}$"
            + "\n"
            + rf"$p_{{w=0.1}}={float(perturbed['order']):.2f}$",
            transform=ax.transAxes,
            fontsize=7.2,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.88, "pad": 2.5},
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(EXPECTED_RESOLUTIONS)
        ax.set_xticklabels([str(value) for value in EXPECTED_RESOLUTIONS])
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel("Cells per side, $N$")
        ax.grid(True, which="both", color="#d1d5db", linewidth=0.5, alpha=0.65)
    axes[0].set_ylabel("Median error over 25 cases")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
        columnspacing=1.2,
        handlelength=2.2,
    )
    fig.suptitle("Circular reconstruction of ellipses: empirical convergence", y=0.99)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.86, bottom=0.25, wspace=0.28)
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Title": "Ellipse convergence analysis",
            "Subject": "Median case metrics and empirical log-log orders",
        },
    )
    plt.close(fig)


def build_report(
    input_path: Path,
    loaded: Mapping[str, object],
    fit_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    points = list(loaded["points"])
    return {
        "schema_version": 1,
        "source": {
            "case_metrics": str(input_path.resolve()),
            "sha256": sha256_file(input_path),
            "selected_row_count": loaded["source_row_count"],
            "context": loaded["context"],
        },
        "selection": {
            "experiment": EXPERIMENT,
            "algorithm": ALGORITHM,
            "metrics": list(METRICS),
            "resolutions_N": list(EXPECTED_RESOLUTIONS),
            "wiggles": sorted({float(point["wiggle"]) for point in points}),
            "aggregation": "median over cases at fixed (metric, wiggle, N)",
        },
        "fit_definition": {
            "model": "error = prefactor * h**order",
            "cell_width": "h = 1 / resolution = 100 / N",
            "logarithm": "natural",
            "sign_convention": "positive order is the slope of log(error) versus log(h)",
            "windows": {
                "all_valid_non_floor": "all positive finite medians above the floor",
                "finest_five": "five largest N values remaining after floor exclusion",
                "finest_four": "four largest N values remaining after floor exclusion",
            },
        },
        "floor_rule": {
            "solver_tolerance": SOLVER_TOLERANCE,
            "plot_floor": PLOT_FLOOR,
            "geometric_metrics": "max(1e-14, 1e-10 * h)",
            "tangent_error": "max(1e-14, 1e-10)",
            "exclusion": "median_error <= numerical_floor",
            "excluded_point_count": sum(bool(point["at_or_below_floor"]) for point in points),
        },
        "median_points": points,
        "fits": list(fit_rows),
        "perturbed_wiggle_sensitivity": build_sensitivity_summary(fit_rows),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-metrics",
        type=Path,
        default=DEFAULT_CASE_METRICS,
        help="case-indexed diagnostics CSV from the July 17 sweep",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for convergence CSV, JSON, and PDF artifacts",
    )
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=25,
        help="required case count per (wiggle, resolution); use 0 to disable",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.case_metrics.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_case_metrics(input_path, expected_cases=args.expected_cases)
    points = list(loaded["points"])
    fit_rows = build_fit_rows(points)

    points_path = output_dir / "ellipse_convergence_points.csv"
    fits_path = output_dir / "ellipse_convergence_fits.csv"
    report_path = output_dir / "ellipse_convergence_report.json"
    figure_path = output_dir / "ellipse_convergence.pdf"

    write_points_csv(points_path, points)
    write_fits_csv(fits_path, fit_rows)
    report = build_report(input_path, loaded, fit_rows)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_convergence(figure_path, points, fit_rows)

    print(f"Selected case rows: {loaded['source_row_count']}")
    print(f"Numerical-floor exclusions: {report['floor_rule']['excluded_point_count']}")
    for metric in METRICS:
        for wiggle in (0.0, 0.1):
            fit = _fit_lookup(fit_rows, metric, wiggle, "finest_five")
            print(
                f"{metric:14s} w={wiggle:<3g} finest-five order="
                f"{float(fit['order']):.4f}, R^2={float(fit['r_squared']):.5f}"
            )
    print(f"Wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
