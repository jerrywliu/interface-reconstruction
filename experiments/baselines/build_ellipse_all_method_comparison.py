#!/usr/bin/env python3
"""Build a matched ellipse comparison across project and external baselines."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments.plotting import add_convergence_order_triangle


DEFAULT_NATIVE_CASES = Path(
    "experiments/baselines/results/common_native_metric_replay_20260813/"
    "case_results.csv"
)
DEFAULT_OURS_CASES = Path(
    "experiments/baselines/results/"
    "ellipse_circular_variants_common_metrics_20260814/case_results.csv"
)
DEFAULT_BASELINE_CASES = (
    Path(
        "experiments/baselines/results/"
        "plvira_project_smoke_20260813_final/case_results.csv"
    ),
    Path(
        "experiments/baselines/results/"
        "pcic_project_smoke_20260813_final/case_results.csv"
    ),
    Path(
        "experiments/baselines/results/"
        "quasi_project_smoke_20260813_qa_fixed/case_results.csv"
    ),
)
DEFAULT_OUTPUT = Path(
    "experiments/baselines/results/ellipse_all_method_comparison_20260814"
)
RESOLUTIONS = (32, 64, 128)
GAP_DISPLAY_FLOOR = 1.0e-7
METRICS = (
    "native_symmetric_hausdorff",
    "geometric_curvature_mean_absolute_error",
    "facet_gap",
)

METHODS = (
    {
        "id": "ours_per_cell",
        "method": "Ours",
        "variant": "per-cell circular",
        "label": "Ours (per-cell circular)",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": ":",
    },
    {
        "id": "ours_graph",
        "method": "Ours",
        "variant": "graph-coordinated circular",
        "label": "Ours (graph-coordinated circular)",
        "color": "#009E73",
        "marker": "s",
        "linestyle": "--",
    },
    {
        "id": "ours_c0",
        "method": "Ours",
        "variant": "graph-coordinated circular + guarded C0",
        "label": "Ours (graph-coordinated circular + C0)",
        "color": "#D55E00",
        "marker": "D",
        "linestyle": "-",
    },
    {
        "id": "plvira",
        "method": "PLVIRA",
        "variant": "PLVIRA",
        "label": "PLVIRA",
        "color": "#CC79A7",
        "marker": "^",
        "linestyle": "-.",
    },
    {
        "id": "pcic_center",
        "method": "PCIC",
        "variant": "bare PCIC (center translation)",
        "label": "PCIC (center translation)",
        "color": "#6B7280",
        "marker": "v",
        "linestyle": "--",
    },
    {
        "id": "pcic_radius",
        "method": "PCIC",
        "variant": "bare PCIC (radius adjustment)",
        "label": "PCIC (radius adjustment)",
        "color": "#8C6D31",
        "marker": "<",
        "linestyle": ":",
    },
    {
        "id": "quasi",
        "method": "QUASI",
        "variant": "QUASI (frozen Cartesian port)",
        "label": "QUASI (frozen port)",
        "color": "#7E57C2",
        "marker": "X",
        "linestyle": "-",
    },
)
METHOD_BY_KEY = {(item["method"], item["variant"]): item for item in METHODS}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if value in (None, "", "nan", "NaN"):
        return math.nan
    return float(value)


def _int(row: Mapping[str, Any], field: str) -> int:
    value = row.get(field)
    return int(float(value)) if value not in (None, "") else 0


def assemble_case_metrics(
    native_cases: Path,
    baseline_cases: Sequence[Path],
    ours_cases: Path,
) -> list[dict[str, Any]]:
    """Join native observables to baseline diagnostics and append project variants."""

    diagnostics: dict[tuple[str, str, int, int], Mapping[str, Any]] = {}
    for path in baseline_cases:
        for row in _read_csv(path):
            if row["benchmark"] != "ellipses":
                continue
            key = (
                row["method"],
                row["variant"],
                _int(row, "cells_per_side"),
                _int(row, "case_index"),
            )
            if key in diagnostics:
                raise ValueError(f"duplicate baseline diagnostic key: {key}")
            diagnostics[key] = row

    result: list[dict[str, Any]] = []
    for native in _read_csv(native_cases):
        if native["benchmark"] != "ellipses":
            continue
        key = (
            native["method"],
            native["variant"],
            _int(native, "cells_per_side"),
            _int(native, "case_index"),
        )
        method = METHOD_BY_KEY.get(key[:2])
        if method is None:
            continue
        diagnostic = diagnostics.get(key)
        if diagnostic is None:
            raise ValueError(f"missing baseline diagnostics for {key}")
        mixed = _int(diagnostic, "mixed_cells")
        reconstructed = _int(diagnostic, "reconstructed_cells") + _int(
            diagnostic, "paper_fallback_cells"
        )
        result.append(
            {
                "method_id": method["id"],
                "display_label": method["label"],
                "method": key[0],
                "variant": key[1],
                "case_index": key[3],
                "cells_per_side": key[2],
                "cell_size": _float(native, "cell_size"),
                "native_symmetric_hausdorff": _float(
                    native, "native_symmetric_hausdorff"
                ),
                "geometric_curvature_mean_absolute_error": _float(
                    native, "geometric_curvature_mean_absolute_error"
                ),
                "facet_gap": _float(diagnostic, "shared_edge_gap_mean"),
                "mixed_cells": mixed,
                "reconstructed_cells": reconstructed,
                "unsupported_cells": _int(diagnostic, "unsupported_cells"),
                "unresolved_cells": _int(diagnostic, "unresolved_cells"),
            }
        )

    for row in _read_csv(ours_cases):
        if row["benchmark"] != "ellipses":
            continue
        key = row["method"], row["variant"]
        method = METHOD_BY_KEY.get(key)
        if method is None:
            continue
        mixed = _int(row, "num_mixed_cells")
        result.append(
            {
                "method_id": method["id"],
                "display_label": method["label"],
                "method": key[0],
                "variant": key[1],
                "case_index": _int(row, "case_index"),
                "cells_per_side": _int(row, "cells_per_side"),
                "cell_size": _float(row, "cell_size"),
                "native_symmetric_hausdorff": _float(row, "native_symmetric_hausdorff"),
                "geometric_curvature_mean_absolute_error": _float(
                    row, "geometric_curvature_mean_absolute_error"
                ),
                "facet_gap": _float(row, "production_facet_gap"),
                "mixed_cells": mixed,
                "reconstructed_cells": mixed,
                "unsupported_cells": 0,
                "unresolved_cells": 0,
            }
        )

    expected = len(METHODS) * len(RESOLUTIONS) * 5
    if len(result) != expected:
        raise ValueError(f"expected {expected} matched case rows, found {len(result)}")
    result.sort(
        key=lambda row: (
            next(
                i
                for i, method in enumerate(METHODS)
                if method["id"] == row["method_id"]
            ),
            row["cells_per_side"],
            row["case_index"],
        )
    )
    return result


def _observed_order(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    resolutions = np.asarray([float(row["cells_per_side"]) for row in rows])
    if len(values) < 3 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        return math.nan
    return float(-np.polyfit(np.log(resolutions), np.log(values), 1)[0])


def summarize_case_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method_id"] == method["id"]]
        method_summary = []
        for resolution in RESOLUTIONS:
            selected = [
                row for row in method_rows if int(row["cells_per_side"]) == resolution
            ]
            item: dict[str, Any] = {
                "method_id": method["id"],
                "display_label": method["label"],
                "method": method["method"],
                "variant": method["variant"],
                "cells_per_side": resolution,
                "case_count": len(selected),
            }
            for metric in METRICS:
                values = np.asarray([float(row[metric]) for row in selected])
                item[f"{metric}_median"] = float(np.median(values))
                item[f"{metric}_q25"] = float(np.quantile(values, 0.25))
                item[f"{metric}_q75"] = float(np.quantile(values, 0.75))
            mixed = sum(int(row["mixed_cells"]) for row in selected)
            reconstructed = sum(int(row["reconstructed_cells"]) for row in selected)
            item["mixed_cells"] = mixed
            item["reconstructed_cells"] = reconstructed
            item["unsupported_cells"] = sum(
                int(row["unsupported_cells"]) for row in selected
            )
            item["unresolved_cells"] = sum(
                int(row["unresolved_cells"]) for row in selected
            )
            item["reconstruction_coverage"] = reconstructed / mixed
            method_summary.append(item)
        for metric in METRICS:
            order = _observed_order(method_summary, f"{metric}_median")
            for item in method_summary:
                item[f"{metric}_fit_order"] = order
        summary.extend(method_summary)
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.7), sharex=True)
    panels = (
        ("native_symmetric_hausdorff", "Native symmetric Hausdorff"),
        (
            "geometric_curvature_mean_absolute_error",
            "Unsigned geometric-curvature MAE",
        ),
        ("facet_gap", "Facet gap"),
    )
    for axis, (metric, ylabel) in zip(axes.ravel()[:3], panels):
        for method in METHODS:
            selected = [row for row in summary if row["method_id"] == method["id"]]
            x = np.asarray([int(row["cells_per_side"]) for row in selected])
            y = np.asarray([float(row[f"{metric}_median"]) for row in selected])
            if metric == "facet_gap" and np.all(y == 0.0):
                y = np.full_like(y, GAP_DISPLAY_FLOOR)
            axis.plot(
                x,
                y,
                color=method["color"],
                marker=method["marker"],
                linestyle=method["linestyle"],
                linewidth=1.35,
                markersize=5.0,
                label=method["label"],
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(RESOLUTIONS, tuple(str(value) for value in RESOLUTIONS))
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)

    for method in METHODS:
        selected = [row for row in summary if row["method_id"] == method["id"]]
        axes[1, 1].plot(
            [int(row["cells_per_side"]) for row in selected],
            [100.0 * float(row["reconstruction_coverage"]) for row in selected],
            color=method["color"],
            marker=method["marker"],
            linestyle=method["linestyle"],
            linewidth=1.35,
            markersize=5.0,
        )
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_xticks(RESOLUTIONS, tuple(str(value) for value in RESOLUTIONS))
    axes[1, 1].set_ylabel("Reconstructed mixed cells (%)")
    axes[1, 1].set_ylim(97.5, 100.15)
    axes[1, 1].grid(True, alpha=0.25)

    add_convergence_order_triangle(
        axes[0, 0], 3.0, order_label="3", anchor=(0.75, 0.10), width=0.13
    )
    add_convergence_order_triangle(
        axes[0, 1], 1.0, order_label="1", anchor=(0.76, 0.13), width=0.13
    )
    add_convergence_order_triangle(
        axes[1, 0], 3.0, order_label="3", anchor=(0.74, 0.24), width=0.13
    )
    axes[1, 0].annotate(
        "QUASI exact zero\n(shown at plotting floor)",
        (64, GAP_DISPLAY_FLOOR),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color="#7E57C2",
        fontsize=7,
    )
    for axis in axes[1, :]:
        axis.set_xlabel("Cells per side")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        frameon=False,
        ncol=4,
        fontsize=7,
    )
    figure.suptitle(
        "Ellipse benchmark: circular variants and higher-order baselines",
        fontsize=11,
        y=0.985,
    )
    figure.text(
        0.5,
        0.935,
        "Medians over five matched Cartesian cases; native geometry and curvature",
        ha="center",
        va="center",
        fontsize=8,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    figure.savefig(
        path,
        bbox_inches="tight",
        dpi=180 if path.suffix.lower() == ".png" else None,
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-cases", type=Path, default=DEFAULT_NATIVE_CASES)
    parser.add_argument("--ours-cases", type=Path, default=DEFAULT_OURS_CASES)
    parser.add_argument(
        "--baseline-cases",
        nargs="+",
        type=Path,
        default=list(DEFAULT_BASELINE_CASES),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = assemble_case_metrics(
        args.native_cases, tuple(args.baseline_cases), args.ours_cases
    )
    summary = summarize_case_metrics(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "case_metrics.csv", rows)
    _write_csv(args.output / "summary.csv", summary)
    plot_summary(summary, args.output / "ellipse_all_methods_metrics.pdf")
    plot_summary(summary, args.output / "ellipse_all_methods_metrics.png")


if __name__ == "__main__":
    main()
